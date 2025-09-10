import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Dict, Optional, Tuple
from lib.models.losses.matry_CE import Matry_CrossEntropyLoss
import torch.nn.functional as F

class OrthogonalProjection(nn.Module):
    def __init__(self, in_dim=768, out_dim=2048):
        super(OrthogonalProjection, self).__init__()
        # Create a regular linear layer first
        self.projector = nn.Linear(in_dim, out_dim, bias=False)
        # Initialize with orthogonal weights (in float32)
        with torch.no_grad():
            nn.init.orthogonal_(self.projector.weight)

    def forward(self, x):
        # Handle both device and dtype mismatches
        if (x.device != self.projector.weight.device or 
            x.dtype != self.projector.weight.dtype):
            # Move and convert projector weights to match input
            self.projector.weight.data = self.projector.weight.data.to(
                device=x.device, dtype=x.dtype
            )
        
        return self.projector(x)
        
    def orthogonal_regularization_loss(self):
        """
        Optional: Add this to your total loss to maintain orthogonality during training
        L_ortho = ||W^T W - I||_F^2
        """
        W = self.projector.weight  # [out_dim, in_dim]
        if W.shape[0] >= W.shape[1]:  # out_dim >= in_dim
            # W^T W should be identity
            WtW = torch.mm(W.t(), W)  # [in_dim, in_dim]
            I = torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
        else:  # out_dim < in_dim  
            # W W^T should be identity
            WWt = torch.mm(W, W.t())  # [out_dim, out_dim]
            I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            WtW = WWt
        
        ortho_loss = torch.norm(WtW - I, p='fro') ** 2
        return ortho_loss

class NC2(Matry_CrossEntropyLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.kd_rate = args.kd_rate
        self.nesting_list = getattr(args, 'mrl_nesting_list', [128, 256, 512, 768])
        self.mrl_efficient = getattr(args, 'mrl_efficient', False)
        
        # NC2 specific hyperparameters
        self.nc2_lambda = getattr(args, 'nc2_lambda', 1.0)  # Weight for NC2 loss
        self.ortho_lambda = getattr(args, 'ortho_lambda', 1)  # Weight for orthogonality loss
        self.ema_momentum = getattr(args, 'ema_momentum', 0.95)  # EMA momentum (beta in paper)
        self.nc2_alpha = getattr(args, 'nc2_alpha', 0.5)  # Interpolation between batch and EMA
        self.epsilon = 1e-8  # Small constant for numerical stability
        self.num_labels = args.num_labels  # Number of classes for classification
        
        # Initialize projection matrices for each nesting dimension
        # Assuming teacher has hidden size 2048
        self.teacher_hidden_size = 2048
        self.projectors = nn.ModuleDict()
        
        for dim in self.nesting_list:
            self.projectors[f'proj_{dim}'] = OrthogonalProjection(
                in_dim=dim, 
                out_dim=self.teacher_hidden_size
            )
        
        # Teacher targets will be set via set_teacher_targets() method
        self.teacher_targets_set = False
        
        # Initialize accumulators for computing class means (not used in current implementation)
        self.accumulators = {}
        for dim in self.nesting_list:
            self.accumulators[dim] = None
    
    def check_and_load_teacher_targets_from_args(self, args):
        """
        Fallback method to load teacher targets from args if they haven't been set directly
        """
        if not self.teacher_targets_set and hasattr(args, '_teacher_class_means') and hasattr(args, '_teacher_gram'):
            self.set_teacher_targets(args._teacher_class_means, args._teacher_gram)
            return True
        return False
    
    def to(self, device, dtype=None):
        """Override to method to ensure projectors are moved to the correct device and dtype"""
        result = super().to(device, dtype=dtype)
        # Ensure projectors are also moved
        for proj in self.projectors.values():
            proj.to(device, dtype=dtype)
        return result
    
    def cuda(self, device=None):
        """Override cuda method to ensure projectors are moved to CUDA"""
        result = super().cuda(device)
        # Ensure projectors are also moved
        for proj in self.projectors.values():
            proj.cuda(device)
        return result
    
    def set_teacher_targets(self, teacher_class_means, teacher_gram):
        """
        Set pre-computed teacher targets from offline computation.
        This should be called after teacher pre-training is complete and 
        teacher targets have been computed over the full dataset.
        
        Args:
            teacher_class_means (torch.Tensor): Pre-computed teacher class means [num_classes, hidden_size]
            teacher_gram (torch.Tensor): Pre-computed normalized teacher Gram matrix [num_classes, num_classes]
        """
        # Remove any existing buffers with these names
        if hasattr(self, 'teacher_class_means'):
            delattr(self, 'teacher_class_means')
        if hasattr(self, 'teacher_gram'):
            delattr(self, 'teacher_gram')
        
        # Register as non-persistent buffers (no gradients, not saved with model)
        self.register_buffer('teacher_class_means', teacher_class_means.detach().clone(), persistent=False)
        self.register_buffer('teacher_gram', teacher_gram.detach().clone(), persistent=False)
        
        self.teacher_targets_set = True
        
        print(f"Set teacher targets: class_means shape {teacher_class_means.shape}, "
              f"gram shape {teacher_gram.shape}")
    
    def compute_student_class_means(self, embeddings, labels, dim, batch_size):
        """
        Compute student class means for a batch
        """
        device = embeddings.device
        dtype = embeddings.dtype
        num_classes = self.num_labels
        
        # Initialize batch class means
        batch_class_means = torch.zeros(num_classes, dim, device=device, dtype=dtype)
        class_counts = torch.zeros(num_classes, device=device)
        
        # Compute class means for classes present in batch
        for i in range(num_classes):
            mask = (labels == i)
            if mask.sum() > 0:
                batch_class_means[i] = embeddings[mask].mean(dim=0)
                class_counts[i] = mask.sum()
        
        return batch_class_means, class_counts
    
    def update_ema_means(self, batch_means, class_counts, dim):
        """
        Update EMA class means for a specific dimension
        """
        ema_buffer_name = f'ema_means_{dim}'
        
        # Check if EMA means buffer exists and is a tensor
        if (hasattr(self, ema_buffer_name) and 
            getattr(self, ema_buffer_name) is not None and
            isinstance(getattr(self, ema_buffer_name), torch.Tensor)):
            ema_means = getattr(self, ema_buffer_name)
        else:
            # Delete existing attribute if it exists as None (from __init__)
            if hasattr(self, ema_buffer_name):
                delattr(self, ema_buffer_name)
            
            # Initialize EMA means as buffer (no gradients)
            self.register_buffer(ema_buffer_name, batch_means.detach().clone(), persistent=False)
            ema_means = getattr(self, ema_buffer_name)
            return ema_means
        
        # Update EMA for classes present in batch
        mask = (class_counts > 0)
        with torch.no_grad():  # Ensure no gradients for EMA updates
            ema_means[mask] = self.ema_momentum * ema_means[mask] + (1 - self.ema_momentum) * batch_means[mask].detach()
        
        return ema_means
    
    def compute_gram_matrix(self, class_means, normalize=True):
        """
        Compute Gram matrix from class means
        """
        gram = torch.mm(class_means, class_means.t())
        if normalize:
            gram = gram / (torch.norm(gram, p='fro') + self.epsilon)
        return gram
    
    def compute_nc2_batch_loss(self, student_embeddings, labels, dim, teacher_gram_subset):
        """
        Compute mini-batch NC2 loss for a specific dimension
        """
        batch_size = student_embeddings.shape[0]
        device = student_embeddings.device
        
        # Project student embeddings
        projector = self.projectors[f'proj_{dim}']
        projected_embeddings = projector(student_embeddings)
        
        # Compute batch class means
        batch_means, class_counts = self.compute_student_class_means(
            projected_embeddings, labels, self.teacher_hidden_size, batch_size
        )
        
        # Get classes present in batch
        present_classes = (class_counts > 0).nonzero(as_tuple=True)[0]
        
        if len(present_classes) > 1:  # Need at least 2 classes for meaningful Gram matrix
            # Extract means for present classes
            batch_means_subset = batch_means[present_classes]
            
            # Compute student Gram matrix for batch
            student_gram = self.compute_gram_matrix(batch_means_subset, normalize=True)
            
            # Compute NC2 loss (Frobenius norm of difference)
            nc2_loss = torch.norm(student_gram - teacher_gram_subset, p='fro') ** 2
        else:
            nc2_loss = torch.tensor(0.0, device=device, dtype=student_embeddings.dtype)
        print(f"nc2_batch_loss: {nc2_loss}")

        return nc2_loss, batch_means, class_counts, projected_embeddings
    
    def compute_nc2_ema_loss(self, student_embeddings, labels, dim):
        """
        Compute EMA-stabilized NC2 loss for a specific dimension
        """
        device = student_embeddings.device
        
        # Project student embeddings
        projector = self.projectors[f'proj_{dim}']
        projected_embeddings = projector(student_embeddings)
        
        # Compute batch class means
        batch_means, class_counts = self.compute_student_class_means(
            projected_embeddings, labels, self.teacher_hidden_size, student_embeddings.shape[0]
        )
        
        # Update EMA means
        ema_means = self.update_ema_means(batch_means, class_counts, dim)
        
        # Compute EMA-based student Gram
        student_gram_ema = self.compute_gram_matrix(ema_means, normalize=True)
        
        # Compute NC2 loss against full teacher Gram (ensure teacher gram is detached)
        if self.teacher_gram is not None:
            teacher_gram_detached = self.teacher_gram.detach()
            nc2_loss = torch.norm(student_gram_ema - teacher_gram_detached, p='fro') ** 2
        else:
            nc2_loss = torch.tensor(0.0, device=device, dtype=student_embeddings.dtype)
        print("ema_nc2_loss:", nc2_loss)
        return nc2_loss
    
    def compute_nc2_loss(
        self, model_outputs, teacher_outputs, output_data, input_data, distiller, log
    ):
        """
        Compute NC2 distillation loss following the mathematical formulation in the paper
        """
        device = output_data["labels"].device
        labels = output_data["labels"]
        batch_size = labels.shape[0]
        
        # Check if teacher targets have been set, if not try fallback
        if not self.teacher_targets_set:
            # Try to get from distiller's args
            if hasattr(distiller, 'args'):
                self.check_and_load_teacher_targets_from_args(distiller.args)
            
            # If still not set, raise error
            if not self.teacher_targets_set:
                raise RuntimeError(
                    "Teacher targets have not been set. Call set_teacher_targets() with "
                    "pre-computed teacher class means and Gram matrix before training."
                )
        # Get student embeddings for different dimensions
        student_embeddings_dict = self.get_student_embeddings_by_dimension(model_outputs, distiller)

        
        # Initialize loss tensors on the correct device
        total_nc2_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        total_ortho_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        # Compute losses for each nesting dimension
        for dim in self.nesting_list:
            if dim not in student_embeddings_dict:
                continue
            
            student_embeddings = student_embeddings_dict[dim]
            
            # Get classes present in batch for teacher Gram subset
            unique_classes = labels.unique()
            if self.teacher_gram is not None and len(unique_classes) > 1:
                # Detach teacher gram to prevent gradient flow
                teacher_gram_subset = self.teacher_gram.detach()[unique_classes][:, unique_classes]
                gram_norm = torch.norm(teacher_gram_subset, p='fro') + self.epsilon
                teacher_gram_subset = teacher_gram_subset / gram_norm
            else:
                teacher_gram_subset = None
            
            # Compute batch NC2 loss
            if teacher_gram_subset is not None:
                batch_nc2_loss, batch_means, class_counts, projected_embeddings = self.compute_nc2_batch_loss(
                    student_embeddings, labels, dim, teacher_gram_subset
                )
            else:
                batch_nc2_loss = torch.tensor(0.0, device=device, dtype=student_embeddings.dtype)
            
            # Compute EMA NC2 loss
            ema_nc2_loss = self.compute_nc2_ema_loss(student_embeddings, labels, dim)
                
            # Combine batch and EMA losses
            nc2_loss_dim = self.nc2_alpha * batch_nc2_loss + (1 - self.nc2_alpha) * ema_nc2_loss
            
            # Add orthogonality regularization for this dimension's projector
            projector = self.projectors[f'proj_{dim}']
            ortho_loss_dim = projector.orthogonal_regularization_loss()
            
            # Accumulate losses
            total_nc2_loss = total_nc2_loss + nc2_loss_dim
            total_ortho_loss = total_ortho_loss + ortho_loss_dim
            
            # Log per-dimension losses (ensure they are detached)
            log[f'nc2_loss_{dim}'] = nc2_loss_dim.detach()
            log[f'ortho_loss_{dim}'] = ortho_loss_dim.detach()
        
        # Combine all losses
        total_loss = self.nc2_lambda * total_nc2_loss + self.ortho_lambda * total_ortho_loss
        
        # Log total losses (ensure they are detached)
        log['nc2_loss_total'] = total_nc2_loss.detach()
        log['ortho_loss_total'] = total_ortho_loss.detach()
        log['nc2_combined_loss'] = total_loss.detach()
        
        return total_loss, log
    
    def get_student_embeddings_by_dimension(self, model_outputs, distiller):
        """
        Helper function to extract student embeddings for different nesting dimensions
        """
        embeddings_dict = {}
        
        # Try to get embeddings from model outputs
        if hasattr(model_outputs, 'hidden_states') and model_outputs.hidden_states is not None:
            # Use the last hidden state
            last_hidden_state = model_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            full_embeddings = last_hidden_state[:, 0]  # [CLS] token: [batch_size, hidden_size]
        elif isinstance(model_outputs, dict) and 'hidden_states' in model_outputs:
            # Dictionary output format
            last_hidden_state = model_outputs['hidden_states'][-1]
            full_embeddings = last_hidden_state[:, 0]
        else:
            # Alternative: try to get from the model directly if we have access to intermediate outputs
            # This might require modifying the forward pass to return hidden states
            raise ValueError("Cannot extract student embeddings. Ensure model returns hidden_states.")
        
        # Create truncated embeddings for each nesting dimension
        for dim in self.nesting_list:
            if dim <= full_embeddings.shape[1]:
                embeddings_dict[dim] = full_embeddings[:, :dim]
            else:
                # Handle case where requested dimension is larger than available
                # Pad with zeros
                embeddings_dict[dim] = torch.cat([
                    full_embeddings, 
                    torch.zeros(full_embeddings.shape[0], dim - full_embeddings.shape[1], 
                               device=full_embeddings.device, dtype=full_embeddings.dtype)
                ], dim=1)
        
        return embeddings_dict
    
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        target = output_data["labels"]
        
        # Student forward pass with hidden states
        model_outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            output_hidden_states=True  # Ensure we get hidden states
        )
        
        # Extract logits dictionary
        if isinstance(model_outputs, dict) and 'logits' in model_outputs:
            logits_dict = model_outputs['logits']
        else:
            # Fallback for non-MRL models
            logits_dict = {'logits_full': model_outputs.logits if hasattr(model_outputs, 'logits') else model_outputs}

        # Compute loss and accuracy for MRL
        if len(logits_dict) > 1 or any('logits_' in key for key in logits_dict.keys()):
            # MRL case: multiple logits for different nesting dimensions
            loss, nll_loss = self.compute_matryoshka_cross_entropy_loss(logits_dict, target)
            correct = self.compute_matryoshka_accuracy(logits_dict, target)
        else:
            # Regular case: single logits
            logits = list(logits_dict.values())[0]
            loss, nll_loss = self.compute_cross_entropy_loss(logits, target)
            correct = self.compute_accuracy(logits, target)

        # Teacher forward pass (no gradient) - CRITICAL: detach outputs properly
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data["teacher_input_ids"],
                attention_mask=input_data["teacher_attention_mask"],
                output_hidden_states=True
            )
            
            # Detach all teacher outputs to prevent gradient flow
            if hasattr(teacher_outputs, 'hidden_states') and teacher_outputs.hidden_states is not None:
                # Detach each hidden state
                detached_hidden_states = [hs.detach() for hs in teacher_outputs.hidden_states]
                # Create a new object with detached hidden states
                class DetachedOutputs:
                    def __init__(self, hidden_states):
                        self.hidden_states = hidden_states
                teacher_outputs_detached = DetachedOutputs(detached_hidden_states)
            elif isinstance(teacher_outputs, dict) and 'hidden_states' in teacher_outputs:
                teacher_outputs_detached = {
                    'hidden_states': [hs.detach() for hs in teacher_outputs['hidden_states']]
                }
            else:
                teacher_outputs_detached = teacher_outputs
        
        # Initialize log dictionary
        log = {}
        
        # Compute NC2 distillation loss using detached teacher outputs
        kd_loss, log = self.compute_nc2_loss(
            model_outputs, teacher_outputs_detached, output_data, input_data, distiller, log
        )
        print("full_nc2_loss:", kd_loss)
        # Combine losses
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        
        # Ensure loss is a tensor
        log["loss"] = loss.detach() if isinstance(loss, torch.Tensor) else torch.tensor(loss, device=target.device)

        # Compute accuracy from the first/main logits (ensure it's a tensor)
        main_logits = list(logits_dict.values())[0]
        accuracy = self.compute_accuracy(main_logits, target)
        log["accuracy"] = accuracy.detach() if isinstance(accuracy, torch.Tensor) else torch.tensor(accuracy, device=target.device)

        # Update logging output
        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss, logging_output