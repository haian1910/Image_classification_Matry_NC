import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal
from functools import partial

# Try relative imports first, fall back to absolute imports
try:
    from .kl_div import KLDivergence
    from .dist_kd import DIST
    from .diffkd import DiffKD
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from kl_div import KLDivergence
    from dist_kd import DIST
    from diffkd import DiffKD

import logging
logger = logging.getLogger()


class OrthogonalProjection(nn.Module):
    """Orthogonal projection layer using PyTorch's built-in orthogonal parameterization.
    
    Maps student features to teacher dimension using guaranteed orthogonal matrices.
    Uses torch.nn.utils.parametrizations.orthogonal for true orthogonality.
    """
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Create linear layer
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        
        # Apply orthogonal parameterization to ensure the weight matrix is always orthogonal
        # This automatically maintains orthogonality during training
        orthogonal(self.linear, 'weight')
        
    def forward(self, x):
        """Apply orthogonal projection."""
        return self.linear(x)


class NC1Loss(nn.Module):
    """Neural Collapse (NC1) distillation loss for Matryoshka models.
    
    Uses PyTorch's built-in orthogonal parameterization to map student dimensions 
    to teacher space, then applies MSE loss. Final loss is sum over all dimensions.
    """
    
    def __init__(self, teacher_dim, student_dims):
        super().__init__()
        self.teacher_dim = teacher_dim
        self.student_dims = student_dims  # List of Matryoshka dimensions
        
        # Orthogonal projections from each student dimension to teacher dimension
        # Using PyTorch's orthogonal parameterization for guaranteed orthogonality
        self.projections = nn.ModuleDict()
        for dim in student_dims:
            self.projections[str(dim)] = OrthogonalProjection(dim, teacher_dim)
        
    def forward(self, teacher_features, student_features_dict):
        """Compute NC1 distillation loss for all Matryoshka dimensions.
        
        Args:
            teacher_features: Teacher feature tensor [B, teacher_dim]
            student_features_dict: Dict of {dim: features} for each Matryoshka dimension
        
        Returns:
            Dictionary containing individual losses and total loss
        """
        total_loss = 0.0
        losses = {}
        
        # Compute MSE losses for each dimension and sum them
        for dim_str, student_features in student_features_dict.items():
            dim = int(dim_str)
            
            # Project student features to teacher dimension using orthogonal projection
            student_projected = self.projections[dim_str](student_features)
            
            # MSE loss between projected student features and teacher features
            mse_loss = F.mse_loss(student_projected, teacher_features)
            
            losses[f'nc1_loss_{dim}'] = mse_loss
            total_loss += mse_loss
        
        # Total loss is sum over all Matryoshka dimensions (no averaging)
        losses['nc1_loss_total'] = total_loss
        
        return losses


KD_MODULES = {
    'cifar_wrn_40_1': dict(modules=['relu', 'fc'], channels=[64, 100]),
    'cifar_wrn_40_2': dict(modules=['relu', 'fc'], channels=[128, 100]),
    'cifar_resnet56': dict(modules=['layer3', 'fc'], channels=[4096, 100]),  # Fixed for CIFAR-100: layer3 = 64*8*8=4096, fc=100 for CIFAR-100
    'cifar_resnet20': dict(modules=['layer3', 'fc'], channels=[4096, 100]),  # Fixed for CIFAR-100: layer3 = 64*8*8=4096, fc=100 for CIFAR-100
    'cifar_resnet110': dict(modules=['avgpool', 'fc'], channels=[64, 100]),  # avgpool gives the final features before classification (64 features after pooling)
    'cifar_resnet101': dict(modules=['layer3', 'fc'], channels=[4096, 100]),  # CIFAR ResNet101, same dimensions as other CIFAR models, CIFAR-100
    'matryoshka_cifar_resnet20': dict(modules=['module.avgpool_flatten', 'module.matryoshka_head'], channels=[64, 100]),  # Hook the final features before matryoshka head
    'tv_resnet50': dict(modules=['layer4', 'fc'], channels=[2048, 1000]),
    'tv_resnet34': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'tv_resnet18': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'tv_resnet101': dict(modules=['layer4', 'fc'], channels=[2048, 1000]),
    'resnet18': dict(modules=['layer4', 'fc'], channels=[512, 1000]),
    'resnet101': dict(modules=['layer4', 'fc'], channels=[2048, 10]),  # Updated for CIFAR-10: layer4 outputs 2048 channels with small spatial size
    'matryoshka_resnet18': dict(modules=['module.backbone.3', 'module.matryoshka_head'], channels=[4096, 10]),  # Fixed: DDP + Matryoshka structure for CIFAR (8x8x64=4096)
    'matryoshka_resnet50': dict(modules=['module.backbone.3', 'module.matryoshka_head'], channels=[2048, 1000]),  # Fixed: DDP + Matryoshka structure
    'tv_mobilenet_v2': dict(modules=['features.18', 'classifier'], channels=[1280, 1000]),
    'nas_model': dict(modules=['features.conv_out', 'classifier'], channels=[1280, 1000]),  # mbv2
    'timm_tf_efficientnet_b0': dict(modules=['conv_head', 'classifier'], channels=[1280, 1000]),
    'mobilenet_v1': dict(modules=['model.13', 'fc'], channels=[1024, 1000]),
    'timm_swin_large_patch4_window7_224': dict(modules=['norm', 'head'], channels=[1536, 1000]),
    'timm_swin_tiny_patch4_window7_224': dict(modules=['norm', 'head'], channels=[768, 1000]),
}



class KDLoss():
    '''
    kd loss wrapper.
    '''

    def __init__(
        self,
        student,
        teacher,
        student_name,
        teacher_name,
        ori_loss,
        kd_method='kdt4',
        ori_loss_weight=1.0,
        kd_loss_weight=1.0,
        kd_loss_kwargs={}
    ):
        self.student = student
        self.teacher = teacher
        self.ori_loss = ori_loss
        self.ori_loss_weight = ori_loss_weight
        self.kd_method = kd_method
        self.kd_loss_weight = kd_loss_weight

        self._teacher_out = None
        self._student_out = None

        # init kd loss
        # module keys for distillation. '': output logits
        teacher_modules = ['',]
        student_modules = ['',]
        if kd_method == 'kd':
            self.kd_loss = KLDivergence(tau=4)
        elif kd_method == 'dist':
            self.kd_loss = DIST(beta=1, gamma=1, tau=1)
        elif kd_method.startswith('dist_t'):
            tau = float(kd_method[6:])
            self.kd_loss = DIST(beta=1, gamma=1, tau=tau)
        elif kd_method.startswith('kdt'):
            tau = float(kd_method[3:])
            self.kd_loss = KLDivergence(tau)
        elif kd_method == 'diffkd':
            # get configs
            ae_channels = kd_loss_kwargs.get('ae_channels', 1024)
            use_ae = kd_loss_kwargs.get('use_ae', True)
            tau = kd_loss_kwargs.get('tau', 1)

            print(kd_loss_kwargs)
            kernel_sizes = [3, 1]  # distillation on feature and logits
            student_modules = KD_MODULES[student_name]['modules']
            student_channels = KD_MODULES[student_name]['channels']
            teacher_modules = KD_MODULES[teacher_name]['modules']
            teacher_channels = KD_MODULES[teacher_name]['channels']
            self.diff = nn.ModuleDict()
            self.kd_loss = nn.ModuleDict()
            for tm, tc, sc, ks in zip(teacher_modules, teacher_channels, student_channels, kernel_sizes):
                self.diff[tm] = DiffKD(sc, tc, kernel_size=ks, use_ae=(ks!=1) and use_ae, ae_channels=ae_channels)
                self.kd_loss[tm] = nn.MSELoss() if ks != 1 else KLDivergence(tau=tau)
            self.diff.cuda()
            # add diff module to student for optimization
            self.student._diff = self.diff
        elif kd_method == 'mse':
            # distillation on feature
            student_modules = KD_MODULES[student_name]['modules'][:1]
            student_channels = KD_MODULES[student_name]['channels'][:1]
            teacher_modules = KD_MODULES[teacher_name]['modules'][:1]
            teacher_channels = KD_MODULES[teacher_name]['channels'][:1]
            self.kd_loss = nn.MSELoss()
            self.align = nn.Conv2d(student_channels[0], teacher_channels[0], 1)
            self.align.cuda()
            # add align module to student for optimization
            self.student._align = self.align
        elif kd_method == 'nc1':
            # NC1 distillation on features - maps student dimensions to teacher space
            # Uses PyTorch's orthogonal parameterization for guaranteed orthogonality
            
            # Teacher: single feature extraction point
            teacher_modules = KD_MODULES[teacher_name]['modules'][:1]  # feature only
            teacher_channels = KD_MODULES[teacher_name]['channels'][:1]
            
            # Student: we need to get features for all Matryoshka dimensions
            # For Matryoshka models, we'll extract the backbone features and then
            # compute projections for each dimension
            student_modules = KD_MODULES[student_name]['modules'][:1]  # backbone features only
            
            # Get Matryoshka dimensions from the student model
            student_dims = getattr(self.student.module, 'matryoshka_dims', [128, 256, 384, 512])
            if hasattr(self.student, 'module') and hasattr(self.student.module, 'matryoshka_head'):
                # Get actual dimensions from the model
                student_dims = list(self.student.module.matryoshka_head.classifiers.keys())
                student_dims = [int(dim) for dim in student_dims]
            
            # Create NC1 loss with orthogonal projections (student_dims -> teacher_dim)
            self.kd_loss = NC1Loss(
                teacher_dim=teacher_channels[0],
                student_dims=student_dims
            )
            self.kd_loss.cuda()
            # add NC1 module to student for optimization
            self.student._nc1_loss = self.kd_loss
            
            # Store student dimensions for later use
            self.matryoshka_dims = student_dims
        else:
            raise RuntimeError(f'KD method {kd_method} not found.')

        # register forward hook
        # dicts that store distillation outputs of student and teacher
        self._teacher_out = {}
        self._student_out = {}

        for student_module, teacher_module in zip(student_modules, teacher_modules):
            self._register_forward_hook(student, student_module, teacher=False)
            self._register_forward_hook(teacher, teacher_module, teacher=True)
        self.student_modules = student_modules
        self.teacher_modules = teacher_modules

        teacher.eval()
        self._iter = 0

    def __call__(self, x, targets):
        with torch.no_grad():
            t_logits = self.teacher(x)

        # compute ori loss of student
        logits = self.student(x)
        ori_loss = self.ori_loss(logits, targets)

        kd_loss = 0

        for tm, sm in zip(self.teacher_modules, self.student_modules):

            # transform student feature
            if self.kd_method == 'diffkd':
                self._student_out[sm], self._teacher_out[tm], diff_loss, ae_loss = \
                    self.diff[tm](self._reshape_BCHW(self._student_out[sm]), self._reshape_BCHW(self._teacher_out[tm]))
            if hasattr(self, 'align'):
                self._student_out[sm] = self.align(self._student_out[sm])
            
            # Special handling for NC1 - extract features for all Matryoshka dimensions
            if self.kd_method == 'nc1':
                teacher_feat = self._teacher_out[tm]
                student_backbone_feat = self._student_out[sm]  # Raw backbone features
                
                # Handle tuples from forward hooks
                if isinstance(teacher_feat, tuple):
                    teacher_feat = teacher_feat[0] if len(teacher_feat) > 0 else teacher_feat
                if isinstance(student_backbone_feat, tuple):
                    student_backbone_feat = student_backbone_feat[0] if len(student_backbone_feat) > 0 else student_backbone_feat
                
                # Ensure we have tensors
                if not isinstance(teacher_feat, torch.Tensor):
                    raise TypeError(f"Expected tensor for teacher features, got {type(teacher_feat)}")
                if not isinstance(student_backbone_feat, torch.Tensor):
                    raise TypeError(f"Expected tensor for student features, got {type(student_backbone_feat)}")
                
                # Flatten spatial dimensions if needed
                if teacher_feat.dim() > 2:
                    teacher_feat = torch.flatten(teacher_feat, 1)
                if student_backbone_feat.dim() > 2:
                    student_backbone_feat = torch.flatten(student_backbone_feat, 1)
                
                # Extract features for each Matryoshka dimension from the backbone features
                student_features_dict = {}
                for dim in self.matryoshka_dims:
                    # Take the first 'dim' features from the backbone
                    if student_backbone_feat.size(1) >= dim:
                        student_features_dict[str(dim)] = student_backbone_feat[:, :dim]
                    else:
                        # If backbone features are smaller than requested dim, pad with zeros
                        padded_feat = torch.zeros(student_backbone_feat.size(0), dim, 
                                                device=student_backbone_feat.device, 
                                                dtype=student_backbone_feat.dtype)
                        padded_feat[:, :student_backbone_feat.size(1)] = student_backbone_feat
                        student_features_dict[str(dim)] = padded_feat
                
                # Use the NC1 loss: project student to teacher dimensions with orthogonal mapping
                nc1_losses = self.kd_loss(teacher_feat, student_features_dict)
                
                # Extract the total loss for backward pass (sum of all dimension losses)
                kd_loss_ = nc1_losses['nc1_loss_total']
                
                # Log individual dimension losses for monitoring
                if self._iter % 50 == 0:
                    # Log MSE losses per dimension
                    dim_losses = [f"dim_{k.split('_')[-1]}: {v.item():.4f}" for k, v in nc1_losses.items() if k.startswith('nc1_loss_') and not k.endswith('_total')]
                    total_loss = f"total: {nc1_losses['nc1_loss_total'].item():.4f}"
                    
                    loss_str = " | ".join(dim_losses + [total_loss])
                    logger.info(f'[{tm}-{sm}] NC1 losses: {loss_str}')
            
            # Special handling for other methods that need flattening
            elif self.kd_method in ['mse']:
                student_feat = self._student_out[sm]
                teacher_feat = self._teacher_out[tm]
                
                # Handle tuples from forward hooks
                if isinstance(student_feat, tuple):
                    student_feat = student_feat[0] if len(student_feat) > 0 else student_feat
                if isinstance(teacher_feat, tuple):
                    teacher_feat = teacher_feat[0] if len(teacher_feat) > 0 else teacher_feat
                
                # Ensure we have tensors
                if not isinstance(student_feat, torch.Tensor):
                    raise TypeError(f"Expected tensor for student features, got {type(student_feat)}")
                if not isinstance(teacher_feat, torch.Tensor):
                    raise TypeError(f"Expected tensor for teacher features, got {type(teacher_feat)}")
                
                # Flatten spatial dimensions if needed
                if student_feat.dim() > 2:
                    student_feat = torch.flatten(student_feat, 1)
                if teacher_feat.dim() > 2:
                    teacher_feat = torch.flatten(teacher_feat, 1)
                    
                # Use flattened features
                self._student_out[sm] = student_feat
                self._teacher_out[tm] = teacher_feat
                
                # compute kd loss
                if isinstance(self.kd_loss, nn.ModuleDict):
                    kd_loss_ = self.kd_loss[tm](self._student_out[sm], self._teacher_out[tm])
                else:
                    kd_loss_ = self.kd_loss(self._student_out[sm], self._teacher_out[tm])
            else:
                # Standard KD loss computation
                if isinstance(self.kd_loss, nn.ModuleDict):
                    kd_loss_ = self.kd_loss[tm](self._student_out[sm], self._teacher_out[tm])
                else:
                    kd_loss_ = self.kd_loss(self._student_out[sm], self._teacher_out[tm])

            if self.kd_method == 'diffkd':
                # add additional losses in DiffKD
                if ae_loss is not None:
                    kd_loss += diff_loss + ae_loss
                    if self._iter % 50 == 0:
                        logger.info(f'[{tm}-{sm}] KD ({self.kd_method}) loss: {kd_loss_.item():.4f} Diff loss: {diff_loss.item():.4f} AE loss: {ae_loss.item():.4f}')
                else:
                    kd_loss += diff_loss
                    if self._iter % 50 == 0:
                        logger.info(f'[{tm}-{sm}] KD ({self.kd_method}) loss: {kd_loss_.item():.4f} Diff loss: {diff_loss.item():.4f}')
            else:
                if self._iter % 50 == 0:
                    logger.info(f'[{tm}-{sm}] KD ({self.kd_method}) loss: {kd_loss_.item():.4f}')
            kd_loss += kd_loss_

        self._teacher_out = {}
        self._student_out = {}

        self._iter += 1
        return ori_loss * self.ori_loss_weight + kd_loss * self.kd_loss_weight

    def _register_forward_hook(self, model, name, teacher=False):
        if name == '':
            # use the output of model
            model.register_forward_hook(partial(self._forward_hook, name=name, teacher=teacher))
        else:
            module = None
            for k, m in model.named_modules():
                if k == name:
                    module = m
                    break
            
            if module is None:
                # Debug: print available modules
                available_modules = [k for k, _ in model.named_modules()]
                model_type = "teacher" if teacher else "student"
                
                # Try to find similar module names
                similar_modules = [m for m in available_modules if any(part in m for part in name.split('.'))]
                
                error_msg = f"Module '{name}' not found in {model_type} model.\n"
                error_msg += f"Available modules ({len(available_modules)} total): {available_modules[:15]}...\n"
                if similar_modules:
                    error_msg += f"Similar modules found: {similar_modules[:10]}"
                
                raise RuntimeError(error_msg)
            
            module.register_forward_hook(partial(self._forward_hook, name=name, teacher=teacher))

    def _forward_hook(self, module, input, output, name, teacher=False):
        if teacher:
            self._teacher_out[name] = output[0] if len(output) == 1 else output
        else:
            self._student_out[name] = output[0] if len(output) == 1 else output

    def _reshape_BCHW(self, x):
        """
        Reshape a 2d (B, C) or 3d (B, N, C) tensor to 4d BCHW format.
        """
        if x.dim() == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)
        elif x.dim() == 3:
            # swin [B, N, C]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x