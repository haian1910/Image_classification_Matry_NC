import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class MatryoshkaLoss(nn.Module):
    """Loss function for Matryoshka models with nested supervision."""
    
    def __init__(self, 
                 matryoshka_dims: List[int],
                 loss_weights: Optional[List[float]] = None,
                 base_loss_fn: nn.Module = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.matryoshka_dims = sorted(matryoshka_dims)
        
        # Set loss weights (higher weight for larger dimensions)
        if loss_weights is None:
            # Default: exponentially increasing weights
            self.loss_weights = [2**i for i in range(len(matryoshka_dims))]
        else:
            assert len(loss_weights) == len(matryoshka_dims)
            self.loss_weights = loss_weights
        
        # Normalize weights
        total_weight = sum(self.loss_weights)
        self.loss_weights = [w / total_weight for w in self.loss_weights]
        
        # Base loss function
        if base_loss_fn is None:
            if label_smoothing > 0:
                from .cross_entropy import CrossEntropyLabelSmooth
                self.base_loss_fn = CrossEntropyLabelSmooth(
                    epsilon=label_smoothing
                )
            else:
                self.base_loss_fn = nn.CrossEntropyLoss()
        else:
            self.base_loss_fn = base_loss_fn
    
    def forward(self, outputs: Dict[int, torch.Tensor], targets: torch.Tensor):
        """
        Args:
            outputs: Dictionary mapping dimension to logits
            targets: Ground truth labels [B]
        
        Returns:
            Total weighted loss
        """
        total_loss = 0.0
        loss_dict = {}
        
        for i, dim in enumerate(self.matryoshka_dims):
            if dim in outputs:
                loss = self.base_loss_fn(outputs[dim], targets)
                weighted_loss = loss * self.loss_weights[i]
                total_loss += weighted_loss
                loss_dict[f'loss_dim_{dim}'] = loss.item()
        
        # Store individual losses for logging
        self.loss_dict = loss_dict
        
        return total_loss
