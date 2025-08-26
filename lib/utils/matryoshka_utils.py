import torch
import numpy as np
from typing import List, Dict, Optional


class MatryoshkaTrainer:
    """Utility class for Matryoshka model training."""
    
    def __init__(self, matryoshka_dims: List[int], 
                 adaptive_training: bool = True,
                 warmup_epochs: int = 10):
        self.matryoshka_dims = sorted(matryoshka_dims)
        self.adaptive_training = adaptive_training
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def get_training_dims(self, epoch: int) -> List[int]:
        """Get dimensions to train for current epoch."""
        self.current_epoch = epoch
        
        if not self.adaptive_training or epoch < self.warmup_epochs:
            # Train all dimensions during warmup
            return self.matryoshka_dims
        
        # Progressive training: start with largest, gradually add smaller
        progress = (epoch - self.warmup_epochs) / max(1, 100 - self.warmup_epochs)
        progress = min(1.0, progress)
        
        num_dims = max(1, int(progress * len(self.matryoshka_dims)) + 1)
        return self.matryoshka_dims[-num_dims:]
    
    def compute_accuracy(self, outputs: Dict[int, torch.Tensor], 
                        targets: torch.Tensor, topk=(1, 5)):
        """Compute accuracy for each dimension."""
        accuracies = {}
        
        for dim, logits in outputs.items():
            top1, top5 = accuracy(logits, targets, topk=topk)
            accuracies[f'top1_dim_{dim}'] = top1 * 100
            accuracies[f'top5_dim_{dim}'] = top5 * 100
        
        return accuracies


def accuracy(output, target, topk=(1,)):
    """Compute accuracy for specified top-k values."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res
