import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
import math


class MatryoshkaLinear(nn.Module):
    """Linear layer that supports multiple nested output dimensions."""
    
    def __init__(self, in_features: int, out_features: int, 
                 matryoshka_dims: List[int], bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.matryoshka_dims = sorted(matryoshka_dims)
        
        # Ensure all dimensions are valid
        assert all(dim <= out_features for dim in matryoshka_dims)
        assert out_features in matryoshka_dims
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x, dim: Optional[int] = None):
        if dim is None:
            dim = self.out_features
        
        assert dim in self.matryoshka_dims, f"Dimension {dim} not in matryoshka_dims {self.matryoshka_dims}"
        
        weight = self.weight[:dim]
        bias = self.bias[:dim] if self.bias is not None else None
        
        return F.linear(x, weight, bias)


class MatryoshkaRepresentation(nn.Module):
    """Module that creates nested representations from backbone features."""
    
    def __init__(self, backbone_dim: int, matryoshka_dims: List[int], 
                 num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.matryoshka_dims = sorted(matryoshka_dims)
        self.num_classes = num_classes
        
        # Feature projection to create nested representations
        self.feature_proj = MatryoshkaLinear(
            backbone_dim, max(matryoshka_dims), matryoshka_dims, bias=False
        )
        
        # Normalization for each dimension
        self.layer_norms = nn.ModuleDict({
            str(dim): nn.LayerNorm(dim) for dim in matryoshka_dims
        })
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Classification heads for each dimension
        self.classifiers = nn.ModuleDict({
            str(dim): nn.Linear(dim, num_classes) for dim in matryoshka_dims
        })
    
    def forward(self, x, dims: Optional[List[int]] = None):
        """
        Args:
            x: Input features from backbone [B, backbone_dim]
            dims: List of dimensions to compute. If None, compute all.
        
        Returns:
            Dictionary mapping dimension to logits
        """
        if dims is None:
            dims = self.matryoshka_dims
        
        outputs = {}
        
        for dim in dims:
            # Get nested representation
            features = self.feature_proj(x, dim)
            features = self.layer_norms[str(dim)](features)
            features = self.dropout(features)
            
            # Classify
            logits = self.classifiers[str(dim)](features)
            outputs[dim] = logits
        
        return outputs


class MatryoshkaModel(nn.Module):
    """Complete Matryoshka model with backbone and nested representations."""
    
    def __init__(self, backbone: nn.Module, backbone_dim: int,
                 matryoshka_dims: List[int], num_classes: int,
                 dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.matryoshka_head = MatryoshkaRepresentation(
            backbone_dim, matryoshka_dims, num_classes, dropout
        )
        self.matryoshka_dims = matryoshka_dims
        
    def forward(self, x, dims: Optional[List[int]] = None):
        # Extract features from backbone
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        # Get nested representations
        return self.matryoshka_head(features, dims)
    
    def get_classifier(self):
        """Return the largest classifier for compatibility."""
        largest_dim = max(self.matryoshka_dims)
        return self.matryoshka_head.classifiers[str(largest_dim)]


def create_matryoshka_resnet(model_name: str = 'resnet50', 
                           matryoshka_dims: List[int] = [64, 128, 256, 512],
                           num_classes: int = 1000,
                           pretrained: bool = False,
                           dropout: float = 0.1):
    """Create a Matryoshka ResNet model."""
    from . import resnet
    import torchvision.models as models
    
    # Create backbone
    if model_name == 'resnet18':
        if pretrained:
            backbone = models.resnet18(pretrained=True)
        else:
            backbone = resnet.resnet18(pretrained=False)
        backbone_dim = 512
    elif model_name == 'resnet34':
        if pretrained:
            backbone = models.resnet34(pretrained=True)
        else:
            backbone = resnet.resnet34(pretrained=False)
        backbone_dim = 512
    elif model_name == 'resnet50':
        if pretrained:
            backbone = models.resnet50(pretrained=True)
        else:
            backbone = resnet.resnet50(pretrained=False)
        backbone_dim = 2048
    elif model_name == 'resnet101':
        if pretrained:
            backbone = models.resnet101(pretrained=True)
        else:
            backbone = resnet.resnet101(pretrained=False)
        backbone_dim = 2048
    elif model_name == 'resnet152':
        if pretrained:
            backbone = models.resnet152(pretrained=True)
        else:
            backbone = resnet.resnet152(pretrained=False)
        backbone_dim = 2048
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Remove the final classification layer
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    
    # Create Matryoshka model
    model = MatryoshkaModel(
        backbone=backbone,
        backbone_dim=backbone_dim,
        matryoshka_dims=matryoshka_dims,
        num_classes=num_classes,
        dropout=dropout
    )
    
    return model
