#!/usr/bin/env python3
"""
Test script for Matryoshka ResNet models
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lib.models.matryoshka import create_matryoshka_resnet

def test_matryoshka_resnet():
    """Test Matryoshka ResNet model creation and forward pass."""
    print("Testing Matryoshka ResNet model...")
    
    # Test parameters
    batch_size = 4
    input_shape = (3, 32, 32)  # CIFAR-10 size
    num_classes = 10
    matryoshka_dims = [64, 128, 256, 512]
    
    # Create model
    model = create_matryoshka_resnet(
        model_name='resnet50',
        matryoshka_dims=matryoshka_dims,
        num_classes=num_classes,
        pretrained=False,
        dropout=0.1
    )
    
    print(f"Model created successfully!")
    print(f"Matryoshka dimensions: {model.matryoshka_dims}")
    
    # Test forward pass
    x = torch.randn(batch_size, *input_shape)
    
    # Test with all dimensions
    print("\nTesting forward pass with all dimensions...")
    outputs = model(x)
    
    for dim, output in outputs.items():
        print(f"Dimension {dim}: output shape {output.shape}")
        assert output.shape == (batch_size, num_classes), f"Wrong output shape for dim {dim}"
    
    # Test with specific dimensions
    print("\nTesting forward pass with specific dimensions...")
    test_dims = [64, 256]
    outputs = model(x, dims=test_dims)
    
    assert len(outputs) == len(test_dims), "Wrong number of outputs"
    for dim in test_dims:
        assert dim in outputs, f"Missing output for dimension {dim}"
        print(f"Dimension {dim}: output shape {outputs[dim].shape}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_matryoshka_resnet()
