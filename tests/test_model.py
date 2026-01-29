import torch
import pytest
from src.model.model import SimpleCNN

def test_model_structure():
    model = SimpleCNN()
    # Check if model has expected layers
    assert hasattr(model, 'conv1')
    assert hasattr(model, 'fc1')
    assert hasattr(model, 'fc3')

def test_model_forward_pass():
    model = SimpleCNN()
    # Create a dummy input tensor: Batch size 1, 3 channels, 224x224 height/width
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Run forward pass
    output = model(dummy_input)
    
    # Check output shape: Should be (1, 1) for binary classification
    assert output.shape == (1, 1)
    
    # Check output range: Sigmoid output should be between 0 and 1
    assert 0 <= output.item() <= 1
