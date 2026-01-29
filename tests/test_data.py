import pytest
import torch
import os
from unittest.mock import patch, MagicMock
from PIL import Image
from src.model.dataset import CatsDogsDataset

@pytest.fixture
def mock_dataset_root(tmp_path):
    # Create a temporary directory structure
    d = tmp_path / "data"
    d.mkdir()
    
    # Create dummy image files
    (d / "cat.0.jpg").touch()
    (d / "dog.1.jpg").touch()
    
    return str(d)

def test_dataset_length(mock_dataset_root):
    dataset = CatsDogsDataset(root_dir=mock_dataset_root)
    assert len(dataset) == 2

@patch("src.model.dataset.Image.open")
def test_getitem(mock_image_open, mock_dataset_root):
    # Mock image opening to return a dummy image
    mock_img = Image.new('RGB', (100, 100))
    mock_image_open.return_value = mock_img
    
    dataset = CatsDogsDataset(root_dir=mock_dataset_root)
    
    # Test getting an item
    img, label = dataset[0] # Should be cat.0.jpg (sorted or not specific, but locally likely)
    
    # Check if tensor
    assert isinstance(img, torch.Tensor)
    # Check shape (default transform resizes to 224x224)
    assert img.shape == (3, 224, 224)
    
    # Check label type
    assert isinstance(label, float)

def test_dataset_empty_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    dataset = CatsDogsDataset(root_dir=str(empty_dir))
    assert len(dataset) == 0
