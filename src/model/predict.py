import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from .model import SimpleCNN

def load_model(model_path, device='cpu'):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_bytes, device='cpu'):
    tensor = transform_image(image_bytes).to(device)
    with torch.no_grad():
        output = model(tensor)
        probability = output.item()
        
    label = "dog" if probability > 0.5 else "cat"
    confidence = probability if label == "dog" else 1 - probability
    
    return label, confidence
