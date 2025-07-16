import torch
import argparse
import yaml
from PIL import Image
from torchvision import transforms
import os

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    image = Image.open(image)
    return transform(image).unsqueeze(0)  # Add batch dim    

def predict(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = "models/FashionMNISTModel_V0.pt"
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    model = torch.jit.load(model_path).to(device)
    model.eval()
    
    classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Sneaker','Bag','Ankle boot']
    
    # assert os.path.exists(image_path), f"Image file not found: {image_path}"
    image = preprocess_image(image).to(device)
    with torch.inference_mode():
        pred_idx = torch.softmax(model(image), dim=1).argmax(dim=1).item()
        pred_class = classes[pred_idx]
    return pred_class