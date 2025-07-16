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