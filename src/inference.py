import torch
import argparse
import yaml
from PIL import Image
from torchvision import transforms
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file path (.yaml) where model configuration lives", type=str, default="test_config.yaml")
    parser.add_argument("--image_path", help="Image path to perform inference on")
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # Add batch dim    

def main():
    args = get_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = config["model"]["path"]
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    model = torch.jit.load(model_path).to(device)
    model.eval()
    
    classes = config["inference"]["classes"]
    image_path = args.image_path
    assert os.path.exists(image_path), f"Image file not found: {image_path}"
    image = preprocess_image(image_path).to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(image), dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
        pred_class = classes[pred_idx]
    print(f'[INFO] Predicted class is "{pred_class}" with confidence of {confidence:.2%}')

if __name__ == "__main__":
    main()