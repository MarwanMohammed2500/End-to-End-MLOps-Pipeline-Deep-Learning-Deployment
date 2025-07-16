import numpy as np
import random
import torch
from pathlib import Path
def set_seed(seed:int=42):
    """
    sets the seed in all needed libraries (torch, torch.cuda, numpy, and random for general python code)

    Arguments:
    seed:int=42, the seed
    """
    np.random.seed(seed) # Sets NumPy random seed
    torch.manual_seed(seed) # Sets PyTorch's random seed
    torch.cuda.manual_seed(seed) # Sets PyTorch's random seed on CUDA
    torch.cuda.manual_seed_all(seed) # Sets PyTorch's random seed on all objects on the GPU
    random.seed(seed) # Sets the random seed on all python objects
    
def save_model(model, model_name, device):
    try:
        scripted_model = torch.jit.script(model)  # safer long term
    except Exception as e:
        print(f"[WARN] Scripting failed, falling back to tracing: {e}")
        example_input = torch.randn(1, 1, 28, 28).to(device)
        scripted_model = torch.jit.trace(model, example_input)
    base_dir = Path("models/")
    base_dir.mkdir(parents=True, exist_ok=True)
    scripted_model.save(f"models/{model_name}.pt")