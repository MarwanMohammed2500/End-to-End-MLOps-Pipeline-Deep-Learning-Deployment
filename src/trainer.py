from dataloader import get_dataloaders
from metrics import set_metrics, classification_report
from model import CNNModelV0, MODEL_REGISTRY
from plot import plot_predictions, plot_images
from preproc import merge_labels
from train import train_test_loop
from utils import set_seed, save_model
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import sys
import argparse
import yaml
writer = SummaryWriter()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file path (.yaml) where model configuration lives", type=str, default="train_config.yaml")
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = get_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Set device to {device}")
    
    model_class = config["model"]["name"]
    try:
        model_reg = MODEL_REGISTRY[model_class]
    except:
        print(f"[ERROR] Model {model_class} is not a valid model class. Please choose only one of {MODEL_REGISTRY.keys()}")
        sys.exit(1)
    
    set_seed(config["training"]["seed"])
    train_dataloader, test_dataloader = get_dataloaders(batch_size=config["training"]["batch_size"])
    train_dataloader.dataset.targets = merge_labels(train_dataloader.dataset.targets)
    test_dataloader.dataset.targets = merge_labels(test_dataloader.dataset.targets)
    print(f"[INFO] There are {len(train_dataloader)} batches of training data")
    print(f"[INFO] There are {len(test_dataloader)} batches of testing data")
    # classes = train_dataloader.dataset.classes
    
    hidden_features = config["training"]["hidden_features"]
    in_features = hidden_features*4*4
    out_features = config["training"]["out_features"]
    kernel_size = config["training"]["kernel_size"]
    stride = config["training"]["stride"]
    padding = config["training"]["padding"]

    model = model_reg(
        in_features=in_features,
        out_features=out_features,
        hidden_feauters=hidden_features,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
                    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    
    lr = 1e-3
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=lr
                        )
    
    epochs = config["training"]["epochs"]
    verbose = config["training"]["verbose"]
    model = train_test_loop(
        epochs=epochs, model=model, loss_fn=loss_fn, optimizer=optimizer,
        train_dataloader=train_dataloader, test_dataloader=test_dataloader,
        writer=writer, device=device, verbose=verbose
        )
    print(f"[INFO] Training Finished. Logs saved to {writer.log_dir}")
    writer.close()
    
    # Save the model
    model_name = config["save_model"]["model_name"]
    save_path = f"models/{model_name}.pt"
    print(f"[INFO] Saving model {model_name} to path '{save_path}'")
    save_model(model, model_name, device)
    

if __name__ == "__main__":
    main()