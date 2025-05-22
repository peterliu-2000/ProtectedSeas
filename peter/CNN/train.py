import argparse
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from tqdm import tqdm
from dataloader_imgs import get_type_datasets
from engine import train_model, save_model
from models import *

sys.path.append(os.path.abspath('..'))
from core.DICT import TYPE2NUM
from torch.optim import lr_scheduler
import re
import time

"""
File to systematically train CNN models on track images with various configurations
Currently only trained on non-transit/stopped vessels from radar/tagged detections
"""

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name: str, num_classes: int, **kwargs):
    name = model_name.lower()
    
    model_constructors = {
        "resnet18": get_resnet18_classifier,
        "resnet50": get_resnet50_classifier,
        "resnet101": get_resnet101_classifier,
    }

    if name in model_constructors:
        return model_constructors[name](num_classes, **kwargs)
    
    # Check for exact match of saved weights file
    model_path = os.path.join("models", f"{model_name}.pth")
    if os.path.exists(model_path):
        print(f"Found saved weights at {model_path}, attempting to load...")

        # Extract architecture prefix using regex (must start with a known architecture)
        match = re.match(r"^(resnet18|resnet50|resnet101)", model_name)
        if match:
            arch = match.group(1)
            model = model_constructors[arch](num_classes, **kwargs)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Weights successfully loaded into {arch}.")
            return model
        else:
            raise ValueError(f"Could not determine architecture from model name: '{model_name}'")
    else:
        raise ValueError(f"Unsupported model '{model_name}' and no saved weights found at {model_path}")

def get_optimizer(optimizer_name: str, model_params, lr: float):
    if optimizer_name.lower() == "adam":
        return optim.Adam(model_params, lr=lr)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
def get_scheduler(scheduler_name: str, optimizer, num_epochs: int = 10):
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "none":
        return None
    elif scheduler_name == "step":
        return lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif scheduler_name == "multistep":
        return lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)
    elif scheduler_name == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == "reduce_on_plateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def main(args):
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths and constants
    label_path = '../../data/labels/full_non_transit_stopped_radar_labels.csv'
    image_path = 'nontransit_track_images/'
    BATCH_SIZE = 32
    NUM_CLASSES = len(TYPE2NUM)

    # Data Loaders
    train_loader, val_loader, test_loader = get_type_datasets(label_path, image_path, batch_size=BATCH_SIZE)

    # Model & Optimizer
    model = get_model(args.model, NUM_CLASSES, dropout = args.dropout).to(device)
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr)

    # Train config
    train_config = {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "device": device,
        "start_epoch": args.start_epoch,
        "num_epochs": args.epochs,
        "scheduler": None,
    }

    # Train
    start_time = time.time()
    TRAIN_LOSSES, TRAIN_ACC, VAL_LOSSES, VAL_ACC, best_model_wts, best_val_loss = train_model(**train_config)
    end_time = time.time()
    training_time = end_time - start_time
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)
    print(f"Training time: {minutes} minutes and {seconds} seconds for {args.epochs} epochs")

    # Save
    model_name = f"{args.model}_dropout{args.dropout}_{args.optimizer}_{args.start_epoch + args.epochs}"
    save_model(TRAIN_LOSSES, TRAIN_ACC, VAL_LOSSES, VAL_ACC, best_val_loss, best_model_wts, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classification model on radar image tracks.")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name (e.g., resnet50)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer (adam or sgd)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()
    main(args)
