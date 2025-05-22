from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch   
import copy
from tqdm import tqdm
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                optimizer: optim.Optimizer, 
                device: torch.device,
                start_epoch: int,
                num_epochs: int,
                scheduler: None, ) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Output:
        TRAIN_LOSSES: list of training losses
        TRAIN_ACC: list of training accuracies
        VAL_LOSSES: list of validation losses
        VAL_ACC: list of validation accuracies
        best_model_wts: best model weights
        best_val_loss: best validation loss
    """
    
    TRAIN_LOSSES = []
    TRAIN_ACC = []
    VAL_LOSSES = []
    VAL_ACC = []

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    criterion = nn.CrossEntropyLoss()

    #tensorboard logs:
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    lr = optimizer.param_groups[0]['lr']
    scheduler_name = scheduler.__class__.__name__ if scheduler is not None else "None"
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_name = f"{model_name}_{optimizer_name}_lr{lr}_sched_{scheduler_name}_epochs{num_epochs}_{datetime_str}"
    tensorboard_dir = os.path.join("tensorboard_logs", log_name)
    writer = SummaryWriter(log_dir = tensorboard_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(device), targets.to(device) 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()    

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
        train_loss = running_loss / total
        train_acc = correct / total
        TRAIN_LOSSES.append(train_loss)
        TRAIN_ACC.append(train_acc)

        if scheduler is not None:
            scheduler.step()

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():   
            for inputs, targets in tqdm(val_loader, desc="Evaluating", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        val_loss = running_loss / total
        val_acc = correct / total
        VAL_LOSSES.append(val_loss)
        VAL_ACC.append(val_acc) 

        # TensorBoard logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch {start_epoch+epoch+1}/{start_epoch+num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    return TRAIN_LOSSES, TRAIN_ACC, VAL_LOSSES, VAL_ACC, best_model_wts, best_val_loss

def test_model(model: nn.Module,
               test_loader: DataLoader,
               device: torch.device,
               loss_fn: nn.Module = nn.CrossEntropyLoss()) -> tuple[float, float]:
    """
    Evaluate the model on the test dataset.

    Args:
        model (nn.Module): Trained model to evaluate.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run the evaluation on.
        loss_fn (nn.Module): Loss function to use. Defaults to CrossEntropyLoss.

    Returns:
        test_loss (float): Average test loss.
        test_acc (float): Average test accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    test_loss = running_loss / total
    test_acc = correct / total

    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc

def save_model(TRAIN_LOSSES, TRAIN_ACC, VAL_LOSSES, VAL_ACC, best_val_loss, best_model_wts, model_name):
    """
    Save best model weights and model specs in models/ folder
    """
    #save best model weights
    torch.save(best_model_wts, f"models/{model_name}.pth")

    model_specs = {
        "model_name": model_name,
        "train_loss": TRAIN_LOSSES,
        "train_acc": TRAIN_ACC,
        "val_loss": VAL_LOSSES,
        "val_acc": VAL_ACC,
        "best_val_loss": best_val_loss,
    }

    #save model specs
    with open(f"models/{model_name}_specs.json", "w") as f:
        json.dump(model_specs, f, indent=4)