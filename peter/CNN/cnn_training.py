from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch   
import copy
from tqdm import tqdm

def train(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                optimizer: optim.Optimizer, 
                device: torch.device,
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

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(device), targets.to(device) 
            optimizer.zero_grad()
            outputs = model(inputs)
            criterion = nn.CrossEntropyLoss()
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    return TRAIN_LOSSES, TRAIN_ACC, VAL_LOSSES, VAL_ACC, best_model_wts, best_val_loss

