import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils.constants import *

def get_optimizer(model, optim_class = "sgd", **kwargs):
    """
    Produce a model parameter optimizer

    Args:
        model: Model to optimize
        optim_class: Supported: "adam", "sgd", "adamW". Defaults to "sgd".
    """
    if optim_class == "sgd":
        return optim.SGD(model.parameters(), **kwargs)
    elif optim_class == "adam":
        return optim.Adam(model.parameters(), **kwargs)
    elif optim_class == "adamW":
        return optim.AdamW(model.parameters(), **kwargs)
    else:
        raise RuntimeError(f"get_optimizer: {optim_class} is not recognized.")
    
def get_loss(class_weights = None):
    """
    Produce the loss function for classification models

    Args:
        class_weights: Rescaling weights for class imbalances. Defaults to None.

    Returns:
        loss_fn to be used for the training routine
    """
    if class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        return nn.CrossEntropyLoss()
        
def get_scheduler(optimizer, type, **kwargs):
    """
    Obtain a learning rate scheduler for the optimizer

    Args:
        optimizer: optimizer given by get_optimizer
        type: Type of learning rate scheduler. Supported: "step", "exponential".
    """
    if type == "step":
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif type == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    else:
        raise RuntimeError(f"get_scheduler: {type} is not recognized.")
    

def CNN_train(model, train_data, valid_data, num_epochs,
               loss_fn, optimizer, scheduler = None, log = None, verbose = False):
    """
    Training routine for CNN Classification Model

    Args:
        model: model given by get_model
        train_data: data loader for training set (see object_detect_data.py)
        valid_data: data loader for validation set (see object_detect_data.py)
        num_epochs: Number of epochs
        loss_fn: Loss criterion by get_loss
        optimizer: Optimizer by get_optimizer
        scheduler: LR Scheduler by get_scheduler. Defaults to None
        log: Log cumulative loss and accuracy. A dictionary with "train_loss", "valid_loss", "train_acc", "valid_acc" fields
        verbose: Report the per epoch train / valid metrics.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
    else:
        device = torch.device("mps")
    
    model = model.to(device)

    # Infer the number of previously trained epochs from log.
    trained_epochs = 0 if log is None else len(log["train_loss"])
    
    def train_one_epoch():
        """
        Helper function for training one epoch
        """
        model.train()
        running_loss = 0.0
        num_obs = 0
        num_correct = 0
        for images, targets in tqdm(train_data, leave=False,
            desc = f"Training {trained_epochs + 1}"):
            # Send all training data to device
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            # Depending on the model architecture
            out = model(images)
            loss = loss_fn(out, targets)
            loss.backward()
            optimizer.step()
            
            # Log the running loss
            batch_size = images.size(0)
            num_obs += batch_size
            running_loss += loss.item() * batch_size 
            
            # Log the correct observations
            preds = torch.argmax(out, dim = 1)
            num_correct += int(torch.sum(preds == targets))
        
        return running_loss / num_obs, num_correct / num_obs
    
    def validate():
        model.eval()
        running_loss = 0.0
        num_obs = 0
        num_correct = 0
        with torch.no_grad():
            for images, targets in tqdm(valid_data, leave=False,
                                    desc = f"Validating {trained_epochs + 1}"):
                # Send all training data to device
                images = images.to(device)
                targets = targets.to(device)
            
                out = model(images)
                loss = loss_fn(out, targets)
                
                # Log the running loss
                batch_size = images.size(0)
                num_obs += batch_size
                running_loss += loss.item() * batch_size 
                
                # Log the correct observations
                preds = torch.argmax(out, dim = 1)
                num_correct += int(torch.sum(preds == targets))
                
        return running_loss / num_obs, num_correct / num_obs
        
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch()
        valid_loss, valid_acc = validate()
        # Update the learning rate and timestamp
        if scheduler: scheduler.step()
        trained_epochs += 1
        # Log the evaluation metrics:
        if log:
            log["train_loss"].append(train_loss)
            log["train_acc"].append(train_acc)
            log["valid_loss"].append(valid_loss)
            log["valid_acc"].append(valid_acc)
        
        # Report the metrics:
        if verbose:
            print(f"Epoch {trained_epochs}: ", end = "")
            print(f"Current LR: {scheduler.get_last_lr()}" if scheduler else "")
            print("Train Loss: {:<8}".format(round(train_loss, 5)), end = "")
            print("Valid Loss: {:<8}".format(round(valid_loss, 5)), end = "")
            print("Train Acc:  {:<8}".format(round(train_acc, 5)), end = "")
            print("Valid Acc:  {:<8}".format(round(valid_acc, 5)))
        
    return
        


    
            
        
