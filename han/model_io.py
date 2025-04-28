import torch
import numpy as np

def save_checkpoint(path, model, optimizer = None, scheduler = None, logs = None):
    """
    Save a checkpoint during training

    Args:
        path: Path to save object to
        model: Trained Model (Mandatory)
        optimizer: Optimizer (Optional). Defaults to None.
        scheduler: Scheduler (Optional). Defaults to None.
        logs: Logs (Optional). Defaults to None.
    """
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None if optimizer is None else optimizer.state_dict(),
            'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
            'log' : logs
        }, path
    )
    print(f"Model Chekcpoint saved to {path}")
    
def load_from_checkpoint(path, model, optimizer = None, scheduler = None):
    """
    Loads states from checkpoint

    Args:
        path: path to the pth checkpoint file
        model: Mandatory: initialized model by (eg. get_model)
        optimizer: Optional (Will not be loaded if the savestate contains no optimizer data)
        scheduler: Optional (Will not be loaded if the savestate contains no scheduler data)
        
    Returns:
        Training logs, if the savestate contains such data. None otherwise
    """
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["log"]

def save_loss(path, loss):
    num_epoch = len(loss)
    loss_df = np.array([1 + np.arange(num_epoch), loss]).T
    np.savetxt(path, loss_df, delimiter=",", comments = "",
               header = "Epoch, Loss", fmt = ["%d", "%.10f"])