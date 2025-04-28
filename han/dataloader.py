import os
import torch
import numpy as np 
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils.constants import *

# Random Seed
SEED = 1

# Image Dimension Constants:
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_FILETYPE = ".jpg"

class VesselActivityDataset(Dataset):
    def __init__(self, label_directory, image_directory, transform = None) -> None:
        self.labels = pd.read_csv(label_directory)
        self.image_path = image_directory
        self.transform = transform if transform is not None else \
            transforms.Compose([
                transforms.Resize((IMAGE_HEIGHT, IMAGE_HEIGHT)), # Resize to ResNet compliant size
                transforms.ToTensor() 
            ])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Read the corresponding row from the label dataset
        record = self.labels.iloc[idx]
        id, act = record["id_track"], record["activity"]
        
        img_name = str(id) + IMAGE_FILETYPE
        img = Image.open(os.path.join(self.image_path, img_name))
        
        if self.transform:
            img = self.transform(img)
            
        # Convert the activity code to label
        y = ACT_to_LABEL[act]
        return img, y

def collate_fn(batch):
    images, targets = zip(*batch)  # Unzip images and targets
    images = torch.stack(images, dim=0)  # Stack images into a batch
    return images, torch.tensor(targets, dtype = torch.int64)
        
def data_split(dataset, batch_size, val_prob = 0.2, num_workers = 1):
    # Use num_workers = 0 for CPU training
    # Generate random split for training and validation sets
    split_threshold= int(np.floor(val_prob * len(dataset)))
    indices = list(range(len(dataset)))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    train_inds, val_inds = indices[split_threshold:], indices[:split_threshold]
    
    train_sampler = SubsetRandomSampler(train_inds)
    val_sampler = SubsetRandomSampler(val_inds)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                              sampler = train_sampler, num_workers=num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=batch_size, 
                            sampler = val_sampler, num_workers=num_workers,
                            collate_fn=collate_fn)
    return (train_loader, val_loader)
    
def get_activity_datasets(label_directory, image_directory, batch_size, val_prob = 0.2):
    dataset = VesselActivityDataset(label_directory, image_directory)
    return data_split(dataset, batch_size, val_prob)
