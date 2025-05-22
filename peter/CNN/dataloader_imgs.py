import os
import torch
import numpy as np 
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import sys

sys.path.append(os.path.abspath('..'))
from core.DICT import  *

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
        y = ACITIVTY2NUM[act]
        return img, y
    
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class VesselTypeDataset(Dataset):
    def __init__(self, label_directory, image_directory, transform=None) -> None:
        self.image_path = image_directory
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_HEIGHT)),  # Resize to ResNet-compliant size
            transforms.ToTensor()
        ])
        self.labels = pd.read_csv(label_directory)

        #drop tracks with no vessel type
        self.labels = self.labels.dropna(subset=["type_m2_agg"]) 
        # Filter out rows with missing image files
        self.labels = self.labels[self.labels["id_track"].apply(
            lambda id: os.path.isfile(os.path.join(self.image_path, str(id) + IMAGE_FILETYPE))
        )].reset_index(drop=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        record = self.labels.iloc[idx]
        id, vessel_type = record["id_track"], record["type_m2_agg"]

        img_name = str(id) + IMAGE_FILETYPE
        img = Image.open(os.path.join(self.image_path, img_name))

        if self.transform:
            img = self.transform(img)

        y = TYPE2NUM[vessel_type]
        return img, y

def collate_fn(batch):
    images, targets = zip(*batch)  # Unzip images and targets
    images = torch.stack(images, dim=0)  # Stack images into a batch
    return images, torch.tensor(targets, dtype = torch.int64)
        
def data_split(dataset, batch_size, val_prob=0.1, test_prob=0.1, num_workers=1):
    indices = list(range(len(dataset)))
    np.random.seed(SEED)
    np.random.shuffle(indices)

    total_len = len(dataset)
    test_split = int(np.floor(test_prob * total_len))
    val_split = int(np.floor(val_prob * total_len))

    test_inds = indices[:test_split]
    val_inds = indices[test_split:test_split + val_split]
    train_inds = indices[test_split + val_split:]

    train_sampler = SubsetRandomSampler(train_inds)
    val_sampler = SubsetRandomSampler(val_inds)
    test_sampler = SubsetRandomSampler(test_inds)

    def get_loader(sampler):
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, collate_fn=collate_fn)

    return get_loader(train_sampler), get_loader(val_sampler), get_loader(test_sampler)

    
def get_activity_datasets(label_directory, image_directory, batch_size):
    dataset = VesselActivityDataset(label_directory, image_directory)
    return data_split(dataset, batch_size)

def get_type_datasets(label_directory, image_directory, batch_size):
    dataset = VesselTypeDataset(label_directory, image_directory)
    return data_split(dataset, batch_size)