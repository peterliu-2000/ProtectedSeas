import os
import pandas as pd
import torch
import numpy as np 
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import sys

sys.path.append(os.path.abspath('..')) 
from core.DICT import ACT_to_LABEL, TYPE_to_LABEL, TYPES_TO_AGG

# Random Seed
SEED = 1

# Image Dimension Constants:
MATRIX_HEIGHT = 224
MATRIX_WIDTH = 224

class VesselActivityDataset(Dataset):
    """
    Input:
        label_directory: path to the label csv file
        matrix_directory: path to the matrix folder

    Returns:
        X: torch.Tensor, shape (5, 224, 224)
        y: torch.Tensor, shape (1)
    """
    def __init__(self, label_directory, matrix_directory) -> None:
        self.labels = pd.read_csv(label_directory)
        self.matrix_path = matrix_directory
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
       
        record = self.labels.iloc[idx]
        id, act = record["id_track"], record["activity"]
        matrix_name = str(id) + ".pt"

        X = torch.load(os.path.join(self.matrix_path, matrix_name))
        y = ACT_to_LABEL[act]
        return X, y
    

class VesselTypeDataset(Dataset):
    def __init__(self, label_directory, matrix_directory) -> None:
        self.labels = pd.read_csv(label_directory)
        self.matrix_path = matrix_directory

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            X: torch.Tensor, shape (5, 224, 224)
            y: torch.Tensor, shape (1)
        """
        
        record = self.labels.iloc[idx]
        id, typ = record["id_track"], record["type_m2"]
        matrix_name = str(id) + ".pt"

        X = torch.load(os.path.join(self.matrix_path, matrix_name))
        y = TYPE_to_LABEL[TYPES_TO_AGG[typ]] #numerical label corresponding to the type_m2_agg
        # EXP:try different dim
        # return X, y 
        return X[:3, :, :], y 

def collate_fn(batch):
    matrices, targets = zip(*batch) 
    matrices = torch.stack(matrices, dim=0)  
    return matrices, torch.tensor(targets, dtype = torch.int64)
        
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

    
def get_activity_datasets(label_directory, matrix_directory, batch_size):
    dataset = VesselActivityDataset(label_directory, matrix_directory)
    return data_split(dataset, batch_size)

def get_type_datasets(label_directory, matrix_directory, batch_size):
    dataset = VesselTypeDataset(label_directory, matrix_directory)
    return data_split(dataset, batch_size)

if __name__ == "__main__":
    label_path = "../../data/labels/ais_type_labels_radar_detections.csv"
    matrix_dir = "track_matrices"
    batch_size = 8

    train_loader, val_loader, test_loader = get_type_datasets(label_path, matrix_dir, batch_size)
    for batch in train_loader:
        X, y = batch
        print(X.shape, y.shape)
        break
