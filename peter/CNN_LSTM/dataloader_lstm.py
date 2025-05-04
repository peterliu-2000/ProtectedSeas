import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from core.sum_stats import SumStatsBaseline
from core.vessel_agg import VesselTypeAggregator
from core.DICT import TYPE_to_LABEL
import json

"""



"""

def setup_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


# Data preprocessing and summary statistics
def preprocess_data(radar_detections, ais_type_labels):
    summary_df = SumStatsBaseline(radar_detections)()
    summary_cols = [col for col in summary_df.columns if col not in ['id_track', 'duration', 'detections']]
    summary_lookup = summary_df.set_index('id_track')[summary_cols].to_dict(orient='index')
    print(f'-- Summary statistics created from SumStatsBaseline --')
    print(f'-- Summary columns: {summary_cols} --')

    merged = radar_detections.merge(ais_type_labels, on='id_track', how='inner')
    
    vessel_type_aggregator = VesselTypeAggregator()
    vessel_type_aggregator.aggregate_vessel_type(merged)

    feature_cols = ['speed', 'course', 'longitude', 'latitude']
    label_col = 'type_m2_agg'
    label_lookup = merged.drop_duplicates('id_track').set_index('id_track')[label_col].map(TYPE_to_LABEL).to_dict()

    def store_model_specs():
        config = {
            'feature_cols': feature_cols,
            'summary_cols': summary_cols
        }
        with open('model_specs/LSTM_specs.json', 'w') as f:
            json.dump(config, f)

    store_model_specs()
    return merged, summary_lookup, feature_cols, label_lookup


# Create dataset for model training
def create_dataset(merged, summary_lookup, feature_cols, label_lookup, device):
    grouped = merged.groupby('id_track')
    track_data = []
    
    for id_track, group in grouped:
        if id_track not in label_lookup:
            continue
        
        features = torch.tensor(group[feature_cols].values, dtype=torch.float32)
        length = features.size(0)
        summary_vector = torch.tensor(list(summary_lookup[id_track].values()), dtype=torch.float32)
        
        track_data.append({
            'features': features.to(device),
            'summary': summary_vector.to(device),
            'length': length,
            'label': torch.tensor(label_lookup[id_track], dtype=torch.long).to(device)
        })
    
    print(f"-- Prepared {len(track_data)} track tensors (raw features) on {device} --")
    return track_data


# Splitting dataset into train, val, and test sets
def split_data(labels, seed=42):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(splitter.split(np.zeros(len(labels)), labels))

    temp_labels = labels[temp_idx]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)  # 50% to test
    val_idx, test_idx = next(splitter2.split(np.zeros(len(temp_labels)), temp_labels))
    
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    return train_idx, val_idx, test_idx


# DataLoader creation
def create_dataloaders(train_set, val_set, test_set, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


# Collate function to pad sequences and prepare batches
def collate_fn(batch):
    batch.sort(key=lambda x: x['length'], reverse=True)
    features = [item['features'] for item in batch]
    summaries = torch.stack([item['summary'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.tensor([seq.size(0) for seq in features])
    padded_features = pad_sequence(features, batch_first=True)  # B, T_max, M

    return padded_features, lengths, summaries, labels


# Dataset class
class VesselDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Check distribution of labels in the dataset
def _check_loader_label_distribution(loader, name=""):
    all_labels = []
    for batch in loader:
        labels = batch[-1]  # labels is the last item
        all_labels.extend(labels.cpu().numpy())

    counter = Counter(all_labels)
    total = sum(counter.values())
    print(f"\n{name} Label Distribution:")
    for label, count in sorted(counter.items()):
        print(f"  Label {label}: {count} ({count/total:.2%})")


# Main function to load data, process, and create DataLoaders
def load_data(radar_detections, ais_type_labels, batch_size):
    device = get_device()
    
    merged, summary_lookup, feature_cols, label_lookup = preprocess_data(radar_detections, ais_type_labels)
    track_data = create_dataset(merged, summary_lookup, feature_cols, label_lookup, device)
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    full_dataset = VesselDataset(track_data)
    labels = torch.stack([item['label'] for item in track_data]).cpu().numpy()

    # Step 1: Train vs temp (val+test)
    train_idx, val_idx, test_idx = split_data(labels, seed)

    # Create datasets
    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(train_set, val_set, test_set, batch_size)

    print(f"Data split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    cleaned_detections_path = '../../data/cleaned_data/preprocessed_radar_detections.csv'
    ais_type_labels_path = '../../data/ais_type_labels.csv'
    batch_size = 32

    radar_detections = pd.read_csv(cleaned_detections_path)
    ais_type_labels = pd.read_csv(ais_type_labels_path)

    train_loader, val_loader, test_loader = load_data(radar_detections, ais_type_labels, batch_size)

    _check_loader_label_distribution(train_loader, name="Train")
    _check_loader_label_distribution(val_loader, name="Validation")
    _check_loader_label_distribution(test_loader, name="Test")
