import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dance_dataset import DanceMotionDataset
from utils.dance_preprocessing import load_motion_data, segment_sequences, normalize_sequences

def prepare_datasets(data_dir, seq_length=50, stride=20, train_ratio=0.8, val_ratio=0.05):
    """
    Prepare train, validation, and test datasets from motion data directory.
    
    Args:
        data_dir (str): Directory containing motion data files
        seq_length (int): Length of each motion sequence
        stride (int): Stride between consecutive sequences
        train_ratio (float): Proportion of data to use for training
        val_ratio (float): Proportion of data to use for validation
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, normalization_params)
    """
    motion_data_list = load_motion_data(data_dir)
    
    all_sequences = []
    for motion_data in motion_data_list:
        sequences = segment_sequences(motion_data, seq_length, stride)
        all_sequences.extend(sequences)
    
    print(f"Total sequences after segmentation: {len(all_sequences)}")
    
    all_sequences, mean, std = normalize_sequences(all_sequences)
    
    indices = list(range(len(all_sequences)))
    np.random.shuffle(indices)
    all_sequences = [all_sequences[i] for i in indices]
    
    n_train = int(len(all_sequences) * train_ratio)
    n_val = int(len(all_sequences) * val_ratio)
    
    train_sequences = all_sequences[:n_train]
    val_sequences = all_sequences[n_train:n_train+n_val]
    test_sequences = all_sequences[n_train+n_val:]
    
    print(f"Train sequences: {len(train_sequences)} ({train_ratio*100:.1f}%)")
    print(f"Validation sequences: {len(val_sequences)} ({val_ratio*100:.1f}%)")
    print(f"Test sequences: {len(test_sequences)} ({(1-train_ratio-val_ratio)*100:.1f}%)")
    
    train_dataset = DanceMotionDataset(train_sequences, augment=True)
    val_dataset = DanceMotionDataset(val_sequences, augment=False)
    test_dataset = DanceMotionDataset(test_sequences, augment=False)
    
    return train_dataset, val_dataset, test_dataset, (mean, std)

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    """
    Create DataLoader objects for train, validation, and test datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size (int): Batch size for DataLoaders
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)
    
    return train_loader, val_loader, test_loader
