import torch
from torch.utils.data import Dataset
import numpy as np
from utils.dance_preprocessing import augment_sequence

class DanceMotionDataset(Dataset):
    """
    Dataset for dance motion sequences with optional data augmentation.
    
    Args:
        sequences (list): List of motion sequences
        augment (bool): Whether to apply data augmentation
    """
    def __init__(self, sequences, augment=False):
        self.augment = augment
        
        if self.augment:
            augmented_sequences = []
            for seq in sequences:
                augmented_sequences.extend(augment_sequence(seq))
            self.sequences = augmented_sequences
            print(f"Expanded dataset with augmentation from {len(sequences)} to {len(self.sequences)} sequences")
        else:
            self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        sequence_tensor = torch.FloatTensor(sequence)
        sequence_tensor = sequence_tensor.permute(1, 0, 2)
        
        return sequence_tensor
