import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from typing import Dict, Tuple, List, Optional

class EmbeddingDataset(Dataset):
    """
    Dataset for loading and processing embeddings for contrastive learning.
    
    Loads embeddings from a NumPy file and prepares them for contrastive learning.
    Supports cluster IDs for supervised contrastive learning and class weighting.
    """
    
    def __init__(
        self, 
        data_path: str, 
        dance_embedding_idx: int = 1, 
        text_embedding_idx: int = 4,
        transform=None
    ):
        """
        Initialize embedding dataset.
        
        Args:
            data_path: Path to the .npy file containing embeddings
            dance_embedding_idx: Index of dance embedding in the data array
            text_embedding_idx: Index of text embedding in the data array
            transform: Optional transform to apply to the data
        
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If embeddings cannot be extracted from the data
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        self.data = np.load(data_path, allow_pickle=True)
        self.dance_embedding_idx = dance_embedding_idx
        self.text_embedding_idx = text_embedding_idx
        self.transform = transform
        
        print(f"Loaded data with shape: {self.data.shape}")
        
        # Extract embeddings
        try:
            self.dance_embeddings = np.stack([item[dance_embedding_idx] for item in self.data])
            self.text_embeddings = np.stack([item[text_embedding_idx] for item in self.data])
            
            # Generate simple cluster IDs based on data indices (can be replaced with real cluster IDs)
            # Here we just use data indices as simple placeholders
            self.cluster_ids = np.arange(len(self.data))
        except (IndexError, ValueError) as e:
            raise ValueError(f"Error extracting embeddings: {e}. Check embedding indices.")
        
        # Check for NaN values in the data
        if np.isnan(self.dance_embeddings).any() or np.isnan(self.text_embeddings).any():
            print("WARNING: NaN values detected in the input data. Replacing with zeros...")
            self.dance_embeddings = np.nan_to_num(self.dance_embeddings)
            self.text_embeddings = np.nan_to_num(self.text_embeddings)
        
        # Convert to torch tensors
        self.dance_embeddings = torch.FloatTensor(self.dance_embeddings)
        self.cluster_ids = torch.LongTensor(self.cluster_ids)
        self.text_embeddings = torch.FloatTensor(self.text_embeddings)
        
        # Print dataset statistics
        print(f"Dataset size: {len(self.data)} samples")
        print(f"Dance embeddings shape: {self.dance_embeddings.shape}")
        print(f"Text embeddings shape: {self.text_embeddings.shape}")
        
        # Calculate class weights for balanced loss (using dummy cluster IDs in this case)
        # In real application, replace with meaningful cluster IDs
        unique_clusters, counts = np.unique(self.cluster_ids.numpy(), return_counts=True)
        total = len(self.cluster_ids)
        self.class_weights = {
            int(cluster): float(total / (count * len(unique_clusters)))
            for cluster, count in zip(unique_clusters, counts)
        }
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def get_original_data(self):
        """
        Return the original data array.
        
        Returns:
            NumPy array containing the original data
        """
        return self.data
    
    def __getitem__(self, idx):
        """
        Get a single data item.
        
        Args:
            idx: Index of the data item to retrieve
            
        Returns:
            Dictionary containing dance embedding, text embedding, cluster ID, and original index
        """
        sample = {
            'dance_embedding': self.dance_embeddings[idx],
            'cluster_id': self.cluster_ids[idx],
            'text_embedding': self.text_embeddings[idx],
            'original_idx': idx,  # Keep track of original index for later use
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def create_data_loaders(
    train_path: str,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    dance_embedding_idx: int = 1,
    text_embedding_idx: int = 4,
    batch_size: int = 32,
    num_workers: int = 2,
    val_split: float = 0.2,
) -> Tuple[Dict[str, DataLoader], Dict[str, Dataset]]:
    """
    Create data loaders for train, validation, and test sets.
    
    Creates DataLoader objects for each data split with appropriate configurations.
    If validation path is not provided, creates a validation split from the training data.
    
    Args:
        train_path: Path to training data file
        val_path: Path to validation data file (optional)
        test_path: Path to test data file (optional)
        dance_embedding_idx: Index of dance embedding in the data
        text_embedding_idx: Index of text embedding in the data
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for data loading
        val_split: Fraction of training data to use for validation if val_path is not provided
        
    Returns:
        Tuple of (dataloaders, datasets) dictionaries with 'train', 'val', and 'test' keys
    """
    # Create datasets
    datasets = {}
    
    # Training dataset
    train_dataset = EmbeddingDataset(
        train_path, 
        dance_embedding_idx=dance_embedding_idx, 
        text_embedding_idx=text_embedding_idx
    )
    
    # If validation set is provided as separate file
    if val_path and os.path.exists(val_path):
        val_dataset = EmbeddingDataset(
            val_path, 
            dance_embedding_idx=dance_embedding_idx, 
            text_embedding_idx=text_embedding_idx
        )
    # Otherwise split training set
    elif val_split > 0:
        train_size = int((1 - val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Split dataset: {train_size} training, {val_size} validation samples")
    else:
        val_dataset = None
    
    # Test dataset if provided
    if test_path and os.path.exists(test_path):
        test_dataset = EmbeddingDataset(
            test_path, 
            dance_embedding_idx=dance_embedding_idx, 
            text_embedding_idx=text_embedding_idx
        )
    else:
        test_dataset = None
    
    # Store datasets
    datasets['train'] = train_dataset
    if val_dataset:
        datasets['val'] = val_dataset
    if test_dataset:
        datasets['test'] = test_dataset
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
    }
    
    if val_dataset:
        dataloaders['val'] = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
    
    if test_dataset:
        dataloaders['test'] = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
    
    return dataloaders, datasets
