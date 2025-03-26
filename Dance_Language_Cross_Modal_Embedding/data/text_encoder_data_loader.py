import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class TextEmbeddingDataset(Dataset):
    """
    Dataset for loading text-embedding pairs from NPY file.
    
    This dataset handles loading and extraction of text and embedding pairs from
    a structured NPY file for training text encoder models.
    """
    
    def __init__(self, data_path, text_index=3, text_subindex=0, embedding_index=1):
        """
        Initialize the TextEmbeddingDataset.
        
        Args:
            data_path (str): Path to the NPY file with data
            text_index (int): Index where the text is located in each item
            text_subindex (int): If text is in a nested structure, this is the subindex
            embedding_index (int): Index where the 256d embedding is located in each item
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        # Load data from NPY file
        self.data = np.load(data_path, allow_pickle=True)
        print(f"Loaded data from {data_path} with {len(self.data)} samples")
        
        self.text_index = text_index
        self.text_subindex = text_subindex
        self.embedding_index = embedding_index
    
    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            dict: Dictionary with 'text' and 'target_vector' keys
        """
        item = self.data[idx]
        
        # Extract text based on the specified indices
        try:
            if isinstance(item[self.text_index], (list, tuple, np.ndarray)) and len(item[self.text_index]) > self.text_subindex:
                text = str(item[self.text_index][self.text_subindex])
            else:
                text = str(item[self.text_index])
        except (IndexError, TypeError) as e:
            print(f"Error extracting text from item {idx}: {e}")
            text = ""  # Fallback to empty string if text extraction fails
        
        # Extract embedding and ensure it's a proper numpy array with float32 type
        try:
            embedding = np.array(item[self.embedding_index], dtype=np.float32)
        except (IndexError, TypeError, ValueError) as e:
            print(f"Error extracting embedding from item {idx}: {e}")
            embedding = np.zeros(256, dtype=np.float32)  # Fallback to zeros if embedding extraction fails
        
        return {
            'text': text,
            'target_vector': torch.tensor(embedding, dtype=torch.float32)
        }

def custom_collate_fn(batch):
    """
    Custom collate function to handle text data properly in batches.
    
    Args:
        batch (list): List of items from the dataset
        
    Returns:
        dict: Dictionary with batched texts and target vectors
    """
    # Extract texts and target vectors
    texts = [item['text'] for item in batch]
    target_vectors = torch.stack([item['target_vector'] for item in batch])
    
    return {
        'text': texts,  # Keep texts as a list of strings
        'target_vector': target_vectors
    }

def get_data_loaders(train_path, val_path, test_path, batch_size=32, num_workers=4, 
                     text_index=3, text_subindex=0, embedding_index=1):
    """
    Create train, validation, and test data loaders from NPY files.
    
    Args:
        train_path (str): Path to training data NPY file
        val_path (str): Path to validation data NPY file
        test_path (str): Path to test data NPY file
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        text_index (int): Index where the text is located in each item
        text_subindex (int): If text is in a nested structure, this is the subindex
        embedding_index (int): Index where the 256d embedding is located in each item
    
    Returns:
        tuple: (train_loader, val_loader, test_loader) - DataLoader objects for each split
    """
    # Create datasets
    train_dataset = TextEmbeddingDataset(
        train_path, text_index=text_index, text_subindex=text_subindex, embedding_index=embedding_index
    )
    
    val_dataset = TextEmbeddingDataset(
        val_path, text_index=text_index, text_subindex=text_subindex, embedding_index=embedding_index
    )
    
    test_dataset = TextEmbeddingDataset(
        test_path, text_index=text_index, text_subindex=text_subindex, embedding_index=embedding_index
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader
