
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import logging
import time
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

def setup_logging(output_dir: str) -> logging.Logger:
    """
    Setup logging configuration for the training process.
    
    Creates a logger that outputs to both console and a file in the specified directory.
    
    Args:
        output_dir: Directory where log file will be saved
        
    Returns:
        Configured logger instance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('contrastive_training')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str) -> Dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict, output_dir: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save
        output_dir: Directory where the configuration will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def check_model_for_nan(model: torch.nn.Module, name: str) -> bool:
    """
    Check if model parameters contain NaN values.
    
    Args:
        model: PyTorch model to check
        name: Name of the model for logging purposes
        
    Returns:
        True if NaN values are detected, False otherwise
    """
    for param_name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in {name} parameters: {param_name}")
            return True
    return False

def plot_training_curves(
    train_losses: List[float], 
    val_losses: List[float], 
    temperatures: List[float],
    output_dir: str
) -> None:
    """
    Plot and save training curves.
    
    Creates a figure with two subplots:
    1. Training and validation loss over epochs
    2. Temperature parameter over epochs
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        temperatures: List of temperature values 
        output_dir: Directory where the plot will be saved
    """
    plt.figure(figsize=(12, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Temperature subplot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(temperatures)+1), temperatures, label='Temperature', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Temperature')
    plt.title('Temperature')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))

def save_learning_curves(
    train_losses: List[float], 
    val_losses: List[float], 
    temperatures: List[float],
    output_dir: str
) -> None:
    """
    Save learning curves data as CSV.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        temperatures: List of temperature values
        output_dir: Directory where the CSV file will be saved
    """
    curves_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses)+1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'temperature': temperatures
    })
    curves_df.to_csv(os.path.join(output_dir, 'learning_curves.csv'), index=False)

def add_projected_embeddings(
    data_path: str,
    dance_projector: torch.nn.Module,
    text_projector: torch.nn.Module,
    device: torch.device,
    dance_embedding_idx: int = 1,
    text_embedding_idx: int = 4,
    output_dance_embedding_idx: int = 5,
    output_text_embedding_idx: int = 6,
    batch_size: int = 32,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Add projected embeddings to the original data array.
    
    Processes the data in batches to generate projected embeddings using the 
    trained projector models and adds them to the data array at the specified indices.
    
    Args:
        data_path: Path to the data file containing embeddings
        dance_projector: Trained dance projector model
        text_projector: Trained text projector model
        device: Device to run the models on (CPU or CUDA)
        dance_embedding_idx: Index of dance embedding in the data array
        text_embedding_idx: Index of text embedding in the data array
        output_dance_embedding_idx: Index to store projected dance embedding
        output_text_embedding_idx: Index to store projected text embedding
        batch_size: Batch size for processing
        output_path: Path to save the output data (if None, no file is saved)
        
    Returns:
        Updated numpy array with projected embeddings added
    """
    # Load data
    data = np.load(data_path, allow_pickle=True)
    print(f"Loaded data shape: {data.shape}")
    
    # Set models to evaluation mode
    dance_projector.eval()
    text_projector.eval()
    
    # Process in batches to avoid memory issues
    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Prepare projected embeddings storage
    dance_projections = []
    text_projections = []
    
    print(f"Processing {num_samples} samples in {num_batches} batches...")
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            # Extract batch
            batch_data = data[start_idx:end_idx]
            
            # Extract embeddings
            try:
                batch_dance_emb = np.stack([item[dance_embedding_idx] for item in batch_data])
                batch_text_emb = np.stack([item[text_embedding_idx] for item in batch_data])
            except (IndexError, ValueError) as e:
                raise ValueError(f"Error extracting embeddings: {e}. Check embedding indices.")
            
            # Convert to tensors
            batch_dance_emb = torch.FloatTensor(batch_dance_emb).to(device)
            batch_text_emb = torch.FloatTensor(batch_text_emb).to(device)
            
            # Project embeddings
            batch_dance_proj = dance_projector(batch_dance_emb)
            batch_text_proj = text_projector(batch_text_emb)
            
            # Store projections
            dance_projections.append(batch_dance_proj.cpu().numpy())
            text_projections.append(batch_text_proj.cpu().numpy())
            
            # Print progress
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"Processed {end_idx}/{num_samples} samples")
    
    # Concatenate all projections
    dance_projections = np.concatenate(dance_projections, axis=0)
    text_projections = np.concatenate(text_projections, axis=0)
    
    print(f"Projection shapes: Dance: {dance_projections.shape}, Text: {text_projections.shape}")
    
    # Create a new array with the expanded size
    max_index = max(output_dance_embedding_idx, output_text_embedding_idx)
    target_size = max_index + 1  # +1 because indices are 0-based
    
    new_data = []
    
    # For each sample in the dataset
    for i in range(num_samples):
        original_item = data[i]
        original_len = len(original_item)
        
        # Create a new array with the target size
        new_item = np.empty(target_size, dtype=object)
        
        # Copy existing data
        for j in range(original_len):
            new_item[j] = original_item[j]
            
        # Fill remaining positions with None up to the target size
        for j in range(original_len, target_size):
            new_item[j] = None
            
        # Add the projections
        new_item[output_dance_embedding_idx] = dance_projections[i]
        new_item[output_text_embedding_idx] = text_projections[i]
        
        new_data.append(new_item)
    
    # Convert to numpy array
    new_data = np.array(new_data, dtype=object)
    
    print(f"New data shape: {new_data.shape}, item length: {target_size}")
    
    # Save updated data if output path is provided
    if output_path:
        print(f"Saving updated data to {output_path}")
        np.save(output_path, new_data)
    
    return new_data
