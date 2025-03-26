import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class DanceMotionDataset(Dataset):
    """
    Dataset for dance motion sequences with optional text conditioning
    """
    
    def __init__(self, data_path, use_text=False):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to .npy file containing data
            use_text: Whether to use text embeddings
        """
        self.data = np.load(data_path, allow_pickle=True)
        self.use_text = use_text
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            motion_data: Motion sequence tensor
            text_embedding or latent_vector: Text embedding if use_text is True, 
                                            otherwise latent vector or None
        """
        # Extract motion data at x[i][0] -> typically shape (seq_len, n_joints, n_dims)
        motion_data = torch.tensor(self.data[idx][0], dtype=torch.float32)
        
        if self.use_text:
            # Extract text embedding at x[i][6]
            text_embedding = torch.tensor(self.data[idx][6], dtype=torch.float32)
            return motion_data, text_embedding
        else:
            # For fine-tuning without text, we'll use latent vectors if available
            latent_vector = None
            if len(self.data[idx]) > 6:
                latent_vector = torch.tensor(self.data[idx][6], dtype=torch.float32)
            
            return motion_data, latent_vector


def create_data_loaders(config, use_text=False):
    """
    Create data loaders for training, validation and testing
    
    Args:
        config: Configuration dictionary
        use_text: Whether to use text embeddings
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each split
    """
    # Training data
    if 'train_path' in config['data']:
        train_dataset = DanceMotionDataset(config['data']['train_path'], use_text)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = None
    
    # Validation data
    if 'val_path' in config['data']:
        val_dataset = DanceMotionDataset(config['data']['val_path'], use_text)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        val_loader = None
    
    # Test data
    if 'test_path' in config['data']:
        test_dataset = DanceMotionDataset(config['data']['test_path'], use_text)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        test_loader = None
    
    return train_loader, val_loader, test_loader


def save_reconstructions(model, data_paths, device, output_dir, epoch=None, final=False):
    """
    Run inference on datasets and save reconstructed sequences
    
    Args:
        model: VAE model
        data_paths: List of paths to data files
        device: Device to use
        output_dir: Output directory
        epoch: Current epoch number (if called during training)
        final: Whether this is the final inference (not during training)
    """
    model.eval()
    
    for path in data_paths:
        # Get dataset name from path
        dataset_name = os.path.basename(path).split('.')[0]
        print(f"\nProcessing {dataset_name} dataset from {path}...")
        
        try:
            data = np.load(path, allow_pickle=True)
            print(f"Loaded {len(data)} samples")
        except Exception as e:
            print(f"Error loading data: {e}")
            continue
        
        # Create a copy of the data that we can modify
        updated_data = []
        total_samples = len(data)
        
        with torch.no_grad():
            for i in tqdm(range(total_samples), desc=f"Reconstructing {dataset_name}"):
                # Get motion data
                motion = data[i][0]
                motion_tensor = torch.tensor(motion, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get text embedding if available
                text_embedding = None
                if len(data[i]) > 6 and isinstance(data[i][6], np.ndarray):
                    text_embedding = torch.tensor(data[i][6], dtype=torch.float32).unsqueeze(0).to(device)
                
                # Generate reconstruction
                reconstructed, _, _ = model(motion_tensor, text_embedding)
                reconstructed = reconstructed.cpu().numpy()[0]  # (seq_len, n_joints, n_dims)
                
                # Create a copy of the original data item
                data_item = list(data[i])
                
                # Add reconstruction at index 7
                if len(data_item) <= 7:
                    # Extend the list if needed
                    data_item.extend([None] * (8 - len(data_item)))
                
                # Store the reconstructed sequence at index 7
                data_item[7] = reconstructed
                updated_data.append(tuple(data_item))
        
        # Create output filename
        if epoch is not None:
            output_filename = f"{dataset_name}_reconstructed_epoch_{epoch}.npy"
        elif final:
            output_filename = f"{dataset_name}_reconstructed_final.npy"
        else:
            output_filename = f"{dataset_name}_reconstructed.npy"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # Save updated data
        try:
            np.save(output_path, np.array(updated_data, dtype=object))
            print(f"Saved {len(updated_data)} reconstructed sequences to {output_path}")
            
            # Verify the save was successful
            verification = np.load(output_path, allow_pickle=True)
            
            # Check that reconstructions were saved correctly
            reconstruction_check = verification[0][7]
            print(f"Verification: First reconstructed sequence shape: {reconstruction_check.shape}")
            
            # Compute average reconstruction error for verification
            orig_motions = np.array([item[0] for item in verification])
            recon_motions = np.array([item[7] for item in verification])
            avg_error = np.mean(np.square(orig_motions - recon_motions))
            print(f"Average reconstruction MSE: {avg_error:.6f}")
            
        except Exception as e:
            print(f"Error saving reconstructions: {e}")


def visualize_reconstruction_samples(original_path, reconstructed_path, num_samples=5, save_dir=None):
    """
    Visualize comparison between original and reconstructed motion sequences
    
    Args:
        original_path: Path to original data file
        reconstructed_path: Path to reconstructed data file
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations (if None, just display)
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Load data
        print(f"Loading data from {reconstructed_path}...")
        data = np.load(reconstructed_path, allow_pickle=True)
        
        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Process the specified number of samples
        for i in range(min(num_samples, len(data))):
            # Get original and reconstructed sequences
            original = data[i][0]
            reconstructed = data[i][7]
            
            # Create a plot comparing a few frames
            n_frames = 5
            frame_indices = np.linspace(0, original.shape[0]-1, n_frames, dtype=int)
            
            fig = plt.figure(figsize=(15, 6))
            for j, frame_idx in enumerate(frame_indices):
                # Original frame
                ax1 = fig.add_subplot(2, n_frames, j+1, projection='3d')
                ax1.scatter(
                    original[frame_idx, :, 0],
                    original[frame_idx, :, 1],
                    original[frame_idx, :, 2],
                    c='blue', s=5
                )
                ax1.set_title(f"Original Frame {frame_idx}")
                ax1.set_xlim([-1, 1])
                ax1.set_ylim([-1, 1])
                ax1.set_zlim([-1, 1])
                
                # Reconstructed frame
                ax2 = fig.add_subplot(2, n_frames, j+n_frames+1, projection='3d')
                ax2.scatter(
                    reconstructed[frame_idx, :, 0],
                    reconstructed[frame_idx, :, 1],
                    reconstructed[frame_idx, :, 2],
                    c='red', s=5
                )
                ax2.set_title(f"Reconstructed Frame {frame_idx}")
                ax2.set_xlim([-1, 1])
                ax2.set_ylim([-1, 1])
                ax2.set_zlim([-1, 1])
            
            plt.tight_layout()
            
            # Save or display
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"comparison_sample_{i}.png"))
                plt.close()
            else:
                plt.show()
        
        print(f"Visualized {num_samples} samples")
        
    except Exception as e:
        print(f"Error during visualization: {e}")


