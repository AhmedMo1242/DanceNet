import os
import time
import argparse
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import sys

# Add the parent directory to Python path to make absolute imports work
# This allows the script to be run from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Keep the absolute imports
from data.dance_data_loader import prepare_datasets, create_data_loaders
from models.vae_model import DanceVAE

def train_epoch(model, train_loader, optimizer, device, beta, clip_grad, alpha, gamma):
    """
    Train the model for one epoch.
    
    Args:
        model: The VAE model
        train_loader: DataLoader for training data
        optimizer: Optimizer for model parameters
        device: Device to use for training
        beta: Weight for KL divergence loss
        clip_grad: Gradient clipping value
        alpha: Weight for L1 loss in combined reconstruction loss
        gamma: Weight for velocity loss
        
    Returns:
        tuple: (avg_loss, avg_recon_loss, avg_kl_loss)
    """
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    for batch in train_loader:
        batch = batch.to(device)
        batch_size, seq_length = batch.shape[0], batch.shape[1]
        
        optimizer.zero_grad()
        
        reconstructed, mu, logvar = model(batch)
        
        # MSE reconstruction loss
        recon_loss_mse = torch.mean((batch - reconstructed) ** 2)
        
        # L1 reconstruction loss
        recon_loss_l1 = torch.mean(torch.abs(batch - reconstructed))
        
        # Combined reconstruction loss
        recon_loss = (1 - alpha) * recon_loss_mse + alpha * recon_loss_l1
        
        # Velocity loss - penalize differences in velocity
        velocity_original = batch[:, 1:] - batch[:, :-1]
        velocity_recon = reconstructed[:, 1:] - reconstructed[:, :-1]
        velocity_loss = torch.mean((velocity_original - velocity_recon) ** 2)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Total loss with beta weighting for KL
        loss = recon_loss + gamma * velocity_loss + beta * kl_loss
        
        loss.backward()
        
        # Gradient clipping
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
    
    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_kl_loss = total_kl_loss / len(train_loader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss

def validate(model, val_loader, device, beta, alpha, gamma):
    """
    Validate the model on validation data.
    
    Args:
        model: The VAE model
        val_loader: DataLoader for validation data
        device: Device to use for validation
        beta: Weight for KL divergence loss
        alpha: Weight for L1 loss in combined reconstruction loss
        gamma: Weight for velocity loss
        
    Returns:
        tuple: (avg_loss, avg_recon_loss, avg_kl_loss)
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch_size, seq_length = batch.shape[0], batch.shape[1]
            
            reconstructed, mu, logvar = model(batch)
            
            # MSE reconstruction loss
            recon_loss_mse = torch.mean((batch - reconstructed) ** 2)
            
            # L1 reconstruction loss
            recon_loss_l1 = torch.mean(torch.abs(batch - reconstructed))
            
            # Combined reconstruction loss
            recon_loss = (1 - alpha) * recon_loss_mse + alpha * recon_loss_l1
            
            # Velocity loss
            velocity_original = batch[:, 1:] - batch[:, :-1]
            velocity_recon = reconstructed[:, 1:] - reconstructed[:, :-1]
            velocity_loss = torch.mean((velocity_original - velocity_recon) ** 2)
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            
            # Total loss
            loss = recon_loss + gamma * velocity_loss + beta * kl_loss
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_recon_loss = total_recon_loss / len(val_loader)
    avg_kl_loss = total_kl_loss / len(val_loader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss

def save_dataset_with_latents(model, dataset, device, filename):
    """
    Save dataset with corresponding latent vectors in numpy format.
    
    Saves data in a structured format where:
    - Each element at index i contains [sequence, latent]
    - sequence has shape (seq_length, n_joints, n_dims)
    - latent has shape (latent_dim,)
    
    Args:
        model: The VAE model
        dataset: Dataset to process
        device: Device to use for computation
        filename: Output filename (.npy)
        
    Returns:
        numpy.ndarray: Array of paired sequences and latent vectors
    """
    model.eval()
    sequences = []
    latents = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            sequence = dataset[idx].unsqueeze(0).to(device)
            latent_vector = model.get_embedding(sequence).cpu().numpy()
            sequence_data = sequence.squeeze(0).cpu().numpy()
            
            sequences.append(sequence_data)
            latents.append(latent_vector.squeeze(0))  # Remove batch dimension
    
    # Prepare the final dataset format: list of [sequence, latent] pairs
    paired_data = [[sequences[i], latents[i]] for i in range(len(sequences))]
    paired_array = np.array(paired_data, dtype=object)
    
    np.save(filename, paired_array)
    
    # Verify the format
    sample_idx = min(5, len(paired_array)-1)
    seq_shape = paired_array[sample_idx][0].shape
    latent_shape = paired_array[sample_idx][1].shape
    print(f"Saved dataset with latent vectors to {filename}")
    print(f"Sample sequence shape: {seq_shape}, latent shape: {latent_shape}")
    
    return paired_array

def main():
    """
    Main training function for the dance VAE model.
    
    Handles configuration loading, dataset preparation, model training,
    validation, checkpointing, and final evaluation.
    """
    parser = argparse.ArgumentParser(description='Train VAE model for dance motion sequences')
    parser.add_argument('--config', type=str, default='config/default_config_dance_vae.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, help='Directory containing motion data')
    parser.add_argument('--save_dir', type=str, help='Directory to save models and results')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension size')
    parser.add_argument('--latent_dim', type=int, help='Latent dimension size')
    parser.add_argument('--n_layers', type=int, help='Number of RNN layers')
    parser.add_argument('--beta_max', type=float, help='Maximum KL weight')
    parser.add_argument('--patience', type=int, help='Patience for early stopping')
    
    args = parser.parse_args()
    
    # Load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.save_dir:
        config['training']['save_dir'] = args.save_dir
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.hidden_dim:
        config['model']['hidden_dim'] = args.hidden_dim
    if args.latent_dim:
        config['model']['latent_dim'] = args.latent_dim
    if args.n_layers:
        config['model']['n_layers'] = args.n_layers
    if args.beta_max:
        config['training']['beta_max'] = args.beta_max
    if args.patience:
        config['training']['patience'] = args.patience
    
    # Extract config values
    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    seq_length = config['data']['seq_length']
    stride = config['data']['stride']
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    
    hidden_dim = config['model']['hidden_dim']
    latent_dim = config['model']['latent_dim']
    n_layers = config['model']['n_layers']
    dropout = config['model']['dropout']
    
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    beta_min = config['training']['beta_min']
    beta_max = config['training']['beta_max']
    alpha = config['training']['alpha']
    gamma = config['training']['gamma']
    clip_grad = config['training']['clip_grad']
    patience = config['training']['patience']
    save_dir = config['training']['save_dir']
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset, _ = prepare_datasets(
        data_dir, seq_length, stride, train_ratio, val_ratio
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )
    
    # Get dataset dimensions
    sample = train_dataset[0]
    _, n_joints, n_dims = sample.size()
    print(f"Data dimensions: sequence length={seq_length}, joints={n_joints}, dimensions={n_dims}")
    
    # Initialize model
    model = DanceVAE(
        n_joints=n_joints,
        n_dims=n_dims,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)
    
    # Use AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
    
    # Use OneCycleLR with longer warmup
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        div_factor=25,
        final_div_factor=1000,
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # Beta annealing - slower increase to avoid posterior collapse
    beta_steps = int(epochs * 0.4)  # First 40% of epochs for annealing
    beta_values = np.linspace(beta_min, beta_max, beta_steps)
    
    # Main training loop
    for epoch in range(epochs):
        # Beta annealing
        current_beta = beta_values[min(epoch, beta_steps-1)] if epoch < beta_steps else beta_max
        
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_recon_loss, train_kl_loss = train_epoch(
            model, train_loader, optimizer, device, current_beta, clip_grad, alpha, gamma
        )
        
        # Validate
        val_loss, val_recon_loss, val_kl_loss = validate(
            model, val_loader, device, current_beta, alpha, gamma
        )
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{epochs} [{epoch_time:.1f}s]")
        print(f"Train Loss: {train_loss:.4f} | Recon Loss: {train_recon_loss:.4f} | KL Loss: {train_kl_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Recon Loss: {val_recon_loss:.4f} | KL Loss: {val_kl_loss:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}, Beta: {current_beta:.6f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save milestone checkpoints every 100 epochs
        if (epoch + 1) % 100 == 0:
            milestone_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, milestone_path)
            print(f"Saved milestone checkpoint at epoch {epoch+1}")
        
        # Checkpointing logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            patience_counter = 0
            print(f"New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (best: {best_val_loss:.4f} at epoch {best_epoch+1})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    test_loss, test_recon_loss, test_kl_loss = validate(model, test_loader, device, beta_max, alpha, gamma)
    print(f"Test Loss: {test_loss:.4f} | Recon Loss: {test_recon_loss:.4f} | KL Loss: {test_kl_loss:.4f}")
    
    # Save datasets with their latent vectors - use .npy extension
    train_latents_file = os.path.join(save_dir, 'train_with_latents.npy')
    val_latents_file = os.path.join(save_dir, 'val_with_latents.npy')
    test_latents_file = os.path.join(save_dir, 'test_with_latents.npy')
    
    save_dataset_with_latents(model, train_dataset, device, train_latents_file)
    save_dataset_with_latents(model, val_dataset, device, val_latents_file)
    save_dataset_with_latents(model, test_dataset, device, test_latents_file)

if __name__ == "__main__":
    main()
