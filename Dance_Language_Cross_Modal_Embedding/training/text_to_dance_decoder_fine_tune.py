import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import time
from datetime import datetime
import argparse
import yaml
import sys
import traceback

# Add the parent directory to Python path to make absolute imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.text_to_dance_decoder_data_loader import create_data_loaders
from models.text_to_dance_decoder_models import OptimizedDanceVAE, vae_loss, motion_loss


def set_seed(seed):
    """
    Set random seed for reproducibility across all libraries
    
    Args:
        seed: Integer random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description='Fine-tune VAE Decoder for dance generation')
    
    # Config file
    parser.add_argument('--config', type=str, default='config/text_to_dance_decoder_config.yaml',
                        help='Path to config file')
    
    # Model parameters
    parser.add_argument('--n_joints', type=int, help='Number of joints')
    parser.add_argument('--n_dims', type=int, help='Number of dimensions per joint')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension size')
    parser.add_argument('--latent_dim', type=int, help='Latent dimension size')
    parser.add_argument('--n_layers', type=int, help='Number of layers in RNNs')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--val_split', type=float, help='Validation split ratio')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for optimizer')
    parser.add_argument('--clip_grad_norm', type=float, help='Gradient clipping norm')
    parser.add_argument('--early_stopping_patience', type=int, help='Early stopping patience')
    parser.add_argument('--scheduler_patience', type=int, help='Scheduler patience')
    parser.add_argument('--scheduler_factor', type=float, help='Scheduler factor')
    parser.add_argument('--beta', type=float, help='Beta value for KL divergence weight')
    parser.add_argument('--alpha', type=float, help='Alpha value for velocity loss weight')
    
    # Data parameters
    parser.add_argument('--seq_length', type=int, help='Sequence length')
    parser.add_argument('--train_path', type=str, help='Path to training data')
    parser.add_argument('--val_path', type=str, help='Path to validation data')
    parser.add_argument('--test_path', type=str, help='Path to test data')
    
    # Paths
    parser.add_argument('--model_path', type=str, help='Path to pretrained model')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, help='Log directory')
    
    # Mode flags - train is true by default
    parser.add_argument('--train', action='store_true', default=True, 
                       help='Run training (default: True)')
    parser.add_argument('--no_train', action='store_false', dest='train',
                       help='Skip training')
    parser.add_argument('--inference', action='store_true', help='Run inference on all datasets')
    parser.add_argument('--save_reconstructions', action='store_true', help='Save reconstructed sequences')
    parser.add_argument('--use_text', action='store_true', help='Use text embeddings')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, help='Device to use (cuda or cpu)')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()


def load_config(args):
    """
    Load configuration from YAML file and update with command line arguments
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        config: Configuration dictionary
    """
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    for key, value in vars(args).items():
        if key == 'config':
            continue
        
        if key in ['n_joints', 'n_dims', 'hidden_dim', 'latent_dim', 'n_layers', 'dropout']:
            if value is not None:
                config['model'][key] = value
        elif key in ['batch_size', 'epochs', 'learning_rate', 'val_split', 'weight_decay', 
                     'clip_grad_norm', 'early_stopping_patience', 'scheduler_patience', 
                     'scheduler_factor', 'beta', 'alpha']:
            if value is not None:
                config['training'][key] = value
        elif key in ['seq_length', 'train_path', 'val_path', 'test_path']:
            if value is not None:
                config['data'][key] = value
        elif key in ['model_path', 'output_dir', 'checkpoint_dir', 'log_dir']:
            if value is not None:
                config['paths'][key] = value
    
    # Ensure numeric values are proper types
    for section in ['model', 'training']:
        if section in config:
            for key, value in config[section].items():
                if key in ['n_joints', 'n_dims', 'hidden_dim', 'latent_dim', 'n_layers',
                          'batch_size', 'epochs', 'early_stopping_patience', 'scheduler_patience']:
                    config[section][key] = int(value) if isinstance(value, str) else value
                elif key in ['dropout', 'learning_rate', 'val_split', 'weight_decay', 
                           'clip_grad_norm', 'scheduler_factor', 'beta', 'alpha']:
                    config[section][key] = float(value) if isinstance(value, str) else value
    
    # Add flags directly to config
    config['use_text'] = args.use_text
    config['debug'] = args.debug
    config['verbose'] = args.verbose
    
    # Create directories
    for path_key in ['output_dir', 'checkpoint_dir', 'log_dir']:
        os.makedirs(config['paths'][path_key], exist_ok=True)
    
    return config


def load_model(config, device):
    """
    Load pretrained VAE model and freeze encoder
    
    Args:
        config: Configuration dictionary
        device: Device to use
        
    Returns:
        model: Loaded model or None if loading failed
    """
    model_path = config['paths']['model_path']
    print(f"Loading pretrained VAE from {model_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Extract model configuration
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        n_joints = model_config.get('n_joints', config['model']['n_joints'])
        n_dims = model_config.get('n_dims', config['model']['n_dims'])
        hidden_dim = model_config.get('hidden_dim', config['model']['hidden_dim'])
        latent_dim = model_config.get('latent_dim', config['model']['latent_dim'])
        n_layers = model_config.get('n_layers', config['model']['n_layers'])
        dropout = model_config.get('dropout', config['model']['dropout'])
    else:
        # Use config from YAML
        n_joints = config['model']['n_joints']
        n_dims = config['model']['n_dims']
        hidden_dim = config['model']['hidden_dim']
        latent_dim = config['model']['latent_dim']
        n_layers = config['model']['n_layers']
        dropout = config['model']['dropout']
    
    # Create model
    model = OptimizedDanceVAE(
        n_joints=n_joints,
        n_dims=n_dims,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        n_layers=n_layers,
        dropout=dropout,
        use_text=config.get('use_text', False)
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move model to device
    model.to(device)
    
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Verify encoder is frozen and decoder is trainable
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    
    print(f"Encoder trainable parameters: {encoder_params} (should be 0)")
    print(f"Decoder trainable parameters: {decoder_params}")
    
    return model


def train_model(model, train_loader, val_loader, config, device):
    """
    Fine-tune the decoder of the VAE model
    
    Args:
        model: VAE model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration dictionary
        device: Device to use
        
    Returns:
        model: Trained model
    """
    # Ensure weight_decay is a float
    weight_decay = float(config['training'].get('weight_decay', 1e-5))
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config['training']['learning_rate']),
        weight_decay=weight_decay
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=float(config['training'].get('scheduler_factor', 0.5)),
        patience=int(config['training'].get('scheduler_patience', 5)),
        verbose=True
    )
    
    # Create checkpoint directory
    checkpoint_dir = config['paths']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    early_stopping_counter = 0
    early_stopping_patience = int(config['training'].get('early_stopping_patience', 10))
    alpha = float(config['training'].get('alpha', 1.0))  # Weight for velocity loss
    beta = float(config['training'].get('beta', 1.0))    # Weight for KL divergence
    verbose = config.get('verbose', False)
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        epoch_start_time = time.time()
        print("\n" + "="*80)
        print(f"Epoch {epoch+1}/{config['training']['epochs']} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Training
        model.train()
        train_loss = 0.0
        recon_loss_sum = 0.0
        vel_loss_sum = 0.0
        kl_loss_sum = 0.0
        batch_count = 0
        
        with tqdm(train_loader, desc="Training") as pbar:
            for batch in pbar:
                # Extract batch data
                if config.get('use_text', False):
                    motions, text_embeddings = batch
                    motions = motions.to(device)
                    text_embeddings = text_embeddings.to(device)
                else:
                    motions, _ = batch
                    motions = motions.to(device)
                    text_embeddings = None
                
                # Forward pass
                reconstructed, mu, logvar = model(motions, text_embeddings)
                
                # Calculate loss
                total_loss, recon_loss, vel_loss = motion_loss(reconstructed, motions, alpha=alpha)
                
                # Add KL divergence loss if using VAE
                kl_loss = 0.0
                if beta > 0:
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
                    total_loss = total_loss + beta * kl_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.decoder.parameters(), 
                    max_norm=config['training'].get('clip_grad_norm', 1.0)
                )
                
                optimizer.step()
                
                # Update statistics
                train_loss += total_loss.item()
                recon_loss_sum += recon_loss.item()
                vel_loss_sum += vel_loss.item()
                kl_loss_sum += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                batch_count += 1
                
                # Update progress bar every N batches to avoid slowdown
                if batch_count % 5 == 0 or batch_count == len(train_loader):
                    pbar.set_postfix({
                        "loss": f"{total_loss.item():.6f}",
                        "recon": f"{recon_loss.item():.6f}",
                        "vel": f"{vel_loss.item():.6f}",
                        "kl": f"{kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss:.6f}"
                    })
        
        # Calculate average losses
        train_loss = train_loss / len(train_loader)
        recon_loss_avg = recon_loss_sum / len(train_loader)
        vel_loss_avg = vel_loss_sum / len(train_loader)
        kl_loss_avg = kl_loss_sum / len(train_loader)
        train_losses.append(train_loss)
        
        print(f"\nTraining Summary:")
        print(f"  Total Loss: {train_loss:.6f}")
        print(f"  Reconstruction Loss: {recon_loss_avg:.6f}")
        print(f"  Velocity Loss: {vel_loss_avg:.6f} (alpha={alpha})")
        print(f"  KL Loss: {kl_loss_avg:.6f} (beta={beta})")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_recon_loss_sum = 0.0
        val_vel_loss_sum = 0.0
        val_kl_loss_sum = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Extract batch data
                if config.get('use_text', False):
                    motions, text_embeddings = batch
                    motions = motions.to(device)
                    text_embeddings = text_embeddings.to(device)
                else:
                    motions, _ = batch
                    motions = motions.to(device)
                    text_embeddings = None
                
                # Forward pass
                reconstructed, mu, logvar = model(motions, text_embeddings)
                
                # Calculate loss
                total_loss, recon_loss, vel_loss = motion_loss(reconstructed, motions, alpha=alpha)
                
                # Add KL divergence loss if using VAE
                kl_loss = 0.0
                if beta > 0:
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
                    total_loss = total_loss + beta * kl_loss
                
                val_loss += total_loss.item()
                val_recon_loss_sum += recon_loss.item()
                val_vel_loss_sum += vel_loss.item()
                val_kl_loss_sum += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        
        # Calculate average validation losses
        val_loss = val_loss / len(val_loader)
        val_recon_loss_avg = val_recon_loss_sum / len(val_loader)
        val_vel_loss_avg = val_vel_loss_sum / len(val_loader)
        val_kl_loss_avg = val_kl_loss_sum / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"\nValidation Summary:")
        print(f"  Total Loss: {val_loss:.6f}")
        print(f"  Reconstruction Loss: {val_recon_loss_avg:.6f}")
        print(f"  Velocity Loss: {val_vel_loss_avg:.6f} (alpha={alpha})")
        print(f"  KL Loss: {val_kl_loss_avg:.6f} (beta={beta})")
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"\nLearning rate adjusted: {old_lr:.6f} → {new_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'decoder_state_dict': model.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'config': config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))
        
        # Save epoch checkpoint if verbose
        if verbose:
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"Saved best model with val loss: {val_loss:.6f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
            
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break
        
        # Print epoch timing information
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch completed in {epoch_time:.2f} seconds")
        
        # Save reconstructions periodically during training if requested
        if config.get('save_reconstructions_during_training', False) and \
           epoch % config.get('reconstruction_save_interval', 10) == 0:
            print("\nSaving current reconstructions...")
            data_paths = []
            if 'val_path' in config['data']:  # Save only validation for speed
                data_paths.append(config['data']['val_path'])
            
            reconstruction_output_dir = os.path.join(
                config['paths']['output_dir'], 
                f"epoch_{epoch+1}"
            )
            os.makedirs(reconstruction_output_dir, exist_ok=True)
            
            if data_paths:
                save_reconstructions_with_progress(model, data_paths[0], device, 
                                                 os.path.join(reconstruction_output_dir, f"val_reconstructed_epoch_{epoch+1}.npy"))
    
    # Save final model
    checkpoint = {
        'epoch': config['training']['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_losses[-1],
        'config': config
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'final_model.pth'))
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config['paths']['output_dir'], 'learning_curves.png'))
    plt.close()
    
    print("Fine-tuning completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return model


def process_and_save_all_datasets(model, config, device):
    """
    Process all available datasets and save reconstructions
    
    Args:
        model: VAE model
        config: Configuration dictionary
        device: Device to use
    """
    print("\nProcessing and saving reconstructions for all datasets...")
    data_paths = []
    dataset_names = []
    
    # Collect all dataset paths
    if 'train_path' in config['data'] and os.path.exists(config['data']['train_path']):
        train_path = config['data']['train_path']
        data_paths.append(train_path)
        dataset_names.append("train")
        print(f"Will process training data: {train_path}")
        
    if 'val_path' in config['data'] and os.path.exists(config['data']['val_path']):
        val_path = config['data']['val_path'] 
        data_paths.append(val_path)
        dataset_names.append("validation")
        print(f"Will process validation data: {val_path}")
        
    if 'test_path' in config['data'] and os.path.exists(config['data']['test_path']):
        test_path = config['data']['test_path']
        data_paths.append(test_path)
        dataset_names.append("test")
        print(f"Will process test data: {test_path}")
    
    if not data_paths:
        print("Error: No data paths found for processing.")
        return
    
    # Create results directory
    output_dir = config['paths']['output_dir']
    results_dir = os.path.join(output_dir, "final_results")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nSaving results to: {results_dir}")
    
    # Process each dataset
    for path, name in zip(data_paths, dataset_names):
        # Define the output path
        output_filename = f"{os.path.basename(path).split('.')[0]}_with_reconstructions.npy"
        output_path = os.path.join(results_dir, output_filename)
        
        print(f"\nProcessing {name} dataset...")
        save_reconstructions_with_progress(model, path, device, output_path)
    
    print("\nAll datasets processed successfully!")
    print(f"Reconstructed data saved to: {results_dir}")


def save_reconstructions_with_progress(model, data_path, device, output_path):
    """
    Process a dataset and save reconstructions with detailed progress tracking
    
    Args:
        model: VAE model
        data_path: Path to data file
        device: Device to use
        output_path: Path to save reconstructions
    """
    model.eval()
    
    # Get dataset name for display
    dataset_name = os.path.basename(data_path).split('.')[0]
    
    try:
        # Load the data
        print(f"Loading data from {data_path}...")
        data = np.load(data_path, allow_pickle=True)
        print(f"Successfully loaded {len(data)} samples")
        
        # Create a copy of the data that we can modify
        updated_data = []
        total_samples = len(data)
        
        # Process each sample
        print(f"Generating reconstructions for {total_samples} samples...")
        with torch.no_grad():
            for i in tqdm(range(total_samples), desc=f"Processing {dataset_name}"):
                # Get motion data
                motion = data[i][0]
                motion_tensor = torch.tensor(motion, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Get text embedding if available
                text_embedding = None
                if len(data[i]) > 6 and isinstance(data[i][6], np.ndarray):
                    text_embedding = torch.tensor(data[i][6], dtype=torch.float32).unsqueeze(0).to(device)
                
                # Generate reconstruction
                reconstructed, mu, logvar = model(motion_tensor, text_embedding)
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
        
        # Save the updated data
        print(f"Saving {len(updated_data)} samples to {output_path}...")
        np.save(output_path, np.array(updated_data, dtype=object))
        
        # Verify the save was successful
        print("Verifying saved data...")
        verification = np.load(output_path, allow_pickle=True)
        
        if len(verification) != len(updated_data):
            print(f"Warning: Saved data has {len(verification)} samples, expected {len(updated_data)}")
        
        # Print data structure information
        print(f"Data structure: tuple with {len(verification[0])} elements")
        print("Elements:")
        for i, element in enumerate(verification[0]):
            if element is not None:
                if isinstance(element, np.ndarray):
                    print(f"  {i}: numpy array with shape {element.shape} and dtype {element.dtype}")
                else:
                    print(f"  {i}: {type(element)}")
            else:
                print(f"  {i}: None")
        
        # Verify reconstructions are stored correctly
        reconstruction_check = verification[0][7]
        print(f"Verification: First reconstructed sequence shape: {reconstruction_check.shape}")
        
        # Compute reconstruction error
        orig_motions = np.array([item[0] for item in verification])
        recon_motions = np.array([item[7] for item in verification])
        mse = np.mean(np.square(orig_motions - recon_motions))
        print(f"Average reconstruction MSE: {mse:.6f}")
        
        print(f"Successfully processed and saved {dataset_name} dataset!")
        
    except Exception as e:
        print(f"Error processing {dataset_name} dataset: {e}")
        traceback.print_exc()


def main():
    """Main function to run the script"""
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args)
    
    # Set random seed
    set_seed(args.seed if args.seed else config.get('seed', 42))
    
    # Set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config, config.get('use_text', False))
    
    # Load model
    model = load_model(config, device)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Training mode
    if args.train:
        if train_loader is None or val_loader is None:
            print("Error: Training and validation data loaders are required for training mode.")
            return
            
        model = train_model(model, train_loader, val_loader, config, device)
        
        # Always process and save all datasets after training completes
        print("\n" + "="*80)
        print("Training complete. Processing all datasets to save reconstructions...")
        print("="*80)
        
        process_and_save_all_datasets(model, config, device)
    
    # Inference mode - save reconstructions for all datasets
    if args.inference or args.save_reconstructions:
        process_and_save_all_datasets(model, config, device)


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
