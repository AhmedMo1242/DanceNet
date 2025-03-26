import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse
import datetime
import warnings
from tqdm import tqdm
import time

# Add parent directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from models.contrastive_models import ProjectionHead, ClusterContrastiveLoss
from data.contrastive_data import create_data_loaders
from utils.contrastive_utils import (
    setup_logging, load_config, save_config, check_model_for_nan,
    plot_training_curves, save_learning_curves, add_projected_embeddings
)

# Handle deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# For PyTorch 2.0+ compatibility, fix deprecated imports
try:
    from torch.amp import GradScaler, autocast
    use_amp = True
    autocast_fn = lambda: autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast
        use_amp = True
        autocast_fn = lambda: autocast()
    except ImportError:
        use_amp = False
        autocast_fn = None

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description='Train contrastive projection models')
    
    # Config
    parser.add_argument('--config', type=str, default='config/contrastive_config.yaml',
                       help='Path to config file')
    
    # Data arguments
    parser.add_argument('--train_data', type=str, 
                       help='Path to training data (overrides config)')
    parser.add_argument('--val_data', type=str, 
                       help='Path to validation data (overrides config)')
    parser.add_argument('--test_data', type=str, 
                       help='Path to test data (overrides config)')
    parser.add_argument('--dance_idx', type=int, 
                       help='Index of dance embedding in data (overrides config)')
    parser.add_argument('--text_idx', type=int, 
                       help='Index of text embedding in data (overrides config)')
    parser.add_argument('--output_dance_idx', type=int, 
                       help='Index to store output dance embedding (overrides config)')
    parser.add_argument('--output_text_idx', type=int, 
                       help='Index to store output text embedding (overrides config)')
    parser.add_argument('--batch_size', type=int, 
                       help='Batch size (overrides config)')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, 
                       help='Input dimension (overrides config)')
    parser.add_argument('--hidden_dim', type=int, 
                       help='Hidden dimension (overrides config)')
    parser.add_argument('--output_dim', type=int, 
                       help='Output dimension (overrides config)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, 
                       help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, 
                       help='Learning rate (overrides config)')
    parser.add_argument('--weight_decay', type=float, 
                       help='Weight decay (overrides config)')
    parser.add_argument('--patience', type=int, 
                       help='Early stopping patience (overrides config)')
    parser.add_argument('--output_dir', type=str, 
                       help='Output directory (overrides config)')
    
    # Loss arguments
    parser.add_argument('--temperature', type=float, 
                       help='Temperature for contrastive loss (overrides config)')
    parser.add_argument('--margin', type=float, 
                       help='Margin for contrastive loss (overrides config)')
    
    # Other arguments
    parser.add_argument('--no_cuda', action='store_true', 
                       help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--log_interval', type=int, default=10, 
                       help='Logging interval')
    parser.add_argument('--inference_only', action='store_true',
                       help='Run inference on test data only')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint for resuming training or inference')
    
    return parser.parse_args()

def update_config_with_args(config, args):
    """
    Update configuration dictionary with command line arguments.
    
    Args:
        config: Configuration dictionary loaded from YAML file
        args: Parsed command line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Data
    if args.train_data:
        config['data']['train_path'] = args.train_data
    if args.val_data:
        config['data']['val_path'] = args.val_data
    if args.test_data:
        config['data']['test_path'] = args.test_data
    if args.dance_idx:
        config['data']['dance_embedding_idx'] = args.dance_idx
    if args.text_idx:
        config['data']['text_embedding_idx'] = args.text_idx
    if args.output_dance_idx:
        config['data']['output_dance_embedding_idx'] = args.output_dance_idx
    if args.output_text_idx:
        config['data']['output_text_embedding_idx'] = args.output_text_idx
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    # Model
    if args.input_dim:
        config['model']['input_dim'] = args.input_dim
    if args.hidden_dim:
        config['model']['hidden_dim'] = args.hidden_dim
    if args.output_dim:
        config['model']['output_dim'] = args.output_dim
    
    # Training
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.weight_decay:
        config['training']['weight_decay'] = args.weight_decay
    if args.patience:
        config['training']['patience'] = args.patience
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    
    # Loss
    if args.temperature:
        config['loss']['temperature'] = args.temperature
    if args.margin:
        config['loss']['margin'] = args.margin
    
    return config

def train_model(config, logger, device):
    """
    Train the contrastive projection models.
    
    Implements the training loop for contrastive learning with dance and text embeddings.
    Includes validation, early stopping, and model checkpointing.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        device: Torch device (CPU or CUDA)
        
    Returns:
        Tuple of (dance_projector, text_projector, best_val_loss, output_dir)
    """
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    save_config(config, output_dir)
    
    # Load data
    logger.info("Loading data...")
    dataloaders, datasets = create_data_loaders(
        train_path=config['data']['train_path'],
        val_path=config.get('data', {}).get('val_path'),
        test_path=config.get('data', {}).get('test_path'),
        dance_embedding_idx=config['data']['dance_embedding_idx'],
        text_embedding_idx=config['data']['text_embedding_idx'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # Get class weights from training dataset
    class_weights = None
    if config['loss']['use_class_weights'] and hasattr(datasets['train'], 'class_weights'):
        class_weights = datasets['train'].class_weights
        logger.info(f"Using class weights: {class_weights}")
    
    # Initialize models
    logger.info("Initializing models...")
    dance_projector = ProjectionHead(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['dance_projector_dropout']
    ).to(device)
    
    text_projector = ProjectionHead(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        dropout=config['model']['text_projector_dropout']
    ).to(device)
    
    # Initialize loss function
    criterion = ClusterContrastiveLoss(
        temperature=config['loss']['temperature'],
        margin=config['loss']['margin'],
        class_weights=class_weights
    )
    
    # Initialize optimizer
    optimizer = optim.SGD([
        {'params': dance_projector.parameters()},
        {'params': text_projector.parameters()}
    ], 
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize LR scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['scheduler_step_size'],
        gamma=config['training']['scheduler_gamma']
    )
    
    # Initialize AMP gradscaler for mixed precision training
    scaler = GradScaler() if use_amp else None
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Training metrics
    train_losses = []
    val_losses = []
    temperatures = []
    
    # Start timer
    start_time = time.time()
    
    # Training loop
    logger.info(f"Starting training for {config['training']['epochs']} epochs...")
    for epoch in range(config['training']['epochs']):
        # Training phase
        dance_projector.train()
        text_projector.train()
        train_loss = 0.0
        epoch_temperatures = []
        
        # Create progress bar for each epoch
        progress_bar = tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            dance_emb = batch['dance_embedding'].to(device)
            cluster_id = batch['cluster_id'].to(device)
            text_emb = batch['text_embedding'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast_fn():
                    dance_proj = dance_projector(dance_emb)
                    text_proj = text_projector(text_emb)
                    
                    # Combine projections for contrastive loss
                    combined_features = torch.cat([dance_proj, text_proj], dim=0)
                    combined_labels = torch.cat([cluster_id, cluster_id], dim=0)
                    
                    # Compute loss
                    loss, temp = criterion(combined_features, combined_labels)
                
                # Skip problematic batches
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at batch {batch_idx}, skipping batch")
                    continue
                
                # Backward pass with mixed precision
                scaler.scale(loss).backward()
                
                # Clip gradients with a small value
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(dance_projector.parameters(), config['training']['grad_clip_norm'])
                nn.utils.clip_grad_norm_(text_projector.parameters(), config['training']['grad_clip_norm'])
                
                # Update weights with mixed precision
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training without mixed precision
                dance_proj = dance_projector(dance_emb)
                text_proj = text_projector(text_emb)
                
                # Combine projections for contrastive loss
                combined_features = torch.cat([dance_proj, text_proj], dim=0)
                combined_labels = torch.cat([cluster_id, cluster_id], dim=0)
                
                # Compute loss
                loss, temp = criterion(combined_features, combined_labels)
                
                # Skip problematic batches
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at batch {batch_idx}, skipping batch")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(dance_projector.parameters(), config['training']['grad_clip_norm'])
                nn.utils.clip_grad_norm_(text_projector.parameters(), config['training']['grad_clip_norm'])
                
                # Update weights
                optimizer.step()
            
            # If NaN detected in model, reload from previous checkpoint or reinitialize
            if check_model_for_nan(dance_projector, "dance_projector") or \
               check_model_for_nan(text_projector, "text_projector"):
                logger.warning("NaN detected in model parameters! Reinitializing models...")
                # Reinitialize models
                dance_projector = ProjectionHead(
                    input_dim=config['model']['input_dim'],
                    hidden_dim=config['model']['hidden_dim'],
                    output_dim=config['model']['output_dim'],
                    dropout=config['model']['dance_projector_dropout'] * 2  # Increase dropout
                ).to(device)
                text_projector = ProjectionHead(
                    input_dim=config['model']['input_dim'],
                    hidden_dim=config['model']['hidden_dim'],
                    output_dim=config['model']['output_dim'],
                    dropout=config['model']['text_projector_dropout'] * 2  # Increase dropout
                ).to(device)
                optimizer = optim.SGD([
                    {'params': dance_projector.parameters()},
                    {'params': text_projector.parameters()}
                ], lr=config['training']['learning_rate'] * 0.1, momentum=0.9, weight_decay=1e-4)
                logger.warning(f"Reduced learning rate to {optimizer.param_groups[0]['lr']}")
                break
            
            # Accumulate loss and temperature
            train_loss += loss.item()
            epoch_temperatures.append(temp.item() if hasattr(temp, 'item') else temp)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'avg_loss': f"{train_loss/(batch_idx+1):.4f}",
                'temp': f"{temp:.4f}" if isinstance(temp, float) else f"{temp.item()::.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log every log_interval batches
            if (batch_idx + 1) % args.log_interval == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloaders['train'])}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Temp: {temp:.4f}" if isinstance(temp, float) else f"{temp.item():.4f}")
        
        # Skip epoch if we had to reinitialize models
        if check_model_for_nan(dance_projector, "dance_projector") or \
           check_model_for_nan(text_projector, "text_projector"):
            logger.warning("Skipping rest of epoch due to NaN issues")
            continue
        
        # Average training loss and temperature
        train_loss = train_loss / len(dataloaders['train']) if len(dataloaders['train']) > 0 else float('inf')
        avg_temp = sum(epoch_temperatures) / len(epoch_temperatures) if epoch_temperatures else 0.5
        train_losses.append(train_loss)
        temperatures.append(avg_temp)
        
        # Validation phase
        val_loss = 0.0
        if 'val' in dataloaders:
            dance_projector.eval()
            text_projector.eval()
            val_temps = []
            
            # Validate with no_grad for efficiency
            with torch.no_grad():
                # Create progress bar for validation
                val_progress = tqdm(dataloaders['val'], desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]")
                
                for batch in val_progress:
                    # Get data
                    dance_emb = batch['dance_embedding'].to(device)
                    cluster_id = batch['cluster_id'].to(device)
                    text_emb = batch['text_embedding'].to(device)
                    
                    # Forward pass
                    dance_proj = dance_projector(dance_emb)
                    text_proj = text_projector(text_emb)
                    
                    # Combine projections
                    combined_features = torch.cat([dance_proj, text_proj], dim=0)
                    combined_labels = torch.cat([cluster_id, cluster_id], dim=0)
                    
                    # Compute loss
                    loss, temp = criterion(combined_features, combined_labels)
                    
                    # Skip NaN loss
                    if torch.isnan(loss):
                        continue
                    
                    # Accumulate loss and temperature
                    val_loss += loss.item()
                    val_temps.append(temp.item() if hasattr(temp, 'item') else temp)
                    
                    # Update progress bar
                    val_progress.set_postfix({
                        'loss': f"{loss.item():.4f}", 
                        'temp': f"{temp:.4f}" if isinstance(temp, float) else f"{temp.item():.4f}"
                    })
            
            # Average validation loss
            val_loss = val_loss / len(dataloaders['val']) if len(dataloaders['val']) > 0 else float('inf')
            val_losses.append(val_loss)
        else:
            # If no validation set, use training loss
            val_loss = train_loss
            val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Temp: {avg_temp:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'dance_projector': dance_projector.state_dict(),
                'text_projector': text_projector.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': config
            }, os.path.join(output_dir, 'best_projectors.pt'))
            
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve for {patience_counter} epochs")
            
            if patience_counter >= config['training']['patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save checkpoint every save_interval epochs
        if (epoch + 1) % config['training']['save_interval'] == 0:
            torch.save({
                'dance_projector': dance_projector.state_dict(),
                'text_projector': text_projector.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': config
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Training completed
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, temperatures, output_dir)
    
    # Save learning curves
    save_learning_curves(train_losses, val_losses, temperatures, output_dir)
    
    # Save final model
    torch.save({
        'dance_projector': dance_projector.state_dict(),
        'text_projector': text_projector.state_dict(),
        'epoch': config['training']['epochs']-1,
        'val_loss': val_losses[-1] if val_losses else None,
        'config': config
    }, os.path.join(output_dir, 'final_projectors.pt'))
    
    # Load best model for return
    checkpoint = torch.load(os.path.join(output_dir, 'best_projectors.pt'))
    dance_projector.load_state_dict(checkpoint['dance_projector'])
    text_projector.load_state_dict(checkpoint['text_projector'])
    
    return dance_projector, text_projector, best_val_loss, output_dir

def run_inference(config, checkpoint_path, device, logger):
    """
    Run inference to add projected embeddings to datasets.
    
    Loads trained models and applies them to add projected embeddings to the specified datasets.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to the model checkpoint
        device: Torch device (CPU or CUDA)
        logger: Logger instance
        
    Returns:
        Path to output directory with processed data
    """
    logger.info(f"Running inference using checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load or create models
    dance_projector = ProjectionHead(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim']
    ).to(device)
    
    text_projector = ProjectionHead(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim']
    ).to(device)
    
    # Load state dictionaries
    dance_projector.load_state_dict(checkpoint['dance_projector'])
    text_projector.load_state_dict(checkpoint['text_projector'])
    
    # Process train, val, and test data
    data_paths = [
        ('train', config['data']['train_path']),
        ('val', config['data']['val_path'] if 'val_path' in config['data'] else None),
        ('test', config['data']['test_path'] if 'test_path' in config['data'] else None)
    ]
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['training']['output_dir'], f"inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    save_config(config, output_dir)
    
    # Process each dataset
    for dataset_name, data_path in data_paths:
        if not data_path or not os.path.exists(data_path):
            logger.info(f"Skipping {dataset_name} data: file not found or not specified")
            continue
        
        logger.info(f"Processing {dataset_name} data from {data_path}")
        
        output_path = os.path.join(output_dir, f"{dataset_name}_with_projections.npy")
        
        # Add projected embeddings to the data
        updated_data = add_projected_embeddings(
            data_path=data_path,
            dance_projector=dance_projector,
            text_projector=text_projector,
            device=device,
            dance_embedding_idx=config['data']['dance_embedding_idx'],
            text_embedding_idx=config['data']['text_embedding_idx'],
            output_dance_embedding_idx=config['data']['output_dance_embedding_idx'],
            output_text_embedding_idx=config['data']['output_text_embedding_idx'],
            batch_size=config['data']['batch_size'],
            output_path=output_path
        )
        
        logger.info(f"Saved {dataset_name} data with projections to {output_path}")
    
    logger.info(f"Inference completed. Results saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Update config with command line arguments
    config = update_config_with_args(config, args)
    
    # Set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    print(f"Using device: {device}")
    
    # Setup logging
    logger = setup_logging(config['training']['output_dir'])
    logger.info(f"Starting contrastive projection training/inference")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading config from: {args.config}")
    
    # Run inference or training
    if args.inference_only:
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            logger.error("Checkpoint path must be provided for inference")
            sys.exit(1)
            
        output_dir = run_inference(config, checkpoint_path, device, logger)
        logger.info(f"Inference completed. Results in {output_dir}")
    else:
        # Train model
        dance_projector, text_projector, best_val_loss, output_dir = train_model(config, logger, device)
        
        # After training, run inference on all data to add projected embeddings
        logger.info("Training completed. Running inference to add projected embeddings...")
        checkpoint_path = os.path.join(output_dir, 'best_projectors.pt')
        inference_output_dir = run_inference(config, checkpoint_path, device, logger)
        
        logger.info(f"All processing completed. Final outputs in {inference_output_dir}")
