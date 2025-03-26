import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from models.text_encoder import combined_loss

def train_model(model, train_loader, val_loader, config, device):
    """
    Train the text encoder model with the given configuration.
    
    This function handles the complete training process including optimization,
    learning rate scheduling, early stopping, and model checkpointing.
    
    Args:
        model: The TextEncoder model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration dictionary containing:
            - num_epochs: Number of training epochs
            - learning_rate: Initial learning rate
            - weight_decay: Weight decay for optimizer
            - patience: Patience for early stopping
            - cosine_weight: Weight for cosine similarity in loss
            - clip_grad_norm: Value for gradient clipping
            - scheduler: Type of learning rate scheduler
            - scheduler_params: Parameters for the chosen scheduler
            - model_save_path: Path to save the best model
        device: Device to run training on
    
    Returns:
        The trained model
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    
    # Move model to device
    model.to(device)
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Define scheduler
    if config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=config['scheduler_params']['T_0'], 
            T_mult=config['scheduler_params']['T_mult']
        )
    elif config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=config['scheduler_params']['factor'], 
            patience=config['scheduler_params']['plateau_patience'], 
            verbose=True
        )
    else:
        raise ValueError(f"Scheduler {config['scheduler']} not recognized")
    
    # Define loss function with weight
    def criterion(pred, target):
        return combined_loss(pred, target, alpha=config['cosine_weight'])
    
    # Setup for monitoring best model
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config['patience']
    
    # Track losses for plotting
    train_losses = []
    val_losses = []
    
    # Add timing for tracking
    total_train_time = 0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            texts = batch['text']
            target_vectors = batch['target_vector'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(texts)
            
            # Calculate loss
            loss = criterion(outputs, target_vectors)
            
            # Backward pass and optimize
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=config['clip_grad_norm']
            )
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch: {epoch+1}/{config["num_epochs"]}, '
                      f'Batch: {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_train_time += epoch_time
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                texts = batch['text']
                target_vectors = batch['target_vector'].to(device)
                
                outputs = model(texts)
                loss = criterion(outputs, target_vectors)
                
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        if config['scheduler'] == 'plateau':
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{config["num_epochs"]}, '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'LR: {current_lr:.6f}, '
              f'Time: {epoch_time:.2f}s')
        
        # Check if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"Validation loss decreased. Saving model to {config['model_save_path']}")
        else:
            patience_counter += 1
            print(f"Validation loss did not decrease. Patience: {patience_counter}/{patience}")
            
        # Check if early stopping should be triggered
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    # Print total training time
    print(f"Total training time: {total_train_time:.2f} seconds")
    
    # Plot training and validation loss
    plot_path = os.path.join(os.path.dirname(config['model_save_path']), 'loss_plot.png')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    
    # Load best model
    model.load_state_dict(torch.load(config['model_save_path']))
    
    return model

def evaluate_model(model, test_loader, device, cosine_weight=0.5):
    """
    Evaluate the model on a test set and report performance metrics.
    
    Args:
        model: The TextEncoder model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        cosine_weight: Weight for cosine similarity in the loss function
    
    Returns:
        tuple: (average_test_loss, average_cosine_similarity)
    """
    model.eval()
    all_outputs = []
    all_targets = []
    all_texts = []
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            texts = batch['text']
            target_vectors = batch['target_vector'].to(device)
            
            outputs = model(texts)
            loss = combined_loss(outputs, target_vectors, alpha=cosine_weight)
            running_loss += loss.item()
            
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(target_vectors.cpu().numpy())
            all_texts.extend(texts)
    
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute average loss
    avg_loss = running_loss / len(test_loader)
    print(f"Test loss: {avg_loss:.4f}")
    
    # Compute average cosine similarity
    cosine_sims = []
    for i in range(len(all_outputs)):
        out_norm = all_outputs[i] / np.linalg.norm(all_outputs[i])
        target_norm = all_targets[i] / np.linalg.norm(all_targets[i])
        sim = np.dot(out_norm, target_norm)
        cosine_sims.append(sim)
    
    avg_cosine_sim = np.mean(cosine_sims)
    print(f"Average cosine similarity on test set: {avg_cosine_sim:.4f}")
    
    # Print some examples
    for i in range(min(5, len(all_texts))):
        print(f"Text: {all_texts[i]}")
        print(f"Cosine similarity: {cosine_sims[i]:.4f}")
        print("---")
    
    return avg_loss, avg_cosine_sim
