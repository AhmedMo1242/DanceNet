import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import yaml
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add the parent directory to Python path to make absolute imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.dance_to_text_decoder import DanceToTextDecoder
from transformers import BertTokenizer, BertModel
from data.dance_to_text_data_loader import create_dance_to_text_dataloader

def create_bert_embedding(text, bert_model, tokenizer, device):
    """Create BERT embedding for a given text"""
    # Tokenize the text
    encoded_input = tokenizer(text, return_tensors='pt', 
                            padding=True, truncation=True, max_length=128)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Get BERT embedding (using [CLS] token representation)
    with torch.no_grad():
        outputs = bert_model(**encoded_input)
        bert_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
    
    return bert_embedding

def evaluate_model(model, data_loader, bert_model, tokenizer, device, top_k_values=[1]):
    """
    Evaluate the model on a dataset
    
    Args:
        model: DanceToTextDecoder model
        data_loader: DataLoader for evaluation
        bert_model: BERT model for text embeddings
        tokenizer: BERT tokenizer
        device: Device to run evaluation on
        top_k_values: List of k values for top-k accuracy
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    results = {f"top_{k}_accuracy": 0.0 for k in top_k_values}
    results["cluster_accuracy"] = 0.0
    total_samples = 0
    
    criterion = nn.CosineEmbeddingLoss(reduction='sum')
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            dance_embeddings = batch['dance_embedding']
            actual_texts = batch['text']
            actual_cluster_ids = batch['cluster_id']
            
            # Forward pass for dance embeddings
            projected_embeddings = model(dance_embeddings)
            
            # Process each sample individually
            for i, (dance_emb, text, cluster_id) in enumerate(zip(dance_embeddings, actual_texts, actual_cluster_ids)):
                # Get BERT embedding for the text
                bert_embedding = create_bert_embedding(text, bert_model, tokenizer, device)
                
                # Project dance embedding
                projected_emb = model(dance_emb.unsqueeze(0))
                
                # Calculate loss
                target = torch.ones(1).to(device)
                loss = criterion(projected_emb, bert_embedding, target)
                total_loss += loss.item()
                
                # Find closest text
                closest_texts = model.find_closest_text(dance_emb, top_k=max(top_k_values))
                predicted_texts = [text for text, _ in closest_texts]
                
                # Check top-k accuracy
                for k in top_k_values:
                    if text in predicted_texts[:k]:
                        results[f"top_{k}_accuracy"] += 1
                
                # Check cluster accuracy - need to get cluster of predicted text
                predicted_text = predicted_texts[0]
                predicted_cluster = None
                
                # Find cluster for predicted text
                for cluster, texts in model.texts_by_cluster.items():
                    if predicted_text in texts:
                        predicted_cluster = cluster
                        break
                
                if predicted_cluster == cluster_id.item():
                    results["cluster_accuracy"] += 1
                
                total_samples += 1
    
    # Calculate metrics
    for k in top_k_values:
        results[f"top_{k}_accuracy"] /= total_samples if total_samples > 0 else 1
    
    results["cluster_accuracy"] /= total_samples if total_samples > 0 else 1
    results["avg_loss"] = total_loss / total_samples if total_samples > 0 else 0
    
    return results

def train_dance_to_text_decoder(config):
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create BERT model for embeddings
    bert_model_name = config['bert_model_name']
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name).to(device)
    bert_model.eval()  # Set to eval mode since we're not training it
    
    # Create dance-to-text decoder
    dance_to_text_model = DanceToTextDecoder(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        bert_model_name=bert_model_name
    ).to(device)
    
    # Create dataloaders - we'll pass None for the text_encoder since we're using BERT directly
    train_loader, unique_texts, cluster_ids = create_dance_to_text_dataloader(
        data_path=config['train_data_path'],
        batch_size=config['batch_size'],
        text_encoder=None,  # No text encoder
        device=device,
        shuffle=True
    )
    
    val_loader, _, _ = create_dance_to_text_dataloader(
        data_path=config['val_data_path'],
        batch_size=config['batch_size'],
        text_encoder=None,  # No text encoder
        device=device,
        shuffle=False
    )
    
    # Create test dataloader if test_data_path is provided
    test_loader = None
    if 'test_data_path' in config and config['test_data_path']:
        test_loader, _, _ = create_dance_to_text_dataloader(
            data_path=config['test_data_path'],
            batch_size=config['batch_size'],
            text_encoder=None,  # No text encoder
            device=device,
            shuffle=False
        )
    
    # Build text embedding database in the model using BERT directly
    # We'll still pass the text_encoder parameter but it won't be used
    dance_to_text_model.build_text_embedding_database(None, unique_texts, cluster_ids)
    
    # Define loss function and optimizer
    criterion = nn.CosineEmbeddingLoss(margin=config['margin'])
    optimizer = optim.Adam(dance_to_text_model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['lr_decay_factor'], 
        patience=config['lr_patience']
    )
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    accuracies = []
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Training phase
        dance_to_text_model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]"):
            optimizer.zero_grad()
            
            dance_embeddings = batch['dance_embedding']
            texts = batch['text']
            
            # Process each sample in batch
            batch_loss = 0.0
            for i, (dance_emb, text) in enumerate(zip(dance_embeddings, texts)):
                # Get BERT embedding for the text
                bert_embedding = create_bert_embedding(text, bert_model, tokenizer, device)
                
                # Ensure dance_emb has the right shape [1, 256]
                if dance_emb.dim() == 1:
                    dance_emb = dance_emb.unsqueeze(0)
                    
                # Forward pass
                projected_emb = dance_to_text_model(dance_emb)
                
                # Target for cosine similarity (1 = similar)
                target = torch.ones(1).to(device)
                
                # Calculate loss - make sure shapes are compatible for cosine similarity
                loss = criterion(projected_emb, bert_embedding, target)
                
                # Accumulate loss for backward pass
                batch_loss += loss
            
            # Average loss over batch
            batch_loss = batch_loss / len(dance_embeddings)
            
            # Backward pass and optimize
            batch_loss.backward()
            optimizer.step()
            
            train_loss += batch_loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        dance_to_text_model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]"):
                dance_embeddings = batch['dance_embedding']
                texts = batch['text']
                cluster_ids = batch['cluster_id']
                
                # Process each sample
                for i, (dance_emb, text, cluster_id) in enumerate(zip(dance_embeddings, texts, cluster_ids)):
                    # Get BERT embedding for the text
                    bert_embedding = create_bert_embedding(text, bert_model, tokenizer, device)
                    
                    # Forward pass
                    projected_emb = dance_to_text_model(dance_emb.unsqueeze(0))
                    
                    # Target for cosine similarity
                    target = torch.ones(1).to(device)
                    
                    # Calculate loss
                    loss = criterion(projected_emb, bert_embedding, target)
                    val_loss += loss.item()
                    
                    # Find closest text and check if cluster matches
                    closest_texts = dance_to_text_model.find_closest_text(dance_emb, top_k=1)
                    predicted_text = closest_texts[0][0]
                    
                    # Find the cluster ID of the predicted text
                    predicted_cluster = None
                    for cluster, texts in dance_to_text_model.texts_by_cluster.items():
                        if predicted_text in texts:
                            predicted_cluster = cluster
                            break
                    
                    # Check if the predicted cluster matches the actual cluster
                    if predicted_cluster == cluster_id.item():
                        correct_predictions += 1
                    
                    total_predictions += 1
        
        val_loss /= total_predictions
        val_losses.append(val_loss)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        accuracies.append(accuracy)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Cluster Prediction Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(dance_to_text_model.state_dict(), config['save_path'])
            print(f"  Model saved to {config['save_path']}")
    
    print("Training complete!")
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    
    # Final evaluation on test set if available
    if test_loader:
        print("\nEvaluating on test set...")
        
        # Load best model
        dance_to_text_model.load_state_dict(torch.load(config['save_path']))
        
        # Evaluate
        top_k_values = config.get('top_k_eval', [1, 3, 5])
        test_results = evaluate_model(
            dance_to_text_model, 
            test_loader, 
            bert_model,
            tokenizer,
            device, 
            top_k_values
        )
        
        # Print test results
        print("Test Results:")
        for metric, value in test_results.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save test results
        if 'results_path' in config:
            os.makedirs(os.path.dirname(config['results_path']), exist_ok=True)
            results = {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'config': config,
                'training': {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'accuracies': accuracies,
                    'best_val_loss': best_val_loss
                },
                'test_results': test_results
            }
            
            with open(config['results_path'], 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to {config['results_path']}")
    
    return dance_to_text_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Dance-to-Text Decoder Model")
    
    # Config file argument
    parser.add_argument("--config", type=str, default="config/dance_to_text_decoder_config.yaml", 
                        help="Path to configuration file")
    
    # Device settings
    parser.add_argument("--device", type=str, help="Device to use for training (cuda or cpu)")
    
    # Model parameters
    parser.add_argument("--bert_model_name", type=str, help="Name of the BERT model to use")
    parser.add_argument("--input_dim", type=int, help="Input dimension of dance embeddings")
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of projection network")
    
    # Loss function parameters
    parser.add_argument("--margin", type=float, help="Margin for cosine embedding loss")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr_decay_factor", type=float, help="Factor by which to decay learning rate")
    parser.add_argument("--lr_patience", type=int, help="Patience for learning rate scheduler")
    
    # Data paths
    parser.add_argument("--train_data_path", type=str, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, help="Path to validation data")
    parser.add_argument("--test_data_path", type=str, help="Path to test data")
    
    # Evaluation parameters
    parser.add_argument("--top_k_eval", type=int, nargs="+", help="K values for top-K accuracy evaluation")
    
    # Output settings
    parser.add_argument("--save_path", type=str, help="Path to save the trained model")
    parser.add_argument("--results_path", type=str, help="Path to save the evaluation results")
    
    args = parser.parse_args()
    
    # Load configuration from file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments if provided
    for arg in vars(args):
        if arg != 'config' and getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)
    
    # Train the model
    train_dance_to_text_decoder(config)
