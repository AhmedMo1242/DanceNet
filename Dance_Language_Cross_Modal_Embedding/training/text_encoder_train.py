import argparse
import yaml
import torch
import os
import numpy as np
import sys

# Add the parent directory to Python path to make absolute imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from models.text_encoder import TextEncoder
from data.text_encoder_data_loader import get_data_loaders
from utils.text_encoder_trainer import train_model, evaluate_model
from utils.text_encoder_processor import process_all_datasets


def parse_args():
    """
    Parse command line arguments for the text encoder training script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Text Encoder Training and Processing')
    
    # Main command arguments
    parser.add_argument('command', type=str, nargs='?', default='train',
                        choices=['train', 'evaluate', 'process'],
                        help='Command to execute: train, evaluate, or process (default: train)')
    
    # Configuration file
    parser.add_argument('--config', type=str, default='config/text_encoder_config.yaml',
                        help='Path to configuration file')
    
    # Data arguments
    parser.add_argument('--train', type=str, help='Path to training data')
    parser.add_argument('--val', type=str, help='Path to validation data')
    parser.add_argument('--test', type=str, help='Path to test data')
    parser.add_argument('--output_dir', type=str, help='Directory to save output files')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, help='Path to saved model')
    parser.add_argument('--bert_model', type=str, help='BERT model name')
    parser.add_argument('--freeze_bert', type=bool, help='Whether to freeze BERT parameters')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    # Device
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], 
                      help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration from a YAML file or return default configuration.
    
    Args:
        config_path (str): Path to the configuration YAML file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        print("Using default configuration")
        # Return default configuration
        return {
            'data': {
                'train_path': 'data/train.npy',
                'val_path': 'data/val.npy',
                'test_path': 'data/test.npy',
                'output_dir': 'data/processed',
                'batch_size': 32,
                'num_workers': 4,
                'text_index': 3,
                'text_subindex': 0,
                'embedding_index': 1
            },
            'model': {
                'bert_model_name': 'bert-base-uncased',
                'freeze_bert': True,
                'output_dim': 256
            },
            'training': {
                'num_epochs': 30,
                'learning_rate': 0.001,
                'weight_decay': 0.01,
                'patience': 5,
                'cosine_weight': 0.5,
                'clip_grad_norm': 1.0,
                'scheduler': 'cosine',
                'scheduler_params': {
                    'T_0': 10,
                    'T_mult': 1,
                    'factor': 0.5,
                    'plateau_patience': 2
                }
            },
            'processing': {
                'process_batch_size': 32,
                'embedding_dim': 256
            }
        }

def get_device(args_device=None):
    """
    Determine the device to use for model training.
    
    Args:
        args_device (str, optional): Device specified in command line arguments
        
    Returns:
        torch.device: Device to use for training
    """
    if args_device:
        return torch.device(args_device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def override_config_with_args(config, args):
    """
    Override configuration values with command line arguments if provided.
    
    Args:
        config (dict): Configuration dictionary
        args (argparse.Namespace): Command line arguments
        
    Returns:
        dict: Updated configuration dictionary
    """
    # Data arguments
    if args.train:
        config['data']['train_path'] = args.train
    if args.val:
        config['data']['val_path'] = args.val
    if args.test:
        config['data']['test_path'] = args.test
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    # Model arguments
    if args.bert_model:
        config['model']['bert_model_name'] = args.bert_model
    if args.freeze_bert is not None:
        config['model']['freeze_bert'] = args.freeze_bert
        
    # Training arguments
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
        
    return config

def main():
    """
    Main function to run the text encoder training, evaluation, or processing.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command line arguments
    config = override_config_with_args(config, args)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Execute the appropriate command
    if args.command == 'train':
        # Load data
        try:
            train_loader, val_loader, test_loader = get_data_loaders(
                config['data']['train_path'],
                config['data']['val_path'],
                config['data']['test_path'],
                batch_size=config['data']['batch_size'],
                num_workers=config['data']['num_workers'],
                text_index=config['data']['text_index'],
                text_subindex=config['data']['text_subindex'],
                embedding_index=config['data']['embedding_index']
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please check that the data files exist and are in the correct format.")
            sys.exit(1)
        
        # Initialize model
        model = TextEncoder(
            bert_model_name=config['model']['bert_model_name'],
            freeze_bert=config['model']['freeze_bert'],
            output_dim=config['model']['output_dim']
        )
        
        # Prepare training configuration
        os.makedirs(config['data']['output_dir'], exist_ok=True)
        train_config = {
            'num_epochs': config['training']['num_epochs'],
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
            'patience': config['training']['patience'],
            'cosine_weight': config['training']['cosine_weight'],
            'clip_grad_norm': config['training']['clip_grad_norm'],
            'scheduler': config['training']['scheduler'],
            'scheduler_params': config['training']['scheduler_params'],
            'model_save_path': os.path.join(config['data']['output_dir'], 'best_model.pth')
        }
        
        # Train the model
        trained_model = train_model(
            model, train_loader, val_loader, train_config, device
        )
        
        # Evaluate on test set
        evaluate_model(
            trained_model, test_loader, device, 
            cosine_weight=config['training']['cosine_weight']
        )
        
        # Save the final model
        final_model_path = os.path.join(config['data']['output_dir'], 'final_model.pth')
        torch.save(trained_model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Process datasets with the trained model
        print("\nProcessing datasets to create 5-dimensional NPY files...")
        try:
            processed_paths = process_all_datasets(
                trained_model,
                config['data']['train_path'],
                config['data']['val_path'],
                config['data']['test_path'],
                config['data']['output_dir'],
                {
                    'process_batch_size': config['processing']['process_batch_size'],
                    'embedding_dim': config['processing']['embedding_dim'],
                    'text_index': config['data']['text_index'],
                    'text_subindex': config['data']['text_subindex'],
                },
                device=device
            )
            
            print("All datasets processed successfully:")
            for split, path in processed_paths.items():
                print(f"  {split}: {path}")
        except Exception as e:
            print(f"Error processing datasets after training: {e}")
            print("You can process datasets later using the 'process' command.")
        
    elif args.command == 'evaluate':
        # Load model
        model_path = args.model_path or os.path.join(config['data']['output_dir'], 'best_model.pth')
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print("Please train a model first or specify a valid model path with --model_path")
            sys.exit(1)
            
        model = TextEncoder(
            bert_model_name=config['model']['bert_model_name'],
            freeze_bert=config['model']['freeze_bert'],
            output_dim=config['model']['output_dim']
        )
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Load test data
        try:
            _, _, test_loader = get_data_loaders(
                config['data']['train_path'],
                config['data']['val_path'],
                config['data']['test_path'],
                batch_size=config['data']['batch_size'],
                num_workers=config['data']['num_workers'],
                text_index=config['data']['text_index'],
                text_subindex=config['data']['text_subindex'],
                embedding_index=config['data']['embedding_index']
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
        
        # Evaluate model
        evaluate_model(
            model, test_loader, device, 
            cosine_weight=config['training']['cosine_weight']
        )
        
    elif args.command == 'process':
        # Load model
        model_path = args.model_path or os.path.join(config['data']['output_dir'], 'best_model.pth')
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print("Please train a model first or specify a valid model path with --model_path")
            sys.exit(1)
            
        model = TextEncoder(
            bert_model_name=config['model']['bert_model_name'],
            freeze_bert=True,  # Always freeze for processing
            output_dim=config['model']['output_dim']
        )
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Process all datasets
        try:
            processed_paths = process_all_datasets(
                model,
                config['data']['train_path'],
                config['data']['val_path'],
                config['data']['test_path'],
                config['data']['output_dir'],
                {
                    'process_batch_size': config['processing']['process_batch_size'],
                    'embedding_dim': config['processing']['embedding_dim'],
                    'text_index': config['data']['text_index'],
                    'text_subindex': config['data']['text_subindex'],
                },
                device=device
            )
            
            print("All datasets processed successfully:")
            for split, path in processed_paths.items():
                print(f"  {split}: {path}")
        except Exception as e:
            print(f"Error processing datasets: {e}")
            sys.exit(1)
            
    else:
        print(f"Unknown command: {args.command}")
        print("Available commands: train, evaluate, process")
        sys.exit(1)

if __name__ == "__main__":
    main()
