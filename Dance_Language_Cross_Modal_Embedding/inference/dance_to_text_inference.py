import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import json

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.dance_to_text_decoder import DanceToTextDecoder
from models.vae_model import DanceVAE
from models.contrastive_models import ProjectionHead

def parse_arguments():
    """Parse command line arguments for dance-to-text inference."""
    parser = argparse.ArgumentParser(description="Dance-to-Text Inference: Convert dance movements to text descriptions")
    
    # Main arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input .npy file containing dance sequences")
    parser.add_argument("--output", type=str, default="dance_to_text_results.json",
                        help="Path to output file for storing text descriptions")
    parser.add_argument("--config", type=str, default="config/dance_to_text_inference_config.yaml",
                        help="Path to configuration file")
    
    # Model paths (can override config)
    parser.add_argument("--vae_model", type=str, 
                        help="Path to VAE model checkpoint")
    parser.add_argument("--projector_model", type=str, 
                        help="Path to projector model checkpoint")
    parser.add_argument("--text_decoder_model", type=str, 
                        help="Path to text decoder model checkpoint")
    
    # Inference settings
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of top text matches to return")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference (cuda or cpu)")
    parser.add_argument("--sample_index", type=int, default=-1,
                        help="Index of specific sample to process (-1 for all)")
    
    # Text database
    parser.add_argument("--texts_file", type=str,
                        help="Path to JSON file containing text vocabulary")
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {str(e)}")
        return {}

def update_config_with_args(config, args):
    """Update configuration with command line arguments."""
    # Update model paths if provided
    if args.vae_model:
        config['models']['vae_model'] = args.vae_model
    if args.projector_model:
        config['models']['projector_model'] = args.projector_model
    if args.text_decoder_model:
        config['models']['text_decoder_model'] = args.text_decoder_model
    if args.texts_file:
        config['data']['texts_file'] = args.texts_file
    
    # Update inference settings
    config['inference']['top_k'] = args.top_k
    config['inference']['batch_size'] = args.batch_size
    config['inference']['device'] = args.device
    config['inference']['sample_index'] = args.sample_index
    
    return config

def load_models(config, device):
    """Load all required models with proper error handling."""
    models = {}
    
    # Load Dance VAE
    try:
        print(f"Loading VAE model from {config['models']['vae_model']}...")
        vae = DanceVAE(
            n_joints=config['models']['n_joints'],
            n_dims=config['models']['n_dims'],
            hidden_dim=config['models']['hidden_dim'],
            latent_dim=config['models']['latent_dim']
        )
        checkpoint = torch.load(config['models']['vae_model'], map_location=device)
        vae.load_state_dict(checkpoint['model_state_dict'])
        models['vae'] = vae.to(device).eval()
        print("VAE model loaded successfully.")
    except Exception as e:
        print(f"Error loading VAE model: {str(e)}")
        raise
    
    # Load Projector model
    try:
        print(f"Loading projector model from {config['models']['projector_model']}...")
        projectors_checkpoint = torch.load(config['models']['projector_model'], map_location=device)
        
        # Create dance projector
        dance_projector = ProjectionHead(
            input_dim=config['models']['latent_dim'],
            hidden_dim=config['models']['projection_hidden_dim'],
            output_dim=config['models']['projection_output_dim']
        ).to(device)
        
        # Load state dict
        dance_projector.load_state_dict(projectors_checkpoint['dance_projector'])
        models['dance_projector'] = dance_projector.eval()
        print("Projector model loaded successfully.")
    except Exception as e:
        print(f"Error loading projector model: {str(e)}")
        raise
    
    # Load Dance-to-Text Decoder
    try:
        print(f"Loading dance-to-text decoder from {config['models']['text_decoder_model']}...")
        text_decoder = DanceToTextDecoder(
            input_dim=config['models']['latent_dim'],
            hidden_dim=config['models']['decoder_hidden_dim'],
            bert_model_name=config['models']['bert_model_name']
        ).to(device)
        
        text_decoder.load_state_dict(torch.load(
            config['models']['text_decoder_model'], 
            map_location=device
        ))
        models['text_decoder'] = text_decoder.eval()
        print("Dance-to-Text decoder loaded successfully.")
    except Exception as e:
        print(f"Error loading Dance-to-Text decoder: {str(e)}")
        raise
    
    return models

def load_texts_and_clusters(config):
    """Load text vocabulary and cluster assignments."""
    texts_file = config['data']['texts_file']
    
    if not texts_file or not os.path.exists(texts_file):
        # Default texts if no file provided
        print("No texts file provided or file not found. Using default text vocabulary.")
        texts = [
            "a slow fluid movement", 
            "an energetic jump",
            "a sharp turn",
            "a smooth wave motion",
            "stretching upward",
            "on your knees",
            "arms extended outward",
            "rapid spinning",
            "bending forward",
            "a balanced pose"
        ]
        # Assign random clusters for demonstration
        cluster_ids = torch.randint(0, 3, (len(texts),))
    else:
        try:
            with open(texts_file, 'r') as f:
                data = json.load(f)
                texts = data['texts']
                cluster_ids = torch.tensor(data['cluster_ids'])
            print(f"Loaded {len(texts)} text descriptions from {texts_file}")
        except Exception as e:
            print(f"Error loading texts file: {str(e)}")
            print("Using default text vocabulary instead.")
            texts = ["a fluid movement", "an energetic jump", "a balanced pose"]
            cluster_ids = torch.randint(0, 3, (len(texts),))
    
    return texts, cluster_ids

def process_dance_sequences(input_path, models, config, device):
    """Process dance sequences and convert to text descriptions."""
    # Load dance sequences
    try:
        print(f"Loading dance sequences from {input_path}...")
        dance_data = np.load(input_path, allow_pickle=True)
        print(f"Loaded {len(dance_data)} dance sequences.")
    except Exception as e:
        print(f"Error loading dance sequences: {str(e)}")
        return None
    
    # Determine which sequences to process
    sample_index = config['inference']['sample_index']
    if sample_index >= 0 and sample_index < len(dance_data):
        # Process only the specified sample
        sequences_to_process = [dance_data[sample_index]]
        indices = [sample_index]
    else:
        # Process all sequences
        sequences_to_process = dance_data
        indices = list(range(len(dance_data)))
    
    # Load text vocabulary and build embedding database
    texts, cluster_ids = load_texts_and_clusters(config)
    models['text_decoder'].build_text_embedding_database(None, texts, cluster_ids)
    
    # Process each sequence
    results = []
    for idx, dance_seq in zip(indices, tqdm(sequences_to_process, desc="Processing dance sequences")):
        # Extract the motion sequence (handle different data formats)
        if isinstance(dance_seq, list) or isinstance(dance_seq, np.ndarray):
            # Handle different data structures
            if len(dance_seq) >= 2:
                # Format [sequence, latent_vector, ...]
                motion = dance_seq[0]
            else:
                # Direct sequence
                motion = dance_seq
        else:
            print(f"Unexpected data format for sequence {idx}. Skipping.")
            continue
        
        # Convert to tensor and add batch dimension if needed
        motion_tensor = torch.tensor(motion, dtype=torch.float32)
        if motion_tensor.dim() == 3:  # [seq_len, n_joints, dims]
            motion_tensor = motion_tensor.unsqueeze(0)  # Add batch dimension
        motion_tensor = motion_tensor.to(device)
        
        # Get latent vector from VAE
        with torch.no_grad():
            _, mu, _ = models['vae'](motion_tensor)
        
        # Project to shared embedding space if projector is available
        if 'dance_projector' in models:
            dance_embeddings = models['dance_projector'](mu)
        else:
            dance_embeddings = mu
        
        # Get text descriptions
        top_k = config['inference']['top_k']
        sample_results = []
        
        for i in range(dance_embeddings.size(0)):
            closest_texts = models['text_decoder'].find_closest_text(
                dance_embeddings[i], top_k=top_k
            )
            
            # Format the results
            matches = []
            for j, (text, similarity) in enumerate(closest_texts, 1):
                matches.append({
                    "rank": j,
                    "text": text,
                    "similarity": float(similarity)  # Convert tensor to Python float
                })
            
            sample_results.append({
                "sequence_index": idx + i,
                "matches": matches
            })
        
        results.extend(sample_results)
    
    return results

def main():
    """Main function for dance-to-text inference."""
    # Parse arguments
    args = parse_arguments()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load models
    models = load_models(config, device)
    
    # Process dance sequences
    results = process_dance_sequences(args.input, models, config, device)
    
    # Save results
    if results:
        with open(args.output, 'w') as f:
            json.dump({
                "config": config,
                "results": results
            }, f, indent=2)
        print(f"Results saved to {args.output}")
        
        # Print sample results to console
        print("\nSample Results:")
        for i, result in enumerate(results[:3]):  # Show first 3 results
            print(f"\nSequence {result['sequence_index']}:")
            for match in result['matches']:
                print(f"  {match['rank']}. {match['text']} (similarity: {match['similarity']:.4f})")
        
        if len(results) > 3:
            print(f"\n... and {len(results) - 3} more sequences")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
