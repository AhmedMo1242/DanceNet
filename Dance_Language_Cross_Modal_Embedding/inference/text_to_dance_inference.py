import os
import sys
import argparse
import yaml
import torch
import numpy as np
from datetime import datetime

# Add the parent directory to Python path to make absolute imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Import model classes
from models.text_encoder import TextEncoder
from models.contrastive_models import ProjectionHead
from models.text_to_dance_decoder_models import EfficientDecoderGRU

def parse_args():
    parser = argparse.ArgumentParser(description="Text to Dance Sequence Generation")
    parser.add_argument("--config", type=str, default="../config/inference_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--text", type=str, required=True,
                        help="Text description to generate dance from")
    parser.add_argument("--output_path", type=str, 
                        help="Path to save the generated dance sequence (overrides config)")
    parser.add_argument("--device", type=str,
                        help="Device to run inference on (overrides config)")
    parser.add_argument("--sequence_length", type=int,
                        help="Length of dance sequence to generate (overrides config)")
    parser.add_argument("--text_encoder_path", type=str,
                        help="Path to text encoder model (overrides config)")
    parser.add_argument("--projectors_path", type=str,
                        help="Path to projection heads model (overrides config)")
    parser.add_argument("--decoder_path", type=str,
                        help="Path to decoder model (overrides config)")
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device(device_arg, config):
    """Determine device to use for inference"""
    if device_arg:
        return device_arg
    if 'inference' in config and 'device' in config['inference']:
        return config['inference']['device']
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_models(config, args, device):
    """Load all required models for inference"""
    print(f"Loading models to {device}...")
    
    # Determine model paths (command line args override config)
    text_encoder_path = args.text_encoder_path or config['model_paths']['text_encoder']
    projectors_path = args.projectors_path or config['model_paths']['projectors']
    decoder_path = args.decoder_path or config['model_paths']['decoder']
    
    # Load text encoder
    text_encoder = TextEncoder(freeze_bert=True).to(device)
    try:
        text_encoder_checkpoint = torch.load(text_encoder_path, map_location=device)
        text_encoder.load_state_dict(text_encoder_checkpoint)
        print(f"Loaded text encoder from {text_encoder_path}")
    except Exception as e:
        print(f"Error loading text encoder: {e}")
        sys.exit(1)

    # Load projection heads
    try:
        projectors = torch.load(projectors_path, map_location=device)
        text_projection = ProjectionHead().to(device)
        dance_projection = ProjectionHead().to(device)
        text_projection.load_state_dict(projectors['other_projector'])
        dance_projection.load_state_dict(projectors['dance_projector'])
        print(f"Loaded projection heads from {projectors_path}")
    except Exception as e:
        print(f"Error loading projection heads: {e}")
        sys.exit(1)

    # Load decoder
    decoder = EfficientDecoderGRU(latent_dim=256, hidden_dim=384, n_joints=55, n_dims=3).to(device)
    try:
        decoder_checkpoint = torch.load(decoder_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'decoder_state_dict' in decoder_checkpoint:
            decoder.load_state_dict(decoder_checkpoint['decoder_state_dict'])
        elif 'model_state_dict' in decoder_checkpoint:
            decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
        else:
            decoder.load_state_dict(decoder_checkpoint)
        print(f"Loaded decoder from {decoder_path}")
    except Exception as e:
        print(f"Error loading decoder: {e}")
        sys.exit(1)
    
    # Set models to evaluation mode
    text_encoder.eval()
    text_projection.eval()
    dance_projection.eval()
    decoder.eval()
    
    return text_encoder, text_projection, dance_projection, decoder

def generate_dance_from_text(text, text_encoder, text_projection, decoder, seq_length, device):
    """Generate dance sequence from text description"""
    print(f"Generating dance sequence for: '{text}'")
    
    with torch.no_grad():
        # Text encoding
        text_emb = text_encoder([text]).to(device)
        
        # Project to shared embedding space
        shared_emb = text_projection(text_emb)
        
        # Decode to motion sequence
        generated = decoder(shared_emb, seq_length)
    
    # Convert to numpy array
    dance_sequence = generated.squeeze(0).cpu().numpy()
    print(f"Generated sequence shape: {dance_sequence.shape}")
    
    return dance_sequence

def save_sequence(dance_sequence, output_path=None, config=None):
    """Save generated dance sequence to file"""
    if output_path is None:
        # Create output filename based on timestamp if not specified
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = config['output']['save_dir'] if config and 'output' in config else '.'
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"generated_dance_{timestamp}.npy")
    
    # Save sequence
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    np.save(output_path, dance_sequence)
    print(f"Dance sequence saved to {output_path}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine device
    device = get_device(args.device, config)
    print(f"Using device: {device}")
    
    # Determine sequence length
    seq_length = args.sequence_length or config['inference']['sequence_length']
    
    # Load models
    text_encoder, text_projection, _, decoder = load_models(config, args, device)
    
    # Generate dance sequence
    dance_sequence = generate_dance_from_text(
        args.text, text_encoder, text_projection, decoder, seq_length, device
    )
    
    # Save generated sequence
    save_sequence(dance_sequence, args.output_path, config)
    
    print("Generation complete!")

if __name__ == "__main__":
    main()
