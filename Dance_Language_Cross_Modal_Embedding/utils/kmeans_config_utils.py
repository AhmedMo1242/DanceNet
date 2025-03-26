import os
import yaml
from pathlib import Path

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Loaded configuration or default configuration if loading fails
    """
    print(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration")
        return {
            "data": {
                "input_files": ["x.npy", "y.npy", "z.npy"],
                "output_dir": "./output",
                "create_folders": True
            },
            "clustering": {
                "method": "kmeans",
                "n_clusters": 3,
                "random_state": 42
            },
            "cluster_descriptions": {
                0: ["Dynamic hip swings", "Rhythmic torso undulations"],
                1: ["Controlled half-turns", "Directional strides"],
                2: ["Low crouching sweeps", "Descending body plunges"]
            },
            "visualization": {
                "create_plots": True,
                "figsize": [12, 10],
                "dpi": 300,
                "colors": ['#FF5733', '#33A8FF', '#33FF57']
            }
        }

def update_config_with_args(config, args):
    """
    Update configuration with command line arguments.
    
    Args:
        config (dict): Current configuration
        args (Namespace): Command line arguments
        
    Returns:
        dict: Updated configuration
    """
    if args.input_files:
        config["data"]["input_files"] = args.input_files
        
    if args.output_dir:
        config["data"]["output_dir"] = args.output_dir
        
    if args.n_clusters:
        config["clustering"]["n_clusters"] = args.n_clusters
        
    return config

def create_output_directories(config):
    """
    Create output directories based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        str: Path to the main output directory
    """
    output_dir = config["data"]["output_dir"]
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories if needed
    if config["data"].get("create_folders", True):
        os.makedirs(os.path.join(output_dir, "processed_data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "cluster_data"), exist_ok=True)
        
    return output_dir
