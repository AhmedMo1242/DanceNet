import os
import numpy as np
import random
from pathlib import Path
import sys
import argparse

# Add the parent directory to Python path to make absolute imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.kmeans_config_utils import load_config, update_config_with_args, create_output_directories
from utils.kmeans_data_utils import load_npy_files, combine_npy_data, add_cluster_id_and_text, split_data, save_processed_data
from clustering.kmeans_clustering import extract_embeddings, perform_kmeans_clustering
from visualization.cluster_vis import apply_pca, plot_clusters, visualize_clusters_with_text

def main():
    """
    Main function for clustering motion data.
    
    This function handles the entire pipeline:
    1. Parse command line arguments
    2. Load and process input .npy files
    3. Perform K-means clustering on extracted embeddings
    4. Add cluster IDs and text descriptions to the data
    5. Generate visualizations
    6. Save processed data
    
    Returns:
        List of paths to saved output files
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process .npy files, cluster them, and add text descriptions")
    
    parser.add_argument("--config", type=str, default="config/kmeans_default_config.yaml",
                        help="Path to the YAML configuration file")
    parser.add_argument("--input_files", nargs="+", default=None,
                        help="List of input .npy files (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output files (overrides config)")
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Number of clusters to use (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config_with_args(config, args)
    
    # Create output directories
    output_dir = create_output_directories(config)
    
    # Load input files
    input_files = [os.path.normpath(f) for f in config["data"]["input_files"]]
    print(f"Loading input files: {input_files}")
    data_list = load_npy_files(input_files)
    
    # Record original sizes for later splitting
    original_sizes = [len(data) for data in data_list]
    print(f"Original file sizes: {original_sizes}")
    
    # Combine data
    combined_data = combine_npy_data(data_list)
    
    # Extract embeddings for clustering
    embeddings = extract_embeddings(combined_data)
    
    # Perform clustering
    n_clusters = config["clustering"]["n_clusters"]
    random_state = config["clustering"].get("random_state", 42)
    cluster_labels, cluster_centers = perform_kmeans_clustering(embeddings, n_clusters, random_state)
    
    # Get text descriptions from config
    cluster_descriptions = config["cluster_descriptions"]
    
    # Add cluster ID and text description to each data point
    updated_data = add_cluster_id_and_text(combined_data, cluster_labels, cluster_descriptions)
    
    # Split data back into original file sizes
    data_splits = split_data(updated_data, original_sizes)
    
    # Prepare output paths
    output_paths = []
    processed_data_dir = output_dir
    
    for i, input_file in enumerate(input_files):
        filename = os.path.basename(input_file)
        output_path = os.path.join(processed_data_dir, filename)
        output_paths.append(output_path)
    
    # Save processed data splits
    saved_paths = save_processed_data(data_splits, output_paths)
    print(f"Saved processed data to: {saved_paths}")
    
    # Create visualizations if enabled
    if config["visualization"].get("create_plots", True):
        print("Creating visualizations...")
        vis_dir = os.path.join(output_dir, "visualizations")
        
        # Apply PCA for visualization
        reduced_embeddings, pca = apply_pca(embeddings)
        
        # Plot clusters
        figsize = tuple(config["visualization"].get("figsize", [12, 10]))
        dpi = config["visualization"].get("dpi", 300)
        colors = config["visualization"].get("colors", None)
        
        # Basic cluster visualization
        plot_path = os.path.join(vis_dir, "cluster_visualization.png")
        plot_clusters(
            reduced_embeddings, 
            cluster_labels, 
            plot_path,
            figsize=figsize,
            dpi=dpi,
            colors=colors,
            title=f'Cluster visualization (k={n_clusters})'
        )
        
        # Extract text descriptions from updated data
        text_descriptions = []
        for item in updated_data:
            if len(item) >= 4 and isinstance(item[3], np.ndarray) and len(item[3]) > 0:
                text_descriptions.append(str(item[3][0]))
            else:
                text_descriptions.append(f"Cluster {cluster_labels[len(text_descriptions)]}")
        
        # Visualization with text descriptions
        text_vis_path = os.path.join(vis_dir, "clusters_with_text.png")
        visualize_clusters_with_text(
            reduced_embeddings,
            cluster_labels,
            text_descriptions,
            text_vis_path,
            figsize=figsize,
            dpi=dpi,
            colors=colors
        )
        
        # Extract original labels if available
        original_labels = []
        for item in combined_data:
            if len(item) >= 2:
                try:
                    if hasattr(item[0], 'label'):
                        original_labels.append(item[0].label)
                    elif isinstance(item[0], (list, tuple, np.ndarray)) and len(item[0]) > 0:
                        original_labels.append(str(item[0]))
                    else:
                        original_labels.append(f"Item_{len(original_labels)}")
                except:
                    original_labels.append(f"Item_{len(original_labels)}")
                
        # Save PCA and cluster data for future use
        np.save(os.path.join(output_dir, "cluster_data", "pca_embeddings.npy"), reduced_embeddings)
        np.save(os.path.join(output_dir, "cluster_data", "cluster_labels.npy"), cluster_labels)
        np.save(os.path.join(output_dir, "cluster_data", "cluster_centers.npy"), cluster_centers)
    
    print(f"Processing complete. Results saved to {output_dir}")
    return saved_paths

if __name__ == "__main__":
    main()
