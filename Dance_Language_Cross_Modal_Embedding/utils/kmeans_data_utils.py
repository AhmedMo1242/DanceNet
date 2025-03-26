import numpy as np
import os
import random

def load_npy_files(file_paths):
    """
    Load and validate multiple .npy files.
    
    Args:
        file_paths (list): List of paths to .npy files
        
    Returns:
        list: List of loaded numpy arrays
        
    Raises:
        ValueError: If no valid .npy files could be loaded
    """
    loaded_data = []
    
    for path in file_paths:
        try:
            data = np.load(path, allow_pickle=True)
            print(f"Loaded {path}, shape: {data.shape}")
            loaded_data.append(data)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
            
    if not loaded_data:
        raise ValueError("No valid .npy files could be loaded")
        
    return loaded_data

def combine_npy_data(data_list):
    """
    Combine multiple numpy arrays into a single array.
    
    Args:
        data_list (list): List of numpy arrays to combine
        
    Returns:
        numpy.ndarray: Combined numpy array
    """
    combined_data = []
    
    for data in data_list:
        if isinstance(data, np.ndarray):
            for item in data:
                if isinstance(item, (list, tuple, np.ndarray)):
                    combined_data.append(item)
                else:
                    print(f"Warning: Skipping an item with unexpected format")
    
    # Convert to numpy array
    combined_data = np.array(combined_data, dtype=object)
    print(f"Combined data shape: {combined_data.shape}")
    
    return combined_data

def add_cluster_id_and_text(data, cluster_labels, cluster_descriptions):
    """
    Add cluster ID and text description to each data point.
    
    Args:
        data (numpy.ndarray): Numpy array of data points
        cluster_labels (numpy.ndarray): Array of cluster labels for each data point
        cluster_descriptions (dict): Dictionary mapping cluster IDs to text descriptions
        
    Returns:
        numpy.ndarray: Updated numpy array with cluster IDs and text descriptions
    """
    updated_data = []
    
    for i, item in enumerate(data):
        # Get cluster ID for this data point
        cluster_id = cluster_labels[i]
        
        # Choose a random text description for this cluster
        if cluster_id in cluster_descriptions and cluster_descriptions[cluster_id]:
            text_desc = random.choice(cluster_descriptions[cluster_id])
        else:
            text_desc = f"Cluster {cluster_id} motion"
        
        # Create updated data point
        if len(item) >= 2:  # Ensure item has at least 2 elements
            # Create a new item with cluster ID and text description
            new_item = list(item)
            
            # Add cluster ID at index 2
            if len(new_item) <= 2:
                new_item.append(np.array([cluster_id]))
            else:
                new_item[2] = np.array([cluster_id])
                
            # Add text description at index 3
            if len(new_item) <= 3:
                new_item.append(np.array([text_desc]))
            else:
                new_item[3] = np.array([text_desc])
                
            updated_data.append(new_item)
        else:
            print(f"Warning: Skipping item at index {i} due to unexpected format")
    
    # Convert to numpy array
    updated_data = np.array(updated_data, dtype=object)
    print(f"Updated data shape: {updated_data.shape}")
    
    return updated_data

def split_data(data, sizes):
    """
    Split combined data into separate arrays.
    
    Args:
        data (numpy.ndarray): Combined numpy array
        sizes (list): List of sizes for each split
        
    Returns:
        list: List of split numpy arrays
    """
    splits = []
    start_idx = 0
    
    for size in sizes:
        if start_idx + size <= len(data):
            splits.append(data[start_idx:start_idx+size])
            start_idx += size
        else:
            # If not enough data for this split, use remaining data
            splits.append(data[start_idx:])
            break
    
    # If there are fewer splits than expected, add empty arrays
    while len(splits) < len(sizes):
        splits.append(np.array([]))
    
    return splits

def save_processed_data(data_splits, output_paths):
    """
    Save processed data splits to output paths.
    
    Args:
        data_splits (list): List of numpy arrays to save
        output_paths (list): List of paths to save to
        
    Returns:
        list: List of saved file paths
    """
    saved_paths = []
    
    for i, (data, path) in enumerate(zip(data_splits, output_paths)):
        try:
            np.save(path, data)
            print(f"Saved processed data to {path}, shape: {data.shape}")
            saved_paths.append(path)
        except Exception as e:
            print(f"Error saving to {path}: {e}")
    
    return saved_paths
