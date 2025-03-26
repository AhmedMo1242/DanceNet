import numpy as np
import torch
import os
from tqdm import tqdm

def process_and_save_npy(model, input_path, output_path, config, device='cuda'):
    """
    Process a dataset file and create a new NPY file with text embeddings at index 4.
    
    This function loads an NPY dataset, extracts text data according to config parameters,
    generates embeddings using the provided model, and saves a new NPY file where each item
    has 5 dimensions with the model's 256d output as the 5th dimension.
    
    Args:
        model: Trained TextEncoder model
        input_path: Path to input NPY file
        output_path: Path to save the processed NPY file
        config: Configuration dictionary with keys:
            - text_index: Index of text data in each item
            - text_subindex: Subindex for nested text data
            - process_batch_size: Batch size for processing
        device: Device to run model on ('cuda' or 'cpu')
    
    Returns:
        str: Path to the saved processed file
    """
    print(f"Processing data from {input_path}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load the original data
    original_data = np.load(input_path, allow_pickle=True)
    print(f"Loaded {len(original_data)} samples")
    
    # Extract required information
    text_index = config['text_index']
    text_subindex = config['text_subindex']
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Extract texts
    texts = []
    valid_indices = []
    
    print("Extracting texts...")
    for idx, item in enumerate(tqdm(original_data)):
        try:
            # Extract the text
            if isinstance(item[text_index], (list, tuple, np.ndarray)) and len(item[text_index]) > text_subindex:
                text = str(item[text_index][text_subindex])
            else:
                text = str(item[text_index])
            
            # Skip empty texts
            if not text.strip():
                continue
                
            texts.append(text)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Warning: Error processing item {idx}: {e}")
    
    print(f"Extracted {len(texts)} valid items")
    
    if len(texts) == 0:
        raise ValueError("No valid items found in the NPY file")
    
    # Generate text embeddings using the model
    print("Generating text embeddings...")
    text_embeddings = []
    batch_size = config['process_batch_size']
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_outputs = model(batch_texts)
            text_embeddings.append(batch_outputs.cpu().numpy())
    
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    
    # Create the new data
    print("Creating processed dataset...")
    new_data = []
    for i, idx in enumerate(valid_indices):
        # Create a new version of the original item
        new_item = list(original_data[idx])
        
        # Ensure the list has at least 5 elements
        while len(new_item) < 4:
            new_item.append(None)
        
        # Add the text embedding at index 4 (5th dimension)
        if len(new_item) > 4:
            new_item[4] = text_embeddings[i]
        else:
            new_item.append(text_embeddings[i])
        
        new_data.append(new_item)
    
    # Save the new NPY file
    np.save(output_path, np.array(new_data, dtype=object))
    print(f"Saved processed data to {output_path} ({len(new_data)} samples)")
    
    return output_path

def process_all_datasets(model, train_path, val_path, test_path, output_dir, config, device='cuda'):
    """
    Process all dataset files (train, val, test) and create 5-dimensional processed versions.
    
    This function applies the text encoder model to all three dataset splits and creates
    new NPY files with the model's 256d output as the 5th dimension.
    
    Args:
        model: Trained TextEncoder model
        train_path: Path to training data NPY file
        val_path: Path to validation data NPY file
        test_path: Path to test data NPY file
        output_dir: Directory to save processed files
        config: Configuration dictionary with processing parameters
        device: Device to run model on ('cuda' or 'cpu')
    
    Returns:
        dict: Dictionary with paths to processed files for each split
    """
    os.makedirs(output_dir, exist_ok=True)
    
    processed_paths = {}
    
    # Process training data
    train_output = os.path.join(output_dir, 'train_processed.npy')
    processed_paths['train'] = process_and_save_npy(
        model, train_path, train_output, config, device
    )
    
    # Process validation data
    val_output = os.path.join(output_dir, 'val_processed.npy')
    processed_paths['val'] = process_and_save_npy(
        model, val_path, val_output, config, device
    )
    
    # Process test data
    test_output = os.path.join(output_dir, 'test_processed.npy')
    processed_paths['test'] = process_and_save_npy(
        model, test_path, test_output, config, device
    )
    
    return processed_paths
