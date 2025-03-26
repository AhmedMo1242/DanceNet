import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation as R

def load_motion_data(data_dir):
    """
    Load motion data from numpy files in the specified directory.
    
    Args:
        data_dir (str): Directory containing motion data files (.npy)
        
    Returns:
        list: List of loaded motion data arrays
    """
    npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
    data_list = []
    
    for npy_file in npy_files:
        try:
            motion_data = np.load(npy_file)
            data_list.append(motion_data)
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
    
    return data_list

def segment_sequences(motion_data, seq_length=50, stride=20):
    """
    Segment motion data into fixed-length sequences with a specified stride.
    
    Args:
        motion_data (numpy.ndarray): Motion data with shape (n_joints, n_timesteps, n_dims)
        seq_length (int): Length of each sequence
        stride (int): Stride between consecutive sequences
        
    Returns:
        list: List of segmented sequences
    """
    n_joints, n_timesteps, n_dims = motion_data.shape
    sequences = []
    
    for start_idx in range(0, n_timesteps - seq_length + 1, stride):
        end_idx = start_idx + seq_length
        sequence = motion_data[:, start_idx:end_idx, :]
        
        if sequence.shape[1] == seq_length:
            sequences.append(sequence)
    
    return sequences

def augment_sequence(sequence):
    """
    Apply various augmentations to a motion sequence.
    
    Args:
        sequence (numpy.ndarray): Motion sequence with shape (n_joints, seq_length, n_dims)
        
    Returns:
        list: List of augmented sequences including the original
    """
    augmentations = []
    
    # Original sequence
    augmentations.append(sequence.copy())
    
    # Rotation around vertical axis by 15 degrees
    augmented = sequence.copy()
    angle = np.pi/12
    rot = R.from_euler('y', angle)
    
    for j in range(augmented.shape[0]):
        for t in range(augmented.shape[1]):
            xz_coords = augmented[j, t, [0, 2]]
            rotated = rot.apply(np.array([xz_coords[0], 0, xz_coords[1]]))
            augmented[j, t, 0] = rotated[0]
            augmented[j, t, 2] = rotated[2]
    
    augmentations.append(augmented)
    
    # Rotation in opposite direction
    augmented = sequence.copy()
    angle = -np.pi/12
    rot = R.from_euler('y', angle)
    
    for j in range(augmented.shape[0]):
        for t in range(augmented.shape[1]):
            xz_coords = augmented[j, t, [0, 2]]
            rotated = rot.apply(np.array([xz_coords[0], 0, xz_coords[1]]))
            augmented[j, t, 0] = rotated[0]
            augmented[j, t, 2] = rotated[2]
    
    augmentations.append(augmented)
    
    # Scaling by 1.05
    augmented = sequence.copy() * 1.05
    augmentations.append(augmented)
    
    # Scaling by 0.95
    augmented = sequence.copy() * 0.95
    augmentations.append(augmented)
    
    # Mirroring along x-axis
    augmented = sequence.copy()
    augmented[:, :, 0] = -augmented[:, :, 0]
    augmentations.append(augmented)
    
    # Time warp
    augmented = sequence.copy()
    seq_length = augmented.shape[1]
    mid_point = seq_length // 2
    
    orig_time = np.arange(seq_length)
    warped_time = np.zeros_like(orig_time, dtype=float)
    
    warped_time[:mid_point] = np.linspace(0, mid_point * 1.2, mid_point)
    warped_time[mid_point:] = np.linspace(mid_point * 1.2, seq_length, seq_length - mid_point)
    
    warped_sequence = sequence.copy()
    for j in range(sequence.shape[0]):
        for d in range(sequence.shape[2]):
            warped_sequence[j, :, d] = np.interp(orig_time, warped_time, sequence[j, :, d])
    
    augmentations.append(warped_sequence)
    
    return augmentations

def normalize_sequences(sequences):
    """
    Normalize motion sequences by computing mean and standard deviation across all data.
    
    Args:
        sequences (list): List of motion sequences
        
    Returns:
        tuple: (normalized_sequences, mean, std)
    """
    all_data = np.concatenate([seq.reshape(-1, seq.shape[-1]) for seq in sequences], axis=0)
    
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    std[std < 1e-10] = 1.0
    
    normalized_sequences = []
    for seq in sequences:
        normalized_seq = (seq - mean) / std
        normalized_sequences.append(normalized_seq)
    
    return normalized_sequences, mean, std
