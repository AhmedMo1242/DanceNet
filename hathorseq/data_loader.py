import numpy as np
from glob import glob
import os
from skeleton_def import point_labels

def load_mocap_data(pattern="data/mariel_*.npy", exclude_joints=False):
    """
    Load mocap data from .npy files
    
    Args:
        pattern: Glob pattern to match .npy files
        exclude_joints: Whether to exclude certain joints (indices 26 and 53)
        
    Returns:
        Dictionary containing raw and processed datasets
    """
    # Load up the datasets, performing some minimal preprocessing
    datasets = {}
    ds_all = []
    
    # Exclude certain markers (indices 26 and 53) only if specified
    exclude_points = [26, 53] if exclude_joints else []
    
    # Expected number of joints from skeleton_def.py
    expected_joints = len(point_labels)
    
    print(f"Loading data from: {pattern}")
    
    # Check if any files were found
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    
    for f in files:
        ds_name = os.path.basename(f)[7:-4] if 'mariel_' in f else os.path.basename(f)[:-4]
        try:
            ds = np.load(f)
            print(f"  Loaded {ds_name}: shape {ds.shape}")
            
            # More validation checks
            if ds.size == 0:
                raise ValueError(f"File {f} contains empty data")
                
            # Make sure the data can be processed as expected
            if len(ds.shape) != 3:
                raise ValueError(f"Expected 3D array, got shape {ds.shape}")
            
            # Determine the data format - the original code assumed (joints, frames, coords)
            # But some files might be in (frames, joints, coords) format
            # Typically coords is 3 (x,y,z), and there are fewer joints than frames
            
            # If the last dimension is 3 (coords), check first two dimensions
            if ds.shape[2] == 3:
                # Detect orientation based on typical patterns
                frames_dim = 1
                joints_dim = 0
                
                # If one dimension is 50-60 (typical joints count) and the other is larger
                if  ds.shape[0] == 55:
                    # First dimension is likely joints
                    frames_dim = 1
                    joint_count = 0
                else:
                    frames_dim = 0
                    joints_dim = 1
                        
                print(f"  Detected format: dimension {joints_dim} is joints, dimension {frames_dim} is frames")
                
                # Create a boolean mask matching the actual number of joints
                joints_count = ds.shape[joints_dim]
                point_mask = np.ones(joints_count, dtype=bool)
                
                # Only apply exclude if requested and indices are within range
                valid_exclude = [idx for idx in exclude_points if idx < joints_count]
                if valid_exclude:
                    point_mask[valid_exclude] = False
                    print(f"  Excluding joint indices: {valid_exclude}")
                else:
                    print(f"  No joints excluded")
                
                # Transpose only if frames are in the second dimension
                if frames_dim == 1 and joints_dim == 0:
                    print(f"  Transposing from (joints, frames, coords) to (frames, joints, coords)")
                    ds = ds.transpose((1, 0, 2))
                    
                    # Now apply masking to the joints dimension (which is now 1) if needed
                    if exclude_joints and len(valid_exclude) > 0:
                        ds = ds[:, point_mask, :]
                else:
                    # If already in (frames, joints, coords) format, apply mask to joints dimension if needed
                    if exclude_joints and len(valid_exclude) > 0:
                        ds = ds[:, point_mask, :]
            else:
                print(f"  Warning: Last dimension is not 3, assuming data is already processed")
                
            # Store the original unfiltered data for later use in exports
            if 'original' not in datasets:
                datasets['original'] = ds.copy()
            
            # REMOVED: Code that forced sequences to be exactly 50 frames
            # We now keep the original frame count
            
            # Pad or trim joints to match expected count if needed (for display purposes)
            joint_count = ds.shape[1]
            if joint_count != expected_joints:
                print(f"  Adjusting joints from {joint_count} to {expected_joints}")
                new_ds = np.zeros((ds.shape[0], expected_joints, ds.shape[2]))
                # Copy existing joints (up to min of actual and expected)
                min_joints = min(joint_count, expected_joints)
                new_ds[:, :min_joints] = ds[:, :min_joints]
                # Replace original with resized version
                ds = new_ds
                
            # Need to invert Z-axis for correct orientation (mocap Z is inverted from matplotlib Z)
            ds[:, :, 2] *= -1  # Re-enable Z inversion - this is actually needed for proper orientation
            
            # Ensure Z is positive for floor contact AFTER inversion
            min_z = np.min(ds[:, :, 2])
            # Shift z values to make minimum zero (or slightly above)
            ds[:, :, 2] -= min_z
            ds[:, :, 2] += 0.05  # Small offset to ensure figure is above ground
            
            datasets[ds_name] = ds
            ds_all.append(ds)
            
        except Exception as e:
            print(f"Error loading {f}: {str(e)}")
            raise

    ds_counts = np.array([ds.shape[0] for ds in ds_all])
    ds_offsets = np.zeros_like(ds_counts)
    ds_offsets[1:] = np.cumsum(ds_counts[:-1])

    ds_all = np.concatenate(ds_all)
    print("Dataset contains {:,} timesteps of {} joints with {} dimensions each.".format(
        ds_all.shape[0], ds_all.shape[1], ds_all.shape[2]))

    # Normalize data with better centering - ensure figure is in center of axes
    low, hi = np.quantile(ds_all, [0.01, 0.99], axis=(0, 1))
    
    # Handle each dimension separately for better control
    for dim in range(2):  # Just handle X and Y for centering
        dim_min = low[dim]
        dim_max = hi[dim]
        dim_center = (dim_min + dim_max) / 2
        # Center the dimension data around origin
        ds_all[:, :, dim] -= dim_center
        # Scale to reasonable range
        dim_range = max(0.1, dim_max - dim_min)  # Avoid division by very small values
        ds_all[:, :, dim] /= dim_range / 1.5
    
    # Scale up to better fit standard axes
    ds_all *= 1.5  # Scale up to make figure appear more prominently

    # Create centered version (remove mean x,y position from each frame)
    ds_all_centered = ds_all.copy()
    ds_all_centered[:, :, :2] -= ds_all_centered[:, :, :2].mean(axis=1, keepdims=True)

    # Process individual datasets with same transformation
    datasets_centered = {}
    for ds_name in datasets:
        if ds_name == 'original':
            continue  # Skip the original unfiltered data
            
        ds = datasets[ds_name].copy()  # Make a copy to avoid modifying original
        
        # Apply same centering as ds_all
        for dim in range(2):  # Just handle X and Y for centering
            dim_min = low[dim]
            dim_max = hi[dim]
            dim_center = (dim_min + dim_max) / 2
            ds[:, :, dim] -= dim_center
            dim_range = max(0.1, dim_max - dim_min)
            ds[:, :, dim] /= dim_range / 1.5
            
        ds *= 1.5  # Match the scaling of ds_all
        
        # Now create the centered version by removing per-frame mean
        ds_centered = ds.copy()
        ds_centered[:, :, :2] -= ds[:, :, :2].mean(axis=1, keepdims=True)
        
        datasets_centered[ds_name] = ds_centered
    
    # Calculate velocities (first velocity is always 0)
    # Adjust to match the actual number of joints
    joints_count = ds_all.shape[1]
    velocities = np.vstack([
        np.zeros((1, joints_count, 3)),
        np.array([35 * (ds_all[t+1, :, :] - ds_all[t, :, :]) for t in range(len(ds_all)-1)])
    ])  # (delta_x/y/z per frame) * (35 frames/sec)
    
    # Stack positions and velocities
    ds_all_with_vel = np.dstack([ds_all, velocities])
    ds_all_centered_with_vel = np.dstack([ds_all_centered, velocities])
    
    # Normalize both datasets
    for data in [ds_all_with_vel, ds_all_centered_with_vel]:
        # Normalize locations & velocities (separately) to [-1, 1]
        loc_min = np.min(data[:, :, :3])
        loc_max = np.max(data[:, :, :3])
        vel_min = np.min(data[:, :, 3:])
        vel_max = np.max(data[:, :, 3:])
        print("loc_min: {:.3f}, loc_max: {:.3f}".format(loc_min, loc_max))
        print("vel_min: {:.3f}, vel_max: {:.3f}".format(vel_min, vel_max))
        data[:, :, :3] = (data[:, :, :3] - loc_min) * 2 / (loc_max - loc_min) - 1
        data[:, :, 3:] = (data[:, :, 3:] - vel_min) * 2 / (vel_max - vel_min) - 1
    
    return {
        'raw': ds_all,
        'centered': ds_all_centered,
        'with_velocity': ds_all_with_vel,
        'centered_with_velocity': ds_all_centered_with_vel,
        'individual': datasets,
        'individual_centered': datasets_centered,
        'counts': ds_counts,
        'original': datasets.get('original', None)  # Add the original unfiltered data
    }
