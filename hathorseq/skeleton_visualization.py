import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import juggle_axes
from tqdm import tqdm
import matplotlib
from skeleton_def import point_labels, skeleton_lines
import os

def generate_edge_indices():
    """Generate indices for skeleton edges and full cloud connections"""
    # Normal, connected skeleton
    skeleton_idxs = []
    for g1, g2 in skeleton_lines:
        entry = []
        entry.append([point_labels.index(l) for l in g1])
        entry.append([point_labels.index(l) for l in g2])
        skeleton_idxs.append(entry)

    # Cloud of every point connected
    cloud_idxs = []
    cloud_idxs_names = []
    for i in range(53):
        for j in range(53):
            if i == j:
                continue  # skip self-loops
            entry = []
            entry.append([i])
            entry.append([j])
            cloud_idxs.append(entry)
            
            name_entry = []
            name_entry.append([point_labels[i]])
            name_entry.append([point_labels[j]])
            cloud_idxs_names.append(name_entry)
    
    all_idxs = skeleton_idxs + cloud_idxs
    
    return {
        'skeleton': skeleton_idxs,
        'cloud': cloud_idxs,
        'cloud_names': cloud_idxs_names,
        'all': all_idxs
    }

def get_line_segments(seq, edge_indices, zcolor=None, cmap=None, edge_types=None, edge_class=None):
    """Calculate the coordinates for the lines in the visualization"""
    all_idxs = edge_indices['all']
    skeleton_idxs = edge_indices['skeleton']
    
    xline = np.zeros((seq.shape[0], len(all_idxs), 3, 2))
    if cmap:
        colors = np.zeros((len(all_idxs), 4))
    
    # Get the maximum valid joint index
    max_joint_idx = seq.shape[1] - 1
    
    for edge, (joint1, joint2) in enumerate(all_idxs):
        # Check if the joint indices are within bounds
        j1_filtered = [j for j in joint1 if j <= max_joint_idx]
        j2_filtered = [j for j in joint2 if j <= max_joint_idx]
        
        # Skip this edge if no valid joints
        if not j1_filtered or not j2_filtered:
            # Set the line to the origin to avoid errors
            xline[:, edge, :, 0] = 0
            xline[:, edge, :, 1] = 0
            continue
            
        # Use only valid joints for calculating line endpoints
        try:
            xline[:, edge, :, 0] = np.mean(seq[:, j1_filtered], axis=1)
            xline[:, edge, :, 1] = np.mean(seq[:, j2_filtered], axis=1)
        except Exception as e:
            print(f"Error processing edge {edge} with joints {joint1}->{joint2}: {e}")
            # Set to origin in case of error
            xline[:, edge, :, 0] = 0
            xline[:, edge, :, 1] = 0
        
        if cmap:
            if edge_types is not None:
                if edge >= len(skeleton_idxs):  # cloud edges
                    if edge_types[edge - len(skeleton_idxs), edge_class] == 1:
                        colors[edge] = cmap(1)
                    else:
                        colors[edge] = cmap(0)
            else:
                colors[edge] = cmap(0)
    
    if cmap:
        return xline, colors
    else:
        return xline, None

# Add a new function to adapt edge indices based on joint count
def adapt_edges_to_sequence(seq, edge_indices):
    """
    Adapts edge indices to match the available joints in the sequence
    
    Args:
        seq: Motion sequence with shape [frames, joints, dimensions]
        edge_indices: Dictionary containing edge definitions
        
    Returns:
        Filtered edge indices dictionary with only valid edges
    """
    # Get the number of joints in the sequence
    num_joints = seq.shape[1]
    
    filtered_indices = {}
    for key in edge_indices:
        if key in ['skeleton', 'cloud', 'cloud_names']:
            # Filter each group to only include valid joint indices
            filtered_list = []
            for joint_pair in edge_indices[key]:
                g1, g2 = joint_pair
                
                # Only keep pairs where all joints are within range
                if (all(j < num_joints for j in g1) and 
                    all(j < num_joints for j in g2)):
                    filtered_list.append(joint_pair)
                    
            filtered_indices[key] = filtered_list
            
    # Regenerate 'all' from filtered skeleton and cloud
    if 'skeleton' in filtered_indices and 'cloud' in filtered_indices:
        filtered_indices['all'] = filtered_indices['skeleton'] + filtered_indices['cloud']
    
    return filtered_indices

def put_lines(ax, segments, edge_indices, color=None, lw=2.5, skeleton=True, skeleton_alpha=0.3, 
             cloud=False, cloud_alpha=0.03, edge_types=None, edge_opacities=None, 
             threshold=0, edge_class=None):
    """Put line segments on the given axis, with given colors"""
    skeleton_idxs = edge_indices['skeleton']
    cloud_idxs = edge_indices['cloud']
    cloud_idxs_names = edge_indices['cloud_names']
    
    lines = []
    # Main skeleton
    for i in tqdm(range(len(skeleton_idxs)), desc="Skeleton lines"):
        if isinstance(color, (list, tuple, np.ndarray)):
            c = color[i]
        else:
            c = color
        
        alpha = skeleton_alpha if skeleton else 0
        
        # Plot the main skeleton
        l = ax.plot(
            np.linspace(segments[i, 0, 0], segments[i, 0, 1], 2),
            np.linspace(segments[i, 1, 0], segments[i, 1, 1], 2),
            np.linspace(segments[i, 2, 0], segments[i, 2, 1], 2),
            color=c,
            alpha=alpha,
            lw=lw
        )[0]
        lines.append(l)
    
    if cloud:
        # Cloud of all-connected joints
        for i in tqdm(range(len(cloud_idxs)), desc="Cloud lines"):
            if isinstance(color, (list, tuple, np.ndarray)):
                c = color[i + len(skeleton_idxs)]
            else:
                c = color
            
            # Plot or don't plot lines based on edge class
            if edge_types is not None and edge_class is not None:
                custom_colors = ["deeppink", "red", "blue", "green", "orange"]
                if edge_types[i][edge_class] == 1:
                    if edge_opacities is not None and edge_opacities[i, edge_class] > threshold:
                        alpha = 0.5
                        print("Surviving edge: {} | ({} -> {}), i.e. ({} -> {})".format(
                            i, cloud_idxs[i][0], cloud_idxs[i][1], 
                            cloud_idxs_names[i][0], cloud_idxs_names[i][1]))
                    else:
                        alpha = cloud_alpha
                    
                    l = ax.plot(
                        np.linspace(segments[i + len(skeleton_idxs), 0, 0], segments[i + len(skeleton_idxs), 0, 1], 2),
                        np.linspace(segments[i + len(skeleton_idxs), 1, 0], segments[i + len(skeleton_idxs), 1, 1], 2),
                        np.linspace(segments[i + len(skeleton_idxs), 2, 0], segments[i + len(skeleton_idxs), 2, 1], 2),
                        color=custom_colors[edge_class],
                        alpha=alpha,
                        lw=lw
                    )[0]
                    lines.append(l)
                else:
                    l = ax.plot(
                        np.linspace(segments[i + len(skeleton_idxs), 0, 0], segments[i + len(skeleton_idxs), 0, 1], 2),
                        np.linspace(segments[i + len(skeleton_idxs), 1, 0], segments[i + len(skeleton_idxs), 1, 1], 2),
                        np.linspace(segments[i + len(skeleton_idxs), 2, 0], segments[i + len(skeleton_idxs), 2, 1], 2),
                        alpha=0,
                        color="white",
                        lw=lw
                    )[0]
                    lines.append(None)
            else:  # regular cloud
                l = ax.plot(
                    np.linspace(segments[i + len(skeleton_idxs), 0, 0], segments[i + len(skeleton_idxs), 0, 1], 2),
                    np.linspace(segments[i + len(skeleton_idxs), 1, 0], segments[i + len(skeleton_idxs), 1, 1], 2),
                    np.linspace(segments[i + len(skeleton_idxs), 2, 0], segments[i + len(skeleton_idxs), 2, 1], 2),
                    alpha=cloud_alpha,
                    lw=lw
                )[0]
                lines.append(l)
    
    return lines

def animate_stick(seq, ghost=None, ghost_shift=0, edge_types=None, edge_opacities=None, 
                  threshold=0, edge_class=None, figsize=(10, 8), zcolor=None, pointer=None, 
                  ax_lims=(-0.4, 0.4), speed=45, dot_size=20, dot_alpha=0.5, lw=2.5, 
                  cmap='cool_r', pointer_color='black', cloud=False, cloud_alpha=0.03, 
                  skeleton=True, skeleton_alpha=0.3, save_path=None, show_background=False,
                  fig=None, ax=None):
    """
    Animate a video of the stick figure.
    
    Args:
        seq: The motion sequence to animate (shape [frames, joints, dimensions])
        ghost: Optional second sequence to superimpose (shape [frames, joints, dimensions])
        ghost_shift: Lateral shift between primary and ghost sequence
        edge_types: Types of edges to highlight
        edge_opacities: Opacity values for edges
        threshold: Threshold for edge opacities
        edge_class: Class of edges to display
        figsize: Figure size (width, height)
        zcolor: Optional N-length array for coloring vertices
        pointer: Optional direction indicator
        ax_lims: Axis limits (min, max)
        speed: Animation speed (milliseconds per frame)
        dot_size: Size of joint markers
        dot_alpha: Opacity of joint markers
        lw: Line width
        cmap: Colormap name
        pointer_color: Color of pointer
        cloud: Whether to draw full cloud of connections
        cloud_alpha: Opacity of cloud connections
        skeleton: Whether to draw skeleton
        skeleton_alpha: Opacity of skeleton
        save_path: Path to save animation (None = don't save)
        show_background: Whether to show background and axes
        fig: Optional existing figure to use
        ax: Optional existing 3D axes to use
        
    Returns:
        Animation object
    """
    if zcolor is None:
        zcolor = np.zeros(seq.shape[1])
    
    # Create figure and 3D axis if not provided
    if fig is None or ax is None:
        fig = plt.figure(figsize=figsize)
        ax = p3.Axes3D(fig)
        
        # Set a better default view to ensure figure is right-side up
        ax.view_init(elev=-10, azim=-70)  # Use negative elevation to fix upside-down view
    
    # Set background and axes visibility
    if not show_background:
        # Eliminate background lines/axes
        ax.axis('off')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
        
        # Set figure background opacity to 0
        fig.patch.set_alpha(0.)
    else:
        # Show background with grid and labels
        ax.axis('on')
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.set_frame_on(True)
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Motion Capture Visualization')
    
    # Apply ghost shift if needed
    if ghost_shift and ghost is not None:
        seq = seq.copy()
        ghost = ghost.copy()
        seq[:, :, 0] -= ghost_shift
        ghost[:, :, 0] += ghost_shift
    
    # Setup colormap
    if isinstance(cmap, str):
        cm = matplotlib.cm.get_cmap(cmap)
    else:
        # Allow passing a custom colormap directly
        cm = cmap
    
    # Generate edge indices and adapt them to the sequence's joint count
    edge_indices = generate_edge_indices()
    
    # Check if we need to adapt edges to the current sequence
    if seq.shape[1] < len(point_labels):
        print(f"Warning: Sequence has {seq.shape[1]} joints, but skeleton expects {len(point_labels)}.")
        print("Adapting edges to match available joints...")
        edge_indices = adapt_edges_to_sequence(seq, edge_indices)
    
    # Setup joint markers
    if edge_types is not None:
        num_joints = seq.shape[1]
        num_connections = np.zeros((num_joints))
        for edge, (joint1, joint2) in enumerate(edge_indices['all']):
            valid_joint1 = [j for j in joint1 if j < num_joints]
            valid_joint2 = [j for j in joint2 if j < num_joints]
            
            if not valid_joint1 or not valid_joint2:
                continue
                
            if edge >= len(edge_indices['skeleton']) and edge_types[edge - len(edge_indices['skeleton']), edge_class] == 1:
                for j in valid_joint1:
                    num_connections[j] += 1
                for j in valid_joint2:
                    num_connections[j] += 1
        
        # Normalize so colormap can use values between 0 and 1
        num_connections = num_connections / np.max(num_connections) if np.max(num_connections) > 0 else num_connections
        dot_color = [cm(num_connections[joint]) for joint in range(num_joints)]
        pts = ax.scatter(seq[0, :, 0], seq[0, :, 1], seq[0, :, 2], c=dot_color, s=dot_size, alpha=dot_alpha)
    else:
        dot_color = "black"
        pts = ax.scatter(seq[0, :, 0], seq[0, :, 1], seq[0, :, 2], c=dot_color, s=dot_size, alpha=dot_alpha)
    
    # Add ghost markers if needed
    ghost_color = 'blue'
    if ghost is not None:
        pts_g = ax.scatter(ghost[0, :, 0], ghost[0, :, 1], ghost[0, :, 2], c=ghost_color, s=dot_size, alpha=dot_alpha)
    
    # Set axis limits to center the figure
    if ax_lims:
        # Calculate the range
        x_range = ax_lims[1] - ax_lims[0]
        
        # Find Z extent of the figure
        z_min = np.min(seq[:, :, 2])
        z_max = np.max(seq[:, :, 2])
        z_range = max(0.1, z_max - z_min)  # Ensure not too small
        
        # Set symmetric limits around origin for X and Y
        ax.set_xlim(*ax_lims)
        ax.set_ylim(*ax_lims)
        
        # Position Z with proper centering vertically
        z_center = (z_min + z_max) / 2
        z_padding = z_range * 0.1  # Add some padding
        ax.set_zlim(z_min - z_padding, z_max + z_padding)
        
        # Set equal aspect ratio 
        ax.set_box_aspect([1, 1, 0.5])  # Adjust Z aspect
    
    # Calculate line segments with adapted edge indices
    try:
        xline, colors = get_line_segments(seq, edge_indices, zcolor, cm, edge_types=edge_types, edge_class=edge_class)
        lines = put_lines(ax, xline[0], edge_indices, color=colors, lw=lw, cloud=cloud, 
                     cloud_alpha=cloud_alpha, edge_types=edge_types, edge_opacities=edge_opacities, 
                     threshold=threshold, edge_class=edge_class, skeleton=skeleton, skeleton_alpha=skeleton_alpha)
    except Exception as e:
        print(f"Error creating line segments: {e}")
        # Return empty animation to avoid crash
        def empty_update(t):
            return []
        anim = animation.FuncAnimation(fig, empty_update, 1, interval=speed, blit=False)
        return anim
    
    # Add ghost lines if needed
    if ghost is not None:
        xline_g, _ = get_line_segments(ghost, edge_indices)
        lines_g = put_lines(ax, xline_g[0], edge_indices, ghost_color, lw=lw, 
                          cloud=cloud, cloud_alpha=cloud_alpha, skeleton=skeleton, skeleton_alpha=skeleton_alpha)
    
    # Add pointer if needed
    if pointer is not None:
        vR = 0.15
        dX, dY = vR * np.cos(pointer), vR * np.sin(pointer)
        
        # Find a valid joint index for CLAV (or use the first joint if not found)
        try:
            zidx = point_labels.index('CLAV')
            if zidx >= seq.shape[1]:
                zidx = 0  # Use first joint if CLAV is out of bounds
        except ValueError:
            zidx = 0  # Use first joint if CLAV is not found
            
        X = seq[:, zidx, 0]
        Y = seq[:, zidx, 1]
        Z = seq[:, zidx, 2]
        quiv = ax.quiver(X[0], Y[0], Z[0], dX[0], dY[0], 0, color=pointer_color)
        ax.quiv = quiv
    
    # Define update function for animation
    def update(t):
        pts._offsets3d = juggle_axes(seq[t, :, 0], seq[t, :, 1], seq[t, :, 2], 'z')
        for i, l in enumerate(lines):
            if l is not None:
                l.set_data(xline[t, i, :2])
                l.set_3d_properties(xline[t, i, 2])
        
        if ghost is not None:
            pts_g._offsets3d = juggle_axes(ghost[t, :, 0], ghost[t, :, 1], ghost[t, :, 2], 'z')
            for i, l in enumerate(lines_g):
                l.set_data(xline_g[t, i, :2])
                l.set_3d_properties(xline_g[t, i, 2])
        
        if pointer is not None:
            ax.quiv.remove()
            ax.quiv = ax.quiver(X[t], Y[t], Z[t], dX[t], dY[t], 0, color=pointer_color)
    
    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        len(seq),
        interval=speed,
        blit=False,
    )
    
    # Save animation if path provided
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=1000/speed)
    
    return anim

def save_animation(anim, filename, fps=25, ffmpeg_path=None):
    """
    Save animation to file with proper error handling
    
    Args:
        anim: Animation object
        filename: Path to save the animation
        fps: Frames per second
        ffmpeg_path: Optional path to ffmpeg executable
    """
    try:
        if ffmpeg_path:
            # Set ffmpeg path if provided
            plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
            
        # Try creating the writer with specified fps
        try:
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='MocapViz'), bitrate=1800)
            anim.save(filename, writer=writer)
            print(f"Animation saved to {filename}")
            return True
        except FileNotFoundError:
            print("FFmpeg not found. Trying alternative save method...")
            
            # If original save fails, try PillowWriter for GIF output
            if filename.lower().endswith('.mp4'):
                gif_filename = filename.replace('.mp4', '.gif')
            else:
                gif_filename = f"{os.path.splitext(filename)[0]}.gif"
                
            try:
                writer = animation.PillowWriter(fps=fps)
                anim.save(gif_filename, writer=writer)
                print(f"Animation saved as GIF to {gif_filename}")
                return True
            except Exception as e:
                print(f"Error saving animation as GIF: {e}")
                
                # Last resort: try to use HTML output
                try:
                    html_filename = f"{os.path.splitext(filename)[0]}.html"
                    html = anim.to_jshtml()
                    with open(html_filename, 'w') as f:
                        f.write(html)
                    print(f"Animation saved as HTML to {html_filename}")
                    return True
                except Exception as e:
                    print(f"Error saving animation as HTML: {e}")
                    return False
    except Exception as e:
        print(f"Error saving animation: {e}")
        return False
