"""
Script to generate multiple visualization styles for all mocap files in a directory
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from data_loader import load_mocap_data
from skeleton_visualization import animate_stick, save_animation

def generate_visualizations(input_dir, output_dir, max_frames=200, formats=None, styles=None, ffmpeg_path=None):
    """
    Generate multiple visualization styles for mocap files
    
    Args:
        input_dir: Directory containing .npy files
        output_dir: Directory to save visualizations
        max_frames: Maximum number of frames to include in each visualization
        formats: List of output formats (mp4, gif)
        styles: List of styles to generate (if None, generate all)
        ffmpeg_path: Path to ffmpeg executable (optional)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Default formats if none specified
    if formats is None or len(formats) == 0:
        formats = ["mp4"]
    
    # Find all .npy files in the input directory
    npy_files = glob(os.path.join(input_dir, "*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {input_dir}. Please check the path.")
        return
    
    print(f"Found {len(npy_files)} .npy files to process")
    
    # Define all available visualization styles
    all_visualization_styles = [
        {
            "name": "lines_only",
            "description": "Only skeleton lines",
            "settings": {
                "skeleton": True,
                "skeleton_alpha": 0.9,
                "dot_size": 0,
                "dot_alpha": 0,
                "cloud": False,
                "show_background": False,
            }
        },
        {
            "name": "dots_only",
            "description": "Only joint markers",
            "settings": {
                "skeleton": False,
                "dot_size": 50,
                "dot_alpha": 1.0,
                "cloud": False,
                "show_background": False,
            }
        },
        {
            "name": "lines_and_dots",
            "description": "Both skeleton lines and joint markers",
            "settings": {
                "skeleton": True,
                "skeleton_alpha": 0.8,
                "dot_size": 30,
                "dot_alpha": 0.8,
                "cloud": False,
                "show_background": False,
            }
        },
        {
            "name": "with_background",
            "description": "Visualization with background and axes",
            "settings": {
                "skeleton": True,
                "skeleton_alpha": 0.8,
                "dot_size": 30,
                "dot_alpha": 0.8,
                "cloud": False,
                "show_background": True,
            }
        },
        {
            "name": "cloud",
            "description": "Joint cloud visualization",
            "settings": {
                "skeleton": True,
                "skeleton_alpha": 0.6,
                "dot_size": 25,
                "dot_alpha": 0.7,
                "cloud": True,
                "cloud_alpha": 0.03,
                "show_background": False,
            }
        }
    ]
    
    # Filter styles if specified
    if styles is not None and len(styles) > 0:
        visualization_styles = [style for style in all_visualization_styles if style["name"] in styles]
        if not visualization_styles:
            print(f"No matching styles found. Available styles: {', '.join([s['name'] for s in all_visualization_styles])}")
            return
    else:
        visualization_styles = all_visualization_styles
    
    # Process each .npy file
    for npy_file in tqdm(npy_files, desc="Processing files"):
        file_basename = os.path.basename(npy_file).split('.')[0]
        print(f"\nProcessing {file_basename}...")
        
        try:
            # Load the data
            data = load_mocap_data(npy_file, exclude_joints=True)
            sequence = data['raw']
            
            print(f"  Loaded sequence with shape {sequence.shape}")
            
            # Use a shorter segment for faster processing
            if sequence.shape[0] <= max_frames:
                segment = sequence
            else:
                segment = sequence[:max_frames]
            
            print(f"  Creating visualizations with {segment.shape[0]} frames")
            
            # Generate visualizations for each style and format
            for style in tqdm(visualization_styles, desc="Generating styles", leave=False):
                print(f"  Generating {style['name']} visualization...")
                
                # Extract settings
                settings = style["settings"].copy()
                
                # Extract show_background to avoid duplicate keyword
                show_background = settings.pop("show_background", False)
                
                # Create figure and animation
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=-10, azim=-70)
                
                anim = animate_stick(
                    segment,
                    figsize=(10, 8),
                    speed=40,
                    lw=3,
                    ax_lims=(-0.6, 0.6),
                    show_background=show_background,
                    fig=fig,
                    ax=ax,
                    **settings
                )
                
                # Save in each requested format
                for fmt in formats:
                    output_file = os.path.join(output_dir, f"{file_basename}_{style['name']}.{fmt}")
                    save_result = save_animation(anim, output_file, fps=25, ffmpeg_path=ffmpeg_path)
                    
                    if save_result:
                        print(f"    Saved to {output_file}")
                    else:
                        print(f"    Failed to save {output_file}")
                
                plt.close(fig)  # Close the figure to free memory
                
        except Exception as e:
            print(f"  Error processing {npy_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nVisualization generation complete!")

def main():
    parser = argparse.ArgumentParser(description="Generate multiple visualizations from mocap data")
    parser.add_argument("--input", "-i", help="Input directory with .npy files", default="./examples")
    parser.add_argument("--output", "-o", help="Output directory", default="./output/visualizations")
    parser.add_argument("--frames", "-f", type=int, help="Maximum frames per visualization", default=200)
    parser.add_argument("--formats", choices=["mp4", "gif"], nargs="+", default=["mp4"], 
                        help="Output formats (can specify multiple)")
    parser.add_argument("--styles", nargs="+", 
                        choices=["lines_only", "dots_only", "lines_and_dots", "with_background", "cloud"],
                        help="Visualization styles to generate (default: all)")
    parser.add_argument("--ffmpeg", help="Path to ffmpeg executable", default=None)
    
    args = parser.parse_args()
    
    generate_visualizations(args.input, args.output, args.frames, args.formats, args.styles, args.ffmpeg)

if __name__ == "__main__":
    main()
