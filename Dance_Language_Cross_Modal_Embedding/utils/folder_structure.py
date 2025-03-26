import os
import argparse

def print_directory_tree(directory, output_file=None, indent='', exclude_dirs=None, exclude_extensions=None):
    """
    Prints the directory structure as a tree and optionally saves it to a file.
    
    Args:
        directory: Root directory to start from
        output_file: Optional file to save the tree structure
        indent: Current indentation (used in recursion)
        exclude_dirs: List of directory names to exclude
        exclude_extensions: List of file extensions to exclude
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', 'venv', '.ipynb_checkpoints']
    
    if exclude_extensions is None:
        exclude_extensions = ['.pyc']
    
    output = []
    
    # Get all items in the directory and sort them (directories first, then files)
    items = sorted(os.listdir(directory))
    dirs = [item for item in items if os.path.isdir(os.path.join(directory, item)) and item not in exclude_dirs]
    files = [item for item in items if os.path.isfile(os.path.join(directory, item)) and 
             not any(item.endswith(ext) for ext in exclude_extensions)]
    
    # Process directories first
    for i, dir_name in enumerate(dirs):
        is_last_dir = (i == len(dirs) - 1 and len(files) == 0)
        connector = '└── ' if is_last_dir else '├── '
        
        dir_path = os.path.join(directory, dir_name)
        output.append(f"{indent}{connector}{dir_name}/")
        
        # Recursively process subdirectory with appropriate indentation
        next_indent = indent + ('    ' if is_last_dir else '│   ')
        sub_output = print_directory_tree(
            dir_path, 
            None, 
            next_indent, 
            exclude_dirs, 
            exclude_extensions
        )
        output.extend(sub_output)
    
    # Process files
    for i, file_name in enumerate(files):
        is_last = (i == len(files) - 1)
        connector = '└── ' if is_last else '├── '
        output.append(f"{indent}{connector}{file_name}")
    
    # If this is the root call and an output file is specified, write to the file
    if indent == '' and output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output))
    
    return output

def main():
    parser = argparse.ArgumentParser(description='Generate a directory tree structure')
    parser.add_argument('--dir', type=str, default='.', help='Root directory to start from')
    parser.add_argument('--output', type=str, help='Output file to save the tree structure')
    parser.add_argument('--exclude-dirs', type=str, nargs='+', help='Directories to exclude')
    parser.add_argument('--exclude-extensions', type=str, nargs='+', help='File extensions to exclude')
    
    args = parser.parse_args()
    
    # Convert the directory path to absolute path if needed
    directory = os.path.abspath(args.dir)
    
    # Print the directory name as the root of the tree
    print(os.path.basename(directory) + "/")
    
    # Generate and print the tree
    tree = print_directory_tree(
        directory,
        args.output,
        exclude_dirs=args.exclude_dirs,
        exclude_extensions=args.exclude_extensions
    )
    print('\n'.join(tree))
    
    if args.output:
        print(f"\nTree structure has been saved to {args.output}")

if __name__ == "__main__":
    main()
