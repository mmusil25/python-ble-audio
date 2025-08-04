#!/usr/bin/env python3
"""
Extract all model components from .task file as .bin files
"""

import sys
import zipfile
from pathlib import Path

def extract_all_models(task_file_path, output_dir=None):
    """Extract all model components as .bin files"""
    
    task_path = Path(task_file_path)
    if not task_path.exists():
        print(f"Error: File not found: {task_path}")
        return False
    
    if output_dir is None:
        output_dir = task_path.parent
    
    output_path = Path(output_dir)
    
    print(f"Extracting all models from {task_path.name}...")
    
    try:
        with zipfile.ZipFile(task_path, 'r') as zip_ref:
            # Extract all TF_LITE components
            for file_info in zip_ref.filelist:
                if file_info.filename.startswith('TF_LITE'):
                    output_file = output_path / f"{file_info.filename}.bin"
                    print(f"\nExtracting: {file_info.filename}")
                    print(f"Size: {file_info.file_size / (1024**2):.1f} MB")
                    print(f"Output: {output_file}")
                    
                    with zip_ref.open(file_info) as source:
                        with open(output_file, 'wb') as target:
                            target.write(source.read())
                    
                    print(f"✓ Saved as {output_file.name}")
            
            print("\nExtraction complete!")
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        task_file = input("Enter path to .task file: ")
    else:
        task_file = sys.argv[1]
    
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    extract_all_models(task_file, output_dir)