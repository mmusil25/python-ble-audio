#!/usr/bin/env python3
"""
Extract TFLite model from .task file
Gemma .task files are often ZIP archives containing the model
"""

import sys
import zipfile
import os
from pathlib import Path

def extract_tflite_from_task(task_file_path, output_dir=None):
    """Extract TFLite model from .task file"""
    
    task_path = Path(task_file_path)
    if not task_path.exists():
        print(f"Error: File not found: {task_path}")
        return False
    
    # If no output directory specified, use the same directory as the input
    if output_dir is None:
        output_dir = task_path.parent
    
    print(f"Analyzing {task_path.name}...")
    print(f"File size: {task_path.stat().st_size / (1024**3):.2f} GB")
    
    # Check if it's a ZIP file
    try:
        with zipfile.ZipFile(task_path, 'r') as zip_ref:
            print("\nFile is a ZIP archive. Contents:")
            
            # List all files
            for info in zip_ref.filelist:
                print(f"  - {info.filename} ({info.file_size / (1024**2):.1f} MB)")
            
            # Look for .tflite files
            tflite_files = [f for f in zip_ref.filelist if f.filename.endswith('.tflite')]
            
            if tflite_files:
                for tflite_file in tflite_files:
                    output_path = Path(output_dir) / tflite_file.filename.split('/')[-1]
                    print(f"\nExtracting {tflite_file.filename} to {output_path}")
                    
                    with zip_ref.open(tflite_file) as source, open(output_path, 'wb') as target:
                        target.write(source.read())
                    
                    print(f"Successfully extracted to: {output_path}")
                    print(f"Extracted size: {output_path.stat().st_size / (1024**3):.2f} GB")
                return True
            else:
                print("\nNo .tflite files found in the archive.")
                
                # Look for TF_LITE model components
                tf_lite_components = [f for f in zip_ref.filelist if f.filename.startswith('TF_LITE')]
                
                if tf_lite_components:
                    print(f"\nFound {len(tf_lite_components)} TensorFlow Lite components.")
                    print("These are likely TFLite models without the .tflite extension.")
                    
                    # Extract the main decoder model
                    decoder_file = next((f for f in tf_lite_components if 'PREFILL_DECODE' in f.filename), None)
                    if decoder_file:
                        output_path = Path(output_dir) / (task_path.stem + '_decoder.tflite')
                        print(f"\nExtracting main decoder model: {decoder_file.filename}")
                        print(f"Output: {output_path}")
                        
                        with zip_ref.open(decoder_file) as source, open(output_path, 'wb') as target:
                            target.write(source.read())
                        
                        print(f"Successfully extracted decoder model!")
                        print(f"Size: {output_path.stat().st_size / (1024**3):.2f} GB")
                        
                        # Also extract the tokenizer if needed
                        tokenizer_file = next((f for f in tf_lite_components if 'TOKENIZER' in f.filename), None)
                        if tokenizer_file:
                            tokenizer_path = Path(output_dir) / (task_path.stem + '_tokenizer.model')
                            print(f"\nExtracting tokenizer: {tokenizer_file.filename}")
                            
                            with zip_ref.open(tokenizer_file) as source, open(tokenizer_path, 'wb') as target:
                                target.write(source.read())
                            
                            print(f"Tokenizer saved to: {tokenizer_path}")
                        
                        return True
                    
                print("\nNo suitable model components found.")
                
                # Check if the archive contains the model in a different format
                bin_files = [f for f in zip_ref.filelist if f.filename.endswith('.bin')]
                if bin_files:
                    print(f"Found {len(bin_files)} .bin files - this might be the model in binary format")
                    
                    # If there's a single large .bin file, it might be the model
                    large_bins = [f for f in bin_files if f.file_size > 1024**3]  # > 1GB
                    if large_bins:
                        print(f"\nLarge binary file found: {large_bins[0].filename}")
                        print("This might be the model file. Extracting and renaming to .tflite...")
                        
                        bin_file = large_bins[0]
                        output_path = Path(output_dir) / task_path.stem + '.tflite'
                        
                        with zip_ref.open(bin_file) as source, open(output_path, 'wb') as target:
                            target.write(source.read())
                        
                        print(f"Extracted to: {output_path}")
                        return True
                
                return False
                
    except zipfile.BadZipFile:
        print("\nFile is not a ZIP archive.")
        
        # Check if it might already be a TFLite file with wrong extension
        with open(task_path, 'rb') as f:
            header = f.read(4)
        
        if header == b'TFL3':
            print("File appears to be a TFLite model with .task extension!")
            output_path = Path(output_dir) / (task_path.stem + '.tflite')
            
            print(f"Copying to: {output_path}")
            import shutil
            shutil.copy2(task_path, output_path)
            print("Done!")
            return True
        else:
            print(f"Unknown file format. Header: {header.hex()}")
            print("\nThe .task file format is not recognized.")
            print("You may need to download the model in .tflite format directly from Kaggle.")
            return False
    
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_tflite_from_task.py <path_to_task_file> [output_directory]")
        sys.exit(1)
    
    task_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = extract_tflite_from_task(task_file, output_dir)
    sys.exit(0 if success else 1)