#!/usr/bin/env python3
"""
Convert Gemma .task file to TensorFlow Lite format
This handles the specific format used by MediaPipe/Gemma models
"""

import os
import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for required libraries
try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    logger.error("Required libraries not found. Install with:")
    logger.error("pip install tensorflow numpy")
    sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks.python.genai import converter
except ImportError:
    logger.warning("MediaPipe not found. Trying alternative conversion method...")
    logger.warning("For best results, install: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False
else:
    MEDIAPIPE_AVAILABLE = True


def convert_with_mediapipe(input_path: str, output_path: str):
    """Convert using MediaPipe's built-in converter"""
    logger.info("Using MediaPipe converter...")
    
    try:
        # MediaPipe genai converter for .task files
        converter.convert_genai_model(
            input_path=input_path,
            output_path=output_path
        )
        logger.info(f"Successfully converted to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"MediaPipe conversion failed: {e}")
        return False


def convert_with_tensorflow(input_path: str, output_path: str):
    """Fallback conversion using TensorFlow"""
    logger.info("Attempting TensorFlow-based conversion...")
    
    try:
        # For .task files, we need to extract the model
        # These files often contain both the model and metadata
        
        # Try to read as a SavedModel format
        if input_path.endswith('.task'):
            # .task files might be zip archives containing the model
            import zipfile
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Try to extract as zip
                    with zipfile.ZipFile(input_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Look for model files
                    model_path = None
                    for root, dirs, files in os.walk(temp_dir):
                        if 'saved_model.pb' in files:
                            model_path = root
                            break
                        elif any(f.endswith('.tflite') for f in files):
                            # Already a tflite file inside
                            tflite_file = next(f for f in files if f.endswith('.tflite'))
                            import shutil
                            shutil.copy(os.path.join(root, tflite_file), output_path)
                            logger.info(f"Found and extracted TFLite model to: {output_path}")
                            return True
                    
                    if model_path:
                        # Convert SavedModel to TFLite
                        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                        converter.target_spec.supported_types = [tf.int8]  # For INT4 quantization
                        tflite_model = converter.convert()
                        
                        with open(output_path, 'wb') as f:
                            f.write(tflite_model)
                        
                        logger.info(f"Successfully converted to: {output_path}")
                        return True
                        
                except zipfile.BadZipFile:
                    logger.warning("Not a valid zip file, trying direct conversion...")
        
        # If not a zip, try direct conversion
        # This might work if it's already in a compatible format
        with open(input_path, 'rb') as f:
            model_data = f.read()
        
        # Check if it might already be a TFLite file with different extension
        if model_data[:4] == b'TFL3':
            logger.info("File appears to already be in TFLite format!")
            with open(output_path, 'wb') as f:
                f.write(model_data)
            logger.info(f"Copied to: {output_path}")
            return True
        
        logger.error("Could not determine model format")
        return False
        
    except Exception as e:
        logger.error(f"TensorFlow conversion failed: {e}")
        return False


def analyze_file(input_path: str):
    """Analyze the .task file to understand its format"""
    logger.info(f"Analyzing file: {input_path}")
    
    file_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    logger.info(f"File size: {file_size:.2f} MB")
    
    # Check file signature
    with open(input_path, 'rb') as f:
        header = f.read(16)
    
    logger.info(f"File header: {header[:8].hex()}")
    
    # Check if it's a zip file
    if header[:2] == b'PK':
        logger.info("File appears to be a ZIP archive")
        try:
            import zipfile
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                logger.info("Contents:")
                for info in zip_ref.filelist:
                    logger.info(f"  - {info.filename} ({info.file_size} bytes)")
        except Exception as e:
            logger.error(f"Failed to read as ZIP: {e}")
    
    # Check if it's already TFLite
    elif header[:4] == b'TFL3':
        logger.info("File appears to already be in TFLite format!")
    
    else:
        logger.info("Unknown file format")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemma .task file to TensorFlow Lite format"
    )
    parser.add_argument(
        "input_file",
        help="Path to input .task file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output .tflite file path (default: same directory as input)",
        default=None
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze the file without converting"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.tflite')
    
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    
    # Analyze file
    analyze_file(str(input_path))
    
    if args.analyze_only:
        return
    
    # Try conversion
    success = False
    
    if MEDIAPIPE_AVAILABLE:
        success = convert_with_mediapipe(str(input_path), str(output_path))
    
    if not success:
        success = convert_with_tensorflow(str(input_path), str(output_path))
    
    if success:
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(f"Conversion successful!")
        logger.info(f"Output file size: {output_size:.2f} MB")
        logger.info(f"Output saved to: {output_path}")
    else:
        logger.error("Conversion failed!")
        logger.error("The .task file format might not be supported.")
        logger.error("Try downloading the model in .tflite format directly from:")
        logger.error("https://www.kaggle.com/models/google/gemma/tfLite/")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments, use the specific file mentioned
        default_file = r"G:\My Drive\Career\Upwork\Gavin McNally\gemma-3n-E2B-it-int4.task"
        if os.path.exists(default_file):
            sys.argv.extend([default_file])
        else:
            print("Usage: python convert_gemma_task_to_tflite.py <input.task> [-o output.tflite]")
            print(f"\nDefault file not found: {default_file}")
            sys.exit(1)
    
    main()