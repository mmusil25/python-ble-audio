#!/usr/bin/env python3
"""
Convert Gemma model to TensorFlow Lite format for mobile deployment
This script helps prepare the Gemma model for Android deployment
"""

import os
import sys
import logging
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
    from transformers import AutoTokenizer, TFAutoModelForCausalLM
except ImportError:
    logger.error("Transformers not found. Install with:")
    logger.error("pip install transformers")
    sys.exit(1)


def create_sample_tflite_model(output_path="gemma_quantized.tflite"):
    """
    Create a sample TFLite model for testing.
    
    In production, you would:
    1. Load the actual Gemma model
    2. Convert it to TensorFlow format
    3. Quantize for mobile deployment
    """
    
    logger.info("Creating sample TFLite model for Android app...")
    
    # For actual Gemma conversion:
    # 1. Load Gemma model using transformers
    # model_name = "google/gemma-2b-it"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = TFAutoModelForCausalLM.from_pretrained(model_name)
    
    # 2. Create a concrete function for inference
    # 3. Convert to TFLite with quantization
    
    # For now, create a simple placeholder model
    # This demonstrates the structure but won't actually run Gemma
    
    # Define a simple model
    input_shape = (1, 128)  # batch_size, sequence_length
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(256000, 128, input_length=128),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(256000, activation='softmax')
    ])
    
    # Build the model
    model.build(input_shape)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable quantization for smaller model size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    logger.info(f"Saved TFLite model to {output_path}")
    logger.info(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    return output_path


def create_conversion_instructions():
    """Create detailed instructions for actual Gemma conversion"""
    
    instructions = """
# Gemma Model Conversion Instructions

## Option 1: Use Google's MediaPipe LLM Inference API (Recommended)

Google provides pre-converted Gemma models optimized for mobile through MediaPipe.
This is the easiest way to run Gemma on Android.

1. Visit: https://developers.google.com/mediapipe/solutions/genai/llm_inference
2. Download the Gemma 2B model for Android
3. Follow the integration guide

## Option 2: Manual TensorFlow Lite Conversion

If you need custom conversion:

```python
# 1. Install requirements
pip install tensorflow transformers tf-keras

# 2. Load Gemma model
from transformers import AutoTokenizer, TFAutoModelForCausalLM
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForCausalLM.from_pretrained(model_name)

# 3. Create inference function
@tf.function
def generate(input_ids):
    return model(input_ids, training=False)

# 4. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_concrete_functions([generate.get_concrete_function(
    tf.TensorSpec(shape=[1, None], dtype=tf.int32)
)])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# 5. Save model
with open('gemma_mobile.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Option 3: Use ONNX Runtime Mobile

1. Convert Gemma to ONNX format
2. Use ONNX Runtime Mobile for Android
3. This provides good performance with easier integration

## Model Optimization Tips

1. **Quantization**: Use INT8 quantization for 4x smaller model
2. **Pruning**: Remove unnecessary weights
3. **Knowledge Distillation**: Train a smaller model to mimic Gemma
4. **Model Splitting**: Run only the decoder on device

## Testing Your Converted Model

```python
# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="gemma_mobile.tflite")
interpreter.allocate_tensors()

# Get input/output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
input_data = np.array([[1, 2, 3]], dtype=np.int32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
```
"""
    
    with open("GEMMA_CONVERSION_GUIDE.md", "w") as f:
        f.write(instructions)
    
    logger.info("Created GEMMA_CONVERSION_GUIDE.md with detailed instructions")


def main():
    """Main conversion process"""
    
    logger.info("Gemma to TFLite Conversion Tool")
    logger.info("=" * 50)
    
    # Create output directory
    output_dir = Path("app/src/main/assets/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample model
    model_path = output_dir / "gemma_quantized.tflite"
    create_sample_tflite_model(str(model_path))
    
    # Create conversion guide
    create_conversion_instructions()
    
    logger.info("\nNext Steps:")
    logger.info("1. Read GEMMA_CONVERSION_GUIDE.md for real model conversion")
    logger.info("2. Consider using Google's MediaPipe for easier integration")
    logger.info("3. The sample model is just a placeholder for testing the app")
    
    # Create a model config file for the app
    config = {
        "model_version": "sample_v1",
        "input_max_length": 128,
        "output_max_length": 128,
        "vocab_size": 256000,
        "model_type": "placeholder",
        "notes": "This is a placeholder model. Replace with actual Gemma TFLite model."
    }
    
    import json
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("Created model_config.json")


if __name__ == "__main__":
    main()