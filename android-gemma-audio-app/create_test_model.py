#!/usr/bin/env python3
"""
Create a small test TFLite model for the Android app
"""

import tensorflow as tf
import numpy as np

def create_test_model():
    """Create a simple text classification model for testing"""
    
    # Create a simple model
    model = tf.keras.Sequential([
        # Input layer for text (as integers)
        tf.keras.layers.Input(shape=(100,), dtype=tf.int32),
        
        # Embedding layer
        tf.keras.layers.Embedding(input_dim=1000, output_dim=64),
        
        # Simple processing
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        
        # Output layer for intent classification
        tf.keras.layers.Dense(7, activation='softmax')  # 7 intent categories
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model created:")
    model.summary()
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save as .tflite
    with open('test_intent_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"\nSaved test_intent_model.tflite")
    print(f"Size: {len(tflite_model) / 1024:.1f} KB")
    
    # Also save as .bin
    with open('test_intent_model.bin', 'wb') as f:
        f.write(tflite_model)
    
    print(f"Also saved as test_intent_model.bin")
    
    # Create a mock Gemma-style response model
    response_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(50,), dtype=tf.int32),
        tf.keras.layers.Embedding(input_dim=500, output_dim=32),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(100, activation='softmax')
    ])
    
    converter2 = tf.lite.TFLiteConverter.from_keras_model(response_model)
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model2 = converter2.convert()
    
    with open('mock_gemma_tiny.bin', 'wb') as f:
        f.write(tflite_model2)
    
    print(f"\nSaved mock_gemma_tiny.bin")
    print(f"Size: {len(tflite_model2) / 1024:.1f} KB")
    
    print("\nThese small models can be used to test the app!")


if __name__ == "__main__":
    create_test_model()