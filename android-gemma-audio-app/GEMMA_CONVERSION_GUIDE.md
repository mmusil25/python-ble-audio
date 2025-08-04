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