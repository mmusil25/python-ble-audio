# MediaPipe LLM Setup Guide for Gemma

This guide explains how to set up and use Google's MediaPipe LLM Inference API to run Gemma on your Android device.

## What is MediaPipe LLM Inference?

MediaPipe LLM Inference is Google's solution for running large language models on-device. It provides:
- Optimized models for mobile devices
- Hardware acceleration (GPU/NPU)
- Efficient memory usage
- Easy-to-use API

## Setup Steps

### 1. Download the Gemma Model

1. Visit the Kaggle Gemma page:
   https://www.kaggle.com/models/google/gemma/tfLite/gemma-2b-it-gpu-int4

2. Sign in to Kaggle (create a free account if needed)

3. Accept the model license terms

4. Download `gemma-2b-it-gpu-int4.bin` (approximately 1.4GB)

5. Transfer the file to your Android device

### 2. Import Model in the App

1. Launch the Gemma Audio Processor app
2. You'll see the "Gemma Model Setup" screen
3. Tap "Import Model"
4. Navigate to your Downloads folder
5. Select the `gemma-2b-it-gpu-int4.bin` file
6. Wait for import to complete (may take 30-60 seconds)

### 3. Model is Ready!

Once imported, the model is saved permanently in the app. You won't need to import it again.

## Model Details

**Model**: Gemma 2B IT (Instruction-Tuned)
- **Size**: ~1.4GB (INT4 quantized)
- **Type**: GPU-optimized with INT4 quantization
- **Performance**: 
  - First inference: 2-5 seconds
  - Subsequent inferences: 0.5-2 seconds
  - Depends on device capabilities

## How It Works

1. **Speech Recognition**: Android's built-in speech-to-text converts audio to text
2. **Prompt Engineering**: The app formats the transcript with instructions for JSON extraction
3. **MediaPipe Inference**: Gemma processes the prompt and generates structured output
4. **JSON Parsing**: The response is parsed to extract intent and entities

## Example Flow

```
User speaks: "Please call John at 3 PM tomorrow"
     ↓
Speech-to-Text: "please call john at 3 pm tomorrow"
     ↓
Prompt to Gemma:
"Extract intent and entities from: 'please call john at 3 pm tomorrow'
Return JSON with intent and entities..."
     ↓
Gemma Response:
{
  "transcript": "please call john at 3 pm tomorrow",
  "timestamp_ms": 1234567890,
  "intent": "request",
  "entities": ["call", "john", "3 pm", "tomorrow"]
}
```

## Performance Tips

1. **Close other apps** to free up memory
2. **First run is slower** as the model loads into memory
3. **Keep prompts concise** for faster inference
4. **The Pixel 9 XL's Tensor G4** provides excellent acceleration

## Troubleshooting

### "Model file too large" error
- Ensure you have at least 2GB free storage
- Try clearing app cache of other apps

### "Failed to load model" error
- Verify the downloaded file is complete (should be ~1.4GB)
- Try re-downloading from Kaggle
- Ensure you downloaded the INT4 version

### Slow performance
- Close background apps
- Restart the device
- The first inference after app launch is always slower

## Alternative Models

If you need different capabilities:

1. **Gemma 2B FP16** - Higher quality, larger size
2. **Gemma 2B CPU** - For devices without GPU support
3. **Custom fine-tuned models** - For specific domains

## Privacy & Security

- All processing happens on-device
- No data is sent to servers
- Audio recordings are stored locally
- Model stays on your device

## Resources

- [MediaPipe Documentation](https://developers.google.com/mediapipe/solutions/genai/llm_inference)
- [Gemma Model Card](https://www.kaggle.com/models/google/gemma)
- [Android ML Guide](https://developer.android.com/ml)