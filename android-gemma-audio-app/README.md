# Gemma Audio Processor - Android App

This Android app runs Gemma 3n locally on your device to process audio transcriptions and extract structured JSON data (intent and entities).

## Features

- **File Processing**: Upload audio files for transcription and analysis
- **Live Recording**: Record audio and get real-time transcription and JSON extraction
- **On-Device ML**: Runs Gemma 3n model locally using TensorFlow Lite
- **Speech Recognition**: Uses Android's built-in speech recognition

## Requirements

- Android 10+ (API level 29+)
- At least 4GB RAM (8GB+ recommended)
- ~500MB storage for the Gemma model

## Setup Instructions

### 1. Install Android Studio

1. Download Android Studio from https://developer.android.com/studio
2. Install Android Studio following the setup wizard
3. During setup, make sure to install:
   - Android SDK
   - Android SDK Platform-Tools
   - Android Virtual Device (optional, for emulator)

### 2. Open the Project

1. Open Android Studio
2. Click "Open" and navigate to this `android-gemma-audio-app` folder
3. Wait for Gradle to sync (this may take a few minutes)

### 3. Download Gemma Model

The app uses TensorFlow Lite for on-device inference. On first launch, the app will guide you through model setup:

1. Download a TFLite-compatible model or convert Gemma using the provided script
2. Use the "Import Model" button to load the model file
3. The model is saved permanently after first import

Note: The app currently runs in demo mode with mock JSON extraction until a proper TFLite model is loaded.

### 4. Enable Developer Mode on Your Pixel 9 XL

1. Go to Settings → About phone
2. Tap "Build number" 7 times
3. Go back to Settings → System → Developer options
4. Enable "USB debugging"

### 5. Build and Install

1. Connect your Pixel 9 XL via USB
2. In Android Studio, select your device from the device dropdown
3. Click the "Run" button (green play icon) or press Shift+F10
4. The app will build and install on your phone

### 6. Grant Permissions

When the app first runs, grant these permissions:
- Microphone (for recording)
- Storage (for file access)

## Usage

### File Processing
1. Tap "Upload Audio File"
2. Select an audio file from your device
3. The app will transcribe and extract JSON

### Live Recording
1. Tap "Start Recording"
2. Speak into the microphone
3. Tap "Stop Recording"
4. View transcription and extracted JSON

## Technical Details

- **Speech Recognition**: Android SpeechRecognizer API
- **ML Framework**: TensorFlow Lite with GPU acceleration
- **Model**: Gemma (requires TFLite conversion)
- **UI**: Jetpack Compose
- **On-Device Processing**: All ML inference runs locally

## Troubleshooting

1. **App crashes on startup**: Ensure you have enough free RAM
2. **Model not loading**: Check that the model file is in the correct location
3. **No transcription**: Ensure microphone permission is granted
4. **Slow performance**: Close other apps to free up memory

## Performance Tips

- The first inference may be slow as the model loads
- Subsequent inferences should be faster
- For best performance, close other apps
- The Pixel 9 XL's Tensor G4 chip provides excellent ML acceleration