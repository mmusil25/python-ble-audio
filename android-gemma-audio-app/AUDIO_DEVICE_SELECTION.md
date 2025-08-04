# Audio Device Selection Feature

## Overview
This update adds audio input device selection to the Android Gemma Audio app, allowing users to choose between different microphones and audio input sources on their device.

## Changes Made

### 1. Fixed Speech Recognition Error Handling
- **Error 7 (NO_MATCH)** is now properly handled with a user-friendly message: "No speech detected. Please speak clearly into the microphone."
- Added comprehensive error messages for all speech recognition error codes
- Users will now see specific guidance based on the error type

### 2. Audio Device Selection
- Added `AudioDeviceManager.kt` to handle device enumeration and selection
- Users can now select from available audio input devices including:
  - Built-in Microphone
  - Bluetooth Headset
  - Wired Headset
  - USB Audio Devices

### 3. UI Improvements
- Added a dropdown menu at the top of the Live Recording screen
- Users can see all available audio devices and select their preferred input
- The selection is disabled while recording to prevent audio interruption
- Selected device shows a microphone icon and "Currently selected" label

## How to Use

1. **Launch the app** and navigate to the "Live Recording" tab
2. **View available devices** in the "Audio Input Device" dropdown at the top
3. **Select your preferred device** from the dropdown menu
4. **Start recording** - the app will use your selected audio device
5. If you get an error, check the specific error message for guidance

## Supported Devices
- **Android 6.0+ (API 23+)**: Full audio device enumeration and selection
- **Older devices**: Falls back to default microphone

## Technical Details

### Audio Device Selection (API 28+)
On Android P and above, the app can set the preferred audio device for MediaRecorder using `setPreferredDevice()`.

### Error Codes Reference
- **ERROR_AUDIO (3)**: Audio recording error
- **ERROR_NO_MATCH (7)**: No speech detected
- **ERROR_NETWORK (2)**: Network connection issue
- **ERROR_SPEECH_TIMEOUT (6)**: No speech input detected

## Troubleshooting

1. **"No speech detected" error**:
   - Ensure you're speaking clearly
   - Check that the correct audio device is selected
   - Verify microphone permissions are granted

2. **Device not appearing in list**:
   - Ensure the device is properly connected
   - For Bluetooth devices, ensure they're paired and connected
   - Try refreshing the screen or restarting the app

3. **Recording still using wrong microphone**:
   - On some devices, the system may override app preferences
   - Try disconnecting other audio devices
   - Check system audio settings