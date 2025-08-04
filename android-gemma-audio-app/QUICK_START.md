# Quick Start Guide - Gemma Audio Processor Android App

## Prerequisites
- Android Studio installed
- Android device with Android 10+ (API 29+)
- USB cable for device connection

## Step 1: Open Project in Android Studio

1. Launch Android Studio
2. Click "Open" 
3. Navigate to `android-gemma-audio-app` folder
4. Click "OK"
5. Wait for Gradle sync to complete (may take 2-5 minutes)

## Step 2: Configure SDK Path

The `local.properties` file should already point to your Android SDK. If you get an error:
1. Go to File → Project Structure → SDK Location
2. Set the Android SDK path (usually `C:\Users\[YourName]\AppData\Local\Android\Sdk` on Windows)

## Step 3: Enable Developer Mode on Your Pixel 9 XL

1. Settings → About phone
2. Tap "Build number" 7 times
3. Go back → System → Developer options
4. Enable "USB debugging"

## Step 4: Connect Your Device

1. Connect your Pixel 9 XL via USB
2. On your phone, tap "Allow" when prompted for USB debugging
3. In Android Studio, you should see your device in the device dropdown

## Step 5: Build and Run

1. Click the green "Run" button (▶️) or press Shift+F10
2. Select your Pixel 9 XL from the device list
3. Click "OK"
4. The app will build and install (first build may take 3-5 minutes)

## Step 6: Grant Permissions

When the app launches:
1. Tap "Grant Permissions"
2. Allow microphone access
3. Allow storage/media access

## Using the App

### File Processing Tab
1. Tap "Select Audio File"
2. Choose an audio file from your device
3. Tap "Process File"
4. View transcription and extracted JSON

### Live Recording Tab
1. Tap the large microphone button to start recording
2. Speak clearly
3. Tap again to stop
4. View real-time transcription and JSON extraction

## Troubleshooting

### "SDK not found" error
- Update `local.properties` with your SDK path
- Or use File → Project Structure → SDK Location

### Build fails with "Could not find dependency"
- Click "Sync Project with Gradle Files" button
- Ensure you have internet connection

### App crashes on launch
- Ensure you have at least 2GB free RAM
- Close other apps
- Try Build → Clean Project, then rebuild

### No device shown
- Check USB cable connection
- Ensure USB debugging is enabled
- Try different USB port
- Install device drivers if on Windows

## Next Steps

To use the actual Gemma model instead of the placeholder:
1. See `GEMMA_CONVERSION_GUIDE.md` for model conversion instructions
2. Replace the placeholder model in `app/src/main/assets/models/`
3. Update `GemmaProcessor.kt` to use the real model

## Performance Notes

- First launch may be slow as resources load
- The Pixel 9 XL's Tensor G4 chip provides excellent ML acceleration
- For best performance, close other apps while using