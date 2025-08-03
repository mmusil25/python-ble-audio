#!/usr/bin/env python3
"""
BLE Audio Capture Module
Captures audio from BLE device (OmiAudio) or microphone and saves as WAV file

BLE Specifications:
- Device name: "OmiAudio"
- 16 kHz / 16-bit mono PCM
- 320-byte notify packets
- Notify UUID: 0000ffb2-0000-1000-8000-00805f9b34fb
"""

import asyncio
import wave
import numpy as np
import sounddevice as sd
from datetime import datetime
from pathlib import Path
import argparse
import logging

# BLE imports (for future implementation)
try:
    import bleak
    from bleak import BleakClient, BleakScanner
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False
    print("Warning: bleak not available. Install with: pip install bleak")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio specifications (matching BLE requirements)
SAMPLE_RATE = 16000  # 16 kHz
CHANNELS = 1         # Mono
DTYPE = np.int16     # 16-bit PCM
BLOCKSIZE = 320      # Matching BLE packet size

# BLE specifications
BLE_DEVICE_NAME = "OmiAudio"
NOTIFY_UUID = "0000ffb2-0000-1000-8000-00805f9b34fb"


class AudioCapture:
    """Base class for audio capture"""
    
    def __init__(self, output_dir="captured_audio", duration=5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.duration = duration
        self.audio_buffer = []
        
    def save_wav(self, audio_data, filename=None):
        """Save audio data as WAV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.wav"
        
        filepath = self.output_dir / filename
        
        # Convert to int16 if needed
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # Save as WAV
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        
        logger.info(f"Saved audio to: {filepath}")
        return filepath


class MicrophoneCapture(AudioCapture):
    """Capture audio from microphone using sounddevice"""
    
    def capture(self):
        """Capture audio from microphone"""
        logger.info(f"Starting microphone capture for {self.duration} seconds...")
        logger.info(f"Sample rate: {SAMPLE_RATE} Hz, Channels: {CHANNELS}, Format: 16-bit PCM")
        
        try:
            # Record audio
            audio_data = sd.rec(
                int(self.duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCKSIZE
            )
            
            # Wait for recording to complete
            sd.wait()
            
            logger.info("Recording complete!")
            
            # Save to WAV
            filepath = self.save_wav(audio_data)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            raise


class BLEAudioCapture(AudioCapture):
    """Capture audio from BLE device (OmiAudio)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not BLEAK_AVAILABLE:
            raise RuntimeError("bleak library not available for BLE capture")
        
        self.device_address = None
        self.client = None
        self.is_capturing = False
        
    async def find_device(self):
        """Find OmiAudio BLE device"""
        logger.info(f"Scanning for BLE device: {BLE_DEVICE_NAME}")
        
        devices = await BleakScanner.discover()
        for device in devices:
            if device.name and BLE_DEVICE_NAME in device.name:
                logger.info(f"Found device: {device.name} [{device.address}]")
                return device.address
        
        logger.warning(f"Device '{BLE_DEVICE_NAME}' not found")
        return None
    
    def notification_handler(self, sender, data):
        """Handle BLE notifications (audio packets)"""
        # Convert bytes to int16 array
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        self.audio_buffer.append(audio_chunk)
        
        # Log progress
        total_samples = sum(len(chunk) for chunk in self.audio_buffer)
        duration_captured = total_samples / SAMPLE_RATE
        logger.debug(f"Received {len(data)} bytes, total duration: {duration_captured:.2f}s")
    
    async def capture_async(self):
        """Capture audio from BLE device"""
        # Find device
        self.device_address = await self.find_device()
        if not self.device_address:
            logger.error("Falling back to microphone capture - BLE device not found")
            # Fallback to microphone
            mic_capture = MicrophoneCapture(self.output_dir, self.duration)
            return mic_capture.capture()
        
        # Connect to device
        async with BleakClient(self.device_address) as client:
            self.client = client
            logger.info(f"Connected to {BLE_DEVICE_NAME}")
            
            # Start notifications
            await client.start_notify(NOTIFY_UUID, self.notification_handler)
            logger.info(f"Started receiving audio notifications for {self.duration} seconds")
            
            # Capture for specified duration
            self.is_capturing = True
            await asyncio.sleep(self.duration)
            self.is_capturing = False
            
            # Stop notifications
            await client.stop_notify(NOTIFY_UUID)
            logger.info("Stopped receiving notifications")
        
        # Combine audio buffer
        if self.audio_buffer:
            audio_data = np.concatenate(self.audio_buffer)
            filepath = self.save_wav(audio_data)
            return filepath
        else:
            logger.error("No audio data received")
            return None
    
    def capture(self):
        """Synchronous wrapper for capture"""
        return asyncio.run(self.capture_async())


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Capture audio from BLE device or microphone")
    parser.add_argument("--source", choices=["ble", "mic"], default="mic",
                        help="Audio source (default: mic)")
    parser.add_argument("--duration", type=float, default=5,
                        help="Recording duration in seconds (default: 5)")
    parser.add_argument("--output-dir", default="captured_audio",
                        help="Output directory for WAV files")
    parser.add_argument("--filename", help="Output filename (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Choose capture method
    if args.source == "ble" and BLEAK_AVAILABLE:
        capture = BLEAudioCapture(args.output_dir, args.duration)
    else:
        if args.source == "ble" and not BLEAK_AVAILABLE:
            logger.warning("BLE requested but bleak not available, using microphone")
        capture = MicrophoneCapture(args.output_dir, args.duration)
    
    # Capture audio
    try:
        filepath = capture.capture()
        print(f"\nSuccess! Audio saved to: {filepath}")
        print(f"Format: {SAMPLE_RATE} Hz, {CHANNELS} channel(s), 16-bit PCM")
        
        # Display file info
        import os
        size = os.path.getsize(filepath)
        print(f"File size: {size:,} bytes ({size/1024:.1f} KB)")
        
    except KeyboardInterrupt:
        print("\nCapture cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()