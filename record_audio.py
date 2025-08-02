#!/usr/bin/env python3
"""
Simple audio recorder for Windows - records and saves to WSL
Run this on Windows (not in WSL) to record from your microphone
"""

import sys
import os
import time

# Check if we have the required module
try:
    import pyaudio
except ImportError:
    print("Please install pyaudio on Windows:")
    print("pip install pyaudio")
    sys.exit(1)

import wave
import threading

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def record_audio(duration=10, output_file="recording.wav"):
    """Record audio from the default microphone"""
    p = pyaudio.PyAudio()
    
    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print(f"Recording for {duration} seconds...")
    print("Speak now!")
    
    frames = []
    
    # Record
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
        
        # Show progress
        elapsed = (i * CHUNK) / RATE
        remaining = duration - elapsed
        print(f"\rRecording... {remaining:.1f}s remaining", end="")
    
    print("\nRecording finished!")
    
    # Stop and close
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"Saved to: {output_file}")
    
    # Try to copy to WSL
    wsl_path = r"\\wsl.localhost\Ubuntu-22.04\home\mark\python-ble-audio\samples"
    if os.path.exists(wsl_path):
        import shutil
        dest = os.path.join(wsl_path, output_file)
        shutil.copy2(output_file, dest)
        print(f"Copied to WSL: {dest}")
    else:
        print(f"To copy to WSL, run:")
        print(f"cp {output_file} {wsl_path}\\")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record audio on Windows")
    parser.add_argument("-d", "--duration", type=int, default=10, help="Duration in seconds")
    parser.add_argument("-o", "--output", default="recording.wav", help="Output filename")
    args = parser.parse_args()
    
    record_audio(args.duration, args.output)