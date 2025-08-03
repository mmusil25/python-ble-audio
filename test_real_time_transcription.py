#!/usr/bin/env python3
"""
Test script for real-time audio transcription
Shows how to use the pipeline programmatically
"""

import asyncio
import sounddevice as sd
from ble_listener import MicrophoneSource, AudioStreamManager
from whisper_tiny_transcription import TranscriptionManager
from gemma_3n_json_extractor import ExtractionManager
from datetime import datetime


def list_audio_devices():
    """List available audio input devices"""
    print("\n🎤 Available Audio Input Devices:")
    print("-" * 60)
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  [{i}] {device['name']} ({device['max_input_channels']} channels)")
            input_devices.append((i, device['name']))
    
    return input_devices


async def test_real_time_transcription(device_index=None, duration=30):
    """Test real-time transcription with selected device"""
    print(f"\n🚀 Starting real-time transcription test...")
    print(f"   Device: {device_index if device_index is not None else 'default'}")
    print(f"   Duration: {duration} seconds")
    print("-" * 60)
    
    # Initialize components
    transcription_manager = TranscriptionManager(engine_type="whisper", model_size="tiny")
    extraction_manager = ExtractionManager(extractor_type="mock")
    
    # Create microphone source with specific device
    mic_source = MicrophoneSource(device_index=device_index)
    stream_manager = AudioStreamManager(mic_source, output_dir="test_recordings")
    
    # Track transcriptions
    transcriptions = []
    
    def transcription_callback(text: str, timestamp: float):
        """Handle transcriptions"""
        if text.strip():
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            print(f"\n[{time_str}] 📝 {text}")
            
            # Extract intent
            extraction = extraction_manager.process_streaming_transcript(text, timestamp)
            print(f"   🎯 Intent: {extraction['intent']}")
            if extraction['entities']:
                print(f"   📌 Entities: {', '.join(extraction['entities'])}")
            
            transcriptions.append({
                'time': time_str,
                'text': text,
                'intent': extraction['intent'],
                'entities': extraction['entities']
            })
    
    # Connect pipeline
    transcription_manager.streaming_transcriber.add_callback(transcription_callback)
    await transcription_manager.transcribe_stream(stream_manager.add_callback)
    
    # Start streaming
    print("\n🎤 Listening... (speak into your microphone)")
    print("   Press Ctrl+C to stop\n")
    
    try:
        # Start streaming
        stream_task = asyncio.create_task(stream_manager.start_streaming())
        
        # Run for specified duration or until interrupted
        await asyncio.sleep(duration)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping...")
    
    finally:
        # Stop streaming
        await transcription_manager.stop_streaming()
        await stream_manager.stop_streaming()
        
        # Summary
        print(f"\n\n📊 Summary:")
        print(f"   Total transcriptions: {len(transcriptions)}")
        
        if transcriptions:
            print(f"\n   Recent transcriptions:")
            for trans in transcriptions[-5:]:
                print(f"   [{trans['time']}] {trans['text'][:50]}...")
        
        # Save results
        if transcriptions:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_transcriptions_{timestamp}.json"
            
            import json
            with open(filename, 'w') as f:
                json.dump(transcriptions, f, indent=2)
            
            print(f"\n💾 Results saved to: {filename}")


async def main():
    """Main test function"""
    # List devices
    devices = list_audio_devices()
    
    if not devices:
        print("\n❌ No audio input devices found!")
        return
    
    # Get user selection
    print("\n📌 Select audio device (or press Enter for default):")
    choice = input("   Device index: ").strip()
    
    device_index = None
    if choice:
        try:
            device_index = int(choice)
            print(f"\n✅ Selected device: {device_index}")
        except ValueError:
            print("\n⚠️  Invalid selection, using default device")
    
    # Test duration
    print("\n⏱️  Enter test duration in seconds (default: 30):")
    duration_input = input("   Duration: ").strip()
    duration = 30
    if duration_input:
        try:
            duration = int(duration_input)
        except ValueError:
            print("   Using default duration: 30 seconds")
    
    # Run test
    await test_real_time_transcription(device_index, duration)


if __name__ == "__main__":
    print("🎙️  Real-Time Audio Transcription Test")
    print("=" * 60)
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()