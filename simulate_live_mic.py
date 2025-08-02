#!/usr/bin/env python3
"""
Simulate live microphone input by streaming an audio file in chunks
This works around WSL audio limitations
"""

import asyncio
import wave
import time
import sys
from pathlib import Path

# Import our modules
from ble_listener import AudioSource, AudioStreamManager
from whisper_tiny_transcription import TranscriptionManager
from gemma_3n_json_extractor import ExtractionManager

class FileStreamSource(AudioSource):
    """Stream an audio file as if it were live microphone input"""
    
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.wave_file = None
        self._connected = False
        self.chunk_size = 1024  # Samples per chunk
        
    async def start(self):
        """Open the audio file"""
        print(f"Starting file stream from: {self.audio_file}")
        self.wave_file = wave.open(self.audio_file, 'rb')
        self._connected = True
        
        # Get info
        channels = self.wave_file.getnchannels()
        rate = self.wave_file.getframerate()
        print(f"Audio: {channels} channel(s), {rate} Hz")
        
    async def stop(self):
        """Close the file"""
        self._connected = False
        if self.wave_file:
            self.wave_file.close()
            self.wave_file = None
        print("File stream stopped")
    
    async def read_chunk(self) -> bytes:
        """Read a chunk and simulate real-time delay"""
        if not self._connected or not self.wave_file:
            return None
        
        # Read chunk
        chunk = self.wave_file.readframes(self.chunk_size)
        if not chunk:
            # Loop back to start
            self.wave_file.rewind()
            chunk = self.wave_file.readframes(self.chunk_size)
        
        # Simulate real-time streaming
        # Calculate delay based on chunk size and sample rate
        delay = self.chunk_size / self.wave_file.getframerate()
        await asyncio.sleep(delay)
        
        return chunk
    
    def is_connected(self) -> bool:
        return self._connected


async def main():
    # Parse arguments
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "samples/harvard.wav"
    
    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        return
    
    print("Simulated Live Microphone Demo")
    print("==============================")
    print(f"Streaming from: {audio_file}")
    print("This simulates real-time microphone input")
    print()
    
    # Create components
    audio_source = FileStreamSource(audio_file)
    audio_manager = AudioStreamManager(audio_source, output_dir="live_output")
    
    # Create transcription manager
    transcription_manager = TranscriptionManager(engine_type="whisper", model_size="tiny")
    extraction_manager = ExtractionManager(extractor_type="mock")
    
    # Connect pipeline
    def transcription_callback(text: str, timestamp: float):
        """Process transcriptions"""
        if text.strip():
            print(f"\n[{time.strftime('%H:%M:%S')}] Transcribed: {text}")
            
            # Extract JSON
            result = extraction_manager.process_streaming_transcript(text, timestamp)
            print(f"Intent: {result['intent']}")
            print(f"Entities: {result['entities']}")
            print("-" * 60)
    
    transcription_manager.streaming_transcriber.add_callback(transcription_callback)
    await transcription_manager.transcribe_stream(audio_manager.add_callback)
    
    # Start streaming
    print("Starting live transcription...")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        await audio_manager.start_streaming()
    except KeyboardInterrupt:
        print("\nStopping...")
        await transcription_manager.stop_streaming()
        await audio_manager.stop_streaming()
        
        # Save results
        transcription_manager.save_transcriptions("live_transcriptions.json")
        extraction_manager.save_extractions("live_extractions.json")
        print("\nResults saved!")


if __name__ == "__main__":
    asyncio.run(main())