#!/usr/bin/env python3
"""
Whisper Tiny Transcription Module
Provides real-time and batch transcription using OpenAI's Whisper Tiny model
"""

import asyncio
import wave
import io
import time
import logging
from typing import Optional, Callable, Dict, List, Tuple
from datetime import datetime
import numpy as np
import queue
import threading

# Whisper imports
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: whisper not installed. Install with: pip install openai-whisper")

# For alternative Deepgram support
import os
import json
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio constants (must match ble_listener.py)
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2


class TranscriptionEngine:
    """Base class for transcription engines"""
    
    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio data to text"""
        raise NotImplementedError
    
    def transcribe_file(self, filepath: str) -> str:
        """Transcribe audio file to text"""
        raise NotImplementedError


class WhisperEngine(TranscriptionEngine):
    """Whisper-based transcription engine"""
    
    def __init__(self, model_size="tiny", device=None, language="en"):
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper not available")
        
        # Auto-detect device if not specified
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size, device=device)
        self.language = language
        logger.info(f"Whisper model loaded (device: {device})")
    
    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio bytes to text"""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe
        result = self.model.transcribe(
            audio_array,
            language=self.language,
            fp16=(str(self.model.device).startswith("cuda")),  # Enable FP16 for GPU
            verbose=False
        )
        
        return result["text"].strip()
    
    def transcribe_file(self, filepath: str) -> str:
        """Transcribe audio file to text"""
        result = self.model.transcribe(
            filepath,
            language=self.language,
            fp16=False,
            verbose=False
        )
        return result["text"].strip()
    
    def transcribe_with_timestamps(self, audio_data: bytes) -> List[Dict]:
        """Transcribe with word-level timestamps"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        result = self.model.transcribe(
            audio_array,
            language=self.language,
            fp16=False,
            verbose=False,
            word_timestamps=True
        )
        
        segments = []
        for segment in result["segments"]:
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })
        
        return segments


class DeepgramEngine(TranscriptionEngine):
    """Deepgram API-based transcription engine"""
    
    def __init__(self, api_key=None, language="en"):
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library not available")
        
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("Deepgram API key not provided")
        
        self.url = "https://api.deepgram.com/v1/listen"
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav"
        }
        self.language = language
        logger.info("Deepgram engine initialized")
    
    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio bytes using Deepgram API"""
        # Create WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_data)
        
        wav_data = wav_buffer.getvalue()
        
        # Send to Deepgram
        params = {
            "model": "nova-2",
            "language": self.language,
            "punctuate": "true"  # Deepgram expects string, not boolean
        }
        
        response = requests.post(
            self.url,
            headers=self.headers,
            params=params,
            data=wav_data
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["results"]["channels"][0]["alternatives"]:
                return result["results"]["channels"][0]["alternatives"][0]["transcript"]
        else:
            logger.error(f"Deepgram error: {response.status_code} - {response.text}")
        
        return ""
    
    def transcribe_file(self, filepath: str) -> str:
        """Transcribe audio file using Deepgram"""
        with open(filepath, 'rb') as f:
            audio_data = f.read()
        
        response = requests.post(
            self.url,
            headers=self.headers,
            params={"model": "nova-2", "language": self.language, "punctuate": "true"},  # String, not boolean
            data=audio_data
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["results"]["channels"][0]["alternatives"]:
                return result["results"]["channels"][0]["alternatives"][0]["transcript"]
        
        return ""


class StreamingTranscriber:
    """Handles streaming transcription with buffering"""
    
    def __init__(self, engine: TranscriptionEngine, buffer_duration=2.0):
        self.engine = engine
        self.buffer_duration = buffer_duration
        self.buffer_size = int(SAMPLE_RATE * buffer_duration * SAMPLE_WIDTH)
        self.audio_buffer = bytearray()
        self.transcription_queue = queue.Queue(maxsize=10)
        self.callbacks = []
        self._running = False
        self._worker_thread = None
    
    def add_callback(self, callback: Callable[[str, float], None]):
        """Add callback for transcription results (text, timestamp)"""
        self.callbacks.append(callback)
    
    def process_audio_chunk(self, chunk: bytes):
        """Add audio chunk to buffer"""
        self.audio_buffer.extend(chunk)
        
        # Log progress
        buffer_percent = (len(self.audio_buffer) / self.buffer_size) * 100
        logger.debug(f"Audio buffer: {buffer_percent:.1f}% full ({len(self.audio_buffer)}/{self.buffer_size} bytes)")
        
        # Check if buffer is full
        if len(self.audio_buffer) >= self.buffer_size:
            # Extract buffer for transcription
            audio_to_transcribe = bytes(self.audio_buffer[:self.buffer_size])
            # Keep last 20% for context overlap
            overlap = int(self.buffer_size * 0.2)
            self.audio_buffer = self.audio_buffer[self.buffer_size - overlap:]
            
            # Queue for transcription
            timestamp = time.time()
            try:
                self.transcription_queue.put_nowait((audio_to_transcribe, timestamp))
            except queue.Full:
                logger.warning("Transcription queue full, dropping audio segment")
    
    def _transcription_worker(self):
        """Background worker for transcription (runs in thread)"""
        while self._running:
            try:
                # Get audio from queue with timeout
                audio_data, timestamp = self.transcription_queue.get(timeout=1.0)
                
                # Transcribe
                start_time = time.time()
                text = self.engine.transcribe(audio_data)
                transcription_time = time.time() - start_time
                
                if text:
                    logger.info(f"Transcribed in {transcription_time:.2f}s: {text}")
                    
                    # Call callbacks
                    for callback in self.callbacks:
                        try:
                            callback(text, timestamp)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription error: {e}")
    
    async def start(self):
        """Start streaming transcriber"""
        self._running = True
        # Start worker thread
        self._worker_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self._worker_thread.start()
        logger.info("Streaming transcriber started")
    
    async def stop(self):
        """Stop streaming transcriber"""
        self._running = False
        # Clear the queue
        while not self.transcription_queue.empty():
            try:
                self.transcription_queue.get_nowait()
            except queue.Empty:
                break
        # Wait for thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        logger.info("Streaming transcriber stopped")
    
    def get_final_transcription(self) -> str:
        """Transcribe any remaining audio in buffer"""
        if len(self.audio_buffer) > 0:
            return self.engine.transcribe(bytes(self.audio_buffer))
        return ""


class TranscriptionManager:
    """High-level transcription manager"""
    
    def __init__(self, engine_type="whisper", **engine_kwargs):
        # Create engine
        if engine_type == "whisper":
            self.engine = WhisperEngine(**engine_kwargs)
        elif engine_type == "deepgram":
            self.engine = DeepgramEngine(**engine_kwargs)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
        
        self.streaming_transcriber = StreamingTranscriber(self.engine)
        self.transcription_history = []
    
    def _save_transcription(self, text: str, timestamp: float):
        """Save transcription to history"""
        self.transcription_history.append({
            "text": text,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat()
        })
    
    async def transcribe_stream(self, audio_callback_registration: Callable):
        """Start streaming transcription"""
        # Add callback to save transcriptions
        self.streaming_transcriber.add_callback(self._save_transcription)
        
        # Start transcriber
        await self.streaming_transcriber.start()
        
        # Register audio callback
        audio_callback_registration(self.streaming_transcriber.process_audio_chunk)
    
    async def stop_streaming(self):
        """Stop streaming transcription"""
        await self.streaming_transcriber.stop()
        
        # Get final transcription
        final_text = self.streaming_transcriber.get_final_transcription()
        if final_text:
            self._save_transcription(final_text, time.time())
    
    def transcribe_file(self, filepath: str) -> Dict:
        """Transcribe a complete audio file"""
        start_time = time.time()
        text = self.engine.transcribe_file(filepath)
        duration = time.time() - start_time
        
        return {
            "filepath": filepath,
            "text": text,
            "duration": duration,
            "timestamp": start_time,
            "datetime": datetime.fromtimestamp(start_time).isoformat()
        }
    
    def get_transcription_history(self) -> List[Dict]:
        """Get all transcriptions"""
        return self.transcription_history
    
    def save_transcriptions(self, filepath: str):
        """Save transcriptions to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.transcription_history, f, indent=2)
        logger.info(f"Saved {len(self.transcription_history)} transcriptions to {filepath}")


async def main():
    """Example usage"""
    import sys
    from ble_listener import AudioStreamManager, MockBLESource
    
    # Check if Whisper is available
    if not WHISPER_AVAILABLE:
        logger.error("Whisper not available. Please install: pip install openai-whisper")
        return
    
    # Parse arguments
    engine_type = sys.argv[1] if len(sys.argv) > 1 else "whisper"
    
    # Create transcription manager
    if engine_type == "deepgram":
        # Check for API key
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            logger.error("DEEPGRAM_API_KEY environment variable not set")
            return
        manager = TranscriptionManager(engine_type="deepgram")
    else:
        manager = TranscriptionManager(engine_type="whisper", model_size="tiny")
    
    # Create audio source and stream manager
    audio_source = MockBLESource()
    audio_manager = AudioStreamManager(audio_source)
    
    # Connect transcription to audio stream
    await manager.transcribe_stream(audio_manager.add_callback)
    
    # Start audio streaming
    logger.info(f"Starting transcription demo with {engine_type} engine")
    logger.info("Press Ctrl+C to stop")
    
    try:
        await audio_manager.start_streaming()
    except KeyboardInterrupt:
        logger.info("\nStopping...")
        await manager.stop_streaming()
        await audio_manager.stop_streaming()
        
        # Save transcriptions
        manager.save_transcriptions("transcriptions.json")
        
        # Print summary
        history = manager.get_transcription_history()
        logger.info(f"\nTranscription Summary:")
        logger.info(f"Total segments: {len(history)}")
        for i, trans in enumerate(history[-5:]):  # Show last 5
            logger.info(f"  [{trans['datetime']}] {trans['text'][:80]}...")


if __name__ == "__main__":
    asyncio.run(main())
