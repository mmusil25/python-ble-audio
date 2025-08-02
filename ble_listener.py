#!/usr/bin/env python3
"""
BLE Audio Listener with multiple input sources
Supports: BLE (real/mock), microphone input, and file playback
"""

import asyncio
import wave
import struct
import time
import os
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple
import logging
import numpy as np

# Audio processing imports
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available. Microphone functionality disabled.")

# BLE imports (optional)
try:
    import bleak
    from bleak import BleakClient, BleakScanner
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False
    print("Warning: bleak not available. BLE functionality disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000  # 16kHz
CHANNELS = 1  # Mono
SAMPLE_WIDTH = 2  # 16-bit
CHUNK_SIZE = 1024  # Samples per chunk


class AudioSource(ABC):
    """Abstract base class for audio sources"""
    
    @abstractmethod
    async def start(self):
        """Start the audio source"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the audio source"""
        pass
    
    @abstractmethod
    async def read_chunk(self) -> Optional[bytes]:
        """Read a chunk of audio data"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if source is connected"""
        pass


class MicrophoneSource(AudioSource):
    """Local microphone audio source using sounddevice"""
    
    def __init__(self, device_index=None):
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice library not available")
        self.device_index = device_index
        self.stream = None
        self.audio_queue = asyncio.Queue()
        self._running = False
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice stream"""
        if status:
            logger.warning(f"Microphone status: {status}")
        
        # Convert float32 to int16 PCM
        audio_int16 = (indata * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Non-blocking put
        try:
            self.audio_queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            logger.warning("Audio queue full, dropping chunk")
    
    async def start(self):
        """Start microphone capture"""
        self._running = True
        
        # List available devices
        devices = sd.query_devices()
        logger.info("Available audio devices:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                logger.info(f"  [{i}] {device['name']} (inputs: {device['max_input_channels']})")
        
        # Create stream
        self.stream = sd.InputStream(
            device=self.device_index,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            callback=self._audio_callback,
            dtype='float32'
        )
        self.stream.start()
        logger.info(f"Microphone started (device: {self.device_index or 'default'})")
    
    async def stop(self):
        """Stop microphone capture"""
        self._running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        logger.info("Microphone stopped")
    
    async def read_chunk(self) -> Optional[bytes]:
        """Read audio chunk from microphone"""
        if not self._running:
            return None
        
        try:
            # Wait for audio with timeout
            chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
            return chunk
        except asyncio.TimeoutError:
            return None
    
    def is_connected(self) -> bool:
        """Check if microphone is active"""
        return self._running and self.stream is not None and self.stream.active


class MockBLESource(AudioSource):
    """Mock BLE audio source for testing without real device"""
    
    def __init__(self, device_name="Mock Omi Device"):
        self.device_name = device_name
        self._connected = False
        self._sample_file = "raynos-demo/sample.wav"
        self._wave_file = None
        self._position = 0
        
    async def start(self):
        """Simulate BLE connection"""
        logger.info(f"Connecting to mock BLE device: {self.device_name}")
        await asyncio.sleep(1)  # Simulate connection delay
        
        # Open sample file if available
        if os.path.exists(self._sample_file):
            self._wave_file = wave.open(self._sample_file, 'rb')
            logger.info(f"Mock BLE using sample file: {self._sample_file}")
        else:
            logger.info("Mock BLE generating sine wave audio")
            
        self._connected = True
        logger.info("Mock BLE connected")
    
    async def stop(self):
        """Simulate BLE disconnection"""
        self._connected = False
        if self._wave_file:
            self._wave_file.close()
            self._wave_file = None
        logger.info("Mock BLE disconnected")
    
    async def read_chunk(self) -> Optional[bytes]:
        """Generate or read mock audio data"""
        if not self._connected:
            return None
        
        await asyncio.sleep(CHUNK_SIZE / SAMPLE_RATE)  # Simulate real-time
        
        if self._wave_file:
            # Read from sample file
            chunk = self._wave_file.readframes(CHUNK_SIZE)
            if not chunk:
                # Loop back to start
                self._wave_file.rewind()
                chunk = self._wave_file.readframes(CHUNK_SIZE)
            return chunk
        else:
            # Generate 440Hz sine wave
            samples = []
            for i in range(CHUNK_SIZE):
                t = (self._position + i) / SAMPLE_RATE
                sample = int(32767 * 0.5 * np.sin(2 * np.pi * 440 * t))
                samples.append(struct.pack('<h', sample))
            self._position += CHUNK_SIZE
            return b''.join(samples)
    
    def is_connected(self) -> bool:
        """Check if mock device is connected"""
        return self._connected


class RealBLESource(AudioSource):
    """Real BLE audio source using bleak"""
    
    def __init__(self, device_name_prefix="Omi", auto_reconnect=True):
        if not BLEAK_AVAILABLE:
            raise RuntimeError("bleak library not available")
        
        self.device_name_prefix = device_name_prefix
        self.auto_reconnect = auto_reconnect
        self.client = None
        self.device = None
        self._connected = False
        self.audio_queue = asyncio.Queue()
        
        # Omi BLE characteristics (these would need to be verified)
        self.AUDIO_SERVICE_UUID = "00001848-0000-1000-8000-00805f9b34fb"
        self.AUDIO_CHAR_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
    
    async def _find_device(self):
        """Scan for BLE device"""
        logger.info(f"Scanning for BLE devices starting with '{self.device_name_prefix}'...")
        
        devices = await BleakScanner.discover(timeout=10.0)
        for device in devices:
            if device.name and device.name.startswith(self.device_name_prefix):
                logger.info(f"Found device: {device.name} ({device.address})")
                return device
        
        return None
    
    def _audio_notification_handler(self, sender, data):
        """Handle incoming BLE audio data"""
        try:
            self.audio_queue.put_nowait(data)
        except asyncio.QueueFull:
            logger.warning("Audio queue full, dropping BLE data")
    
    async def start(self):
        """Connect to BLE device"""
        while True:
            try:
                # Find device
                self.device = await self._find_device()
                if not self.device:
                    if self.auto_reconnect:
                        logger.warning("Device not found, retrying in 5s...")
                        await asyncio.sleep(5)
                        continue
                    else:
                        raise RuntimeError("BLE device not found")
                
                # Connect
                self.client = BleakClient(self.device.address)
                await self.client.connect()
                self._connected = True
                logger.info(f"Connected to {self.device.name}")
                
                # Subscribe to audio notifications
                await self.client.start_notify(
                    self.AUDIO_CHAR_UUID,
                    self._audio_notification_handler
                )
                
                break
                
            except Exception as e:
                logger.error(f"BLE connection error: {e}")
                if not self.auto_reconnect:
                    raise
                await asyncio.sleep(5)
    
    async def stop(self):
        """Disconnect from BLE device"""
        self._connected = False
        if self.client:
            try:
                await self.client.stop_notify(self.AUDIO_CHAR_UUID)
                await self.client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
            self.client = None
        logger.info("BLE disconnected")
    
    async def read_chunk(self) -> Optional[bytes]:
        """Read audio chunk from BLE"""
        if not self._connected:
            return None
        
        try:
            chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
            return chunk
        except asyncio.TimeoutError:
            return None
    
    def is_connected(self) -> bool:
        """Check if BLE is connected"""
        return self._connected and self.client and self.client.is_connected


class AudioStreamManager:
    """Manages audio streaming from various sources"""
    
    def __init__(self, source: AudioSource, output_dir="audio_recordings"):
        self.source = source
        self.output_dir = output_dir
        self.callbacks = []
        self._running = False
        self._current_file = None
        self._wave_writer = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def add_callback(self, callback: Callable[[bytes], None]):
        """Add callback for audio chunks"""
        self.callbacks.append(callback)
    
    def _start_new_file(self):
        """Start a new WAV file"""
        if self._wave_writer:
            self._wave_writer.close()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        filepath = os.path.join(self.output_dir, filename)
        
        self._current_file = filepath
        self._wave_writer = wave.open(filepath, 'wb')
        self._wave_writer.setnchannels(CHANNELS)
        self._wave_writer.setsampwidth(SAMPLE_WIDTH)
        self._wave_writer.setframerate(SAMPLE_RATE)
        
        logger.info(f"Started recording to: {filepath}")
    
    async def start_streaming(self, max_file_duration=300):
        """Start streaming audio from source"""
        self._running = True
        
        try:
            # Start audio source
            await self.source.start()
            
            # Start new file
            self._start_new_file()
            file_start_time = time.time()
            
            # Reconnection logic
            reconnect_attempts = 0
            max_reconnect_attempts = 5
            
            while self._running:
                # Check connection
                if not self.source.is_connected():
                    logger.warning("Source disconnected")
                    
                    if isinstance(self.source, (RealBLESource, MockBLESource)):
                        # Auto-reconnect for BLE sources
                        if reconnect_attempts < max_reconnect_attempts:
                            reconnect_attempts += 1
                            logger.info(f"Attempting reconnection {reconnect_attempts}/{max_reconnect_attempts}")
                            await asyncio.sleep(2)
                            await self.source.start()
                            continue
                        else:
                            logger.error("Max reconnection attempts reached")
                            break
                    else:
                        break
                
                # Reset reconnect counter on successful connection
                reconnect_attempts = 0
                
                # Read audio chunk
                chunk = await self.source.read_chunk()
                if chunk:
                    # Write to file
                    if self._wave_writer:
                        self._wave_writer.writeframes(chunk)
                    
                    # Call callbacks
                    for callback in self.callbacks:
                        try:
                            callback(chunk)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                    
                    # Check if we need a new file
                    if time.time() - file_start_time > max_file_duration:
                        self._start_new_file()
                        file_start_time = time.time()
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
        finally:
            await self.stop_streaming()
    
    async def stop_streaming(self):
        """Stop streaming"""
        self._running = False
        
        # Stop source
        await self.source.stop()
        
        # Close file
        if self._wave_writer:
            self._wave_writer.close()
            self._wave_writer = None
            logger.info(f"Stopped recording: {self._current_file}")


async def main():
    """Example usage"""
    import sys
    
    # Parse command line arguments
    source_type = sys.argv[1] if len(sys.argv) > 1 else "mock"
    
    # Create audio source
    if source_type == "mic":
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("Microphone source requested but sounddevice not available")
            logger.info("Falling back to mock BLE source")
            source = MockBLESource()
        else:
            source = MicrophoneSource()
    elif source_type == "ble" and BLEAK_AVAILABLE:
        source = RealBLESource()
    else:
        source = MockBLESource()
    
    # Create stream manager
    manager = AudioStreamManager(source)
    
    # Add a simple callback to show audio is flowing
    def audio_callback(chunk):
        # Calculate RMS for simple level meter
        samples = struct.unpack(f'<{len(chunk)//2}h', chunk)
        rms = int(np.sqrt(np.mean(np.square(samples))))
        level = min(50, rms // 1000)
        print(f"\rAudio level: {'█' * level}{' ' * (50 - level)}", end='', flush=True)
    
    manager.add_callback(audio_callback)
    
    # Start streaming
    logger.info(f"Starting audio streaming with source: {source_type}")
    logger.info("Press Ctrl+C to stop")
    
    try:
        await manager.start_streaming()
    except KeyboardInterrupt:
        logger.info("\nStopping...")
        await manager.stop_streaming()


if __name__ == "__main__":
    asyncio.run(main())
