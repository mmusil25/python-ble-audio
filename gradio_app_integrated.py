#!/usr/bin/env python3
"""
Integrated Gradio UI with Whisper → Gemma JSON Pipeline
Real-time audio transcription with JSON extraction
"""

import gradio as gr
import threading
import queue
import time
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import whisper
import json
import torch

# Import our modules
from gemma_3n_json_extractor import ExtractionManager
from pathlib import Path

# Try importing sounddevice, but make it optional
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Note: sounddevice not available - desktop streaming disabled")

# Check for Deepgram API key
if os.environ.get("DEEPGRAM_API_KEY"):
    print("✓ Deepgram API key detected")
else:
    print("ℹ️ No DEEPGRAM_API_KEY found - Deepgram option will be disabled")

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1

# Global state
is_streaming = False
audio_queue = None
stream_thread = None
transcription_thread = None
transcriptions = []
json_extractions = []
whisper_model = None
extraction_manager = None
current_device = None

# Mobile streaming state
mobile_audio_buffer = []
mobile_buffer_size = int(SAMPLE_RATE * 3)  # 3 seconds of audio
last_mobile_process_time = 0
mobile_processing = False

# Transcription settings
current_transcription_engine = "whisper"
current_whisper_model_size = "tiny"
current_language = "en"
current_buffer_duration = 3.0


def get_audio_devices():
    """Get list of available audio input devices"""
    devices = []
    
    print("\n=== Querying Audio Devices ===")
    
    if SOUNDDEVICE_AVAILABLE:
        try:
            # Try to get system audio devices
            device_list = sd.query_devices()
            print(f"Found {len(device_list)} total audio devices")
            
            # Log all devices to console
            print("\nAll Audio Devices:")
            for i, device in enumerate(device_list):
                print(f"  [{i}] {device['name']}")
                print(f"      - Input channels: {device['max_input_channels']}")
                print(f"      - Output channels: {device['max_output_channels']}")
                print(f"      - Default sample rate: {device['default_samplerate']}")
                print(f"      - Host API: {device['hostapi']}")
            
            # Filter for input devices
            print("\nInput Devices Available for Selection:")
            for i, device in enumerate(device_list):
                if device['max_input_channels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'label': f"[{i}] {device['name']} ({device['max_input_channels']} ch)"
                    })
                    print(f"  [{i}] {device['name']} - {device['max_input_channels']} channels")
            
            # Also log the default input device
            try:
                default_input = sd.query_devices(kind='input')
                print(f"\nDefault Input Device: [{default_input['index']}] {default_input['name']}")
            except Exception as e:
                print(f"Could not query default input device: {e}")
                
        except Exception as e:
            print(f"Error querying system audio devices: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("sounddevice module not available - using browser default")
    
    if not devices:
        # Return default option for mobile/web or when sounddevice not available
        devices.append({
            'index': None,
            'name': 'Default',
            'channels': 1,
            'label': 'Browser Default Microphone'
        })
        print("No system audio devices found - using browser default")
    
    print(f"\nTotal input devices available: {len(devices)}")
    print("=== End Device Query ===\n")
    
    return devices


def refresh_audio_devices():
    """Refresh the list of audio devices and return updated dropdown"""
    devices = get_audio_devices()
    device_choices = [d['label'] for d in devices]
    
    # Return the updated choices for the dropdown
    return gr.Dropdown(
        choices=device_choices,
        value=device_choices[0] if device_choices else None,
        visible=len(device_choices) > 1
    )


def audio_callback(indata, frames, time_info, status):
    """Callback for audio stream"""
    if status:
        print(f"Audio callback status: {status}")
    if audio_queue is not None:
        try:
            audio_queue.put(indata.copy())
        except Exception as e:
            print(f"Audio callback error: {e}")


def process_audio_stream():
    """Process audio from queue and transcribe"""
    global transcriptions, json_extractions, current_buffer_duration, current_language
    
    # Buffer for accumulating audio
    audio_buffer = []
    buffer_duration = current_buffer_duration  # Use configurable duration
    buffer_size = int(buffer_duration * SAMPLE_RATE)
    
    while is_streaming:
        try:
            # Get audio from queue
            chunk = audio_queue.get(timeout=0.1)
            audio_buffer.extend(chunk.flatten())
            
            # Process when buffer is full
            if len(audio_buffer) >= buffer_size:
                # Convert to numpy array
                audio_data = np.array(audio_buffer[:buffer_size], dtype=np.float32)
                
                # Transcribe with configured engine
                if current_transcription_engine == "whisper" and whisper_model is not None:
                    result = whisper_model.transcribe(audio_data, language=current_language)
                    transcript = result['text'].strip()
                elif current_transcription_engine == "deepgram":
                    # Use Deepgram for desktop streaming
                    try:
                        from whisper_tiny_transcription import TranscriptionManager
                        temp_manager = TranscriptionManager(
                            engine_type="deepgram",
                            language=current_language
                        )
                        # Convert audio to bytes for Deepgram
                        audio_bytes = (audio_data * 32768).astype(np.int16).tobytes()
                        transcript = temp_manager.engine.transcribe(audio_bytes)
                    except Exception as e:
                        print(f"Deepgram error: {e}")
                        transcript = f"[Deepgram error: {str(e)}]"
                else:
                    transcript = ""
                    
                    if transcript:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        
                        # Add transcription
                        transcriptions.append({
                            'time': timestamp,
                            'text': transcript
                        })
                        
                        # Keep only last 20 transcriptions
                        transcriptions = transcriptions[-20:]
                        
                        # Extract JSON with Gemma
                        if extraction_manager is not None:
                            try:
                                # Calculate timestamp in milliseconds
                                timestamp_ms = int(time.time() * 1000)
                                json_result = extraction_manager.extract_from_transcript(transcript, timestamp_ms)
                                json_extractions.append({
                                    'time': timestamp,
                                    'transcript': transcript,
                                    'json': json_result
                                })
                                # Keep only last 10 extractions
                                json_extractions = json_extractions[-10:]
                                
                                print(f"[{timestamp}] Extracted: {json_result.get('intent', 'N/A')}")
                            except Exception as e:
                                print(f"Extraction error: {e}")
                
                # Clear processed audio from buffer
                audio_buffer = audio_buffer[buffer_size:]
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Processing error: {e}")


def update_transcription_settings(engine, model_size, language, buffer_duration):
    """Update transcription settings"""
    global current_transcription_engine, current_whisper_model_size, current_language, current_buffer_duration
    global whisper_model, mobile_buffer_size
    
    # Update settings
    current_transcription_engine = engine
    current_whisper_model_size = model_size
    current_language = language
    current_buffer_duration = float(buffer_duration)
    
    # Update mobile buffer size
    mobile_buffer_size = int(SAMPLE_RATE * current_buffer_duration)
    
    # Reload Whisper model if size changed
    if engine == "whisper" and whisper_model is not None:
        # Check if we need to reload the model
        current_size = getattr(whisper_model, 'model_size', 'tiny')
        if current_size != model_size:
            print(f"Reloading Whisper model: {model_size}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_model = whisper.load_model(model_size, device=device)
            print(f"Whisper {model_size} model loaded on {device}")
    
    return f"Settings updated: {engine} ({model_size if engine == 'whisper' else 'API'}), Language: {language}, Buffer: {buffer_duration}s"


def start_streaming(audio_device=None):
    """Start audio streaming"""
    global is_streaming, audio_queue, stream_thread, transcription_thread, current_device
    global whisper_model, extraction_manager, current_whisper_model_size
    
    if is_streaming:
        return "⚠️ Already streaming!", "", ""
    
    # Check if sounddevice is available
    if not SOUNDDEVICE_AVAILABLE:
        return "❌ Desktop streaming not available. Please use the mobile recording option.", "", ""
    
    # Initialize models if not already done
    if current_transcription_engine == "whisper" and whisper_model is None:
        print(f"Loading Whisper {current_whisper_model_size} model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model(current_whisper_model_size, device=device)
        whisper_model.model_size = current_whisper_model_size  # Store for later reference
        print(f"Whisper {current_whisper_model_size} model loaded on {device}")
    elif current_transcription_engine == "deepgram":
        if not os.environ.get("DEEPGRAM_API_KEY"):
            return "❌ Error: DEEPGRAM_API_KEY not set in environment", "", ""
        print("Using Deepgram API for transcription")
    
    if extraction_manager is None:
        print("Loading Gemma extraction model...")
        extraction_manager = ExtractionManager(
            extractor_type="gemma",
            model_id="unsloth/gemma-3n-e4b-it"  # Use Unsloth optimized model
        )
        print("Gemma model loaded")
    
    # Parse device index
    device_idx = None
    if audio_device and audio_device != "Default Microphone" and audio_device != "Browser Default Microphone":
        try:
            # Extract device index from label "[idx] Device Name"
            device_idx = int(audio_device.split(']')[0].strip('['))
            print(f"Selected device index: {device_idx} from '{audio_device}'")
        except Exception as e:
            print(f"Warning: Could not parse device index from '{audio_device}': {e}")
            device_idx = None
    else:
        print(f"Using default device (device selection: '{audio_device}')")
    
    # Initialize audio
    audio_queue = queue.Queue()
    is_streaming = True
    current_device = device_idx
    
    # Start processing thread
    transcription_thread = threading.Thread(target=process_audio_stream, daemon=True)
    transcription_thread.start()
    
    # Start audio stream
    try:
        print(f"\nAttempting to start audio stream...")
        print(f"  Device index: {device_idx}")
        print(f"  Sample rate: {SAMPLE_RATE} Hz")
        print(f"  Channels: {CHANNELS}")
        print(f"  Block size: {CHUNK_SIZE}")
        
        # Query the specific device info before starting
        if device_idx is not None:
            try:
                device_info = sd.query_devices(device_idx)
                print(f"  Selected device: {device_info['name']}")
                print(f"  Max input channels: {device_info['max_input_channels']}")
                print(f"  Default sample rate: {device_info['default_samplerate']}")
            except Exception as e:
                print(f"  Warning: Could not query device {device_idx}: {e}")
        
        stream = sd.InputStream(
            device=device_idx,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        )
        stream.start()
        
        # Store stream reference
        stream_thread = stream
        
        device_name = audio_device if audio_device else "default"
        print(f"✅ Audio stream started successfully!")
        return f"✅ Streaming started! Using device: {device_name}", "", ""
        
    except Exception as e:
        is_streaming = False
        error_msg = str(e)
        
        # Provide more helpful error messages
        if "Invalid device" in error_msg:
            error_msg += f" (Device index {device_idx} not found. Try refreshing devices.)"
        elif "Invalid number of channels" in error_msg:
            error_msg += f" (Device doesn't support {CHANNELS} channel(s))"
        elif "Invalid sample rate" in error_msg:
            error_msg += f" (Device doesn't support {SAMPLE_RATE} Hz sample rate)"
        
        print(f"❌ Failed to start audio stream: {error_msg}")
        import traceback
        traceback.print_exc()
        
        return f"❌ Error: {error_msg}", "", ""


def stop_streaming():
    """Stop audio streaming"""
    global is_streaming, stream_thread, audio_queue
    
    if not is_streaming:
        return "⚠️ Not currently streaming", "", ""
    
    is_streaming = False
    
    # Stop audio stream
    if stream_thread:
        stream_thread.stop()
        stream_thread.close()
        stream_thread = None
    
    # Clear queue
    if audio_queue:
        while not audio_queue.empty():
            audio_queue.get()
    
    return "🛑 Streaming stopped", "", ""


def refresh_display():
    """Refresh the display with latest transcriptions and extractions"""
    # Format transcriptions
    trans_html = '<div style="font-family: monospace; font-size: 14px;">'
    for t in reversed(transcriptions[-10:]):  # Show last 10
        trans_html += f'<div style="margin: 5px 0;"><span style="color: #666;">[{t["time"]}]</span> {t["text"]}</div>'
    trans_html += '</div>'
    
    # Format JSON extractions
    json_html = '<div style="font-family: monospace; font-size: 12px;">'
    for e in reversed(json_extractions[-5:]):  # Show last 5
        json_data = e['json']
        # Format the JSON with all required fields highlighted
        json_formatted = {
            "transcript": json_data.get("transcript", ""),
            "timestamp_ms": json_data.get("timestamp_ms", 0),
            "intent": json_data.get("intent", "unknown"),
            "entities": json_data.get("entities", [])
        }
        json_str = json.dumps(json_formatted, indent=2)
        
        # Create human-readable timestamp from milliseconds
        ts_seconds = json_data.get("timestamp_ms", 0) / 1000
        readable_ts = datetime.fromtimestamp(ts_seconds).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        json_html += f'''
        <div style="margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; border: 1px solid #ddd;">
            <div style="color: #666; margin-bottom: 5px;">
                <strong>[{e["time"]}]</strong> Intent: <span style="color: #0066cc; font-weight: bold;">{json_data.get("intent", "unknown")}</span>
                | Keywords: <span style="color: #008800;">{", ".join(json_data.get("entities", []))[:100] or "none"}</span>
            </div>
            <pre style="margin: 0; overflow-x: auto; background: white; padding: 8px; border-radius: 3px;">{json_str}</pre>
            <div style="color: #999; font-size: 11px; margin-top: 5px;">Timestamp: {readable_ts}</div>
        </div>
        '''
    json_html += '</div>'
    
    # Status
    if is_streaming:
        status = f"🟢 Streaming active | 📝 {len(transcriptions)} transcriptions | 🔍 {len(json_extractions)} extractions"
    else:
        status = "⚫ Not streaming"
    
    return trans_html, json_html, status


def process_audio_file(audio_file):
    """Process uploaded audio file"""
    global whisper_model, extraction_manager, transcriptions, json_extractions
    global current_transcription_engine, current_whisper_model_size, current_language
    
    if audio_file is None:
        return "Please upload an audio file", "", ""
    
    # Initialize models if not already done
    if current_transcription_engine == "whisper" and whisper_model is None:
        print(f"Loading Whisper {current_whisper_model_size} model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model(current_whisper_model_size, device=device)
        whisper_model.model_size = current_whisper_model_size
        print(f"Whisper {current_whisper_model_size} model loaded on {device}")
    elif current_transcription_engine == "deepgram":
        if not os.environ.get("DEEPGRAM_API_KEY"):
            return "❌ DEEPGRAM_API_KEY not set", "", ""
    
    if extraction_manager is None:
        print("Loading Gemma extraction model...")
        extraction_manager = ExtractionManager(
            extractor_type="gemma",
            model_id="unsloth/gemma-3n-e4b-it"
        )
        print("Gemma model loaded")
    
    try:
        # Transcribe file
        if current_transcription_engine == "whisper":
            result = whisper_model.transcribe(audio_file, language=current_language)
            transcript = result['text'].strip()
        elif current_transcription_engine == "deepgram":
            # Use Deepgram for file transcription
            try:
                from whisper_tiny_transcription import TranscriptionManager
                temp_manager = TranscriptionManager(
                    engine_type="deepgram",
                    language=current_language
                )
                transcript = temp_manager.transcribe_file(audio_file)['text']
            except Exception as e:
                print(f"Deepgram error: {e}")
                transcript = f"[Deepgram error: {str(e)}]"
        else:
            transcript = ""
        
        if transcript:
            # Extract JSON with timestamp
            timestamp_ms = int(time.time() * 1000)
            json_result = extraction_manager.extract_from_transcript(transcript, timestamp_ms)
            
            # Add to history (for mobile recording mode)
            timestamp = datetime.now().strftime("%H:%M:%S")
            transcriptions.append({
                'time': timestamp,
                'text': transcript
            })
            json_extractions.append({
                'time': timestamp,
                'transcript': transcript,
                'json': json_result
            })
            
            # Keep only recent items
            transcriptions = transcriptions[-20:]
            json_extractions = json_extractions[-10:]
            
            # Format results
            trans_html = f'<div style="font-family: monospace; font-size: 14px;">'
            trans_html += f'<h3>Transcription:</h3>'
            trans_html += f'<div style="padding: 10px; background: #f5f5f5; border-radius: 5px;">{transcript}</div>'
            trans_html += f'</div>'
            
            json_html = f'<div style="font-family: monospace; font-size: 12px;">'
            json_html += f'<h3>JSON Extraction:</h3>'
            
            # Format the JSON with all fields
            json_formatted = {
                "transcript": json_result.get("transcript", ""),
                "timestamp_ms": json_result.get("timestamp_ms", 0),
                "intent": json_result.get("intent", "unknown"),
                "entities": json_result.get("entities", [])
            }
            
            # Create human-readable timestamp
            ts_seconds = json_result.get("timestamp_ms", 0) / 1000
            readable_ts = datetime.fromtimestamp(ts_seconds).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            json_html += f'''
            <div style="padding: 10px; background: #f5f5f5; border-radius: 5px; border: 1px solid #ddd;">
                <div style="margin-bottom: 10px;">
                    <strong>Intent:</strong> <span style="color: #0066cc; font-weight: bold;">{json_result.get("intent", "unknown")}</span><br>
                    <strong>Keywords:</strong> <span style="color: #008800;">{", ".join(json_result.get("entities", [])) or "none"}</span><br>
                    <strong>Timestamp:</strong> <span style="color: #666;">{readable_ts}</span>
                </div>
                <pre style="margin: 0; padding: 10px; background: white; border-radius: 3px; overflow-x: auto;">{json.dumps(json_formatted, indent=2)}</pre>
            </div>
            '''
            json_html += f'</div>'
            
            # Also update the streaming display for mobile
            trans_display, json_display, _ = refresh_display()
            
            return f"✅ Processed! File: {Path(audio_file).name}", trans_display, json_display
        else:
            return "⚠️ No speech detected in audio", "", ""
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""


def process_mobile_stream(audio_chunk):
    """Process streaming audio from mobile devices"""
    global whisper_model, extraction_manager, transcriptions, json_extractions
    global mobile_audio_buffer, last_mobile_process_time, mobile_buffer_size, mobile_processing
    global current_transcription_engine, current_whisper_model_size, current_language
    
    if audio_chunk is None:
        return "🔴 No audio received", refresh_display()[0], refresh_display()[1]
    
    # Initialize models if needed
    if current_transcription_engine == "whisper" and whisper_model is None:
        print(f"Loading Whisper {current_whisper_model_size} model for mobile...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model(current_whisper_model_size, device=device)
        whisper_model.model_size = current_whisper_model_size
        print(f"Whisper {current_whisper_model_size} model loaded on {device}")
    elif current_transcription_engine == "deepgram":
        if not os.environ.get("DEEPGRAM_API_KEY"):
            return "❌ DEEPGRAM_API_KEY not set", refresh_display()[0], refresh_display()[1]
    
    if extraction_manager is None:
        print("Loading Gemma extraction model for mobile...")
        extraction_manager = ExtractionManager(
            extractor_type="gemma",
            model_id="unsloth/gemma-3n-e4b-it"
        )
        print("Gemma model loaded")
    
    try:
        # Extract audio data
        if isinstance(audio_chunk, tuple):
            sample_rate, audio_data = audio_chunk
            print(f"Received audio: sample_rate={sample_rate}, shape={audio_data.shape}, dtype={audio_data.dtype}")
            
            # Resample if needed
            if sample_rate != SAMPLE_RATE:
                # Simple resampling by taking every nth sample or interpolating
                resample_ratio = sample_rate / SAMPLE_RATE
                if resample_ratio > 1:  # Downsample
                    step = int(resample_ratio)
                    audio_data = audio_data[::step]
                else:  # Upsample (simple repeat)
                    repeat = int(1 / resample_ratio)
                    audio_data = np.repeat(audio_data, repeat)
                print(f"Resampled from {sample_rate}Hz to {SAMPLE_RATE}Hz")
            
            # Convert to float32 if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
                print("Converted stereo to mono")
        else:
            # Assume it's already float32 numpy array
            audio_data = audio_chunk
            sample_rate = SAMPLE_RATE
        
        # Add to buffer
        mobile_audio_buffer.extend(audio_data.flatten())
        
        # Debug info
        buffer_seconds = len(mobile_audio_buffer) / SAMPLE_RATE
        print(f"Mobile buffer: {buffer_seconds:.1f}s ({len(mobile_audio_buffer)} samples)")
        
        # Process when buffer has at least 3 seconds of audio
        current_time = time.time()
        time_since_last = current_time - last_mobile_process_time
        
        # Process when buffer has enough audio OR buffer is getting too large
        should_process = (len(mobile_audio_buffer) >= mobile_buffer_size or 
                         len(mobile_audio_buffer) >= mobile_buffer_size * 2) and not mobile_processing
        
        if should_process:
            mobile_processing = True
            
            # Get exactly 3 seconds of audio to process
            audio_to_process = np.array(mobile_audio_buffer[:mobile_buffer_size], dtype=np.float32)
            
            # Clear the buffer, keeping a small overlap
            overlap = int(mobile_buffer_size * 0.1)  # 10% overlap
            mobile_audio_buffer = mobile_audio_buffer[mobile_buffer_size - overlap:]
            last_mobile_process_time = current_time
            
            print(f"Processing {len(audio_to_process)/SAMPLE_RATE:.1f}s of audio...")
            
            # Normalize audio if needed
            if np.max(np.abs(audio_to_process)) > 1.0:
                audio_to_process = audio_to_process / np.max(np.abs(audio_to_process))
            
            # Transcribe
            print(f"Transcribing audio with {current_transcription_engine}...")
            
            if current_transcription_engine == "whisper":
                result = whisper_model.transcribe(audio_to_process, language=current_language, fp16=False)
                transcript = result['text'].strip()
            elif current_transcription_engine == "deepgram":
                # Import and use Deepgram if available
                try:
                    from whisper_tiny_transcription import TranscriptionManager
                    # Create temporary manager for Deepgram with language setting
                    temp_manager = TranscriptionManager(
                        engine_type="deepgram",
                        language=current_language
                    )
                    # Convert audio to bytes for Deepgram
                    audio_bytes = (audio_to_process * 32768).astype(np.int16).tobytes()
                    transcript = temp_manager.engine.transcribe(audio_bytes)
                except Exception as e:
                    print(f"Deepgram error: {e}")
                    transcript = f"[Deepgram error: {str(e)}]"
            else:
                transcript = ""
            
            print(f"Transcription result: '{transcript}'")
            
            if transcript:
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Add transcription
                transcriptions.append({
                    'time': timestamp,
                    'text': transcript
                })
                transcriptions = transcriptions[-20:]  # Keep last 20
                
                # Extract JSON
                try:
                    timestamp_ms = int(time.time() * 1000)
                    json_result = extraction_manager.extract_from_transcript(transcript, timestamp_ms)
                    json_extractions.append({
                        'time': timestamp,
                        'transcript': transcript,
                        'json': json_result
                    })
                    json_extractions = json_extractions[-10:]  # Keep last 10
                    
                    print(f"[{timestamp}] Mobile stream - Extracted: {json_result.get('intent', 'N/A')}")
                except Exception as e:
                    print(f"Mobile extraction error: {e}")
                
                # Update display
                trans_display, json_display, _ = refresh_display()
                mobile_processing = False
                return f"🔴 Streaming... Last: {timestamp}", trans_display, json_display
            
            # No speech detected
            mobile_processing = False
            return f"🟠 Processing... (no speech detected)", refresh_display()[0], refresh_display()[1]
        
        # Still buffering
        buffer_percent = min((len(mobile_audio_buffer) / mobile_buffer_size) * 100, 100)
        return f"🟡 Buffering... {buffer_percent:.0f}% ({buffer_seconds:.1f}s)", refresh_display()[0], refresh_display()[1]
        
    except Exception as e:
        print(f"Mobile streaming error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", refresh_display()[0], refresh_display()[1]


def clear_mobile_buffer():
    """Clear the mobile audio buffer when recording stops"""
    global mobile_audio_buffer, last_mobile_process_time, mobile_processing
    mobile_audio_buffer = []
    last_mobile_process_time = 0
    mobile_processing = False
    print("Mobile buffer cleared")
    return "🔴 Buffer cleared - ready to stream"


def create_interface():
    """Create Gradio interface"""
    print("\nIntegrated Whisper → Gemma Pipeline")
    print("=" * 50)
    print("Loading application...")
    
    # Get audio devices
    devices = get_audio_devices()
    device_choices = [d['label'] for d in devices]
    
    # Custom CSS for mobile responsiveness
    custom_css = """
    @media (max-width: 768px) {
        .gradio-container {
            padding: 10px !important;
        }
        .gr-box {
            margin: 5px !important;
        }
        .gr-button {
            font-size: 16px !important;
            padding: 12px !important;
        }
        /* Make audio component more prominent on mobile */
        .audio-container {
            min-height: 120px !important;
        }
    }
    """
    
    with gr.Blocks(title="Real-time Audio → JSON Pipeline", css=custom_css) as app:
        gr.Markdown("""
        # 🎤 Real-time Audio → JSON Pipeline
        
        **Whisper** transcription → **Gemma** JSON extraction with Unsloth optimization
        """)
        
        with gr.Tab("⚙️ Settings"):
            gr.Markdown("### Transcription Settings")
            
            with gr.Row():
                with gr.Column():
                    transcription_engine = gr.Radio(
                        ["whisper", "deepgram"],
                        value="whisper",
                        label="Transcription Engine",
                        info="Choose between local Whisper or Deepgram API"
                    )
                    
                    whisper_model_size = gr.Dropdown(
                        ["tiny", "base", "small", "medium", "large"],
                        value="tiny",
                        label="Whisper Model Size",
                        info="Larger models are more accurate but slower",
                        visible=True
                    )
                    
                    # Show/hide Whisper options based on engine selection
                    def update_model_visibility(engine):
                        return gr.update(visible=(engine == "whisper"))
                    
                    transcription_engine.change(
                        fn=update_model_visibility,
                        inputs=[transcription_engine],
                        outputs=[whisper_model_size]
                    )
                
                with gr.Column():
                    language = gr.Dropdown(
                        ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                        value="en",
                        label="Language",
                        info="Language code for transcription"
                    )
                    
                    buffer_duration = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=3.0,
                        step=0.5,
                        label="Buffer Duration (seconds)",
                        info="How much audio to accumulate before processing"
                    )
                    
                    apply_settings_btn = gr.Button("Apply Settings", variant="primary")
                    settings_status = gr.Textbox(
                        label="Settings Status",
                        value="Default settings active",
                        interactive=False
                    )
            
            gr.Markdown("""
            ### Notes:
            - **Whisper**: Runs locally, no API key needed. Tiny/Base are fast, Large is most accurate.
            - **Deepgram**: Requires API key set as environment variable: `export DEEPGRAM_API_KEY=your_key`
            - **Buffer Duration**: Shorter = more responsive but may miss context. Longer = better context but more delay.
            """)
        
        with gr.Tab("🎤 Live Microphone"):
            gr.Markdown("""
            ### Live Audio Streaming
            
            **Desktop**: Uses system microphone for real-time streaming
            
            **Mobile**: Click the microphone icon to start live streaming. Audio is processed based on buffer duration.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ Controls")
                    
                    # Audio device selection (hidden on mobile)
                    with gr.Group() as device_group:
                        audio_device = gr.Dropdown(
                            choices=device_choices,
                            label="Audio Input Device (Desktop only)",
                            value=device_choices[0] if device_choices else None,
                            visible=len(device_choices) > 1  # Hide if only default option
                        )
                        refresh_devices_btn = gr.Button("🔄 Refresh Devices", size="sm")
                    
                    # Desktop streaming controls
                    if SOUNDDEVICE_AVAILABLE:
                        with gr.Group() as desktop_controls:
                            gr.Markdown("**Desktop Streaming:**")
                            with gr.Row():
                                start_btn = gr.Button("▶️ Start Streaming", variant="primary")
                                stop_btn = gr.Button("⏹️ Stop", variant="stop")
                    else:
                        desktop_controls = None
                        start_btn = gr.Button("▶️ Start Streaming", variant="primary", visible=False)
                        stop_btn = gr.Button("⏹️ Stop", variant="stop", visible=False)
                    
                    # Mobile audio streaming
                    with gr.Group() as mobile_controls:
                        gr.Markdown("**Mobile/Browser Streaming:**")
                        mobile_audio = gr.Audio(
                            label="Live Audio Stream (Click mic to start)",
                            type="numpy",
                            sources=["microphone"],
                            streaming=True,  # Enable streaming mode
                            show_download_button=False,
                            elem_id="mobile-audio-stream"
                        )
                        mobile_status = gr.Textbox(
                            label="Mobile Status",
                            value="🔴 Not streaming - Click microphone to start",
                            interactive=False
                        )
                        clear_buffer_btn = gr.Button("🗑️ Clear Buffer", size="sm")
                    
                    refresh_btn = gr.Button("🔄 Refresh Display")
                    
                    # Status
                    status_text = gr.Textbox(
                        label="Status",
                        value="⚫ Ready - Desktop: Click Start | Mobile: Click Microphone",
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📝 Live Transcriptions")
                    transcriptions_display = gr.HTML(
                        value='<div style="color: #666;">No transcriptions yet...</div>',
                        elem_id="transcriptions"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 🔍 JSON Extractions")
                    json_display = gr.HTML(
                        value='<div style="color: #666;">No extractions yet...</div>',
                        elem_id="extractions"
                    )
        
        with gr.Tab("📁 File Upload"):
            gr.Markdown("""
            ### Upload Audio Files
            
            **Supported formats:** WAV, MP3, FLAC, OGG, M4A, AAC, WMA, OPUS, WebM, MP4, M4B
            """)
            
            with gr.Row():
                with gr.Column():
                    file_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    
                    process_file_btn = gr.Button("🎵 Process File", variant="primary", size="lg")
                    
                    file_status = gr.Textbox(
                        label="Status",
                        value="Ready to process files",
                        interactive=False
                    )
                
                with gr.Column():
                    file_transcription = gr.HTML(
                        value='<div style="color: #666;">Upload a file to see transcription...</div>'
                    )
                
                with gr.Column():
                    file_json = gr.HTML(
                        value='<div style="color: #666;">Upload a file to see JSON extraction...</div>'
                    )
        
        # Wire up events for desktop streaming
        start_btn.click(
            fn=start_streaming,
            inputs=[audio_device],
            outputs=[status_text, transcriptions_display, json_display]
        )
        
        stop_btn.click(
            fn=stop_streaming,
            outputs=[status_text, transcriptions_display, json_display]
        )
        
        refresh_btn.click(
            fn=refresh_display,
            outputs=[transcriptions_display, json_display, status_text]
        )
        
        # Wire up events for device refresh
        refresh_devices_btn.click(
            fn=refresh_audio_devices,
            outputs=[audio_device]
        )
        
        # Wire up events for mobile audio streaming
        mobile_audio.stream(
            fn=process_mobile_stream,
            inputs=[mobile_audio],
            outputs=[mobile_status, transcriptions_display, json_display],
            show_progress="hidden"
        )
        
        # Clear buffer button
        clear_buffer_btn.click(
            fn=clear_mobile_buffer,
            outputs=[mobile_status]
        )
        
        # Wire up events for settings
        apply_settings_btn.click(
            fn=update_transcription_settings,
            inputs=[transcription_engine, whisper_model_size, language, buffer_duration],
            outputs=[settings_status]
        )
        
        # Wire up events for file processing
        process_file_btn.click(
            fn=process_audio_file,
            inputs=[file_input],
            outputs=[file_status, file_transcription, file_json]
        )
        
        # Note: Manual refresh needed for live streaming - click "Refresh Display" button
        # Auto-refresh requires Gradio 4.0+ with 'every' parameter
    
    return app


if __name__ == "__main__":
    # Check for GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Running on CPU")
    
    # Create and launch interface
    app = create_interface()
    
    # Parse command line for share option
    import sys
    share = "--share" in sys.argv
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        show_error=True,
        inbrowser=not share,  # Open browser only if not sharing
        quiet=False,
        ssl_verify=False  # Allow self-signed certificates for mobile
    )