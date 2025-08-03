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
import sounddevice as sd
import numpy as np
import whisper
import json
import torch

# Import our modules
from gemma_3n_json_extractor import ExtractionManager
from pathlib import Path

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


def get_audio_devices():
    """Get list of available audio input devices"""
    devices = []
    try:
        device_list = sd.query_devices()
        for i, device in enumerate(device_list):
            if device['max_input_channels'] > 0:
                devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'label': f"[{i}] {device['name']} ({device['max_input_channels']} ch)"
                })
    except Exception as e:
        print(f"Error getting audio devices: {e}")
    return devices


def audio_callback(indata, frames, time_info, status):
    """Callback for audio stream"""
    if status:
        print(f"Audio callback status: {status}")
    if audio_queue is not None:
        audio_queue.put(indata.copy())


def process_audio_stream():
    """Process audio from queue and transcribe"""
    global transcriptions, json_extractions
    
    # Buffer for accumulating audio
    audio_buffer = []
    buffer_duration = 3.0  # Process every 3 seconds
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
                
                # Transcribe with Whisper
                if whisper_model is not None:
                    result = whisper_model.transcribe(audio_data, language='en')
                    transcript = result['text'].strip()
                    
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
                                json_result = extraction_manager.extract_from_transcript(transcript)
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


def start_streaming(audio_device=None):
    """Start audio streaming"""
    global is_streaming, audio_queue, stream_thread, transcription_thread, current_device
    global whisper_model, extraction_manager
    
    if is_streaming:
        return "⚠️ Already streaming!", "", ""
    
    # Initialize models if not already done
    if whisper_model is None:
        print("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model("tiny", device=device)
        print(f"Whisper model loaded on {device}")
    
    if extraction_manager is None:
        print("Loading Gemma extraction model...")
        extraction_manager = ExtractionManager(
            extractor_type="gemma",
            model_id="unsloth/gemma-3n-e4b-it"  # Use Unsloth optimized model
        )
        print("Gemma model loaded")
    
    # Parse device index
    device_idx = None
    if audio_device:
        try:
            # Extract device index from label "[idx] Device Name"
            device_idx = int(audio_device.split(']')[0].strip('['))
        except:
            device_idx = None
    
    # Initialize audio
    audio_queue = queue.Queue()
    is_streaming = True
    current_device = device_idx
    
    # Start processing thread
    transcription_thread = threading.Thread(target=process_audio_stream, daemon=True)
    transcription_thread.start()
    
    # Start audio stream
    try:
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
        return f"✅ Streaming started! Using device: {device_name}", "", ""
        
    except Exception as e:
        is_streaming = False
        return f"❌ Error: {str(e)}", "", ""


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
        json_str = json.dumps(e['json'], indent=2)
        json_html += f'''
        <div style="margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px;">
            <div style="color: #666; margin-bottom: 5px;">[{e["time"]}] {e["transcript"][:50]}...</div>
            <pre style="margin: 0; overflow-x: auto;">{json_str}</pre>
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
    global whisper_model, extraction_manager
    
    if audio_file is None:
        return "Please upload an audio file", "", ""
    
    # Initialize models if not already done
    if whisper_model is None:
        print("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model("tiny", device=device)
        print(f"Whisper model loaded on {device}")
    
    if extraction_manager is None:
        print("Loading Gemma extraction model...")
        extraction_manager = ExtractionManager(
            extractor_type="gemma",
            model_id="unsloth/gemma-3n-e4b-it"
        )
        print("Gemma model loaded")
    
    try:
        # Transcribe file
        result = whisper_model.transcribe(audio_file)
        transcript = result['text'].strip()
        
        # Extract JSON
        json_result = extraction_manager.extract_from_transcript(transcript)
        
        # Format results
        trans_html = f'<div style="font-family: monospace; font-size: 14px;">'
        trans_html += f'<h3>Transcription:</h3>'
        trans_html += f'<div style="padding: 10px; background: #f5f5f5; border-radius: 5px;">{transcript}</div>'
        trans_html += f'</div>'
        
        json_html = f'<div style="font-family: monospace; font-size: 12px;">'
        json_html += f'<h3>JSON Extraction:</h3>'
        json_html += f'<pre style="padding: 10px; background: #f5f5f5; border-radius: 5px; overflow-x: auto;">{json.dumps(json_result, indent=2)}</pre>'
        json_html += f'</div>'
        
        return f"✅ File processed successfully!", trans_html, json_html
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""


def create_interface():
    """Create Gradio interface"""
    print("\nIntegrated Whisper → Gemma Pipeline")
    print("=" * 50)
    print("Loading application...")
    
    # Get audio devices
    devices = get_audio_devices()
    device_choices = [d['label'] for d in devices]
    
    with gr.Blocks(title="Real-time Audio → JSON Pipeline") as app:
        gr.Markdown("""
        # 🎤 Real-time Audio → JSON Pipeline
        
        **Whisper** transcription → **Gemma** JSON extraction with Unsloth optimization
        """)
        
        with gr.Tab("🎤 Live Microphone"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ Controls")
                    
                    # Audio device selection
                    audio_device = gr.Dropdown(
                        choices=device_choices,
                        label="Audio Input Device",
                        value=device_choices[0] if device_choices else None
                    )
                    
                    # Control buttons
                    with gr.Row():
                        start_btn = gr.Button("▶️ Start Streaming", variant="primary")
                        stop_btn = gr.Button("⏹️ Stop", variant="stop")
                    
                    refresh_btn = gr.Button("🔄 Refresh Display")
                    
                    # Status
                    status_text = gr.Textbox(
                        label="Status",
                        value="⚫ Ready to start",
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
        
        # Wire up events for live streaming
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
        share=share,
        show_error=True
    )