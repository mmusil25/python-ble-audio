#!/usr/bin/env python3
"""
Simplified Gradio UI for BLE Audio → Whisper → Gemma Pipeline
Uses threading instead of asyncio to avoid event loop conflicts
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
whisper_model = None
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
    """Callback for sounddevice stream"""
    global audio_queue
    if status:
        print(f"Audio status: {status}")
    
    if audio_queue is not None:
        # Convert float32 to int16
        audio_int16 = (indata * 32767).astype(np.int16)
        try:
            audio_queue.put_nowait(audio_int16.tobytes())
        except queue.Full:
            pass  # Drop frame if queue is full


def transcription_worker():
    """Worker thread for transcription"""
    global audio_queue, is_streaming, transcriptions, whisper_model
    
    buffer = b''
    chunk_duration = 2.0  # seconds
    chunk_size = int(SAMPLE_RATE * chunk_duration * 2)  # 2 bytes per sample
    
    while is_streaming:
        try:
            # Get audio data from queue
            audio_data = audio_queue.get(timeout=0.1)
            buffer += audio_data
            
            # Process when we have enough data
            if len(buffer) >= chunk_size:
                # Convert to numpy array
                audio_np = np.frombuffer(buffer[:chunk_size], dtype=np.int16)
                audio_float = audio_np.astype(np.float32) / 32768.0
                
                # Transcribe
                try:
                    result = whisper_model.transcribe(
                        audio_float,
                        language='en',
                        fp16=False
                    )
                    
                    text = result['text'].strip()
                    if text:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        transcriptions.append({
                            'time': timestamp,
                            'text': text
                        })
                        print(f"[{timestamp}] {text}")
                        
                except Exception as e:
                    print(f"Transcription error: {e}")
                
                # Remove processed data
                buffer = buffer[chunk_size:]
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker error: {e}")


def start_streaming(device_selection):
    """Start audio streaming"""
    global is_streaming, audio_queue, stream_thread, transcription_thread, transcriptions, whisper_model, current_device
    
    if is_streaming:
        return "Already streaming!", get_transcriptions_html()
    
    # Initialize Whisper if needed
    if whisper_model is None:
        print("Loading Whisper model...")
        whisper_model = whisper.load_model("tiny")
        print("Whisper model loaded")
    
    # Parse device selection
    device_index = None
    if device_selection and device_selection != "default":
        try:
            device_index = int(device_selection.split(']')[0].strip('['))
        except:
            device_index = None
    
    current_device = device_index
    
    # Clear previous data
    transcriptions = []
    audio_queue = queue.Queue(maxsize=100)
    
    # Start audio stream
    try:
        stream = sd.InputStream(
            device=device_index,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
            dtype='float32'
        )
        stream.start()
        
        # Start transcription thread
        is_streaming = True
        transcription_thread = threading.Thread(target=transcription_worker)
        transcription_thread.start()
        
        # Store stream reference
        stream_thread = stream
        
        device_name = device_selection if device_selection else "default"
        return f"✅ Streaming started! Device: {device_name}", get_transcriptions_html()
        
    except Exception as e:
        is_streaming = False
        return f"❌ Error: {str(e)}", get_transcriptions_html()


def stop_streaming():
    """Stop audio streaming"""
    global is_streaming, stream_thread, transcription_thread
    
    if not is_streaming:
        return "Not currently streaming", get_transcriptions_html()
    
    # Stop streaming
    is_streaming = False
    
    # Stop audio stream
    if stream_thread:
        stream_thread.stop()
        stream_thread.close()
        stream_thread = None
    
    # Wait for transcription thread
    if transcription_thread:
        transcription_thread.join(timeout=2.0)
        transcription_thread = None
    
    return "✅ Streaming stopped", get_transcriptions_html()


def get_transcriptions_html():
    """Get HTML display of transcriptions"""
    if not transcriptions:
        return '<div style="padding: 20px; text-align: center; color: #666;">🎤 Waiting for speech...</div>'
    
    html = '<div style="max-height: 400px; overflow-y: auto;">'
    html += '<table style="width: 100%; border-collapse: collapse;">'
    html += '<tr style="background-color: #f0f0f0;"><th style="padding: 8px;">Time</th><th style="padding: 8px;">Transcription</th></tr>'
    
    # Show last 20 transcriptions
    for item in transcriptions[-20:]:
        html += f'<tr style="border-bottom: 1px solid #ddd;">'
        html += f'<td style="padding: 8px; white-space: nowrap;">{item["time"]}</td>'
        html += f'<td style="padding: 8px;">{item["text"]}</td>'
        html += '</tr>'
    
    html += '</table></div>'
    return html


def refresh_display():
    """Refresh the transcription display"""
    status = f"🔴 Streaming active - {len(transcriptions)} transcriptions" if is_streaming else "⚫ Not streaming"
    return get_transcriptions_html(), status


# Create Gradio interface
def create_app():
    """Create Gradio application"""
    
    with gr.Blocks(title="Real-time Audio Transcription", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎤 Real-time Audio Transcription
        
        Simple, working real-time audio transcription using Whisper.
        Select your microphone and start streaming!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Get available devices
                devices = get_audio_devices()
                device_choices = ["default"] + [dev['label'] for dev in devices]
                
                audio_device = gr.Dropdown(
                    choices=device_choices,
                    value="default",
                    label="Audio Input Device",
                    interactive=True
                )
                
                with gr.Row():
                    start_btn = gr.Button("🎤 Start Streaming", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop Streaming", variant="stop")
                
                refresh_btn = gr.Button("🔄 Refresh Display")
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready to stream",
                    lines=1,
                    interactive=False
                )
            
            with gr.Column(scale=2):
                transcriptions_display = gr.HTML(
                    label="Transcriptions",
                    value=get_transcriptions_html()
                )
        
        # Event handlers
        start_btn.click(
            fn=start_streaming,
            inputs=[audio_device],
            outputs=[status_text, transcriptions_display]
        )
        
        stop_btn.click(
            fn=stop_streaming,
            outputs=[status_text, transcriptions_display]
        )
        
        refresh_btn.click(
            fn=refresh_display,
            outputs=[transcriptions_display, status_text]
        )
        
        # Note: Manual refresh needed - click "Refresh Display" button
        # Auto-refresh requires Gradio 4.0+ with 'every' parameter
    
    return app


# Main entry point
if __name__ == "__main__":
    import sys
    
    print("Real-time Audio Transcription")
    print("=============================")
    print("Loading application...")
    
    # Create and launch app
    app = create_app()
    
    # Check for share flag
    share = "--share" in sys.argv
    
    try:
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=share,
            show_api=False,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        if is_streaming:
            stop_streaming()