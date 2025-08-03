#!/usr/bin/env python3
"""
Gradio UI for BLE Audio → Whisper → Gemma Pipeline
Provides web interface for audio processing with live transcription
"""

import gradio as gr
import asyncio
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import numpy as np
import sounddevice as sd

# Import our modules
from ble_listener import (
    AudioStreamManager, 
    MockBLESource, 
    MicrophoneSource,
    RealBLESource,
    BLEAK_AVAILABLE,
    SOUNDDEVICE_AVAILABLE
)
from whisper_tiny_transcription import (
    TranscriptionManager,
    WHISPER_AVAILABLE
)
from gemma_3n_json_extractor import (
    ExtractionManager,
    TRANSFORMERS_AVAILABLE
)
from simulate_live_mic import FileStreamSource

# Global state
streaming_manager = None
transcription_manager = None
extraction_manager = None
is_streaming = False
transcription_history = []
extraction_history = []


def get_audio_devices():
    """Get list of available audio input devices"""
    if not SOUNDDEVICE_AVAILABLE:
        return []
    
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


def initialize_pipeline(transcription_engine="whisper", extraction_model="mock"):
    """Initialize the processing pipeline"""
    global transcription_manager, extraction_manager
    
    # Initialize transcription
    if transcription_engine == "deepgram":
        if not os.environ.get("DEEPGRAM_API_KEY"):
            return "Error: DEEPGRAM_API_KEY not set", None
        transcription_manager = TranscriptionManager(engine_type="deepgram")
    else:
        if not WHISPER_AVAILABLE:
            return "Error: Whisper not available. Install with: pip install openai-whisper", None
        transcription_manager = TranscriptionManager(
            engine_type="whisper", 
            model_size="tiny"
        )
    
    # Initialize extraction
    extraction_manager = ExtractionManager(extractor_type=extraction_model)
    
    return "Pipeline initialized successfully!", None


def process_audio_file(audio_file, transcription_engine="whisper", extraction_model="mock"):
    """Process uploaded audio file"""
    if audio_file is None:
        return "Please upload an audio file", None, None, None
    
    # Initialize if needed
    if transcription_manager is None:
        status, _ = initialize_pipeline(transcription_engine, extraction_model)
        if "Error" in status:
            return status, None, None, None
    
    # Process file
    try:
        # Transcribe
        result = transcription_manager.transcribe_file(audio_file)
        transcript = result["text"]
        
        # Extract JSON
        extraction = extraction_manager.extract_from_transcript(
            transcript,
            int(time.time() * 1000)
        )
        
        # Format results
        transcription_display = f"**Transcription:**\n{transcript}\n\n**Processing Time:** {result['duration']:.2f}s"
        
        json_display = json.dumps(extraction, indent=2)
        
        # Create summary
        summary = f"""
### Summary
- **File:** {Path(audio_file).name}
- **Transcription Length:** {len(transcript.split())} words
- **Intent:** {extraction['intent']}
- **Entities:** {', '.join(extraction['entities']) if extraction['entities'] else 'None'}
- **Processing Time:** {result['duration']:.2f}s
"""
        
        return "File processed successfully!", transcription_display, json_display, summary
        
    except Exception as e:
        return f"Error processing file: {str(e)}", None, None, None


async def start_streaming(audio_source="mock", audio_device=None, transcription_engine="whisper", extraction_model="mock"):
    """Start live streaming"""
    global streaming_manager, is_streaming, transcription_history, extraction_history, transcription_manager, extraction_manager
    
    if is_streaming:
        return "Already streaming!", None, None
    
    # Always reinitialize to ensure fresh event loop context
    status, _ = initialize_pipeline(transcription_engine, extraction_model)
    if "Error" in status:
        return status, None, None
    
    # Clear history
    transcription_history = []
    extraction_history = []
    
    # Create audio source
    if audio_source == "mic" and SOUNDDEVICE_AVAILABLE:
        # Parse device index from dropdown selection
        device_index = None
        if audio_device and audio_device != "default":
            try:
                # Extract device index from "[index] device name" format
                device_index = int(audio_device.split(']')[0].strip('['))
            except:
                device_index = None
        source = MicrophoneSource(device_index=device_index)
    elif audio_source == "ble" and BLEAK_AVAILABLE:
        source = RealBLESource()
    else:
        source = MockBLESource()
    
    # Create stream manager
    streaming_manager = AudioStreamManager(source, output_dir="gradio_recordings")
    
    # Transcription callback
    def transcription_callback(text: str, timestamp: float):
        if text.strip():
            # Process transcription
            extraction = extraction_manager.process_streaming_transcript(text, timestamp)
            
            # Add to history
            transcription_history.append({
                "time": datetime.fromtimestamp(timestamp).strftime("%H:%M:%S"),
                "text": text
            })
            
            extraction_history.append({
                "time": datetime.fromtimestamp(timestamp).strftime("%H:%M:%S"),
                "intent": extraction['intent'],
                "entities": ', '.join(extraction['entities'])
            })
    
    # Connect pipeline
    transcription_manager.streaming_transcriber.add_callback(transcription_callback)
    await transcription_manager.transcribe_stream(streaming_manager.add_callback)
    
    # Start streaming
    is_streaming = True
    
    # Run in background
    asyncio.create_task(streaming_manager.start_streaming())
    
    # Status message
    if audio_source == "mic":
        device_info = f" (Device: {audio_device or 'default'})"
    else:
        device_info = ""
    
    status_msg = f"✅ Streaming started! Source: {audio_source}{device_info}\n📊 Real-time transcription active..."
    
    return status_msg, None, None


async def stop_streaming():
    """Stop live streaming"""
    global streaming_manager, is_streaming
    
    if not is_streaming:
        return "Not currently streaming", None, None
    
    # Stop streaming
    is_streaming = False
    if streaming_manager:
        await transcription_manager.stop_streaming()
        await streaming_manager.stop_streaming()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save transcriptions
    if transcription_history:
        trans_file = f"gradio_transcriptions_{timestamp}.json"
        transcription_manager.save_transcriptions(trans_file)
    
    # Save extractions
    if extraction_history:
        ext_file = f"gradio_extractions_{timestamp}.json"
        extraction_manager.save_extractions(ext_file)
    
    return "Streaming stopped", None, f"Results saved to files"


def update_streaming_display():
    """Update the streaming display with latest results"""
    if not is_streaming:
        return None, None, "Not currently streaming"
    
    # Format transcription history with better styling
    if transcription_history:
        trans_html = '<div style="max-height: 400px; overflow-y: auto;">'
        trans_html += '<table style="width: 100%; border-collapse: collapse;">'
        trans_html += '<tr style="background-color: #f0f0f0;"><th style="padding: 8px;">Time</th><th style="padding: 8px;">Transcription</th></tr>'
        
        for item in transcription_history[-15:]:  # Last 15 entries
            trans_html += f'<tr style="border-bottom: 1px solid #ddd;">'
            trans_html += f'<td style="padding: 8px; white-space: nowrap;">{item["time"]}</td>'
            trans_html += f'<td style="padding: 8px;">{item["text"]}</td>'
            trans_html += '</tr>'
        
        trans_html += '</table></div>'
        trans_display = trans_html
    else:
        trans_display = '<div style="padding: 20px; text-align: center; color: #666;">🎤 Waiting for speech...</div>'
    
    # Format extraction history with better styling
    if extraction_history:
        ext_html = '<div style="max-height: 400px; overflow-y: auto;">'
        ext_html += '<table style="width: 100%; border-collapse: collapse;">'
        ext_html += '<tr style="background-color: #f0f0f0;"><th style="padding: 8px;">Time</th><th style="padding: 8px;">Intent</th><th style="padding: 8px;">Entities</th></tr>'
        
        for item in extraction_history[-15:]:  # Last 15 entries
            ext_html += f'<tr style="border-bottom: 1px solid #ddd;">'
            ext_html += f'<td style="padding: 8px; white-space: nowrap;">{item["time"]}</td>'
            ext_html += f'<td style="padding: 8px;">{item["intent"]}</td>'
            ext_html += f'<td style="padding: 8px;">{item["entities"]}</td>'
            ext_html += '</tr>'
        
        ext_html += '</table></div>'
        ext_display = ext_html
    else:
        ext_display = '<div style="padding: 20px; text-align: center; color: #666;">📋 No extractions yet...</div>'
    
    # Status with emoji
    status = f"🔴 Streaming active | 📝 {len(transcription_history)} transcriptions | 🎯 {len(extraction_history)} extractions"
    
    return trans_display, ext_display, status


# Create Gradio interface
def create_app():
    """Create Gradio application"""
    
    with gr.Blocks(title="BLE Audio Pipeline", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎤 BLE Audio → Whisper → Gemma Pipeline
        
        Real-time audio transcription and intent extraction pipeline.
        Supports file upload and live streaming from BLE/microphone sources.
        """)
        
        with gr.Tab("File Processing"):
            gr.Markdown("### Upload and process audio files")
            
            with gr.Row():
                with gr.Column():
                    file_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    
                    with gr.Row():
                        trans_engine = gr.Radio(
                            ["whisper", "deepgram"],
                            value="whisper",
                            label="Transcription Engine"
                        )
                        ext_model = gr.Radio(
                            ["mock", "gemma"],
                            value="mock",
                            label="Extraction Model"
                        )
                    
                    process_btn = gr.Button("Process File", variant="primary")
                
                with gr.Column():
                    file_status = gr.Textbox(label="Status", lines=1)
                    file_summary = gr.Markdown(label="Summary")
            
            with gr.Row():
                file_transcription = gr.Markdown(label="Transcription")
                file_json = gr.Code(label="Extracted JSON", language="json")
        
        with gr.Tab("Live Streaming"):
            gr.Markdown("### Stream audio from BLE device or microphone")
            
            with gr.Row():
                with gr.Column():
                    audio_source = gr.Radio(
                        ["mock", "mic", "ble"],
                        value="mic",
                        label="Audio Source"
                    )
                    
                    # Get available audio devices
                    audio_devices = get_audio_devices()
                    device_choices = ["default"] + [dev['label'] for dev in audio_devices]
                    
                    audio_device_dropdown = gr.Dropdown(
                        choices=device_choices,
                        value="default",
                        label="Audio Input Device",
                        visible=True,
                        interactive=True
                    )
                    
                    # Update dropdown visibility based on audio source
                    def update_device_visibility(source):
                        return gr.update(visible=(source == "mic"))
                    
                    audio_source.change(
                        fn=update_device_visibility,
                        inputs=[audio_source],
                        outputs=[audio_device_dropdown]
                    )
                    
                    with gr.Row():
                        stream_trans = gr.Radio(
                            ["whisper", "deepgram"],
                            value="whisper",
                            label="Transcription Engine"
                        )
                        stream_ext = gr.Radio(
                            ["mock", "gemma"],
                            value="mock",
                            label="Extraction Model"
                        )
                    
                    with gr.Row():
                        start_btn = gr.Button("Start Streaming", variant="primary")
                        stop_btn = gr.Button("Stop Streaming", variant="stop")
                    
                    stream_status = gr.Textbox(label="Status", lines=2)
                
                with gr.Column():
                    gr.Markdown("### Live Results (Real-time)")
                    with gr.Row():
                        refresh_btn = gr.Button("Refresh Display")
                        refresh_devices_btn = gr.Button("Refresh Audio Devices")
                    
                    with gr.Row():
                        live_transcriptions = gr.HTML(label="Recent Transcriptions", elem_id="live_trans")
                        live_extractions = gr.HTML(label="Recent Extractions", elem_id="live_ext")
                    
                    # Auto-refresh notice
                    gr.Markdown("*💡 Click 'Refresh Display' to see latest transcriptions, or enable auto-refresh below*")
                    
                    # Refresh audio devices
                    def refresh_audio_devices():
                        devices = get_audio_devices()
                        choices = ["default"] + [dev['label'] for dev in devices]
                        return gr.update(choices=choices)
                    
                    refresh_devices_btn.click(
                        fn=refresh_audio_devices,
                        outputs=[audio_device_dropdown]
                    )
        
        with gr.Tab("Settings"):
            gr.Markdown("""
            ### Configuration
            
            - **Whisper Model:** Tiny (fast, runs on CPU/GPU)
            - **Audio Format:** 16kHz, mono, 16-bit PCM
            - **Streaming Buffer:** 2 seconds
            - **Output Directory:** `gradio_recordings/`
            
            ### API Keys
            
            For Deepgram transcription, set environment variable:
            ```bash
            export DEEPGRAM_API_KEY=your_api_key_here
            ```
            
            ### Hardware Requirements
            
            - **Microphone:** Requires native audio (not available in WSL)
            - **BLE:** Requires Bluetooth adapter and `bleak` library
            - **GPU:** Optional but recommended for Whisper/Gemma
            """)
        
        with gr.Tab("About"):
            gr.Markdown("""
            ### BLE Audio Pipeline Demo
            
            This demonstration showcases a complete audio processing pipeline:
            
            1. **Audio Input**: BLE device (Omi pendant), microphone, or files
            2. **Transcription**: OpenAI Whisper or Deepgram API
            3. **Intent Extraction**: Rule-based or Gemma 3n model
            4. **Output**: Structured JSON following the schema
            
            ### Schema
            
            ```json
            {
                "transcript": "the transcribed text",
                "timestamp_ms": 1234567890,
                "intent": "statement|question|request|...",
                "entities": ["extracted", "entities"]
            }
            ```
            
            ### Limitations
            
            - No actual Omi device (using mock BLE source)
            - WSL has audio limitations (recommend native Linux)
            - Gemma requires significant RAM/VRAM
            
            Built with ❤️ using Gradio
            """)
        
        # Event handlers
        process_btn.click(
            fn=process_audio_file,
            inputs=[file_input, trans_engine, ext_model],
            outputs=[file_status, file_transcription, file_json, file_summary]
        )
        
        # Use sync wrappers to avoid event loop conflicts
        def start_streaming_sync(*args):
            """Synchronous wrapper for start_streaming"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(start_streaming(*args))
            finally:
                loop.close()
        
        def stop_streaming_sync():
            """Synchronous wrapper for stop_streaming"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(stop_streaming())
            finally:
                loop.close()
        
        start_btn.click(
            fn=start_streaming_sync,
            inputs=[audio_source, audio_device_dropdown, stream_trans, stream_ext],
            outputs=[stream_status, live_transcriptions, live_extractions]
        )
        
        stop_btn.click(
            fn=stop_streaming_sync,
            outputs=[stream_status, live_transcriptions, live_extractions]
        )
        
        refresh_btn.click(
            fn=update_streaming_display,
            outputs=[live_transcriptions, live_extractions, stream_status]
        )
        
        # Auto-refresh for live streaming
        # Note: 'every' parameter requires Gradio 4.0+
        # For older versions, users need to manually click refresh
    
    return app


# Main entry point
if __name__ == "__main__":
    import sys
    
    # Check dependencies
    print("BLE Audio Pipeline - Gradio UI")
    print("==============================")
    print(f"Whisper available: {WHISPER_AVAILABLE}")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    print(f"BLE available: {BLEAK_AVAILABLE}")
    print(f"Microphone available: {SOUNDDEVICE_AVAILABLE}")
    print()
    
    # Create and launch app
    app = create_app()
    
    # Launch options
    share = "--share" in sys.argv
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,  # Creates public URL if True
        show_api=False,
        show_error=True
    )