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


async def start_streaming(audio_source="mock", transcription_engine="whisper", extraction_model="mock"):
    """Start live streaming"""
    global streaming_manager, is_streaming, transcription_history, extraction_history
    
    if is_streaming:
        return "Already streaming!", None, None
    
    # Initialize if needed
    if transcription_manager is None:
        status, _ = initialize_pipeline(transcription_engine, extraction_model)
        if "Error" in status:
            return status, None, None
    
    # Clear history
    transcription_history = []
    extraction_history = []
    
    # Create audio source
    if audio_source == "mic" and SOUNDDEVICE_AVAILABLE:
        source = MicrophoneSource()
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
    
    return "Streaming started!", None, None


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
        return None, None, None
    
    # Format transcription history
    if transcription_history:
        trans_df = pd.DataFrame(transcription_history[-10:])  # Last 10
        trans_display = trans_df.to_html(index=False)
    else:
        trans_display = "No transcriptions yet..."
    
    # Format extraction history
    if extraction_history:
        ext_df = pd.DataFrame(extraction_history[-10:])  # Last 10
        ext_display = ext_df.to_html(index=False)
    else:
        ext_display = "No extractions yet..."
    
    # Status
    status = f"Streaming active - {len(transcription_history)} transcriptions"
    
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
                        value="mock",
                        label="Audio Source"
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
                    gr.Markdown("### Live Results")
                    refresh_btn = gr.Button("Refresh Display")
                    
                    with gr.Row():
                        live_transcriptions = gr.HTML(label="Recent Transcriptions")
                        live_extractions = gr.HTML(label="Recent Extractions")
        
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
        
        start_btn.click(
            fn=lambda *args: asyncio.run(start_streaming(*args)),
            inputs=[audio_source, stream_trans, stream_ext],
            outputs=[stream_status, live_transcriptions, live_extractions]
        )
        
        stop_btn.click(
            fn=lambda: asyncio.run(stop_streaming()),
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