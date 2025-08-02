#!/usr/bin/env python3
"""
Demo wrapper for BLE Audio → Whisper → Gemma pipeline
Supports both file processing and live streaming
"""

import asyncio
import json
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Import our modules
from ble_listener import (
    AudioStreamManager, 
    MicrophoneSource, 
    MockBLESource, 
    RealBLESource,
    SOUNDDEVICE_AVAILABLE,
    BLEAK_AVAILABLE
)
from whisper_tiny_transcription import (
    TranscriptionManager,
    WHISPER_AVAILABLE
)
from gemma_3n_json_extractor import (
    ExtractionManager,
    TRANSFORMERS_AVAILABLE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, 
                 transcription_engine="whisper",
                 extraction_model="mock",
                 output_dir="pipeline_output"):
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        # Transcription
        if transcription_engine == "deepgram":
            self.transcription_manager = TranscriptionManager(engine_type="deepgram")
        else:
            if WHISPER_AVAILABLE:
                self.transcription_manager = TranscriptionManager(
                    engine_type="whisper", 
                    model_size="tiny"
                )
            else:
                logger.error("Whisper not available. Please install: pip install openai-whisper")
                sys.exit(1)
        
        # Extraction
        self.extraction_manager = ExtractionManager(extractor_type=extraction_model)
        
        # Pipeline state
        self.is_streaming = False
        self.processed_files = []
        self.pipeline_results = []
    
    def _create_result_entry(self, 
                           audio_file: str,
                           transcription: Dict,
                           extraction: Dict) -> Dict:
        """Create a pipeline result entry"""
        return {
            "audio_file": audio_file,
            "timestamp": datetime.now().isoformat(),
            "transcription": transcription,
            "extraction": extraction,
            "pipeline_time_ms": (
                transcription.get("duration", 0) * 1000 + 
                extraction.get("extraction_time_ms", 0)
            )
        }
    
    async def process_file(self, audio_file: str) -> Dict:
        """Process a single audio file through the pipeline"""
        logger.info(f"Processing file: {audio_file}")
        
        # Transcribe
        start_time = time.time()
        transcription = self.transcription_manager.transcribe_file(audio_file)
        
        # Extract JSON
        extraction = self.extraction_manager.extract_from_transcript(
            transcription["text"],
            int(transcription["timestamp"] * 1000)
        )
        
        # Create result
        result = self._create_result_entry(audio_file, transcription, extraction)
        self.pipeline_results.append(result)
        
        # Log summary
        total_time = time.time() - start_time
        logger.info(f"Processed in {total_time:.2f}s")
        logger.info(f"Transcription: {transcription['text'][:100]}...")
        logger.info(f"Intent: {extraction['intent']}, Entities: {extraction['entities']}")
        
        return result
    
    async def process_directory(self, directory: str, pattern="*.wav") -> List[Dict]:
        """Process all audio files in a directory"""
        audio_dir = Path(directory)
        audio_files = list(audio_dir.glob(pattern))
        
        logger.info(f"Found {len(audio_files)} audio files in {directory}")
        
        results = []
        for audio_file in audio_files:
            result = await self.process_file(str(audio_file))
            results.append(result)
        
        return results
    
    async def start_streaming(self, audio_source) -> None:
        """Start live streaming pipeline"""
        self.is_streaming = True
        
        # Create audio stream manager
        audio_manager = AudioStreamManager(
            audio_source, 
            output_dir=str(self.output_dir / "recordings")
        )
        
        # Callback to process transcriptions
        def transcription_callback(text: str, timestamp: float):
            """Process transcription through extraction"""
            if text.strip():
                extraction = self.extraction_manager.process_streaming_transcript(
                    text, timestamp
                )
                
                # Create and save result
                result = {
                    "mode": "streaming",
                    "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                    "transcription": {"text": text, "timestamp": timestamp},
                    "extraction": extraction
                }
                self.pipeline_results.append(result)
                
                # Print live result
                print(f"\n[{result['timestamp']}]")
                print(f"Transcript: {text}")
                print(f"Intent: {extraction['intent']}")
                print(f"Entities: {extraction['entities']}")
                print("-" * 80)
        
        # Connect pipeline
        self.transcription_manager.streaming_transcriber.add_callback(
            transcription_callback
        )
        await self.transcription_manager.transcribe_stream(
            audio_manager.add_callback
        )
        
        # Start streaming
        logger.info("Starting live streaming pipeline...")
        logger.info("Audio buffer fills every 2 seconds before transcription")
        logger.info("Press Ctrl+C to stop")
        
        try:
            await audio_manager.start_streaming()
        except KeyboardInterrupt:
            logger.info("\nStopping pipeline...")
        finally:
            self.is_streaming = False
            await self.transcription_manager.stop_streaming()
            await audio_manager.stop_streaming()
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save pipeline results to JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare summary
        summary = {
            "pipeline_run": {
                "timestamp": datetime.now().isoformat(),
                "total_processed": len(self.pipeline_results),
                "transcription_engine": self.transcription_manager.engine.__class__.__name__,
                "extraction_model": self.extraction_manager.extractor.__class__.__name__,
            },
            "results": self.pipeline_results
        }
        
        # Save
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved results to: {filepath}")
        return str(filepath)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="BLE Audio → Whisper → Gemma Pipeline Demo"
    )
    
    # Mode selection
    parser.add_argument(
        "mode",
        choices=["file", "directory", "stream"],
        help="Processing mode"
    )
    
    # Input source
    parser.add_argument(
        "--input",
        help="Input file/directory for file/directory mode"
    )
    
    # Audio source for streaming
    parser.add_argument(
        "--audio-source",
        choices=["ble", "mic", "mock"],
        default="mock",
        help="Audio source for streaming mode (default: mock)"
    )
    
    # Engine options
    parser.add_argument(
        "--transcription",
        choices=["whisper", "deepgram"],
        default="whisper",
        help="Transcription engine (default: whisper)"
    )
    
    parser.add_argument(
        "--extraction",
        choices=["gemma", "mock"],
        default="mock",
        help="Extraction model (default: mock)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        default="pipeline_output",
        help="Output directory (default: pipeline_output)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.mode in ["file", "directory"] and not args.input:
        parser.error(f"{args.mode} mode requires --input")
    
    # Check dependencies
    if args.transcription == "whisper" and not WHISPER_AVAILABLE:
        logger.error("Whisper not available. Install with: pip install openai-whisper")
        return
    
    if args.extraction == "gemma" and not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers not available for Gemma. Using mock extractor.")
        args.extraction = "mock"
    
    if args.audio_source == "mic" and not SOUNDDEVICE_AVAILABLE:
        logger.warning("Sounddevice not available. Using mock audio source.")
        args.audio_source = "mock"
    
    if args.audio_source == "ble" and not BLEAK_AVAILABLE:
        logger.warning("Bleak not available. Using mock audio source.")
        args.audio_source = "mock"
    
    # Create pipeline
    pipeline = AudioPipeline(
        transcription_engine=args.transcription,
        extraction_model=args.extraction,
        output_dir=args.output_dir
    )
    
    # Process based on mode
    if args.mode == "file":
        # Process single file
        await pipeline.process_file(args.input)
        
    elif args.mode == "directory":
        # Process directory
        await pipeline.process_directory(args.input)
        
    elif args.mode == "stream":
        # Create audio source
        if args.audio_source == "ble":
            audio_source = RealBLESource()
        elif args.audio_source == "mic":
            audio_source = MicrophoneSource()
        else:
            audio_source = MockBLESource()
        
        # Start streaming
        await pipeline.start_streaming(audio_source)
    
    # Save results
    if pipeline.pipeline_results:
        output_file = pipeline.save_results()
        print(f"\nPipeline complete. Results saved to: {output_file}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Total processed: {len(pipeline.pipeline_results)}")
        if args.mode == "stream":
            print(f"  Recordings saved to: {pipeline.output_dir / 'recordings'}")


if __name__ == "__main__":
    # Example usage messages
    if len(sys.argv) == 1:
        print("BLE Audio → Whisper → Gemma Pipeline Demo")
        print("\nExamples:")
        print("  # Process a single file:")
        print("  python demo.py file --input audio.wav")
        print()
        print("  # Process all WAV files in a directory:")
        print("  python demo.py directory --input ./audio_recordings")
        print()
        print("  # Live streaming with mock BLE device:")
        print("  python demo.py stream")
        print()
        print("  # Live streaming with microphone:")
        print("  python demo.py stream --audio-source mic")
        print()
        print("  # Live streaming with Gemma extraction:")
        print("  python demo.py stream --extraction gemma")
        print()
        sys.exit(0)
    
    asyncio.run(main())
