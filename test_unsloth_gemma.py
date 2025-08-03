#!/usr/bin/env python3
"""
Test script to verify Unsloth Gemma 3n integration
"""

import torch
import time
from gemma_3n_json_extractor import ExtractionManager, UNSLOTH_AVAILABLE, TRANSFORMERS_AVAILABLE

def test_unsloth_gemma():
    print("🚀 Testing Unsloth Gemma 3n Integration")
    print("=" * 60)
    
    # Check dependencies
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    print(f"Unsloth available: {UNSLOTH_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers not available. Cannot proceed.")
        return
    
    if not UNSLOTH_AVAILABLE:
        print("⚠️  Unsloth not available. Will use standard transformers.")
    else:
        print("✅ Unsloth is available! Using optimized Gemma 3n model.")
    
    # Initialize extraction manager with Gemma
    print("\n📋 Initializing Gemma extraction manager...")
    start_time = time.time()
    
    try:
        manager = ExtractionManager(extractor_type="gemma")
        init_time = time.time() - start_time
        print(f"✅ Model loaded in {init_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test transcripts
    test_transcripts = [
        "Could you please set a reminder for tomorrow at 2 PM to call John about the project?",
        "The weather today is sunny with temperatures around 72 degrees.",
        "I need to buy milk, eggs, bread, and cheese from the grocery store.",
        "What time does the movie start tonight?",
        "Thank you for helping me with this task.",
    ]
    
    print("\n🔍 Testing JSON extraction...")
    print("-" * 60)
    
    total_extraction_time = 0
    
    for i, transcript in enumerate(test_transcripts, 1):
        print(f"\nTest {i}: {transcript}")
        
        start_time = time.time()
        result = manager.extract_from_transcript(transcript)
        extraction_time = time.time() - start_time
        total_extraction_time += extraction_time
        
        print(f"  Intent: {result['intent']}")
        print(f"  Entities: {', '.join(result['entities']) if result['entities'] else 'None'}")
        print(f"  Time: {extraction_time:.3f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print(f"  Total transcripts: {len(test_transcripts)}")
    print(f"  Total extraction time: {total_extraction_time:.2f}s")
    print(f"  Average time per transcript: {total_extraction_time/len(test_transcripts):.3f}s")
    
    if UNSLOTH_AVAILABLE:
        print("\n✨ Unsloth optimization is active - expect faster inference!")
    
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_unsloth_gemma()