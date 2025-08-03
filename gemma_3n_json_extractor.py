#!/usr/bin/env python3
"""
Gemma 3n JSON Extractor Module
Extracts structured JSON data from transcripts using Google's Gemma 3n model
"""

import json
import time
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import re

# Try to import Unsloth first (before transformers) for optimal performance
UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("✅ Unsloth available for accelerated inference!")
except ImportError:
    pass

# Model loading options
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    if not UNSLOTH_AVAILABLE:
        print("Warning: Unsloth not available. Using standard transformers.")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/torch not available. Install with: pip install transformers torch")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JSONExtractor:
    """Base class for JSON extraction"""
    
    def extract(self, transcript: str, timestamp_ms: int = None) -> Dict:
        """Extract structured data from transcript"""
        raise NotImplementedError


class GemmaExtractor(JSONExtractor):
    """Gemma-based JSON extractor using Transformers with Unsloth optimization"""
    
    def __init__(self, model_id="google/gemma-2b-it", device=None, max_length=512, use_unsloth=True):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library not available")
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        
        # Use Unsloth for Gemma 3n model if available and requested
        if use_unsloth and UNSLOTH_AVAILABLE and "gemma-3n" in model_id.lower():
            logger.info(f"Loading Gemma model with Unsloth optimization: gemma-3n-e4b-it")
            # Load with 4bit quantization for faster inference
            self.model, self.processor = FastLanguageModel.from_pretrained(
                "unsloth/gemma-3n-e4b-it",  # Use the specific Unsloth model
                dtype=None,  # Let Unsloth handle dtype
                load_in_4bit=True,  # Use 4-bit quantization for efficiency
            )
            # Use the fast inference mode
            FastLanguageModel.for_inference(self.model)
            # Store the actual tokenizer from the processor
            self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
            logger.info(f"Gemma 3n model loaded with Unsloth on {device}")
        else:
            # Fallback to standard transformers
            logger.info(f"Loading Gemma model: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device
            )
            self.model.eval()
            logger.info(f"Gemma model loaded on {device}")
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _create_prompt(self, transcript: str) -> str:
        """Create prompt for JSON extraction"""
        # Use apply_chat_template - check processor first, then tokenizer
        apply_template_fn = None
        if hasattr(self, 'processor') and hasattr(self.processor, 'apply_chat_template'):
            apply_template_fn = self.processor.apply_chat_template
        elif hasattr(self.tokenizer, 'apply_chat_template'):
            apply_template_fn = self.tokenizer.apply_chat_template
            
        if apply_template_fn:
            messages = [
                {
                    "role": "user",
                    "content": f"""Extract structured information from the following transcript and return it as JSON.

The JSON should follow this schema:
{{
  "transcript": "the original transcript text",
  "timestamp_ms": "current timestamp in milliseconds",
  "intent": "the main intent or purpose of the speaker",
  "entities": ["list", "of", "key", "entities", "mentioned"]
}}

Transcript: "{transcript}"

Return only valid JSON, no additional text."""
                }
            ]
            return apply_template_fn(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback to manual template
            prompt = f"""<start_of_turn>user
Extract structured information from the following transcript and return it as JSON.

The JSON should follow this schema:
{{
  "transcript": "the original transcript text",
  "timestamp_ms": "current timestamp in milliseconds",
  "intent": "the main intent or purpose of the speaker",
  "entities": ["list", "of", "key", "entities", "mentioned"]
}}

Transcript: "{transcript}"

Return only valid JSON, no additional text.<end_of_turn>
<start_of_turn>model
"""
            return prompt
    
    def extract(self, transcript: str, timestamp_ms: int = None) -> Dict:
        """Extract structured data from transcript"""
        if not transcript.strip():
            return {
                "transcript": "",
                "timestamp_ms": timestamp_ms or int(time.time() * 1000),
                "intent": "empty",
                "entities": []
            }
        
        # Create prompt
        prompt = self._create_prompt(transcript)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,  # Pass string directly 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
            add_special_tokens=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        json_str = self._extract_json_from_text(generated)
        
        try:
            result = json.loads(json_str)
            # Ensure all required fields
            result["transcript"] = result.get("transcript", transcript)
            result["timestamp_ms"] = result.get("timestamp_ms", timestamp_ms or int(time.time() * 1000))
            result["intent"] = result.get("intent", "unknown")
            result["entities"] = result.get("entities", [])
            return result
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from model output: {json_str}")
            return self._fallback_extraction(transcript, timestamp_ms)
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from generated text"""
        # Split by model turn marker
        if "<start_of_turn>model" in text:
            text = text.split("<start_of_turn>model")[-1]
        
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # Try to find JSON between code blocks
        code_match = re.search(r'```(?:json)?\s*(\{[^{}]*\})\s*```', text, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        return text.strip()
    
    def _fallback_extraction(self, transcript: str, timestamp_ms: int = None) -> Dict:
        """Simple rule-based extraction as fallback"""
        # Basic intent detection
        intent = "unknown"
        transcript_lower = transcript.lower()
        
        if any(word in transcript_lower for word in ["what", "when", "where", "who", "how", "why"]):
            intent = "question"
        elif any(word in transcript_lower for word in ["please", "can you", "could you", "would you"]):
            intent = "request"
        elif any(word in transcript_lower for word in ["thank", "thanks", "appreciate"]):
            intent = "gratitude"
        elif any(word in transcript_lower for word in ["hello", "hi", "hey", "good morning"]):
            intent = "greeting"
        elif any(word in transcript_lower for word in ["bye", "goodbye", "see you"]):
            intent = "farewell"
        else:
            intent = "statement"
        
        # Basic entity extraction (proper nouns, numbers)
        entities = []
        
        # Extract capitalized words (potential names/places)
        words = transcript.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 1:
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word and clean_word not in ["I", "The", "A", "An"]:
                    entities.append(clean_word)
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\b', transcript)
        entities.extend(numbers)
        
        # Remove duplicates
        entities = list(dict.fromkeys(entities))
        
        return {
            "transcript": transcript,
            "timestamp_ms": timestamp_ms or int(time.time() * 1000),
            "intent": intent,
            "entities": entities[:10]  # Limit to 10 entities
        }


class MockExtractor(JSONExtractor):
    """Mock extractor for testing without loading models"""
    
    def __init__(self):
        logger.info("Using mock JSON extractor")
    
    def extract(self, transcript: str, timestamp_ms: int = None) -> Dict:
        """Simple mock extraction"""
        # Simulate processing time
        time.sleep(0.1)
        
        # Basic extraction
        words = transcript.split()
        entities = [w for w in words if len(w) > 4 and w[0].isupper()]
        
        intent = "statement"
        if "?" in transcript:
            intent = "question"
        elif any(cmd in transcript.lower() for cmd in ["please", "could", "can you"]):
            intent = "request"
        
        return {
            "transcript": transcript,
            "timestamp_ms": timestamp_ms or int(time.time() * 1000),
            "intent": intent,
            "entities": entities[:5]
        }


class ExtractionManager:
    """Manages JSON extraction from transcripts"""
    
    def __init__(self, extractor_type="gemma", **extractor_kwargs):
        # Create extractor
        if extractor_type == "gemma" and TRANSFORMERS_AVAILABLE:
            # Use Gemma 3n model with Unsloth if available
            if UNSLOTH_AVAILABLE:
                logger.info("Using Gemma 3n with Unsloth optimization")
                # Override model_id to use Gemma 3n for Unsloth optimization
                extractor_kwargs["model_id"] = "gemma-3n-e4b-it"
                extractor_kwargs["use_unsloth"] = True
            self.extractor = GemmaExtractor(**extractor_kwargs)
        else:
            if extractor_type == "gemma" and not TRANSFORMERS_AVAILABLE:
                logger.warning("Gemma requested but transformers not available, using mock")
            self.extractor = MockExtractor()
        
        self.extraction_history = []
    
    def extract_from_transcript(self, transcript: str, timestamp_ms: int = None) -> Dict:
        """Extract JSON from single transcript"""
        start_time = time.time()
        
        result = self.extractor.extract(transcript, timestamp_ms)
        
        # Add metadata
        result["extraction_time_ms"] = int((time.time() - start_time) * 1000)
        result["datetime"] = datetime.now().isoformat()
        
        # Save to history
        self.extraction_history.append(result)
        
        return result
    
    def extract_from_transcripts(self, transcripts: List[Dict]) -> List[Dict]:
        """Extract JSON from multiple transcripts"""
        results = []
        
        for transcript_data in transcripts:
            if isinstance(transcript_data, dict):
                text = transcript_data.get("text", "")
                timestamp = transcript_data.get("timestamp", time.time())
                timestamp_ms = int(timestamp * 1000)
            else:
                text = str(transcript_data)
                timestamp_ms = int(time.time() * 1000)
            
            result = self.extract_from_transcript(text, timestamp_ms)
            results.append(result)
        
        return results
    
    def process_streaming_transcript(self, transcript: str, timestamp: float):
        """Process transcript from streaming source"""
        timestamp_ms = int(timestamp * 1000)
        result = self.extract_from_transcript(transcript, timestamp_ms)
        logger.info(f"Extracted: intent={result['intent']}, entities={result['entities']}")
        return result
    
    def get_extraction_history(self) -> List[Dict]:
        """Get all extractions"""
        return self.extraction_history
    
    def save_extractions(self, filepath: str):
        """Save extractions to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.extraction_history, f, indent=2)
        logger.info(f"Saved {len(self.extraction_history)} extractions to {filepath}")
    
    def validate_against_schema(self, data: Dict) -> bool:
        """Validate extracted data against schema"""
        required_fields = ["transcript", "timestamp_ms"]
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                return False
        
        # Check types
        if not isinstance(data.get("transcript"), str):
            return False
        if not isinstance(data.get("timestamp_ms"), int):
            return False
        if "intent" in data and not isinstance(data["intent"], str):
            return False
        if "entities" in data and not isinstance(data["entities"], list):
            return False
        
        return True


async def main():
    """Example usage"""
    import sys
    
    # Test transcripts
    test_transcripts = [
        "Hello, my name is John Smith and I'd like to schedule a meeting for tomorrow at 3 PM.",
        "What's the weather like in San Francisco today?",
        "Please remind me to call Sarah about the project deadline.",
        "The quarterly report shows revenue increased by 15 percent this quarter.",
        "Thank you for your help with the presentation yesterday."
    ]
    
    # Parse arguments
    model_type = sys.argv[1] if len(sys.argv) > 1 else "mock"
    
    # Create extraction manager
    if model_type == "gemma":
        # Use smaller 2B model by default
        manager = ExtractionManager(
            extractor_type="gemma",
            model_id="google/gemma-2b-it"
        )
    else:
        manager = ExtractionManager(extractor_type="mock")
    
    logger.info(f"Running extraction demo with {model_type} extractor")
    
    # Process test transcripts
    for transcript in test_transcripts:
        logger.info(f"\nProcessing: {transcript[:50]}...")
        result = manager.extract_from_transcript(transcript)
        
        # Validate
        is_valid = manager.validate_against_schema(result)
        logger.info(f"Valid schema: {is_valid}")
        
        # Pretty print result
        print(json.dumps(result, indent=2))
    
    # Save results
    manager.save_extractions("extracted_data.json")
    
    # Summary
    logger.info(f"\nProcessed {len(test_transcripts)} transcripts")
    logger.info(f"Average extraction time: {sum(r['extraction_time_ms'] for r in manager.extraction_history) / len(manager.extraction_history):.0f}ms")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
