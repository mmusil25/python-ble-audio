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
    
    def _create_prompt(self, transcript: str, timestamp_ms: int = None) -> str:
        """Create prompt for JSON extraction"""
        # Get current timestamp if not provided
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
            
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
                    "content": f"""Extract keywords and intent from this text:
"{transcript}"

Intent categories: question, request, statement, greeting, farewell, gratitude, command

Extract ALL important keywords including:
- Nouns (people, places, things, concepts)
- Verbs (actions, activities) 
- Times, dates, numbers
- Descriptive words
- Any word that carries meaning

Return JSON:
{{
  "transcript": "{transcript}",
  "timestamp_ms": {timestamp_ms},
  "intent": "<intent>",
  "entities": ["<extract ALL meaningful keywords>"]
}}

Example: "Please call John Smith at 3 PM tomorrow about the project deadline"
{{
  "transcript": "Please call John Smith at 3 PM tomorrow about the project deadline",
  "timestamp_ms": 1234567890,
  "intent": "request",
  "entities": ["call", "John", "Smith", "3 PM", "tomorrow", "project", "deadline"]
}}"""
                }
            ]
            return apply_template_fn(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback to manual template
            prompt = f"""<start_of_turn>user
Extract keywords and intent from this text:
"{transcript}"

Intent categories: question, request, statement, greeting, farewell, gratitude, command

Extract ALL important keywords including:
- Nouns (people, places, things, concepts)
- Verbs (actions, activities) 
- Times, dates, numbers
- Descriptive words
- Any word that carries meaning

Return JSON:
{{
  "transcript": "{transcript}",
  "timestamp_ms": {timestamp_ms},
  "intent": "<intent>",
  "entities": ["<extract ALL meaningful keywords>"]
}}

Example: "Please call John Smith at 3 PM tomorrow about the project deadline"
{{
  "transcript": "Please call John Smith at 3 PM tomorrow about the project deadline",
  "timestamp_ms": 1234567890,
  "intent": "request",
  "entities": ["call", "John", "Smith", "3 PM", "tomorrow", "project", "deadline"]
}}<end_of_turn>
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
        prompt = self._create_prompt(transcript, timestamp_ms)
        
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
                max_new_tokens=200,  # Reduced for faster response
                temperature=0.3,  # Lower for more deterministic output
                do_sample=True,
                top_p=0.9,
                top_k=40,  # Adjusted for better quality
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.05  # Slight penalty for repetition
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Log the raw model output for debugging
        logger.info(f"Model generated: {generated[:200]}...")
        
        # Extract JSON from response
        json_str = self._extract_json_from_text(generated)
        logger.info(f"Extracted JSON string: {json_str[:200]}...")
        
        try:
            result = json.loads(json_str)
            # Ensure all required fields
            result["transcript"] = result.get("transcript", transcript)
            result["timestamp_ms"] = result.get("timestamp_ms", timestamp_ms or int(time.time() * 1000))
            result["intent"] = result.get("intent", "unknown")
            result["entities"] = result.get("entities", [])
            
            logger.info(f"Successfully parsed JSON - intent: {result['intent']}, entities: {result['entities']}")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from model output: {e}")
            logger.warning(f"JSON string was: {json_str}")
            logger.info("Falling back to rule-based extraction")
            return self._fallback_extraction(transcript, timestamp_ms)
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from generated text"""
        # Split by model turn marker
        if "<start_of_turn>model" in text:
            text = text.split("<start_of_turn>model")[-1]
        
        # Try to find JSON object with better regex that handles nested structures
        # Look for opening brace and find its matching closing brace
        brace_count = 0
        start_idx = text.find('{')
        if start_idx != -1:
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start_idx:i+1]
        
        # Try to find JSON between code blocks
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Last resort - try to parse everything after cleaning
        cleaned = text.strip()
        if cleaned.startswith('{') and cleaned.endswith('}'):
            return cleaned
        
        return text.strip()
    
    def _fallback_extraction(self, transcript: str, timestamp_ms: int = None) -> Dict:
        """Simple rule-based extraction as fallback"""
        # Basic intent detection
        intent = "unknown"
        transcript_lower = transcript.lower()
        words = transcript.split()
        
        # More accurate intent detection
        # Question - check for question words or question mark
        if (transcript.strip().endswith("?") or 
            any(transcript_lower.startswith(qw + " ") for qw in ["what", "when", "where", "who", "how", "why", "which", "whose", "is", "are", "do", "does", "did", "can", "could", "would", "will", "should"]) or
            any(phrase in transcript_lower for phrase in ["what's", "where's", "who's", "how's", "when's", "why's"])):
            intent = "question"
        # Request - polite commands
        elif any(phrase in transcript_lower for phrase in ["please", "can you", "could you", "would you", "will you", "may i", "might i", "i need", "i'd like", "i would like"]):
            intent = "request"
        # Gratitude
        elif any(word in transcript_lower for word in ["thank", "thanks", "appreciate", "grateful"]):
            intent = "gratitude"
        # Greeting
        elif any(word in transcript_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"]):
            intent = "greeting"
        # Farewell
        elif any(word in transcript_lower for word in ["bye", "goodbye", "see you", "farewell", "take care", "later", "night"]):
            intent = "farewell"
        # Command - imperative sentences
        elif (transcript.strip().endswith("!") or
              (len(words) > 0 and words[0].lower() in ["do", "don't", "stop", "start", "go", "come", "wait", "listen", "look", "watch", "take", "put", "give", "get", "make", "let", "keep", "turn", "move", "run", "walk", "sit", "stand"])):
            intent = "command"
        # Default to statement
        else:
            intent = "statement"
        
        # Enhanced keyword extraction
        keywords = []
        
        # Common words to exclude (stopwords)
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'shall', 'can', 'need',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under',
            'again', 'further', 'then', 'once',
            'here', 'there', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            's', 't', 'd', 'll', 've', 'm', 're',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'this', 'that', 'these', 'those',
            'am', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t', 'haven\'t',
            'hadn\'t', 'doesn\'t', 'don\'t', 'didn\'t', 'won\'t', 'wouldn\'t', 'couldn\'t',
            'shouldn\'t', 'mightn\'t', 'mustn\'t', 'needn\'t', 'shan\'t', 'can\'t',
            'it\'s', 'let\'s', 'that\'s', 'who\'s', 'what\'s', 'here\'s', 'there\'s',
            'when\'s', 'where\'s', 'why\'s', 'how\'s',
            'as', 'if', 'because', 'while', 'until', 'although', 'since', 'unless',
            'and', 'but', 'or', 'yet', 'nor', 'not',
            'just', 'like', 'yeah', 'yes', 'no', 'ok', 'okay', 'oh', 'ah', 'um', 'uh',
            'well', 'really', 'actually', 'basically', 'seriously', 'honestly',
            'mean', 'know', 'think', 'want', 'get', 'got', 'go', 'going', 'went',
            'come', 'came', 'say', 'said', 'tell', 'told', 'see', 'saw', 'look', 'looked'
        }
        
        # First, extract all words and clean them
        all_words = re.findall(r'\b[a-zA-Z0-9]+(?:\'[a-zA-Z]+)?\b', transcript)
        
        for word in all_words:
            word_lower = word.lower()
            
            # Skip stopwords and very short words
            if word_lower in stopwords or len(word) < 2:
                continue
            
            # Add the word (keep original capitalization for proper nouns)
            if word[0].isupper() and word not in keywords:
                keywords.append(word)
            elif word_lower not in [k.lower() for k in keywords]:
                keywords.append(word_lower)
        
        # Extract all numbers (including times, dates, etc.)
        numbers = re.findall(r'\b\d+(?::\d+)?(?:\s*[AaPp][Mm])?\b', transcript)
        keywords.extend(numbers)
        
        # Extract compound terms (e.g., "San Francisco", "project deadline")
        # Two-word phrases where at least one word is capitalized
        compound_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Both capitalized
            r'\b[A-Z][a-z]+\s+[a-z]+\b',       # First capitalized
            r'\b[a-z]+\s+[A-Z][a-z]+\b',       # Second capitalized
        ]
        
        for pattern in compound_patterns:
            compounds = re.findall(pattern, transcript)
            for compound in compounds:
                # Check if it's meaningful (not just articles/prepositions)
                words_in_compound = compound.split()
                if all(w.lower() not in stopwords for w in words_in_compound):
                    keywords.append(compound)
        
        # Extract special patterns
        # Times with AM/PM
        times = re.findall(r'\b\d{1,2}(?::\d{2})?\s*[AaPp][Mm]\b', transcript, re.IGNORECASE)
        keywords.extend(times)
        
        # Dates in various formats
        date_patterns = [
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\b',
            r'\b\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?\b',
            r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
            r'\b(?:tomorrow|yesterday|today)\b',
            r'\b(?:next|last|this)\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, transcript, re.IGNORECASE)
            keywords.extend(dates)
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', transcript)
        keywords.extend(emails)
        
        # Extract URLs
        urls = re.findall(r'https?://[^\s]+', transcript)
        keywords.extend(urls)
        
        # Extract phone numbers
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', transcript)
        keywords.extend(phones)
        
        # Important: extract action words (verbs) that are meaningful
        important_verbs = {
            'call', 'email', 'send', 'schedule', 'meet', 'discuss', 'review', 'prepare',
            'submit', 'approve', 'cancel', 'postpone', 'remind', 'check', 'update', 'create',
            'delete', 'edit', 'write', 'read', 'sign', 'pay', 'buy', 'sell', 'book', 'reserve',
            'plan', 'organize', 'arrange', 'confirm', 'notify', 'inform', 'request', 'order',
            'deliver', 'ship', 'receive', 'process', 'complete', 'finish', 'start', 'begin',
            'continue', 'stop', 'pause', 'resume', 'fix', 'repair', 'install', 'configure',
            'build', 'design', 'develop', 'test', 'debug', 'deploy', 'launch', 'release'
        }
        
        for verb in important_verbs:
            # Check for different forms
            patterns = [verb, verb + 'ing', verb + 'ed', verb + 's']
            for pattern in patterns:
                if pattern in transcript_lower and pattern not in [k.lower() for k in keywords]:
                    keywords.append(pattern)
                    break
        
        # Clean up and deduplicate keywords
        cleaned_keywords = []
        seen_lower = set()
        
        for keyword in keywords:
            # Clean the keyword
            keyword_clean = keyword.strip()
            if not keyword_clean:
                continue
                
            # Normalize for deduplication
            keyword_lower = keyword_clean.lower()
            
            # Skip if we've seen it
            if keyword_lower in seen_lower:
                continue
            
            seen_lower.add(keyword_lower)
            cleaned_keywords.append(keyword_clean)
        
        # Sort keywords by importance (proper nouns first, then numbers, then others)
        def keyword_priority(kw):
            if kw[0].isupper():  # Proper noun
                return 0
            elif re.match(r'^\d', kw):  # Starts with number
                return 1
            elif any(c.isdigit() for c in kw):  # Contains number
                return 2
            else:
                return 3
        
        cleaned_keywords.sort(key=keyword_priority)
        
        # Log what we extracted
        logger.info(f"Keyword extraction: intent={intent}, keywords={cleaned_keywords[:15]}")
        
        return {
            "transcript": transcript,
            "timestamp_ms": timestamp_ms or int(time.time() * 1000),
            "intent": intent,
            "entities": cleaned_keywords[:15]  # Return up to 15 keywords
        }


class MockExtractor(JSONExtractor):
    """Mock extractor for testing without loading models"""
    
    def __init__(self):
        logger.info("Using mock JSON extractor")
        # Create an instance of GemmaExtractor just to use its fallback method
        self._fallback_extractor = type('FallbackExtractor', (), {
            '_fallback_extraction': GemmaExtractor._fallback_extraction
        })()
    
    def extract(self, transcript: str, timestamp_ms: int = None) -> Dict:
        """Use the same fallback extraction as GemmaExtractor"""
        # Simulate minimal processing time
        time.sleep(0.05)
        
        # Use the sophisticated fallback extraction
        return self._fallback_extractor._fallback_extraction(transcript, timestamp_ms)


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
