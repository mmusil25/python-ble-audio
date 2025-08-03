#!/usr/bin/env python3

from unsloth import FastLanguageModel

print("Loading model...")
model, processor = FastLanguageModel.from_pretrained(
    "unsloth/gemma-3n-e4b-it",
    dtype=None,
    load_in_4bit=True,
)

# Apply chat template first
messages = [{"role": "user", "content": "Extract JSON from: Hello world"}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Chat template output:\n{text}")

# Check if processor has tokenizer attribute
if hasattr(processor, 'tokenizer'):
    print(f"\nProcessor has tokenizer: {type(processor.tokenizer)}")
    try:
        inputs = processor.tokenizer(text, return_tensors="pt")
        print(f"Tokenizer success! Keys: {inputs.keys()}")
    except Exception as e:
        print(f"Tokenizer error: {e}")

# Try using processor directly with different methods
print("\nTrying processor methods:")

# Method 1: Pass as text kwarg
try:
    inputs = processor(text=text, return_tensors="pt")
    print(f"Method 1 success! Keys: {inputs.keys()}")
except Exception as e:
    print(f"Method 1 error: {e}")

# Method 2: Check processor attributes
print(f"\nProcessor text-related attributes:")
text_attrs = [attr for attr in dir(processor) if 'text' in attr or 'token' in attr]
print(text_attrs[:20])