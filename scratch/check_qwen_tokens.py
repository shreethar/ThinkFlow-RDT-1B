from transformers import AutoProcessor
import sys

processor = AutoProcessor.from_pretrained("shreethar/stage1_unsloth")

text = "</think>"
tokens = processor.tokenizer.encode(text, add_special_tokens=False)
print(f"Token IDs for '</think>': {tokens}")

for t in tokens:
    print(f"  {t}: {processor.tokenizer.decode(t)}")

# Try with a chat template
messages = [
    {"role": "user", "content": "Hello!"}
]
out = processor.apply_chat_template(messages, add_generation_prompt=True)
print("\nChat template output:")
print(out)
