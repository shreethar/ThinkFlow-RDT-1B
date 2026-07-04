import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("shreethar/stage1_unsloth")
vlm = AutoModelForImageTextToText.from_pretrained(
    "shreethar/stage1_unsloth",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

messages = [
    {"role": "user", "content": "Hello!"}
]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

out = vlm(**inputs, use_cache=True, output_hidden_states=False)

if isinstance(out.past_key_values, tuple):
    print("Tuple format")
    K, V = out.past_key_values[7]
else:
    print("DynamicCache format")
    K = out.past_key_values.key_cache[7]
    V = out.past_key_values.value_cache[7]

print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")
