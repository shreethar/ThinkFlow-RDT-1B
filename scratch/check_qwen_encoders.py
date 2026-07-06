import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

vlm_model_id = "shreethar/stage1_unsloth"
processor = AutoProcessor.from_pretrained(vlm_model_id)

vlm = AutoModelForImageTextToText.from_pretrained(
    vlm_model_id,
    device_map="cpu",
    torch_dtype=torch.bfloat16
)

# Dummy image and text
img = Image.new('RGB', (224, 224), color = 'red')
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Hello world!"}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[[img]], return_tensors="pt", padding=True).to("cpu")

print("input_ids shape:", inputs["input_ids"].shape)
if "pixel_values" in inputs:
    print("pixel_values shape:", inputs["pixel_values"].shape)

out = vlm.generate(
    **inputs,
    max_new_tokens=10,
    return_dict_in_generate=True,
    output_hidden_states=True
)

prompt_hidden_states = out.hidden_states[0] # Tuple of layers for the prompt
input_embeddings = prompt_hidden_states[0] # Layer 0 (input to first block)

print("input_embeddings shape:", input_embeddings.shape)

