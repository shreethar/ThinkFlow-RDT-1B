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

out = vlm.generate(
    **inputs,
    max_new_tokens=10,
    return_dict_in_generate=True,
    output_hidden_states=True
)

prompt_hidden_states = out.hidden_states[0] # Tuple of layers for the prompt
input_embeddings = prompt_hidden_states[0] # Layer 0 (input to first block)

print("input_embeddings shape:", input_embeddings.shape)

# Let's see if Qwen has an inputs builder that merges tokens
print("Testing dynamic mapping logic...")
ids = inputs["input_ids"][0]
emb = input_embeddings[0]

emb_is_img = []
emb_is_lang = []

total_img_tokens_in_ids = (ids == 151655).sum().item()
total_img_tokens_in_emb = emb.shape[0] - (len(ids) - total_img_tokens_in_ids)
ratio = total_img_tokens_in_ids // max(1, total_img_tokens_in_emb)

i = 0
while i < len(ids):
    if ids[i] == 151655:
        block_len = 0
        while i < len(ids) and ids[i] == 151655:
            block_len += 1
            i += 1
        num_compressed = block_len // ratio
        emb_is_img.extend([True] * num_compressed)
        emb_is_lang.extend([False] * num_compressed)
    else:
        is_pad_token = (ids[i] == processor.tokenizer.pad_token_id)
        emb_is_img.append(False)
        emb_is_lang.append(not is_pad_token)
        i += 1

print(f"emb length: {emb.shape[0]}")
print(f"emb_is_img length: {len(emb_is_img)}")

emb_is_img = torch.tensor(emb_is_img, dtype=torch.bool, device=emb.device)
emb_is_lang = torch.tensor(emb_is_lang, dtype=torch.bool, device=emb.device)

img_emb = emb[emb_is_img]
lang_emb = emb[emb_is_lang]

print("Extracted img_emb shape:", img_emb.shape)
print("Extracted lang_emb shape:", lang_emb.shape)
