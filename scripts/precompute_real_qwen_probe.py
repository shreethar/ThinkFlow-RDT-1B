import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch.nn.functional as F
import os

def main():
    torch.backends.cudnn.enabled = False
    model_id = "shreethar/stage1_unsloth"
    print(f"Loading processor and model from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    vlm = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa"
    )
    vlm.eval()
    vlm.requires_grad_(False)

    print("Loading images...")
    primary_img_path = "tests/primary.jpg"
    wrist_img_path = "tests/wrist.jpg"
    
    if not os.path.exists(primary_img_path) or not os.path.exists(wrist_img_path):
        raise FileNotFoundError(f"Ensure {primary_img_path} and {wrist_img_path} exist.")

    primary_img = Image.open(primary_img_path)
    wrist_img = Image.open(wrist_img_path)

    instruction = "move the gripper to the corn"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": primary_img},
                {"type": "image", "image": wrist_img},
                {"type": "text", "text": instruction},
            ]
        }
    ]

    print("Preparing inputs...")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[primary_img, wrist_img],
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    print("Running forward pass...")
    with torch.no_grad():
        output = vlm(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    # Extract layer 24 hidden states (shape: [1, seq_len, 2560])
    hidden = output.hidden_states[24][0]
    
    # Locate image and language tokens
    image_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    im_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    input_ids = inputs["input_ids"][0]
    is_image_token = (input_ids == image_pad_token_id)
    
    grid_thw = inputs["image_grid_thw"] # [2, 3] (T, H, W for each image)
    N1 = grid_thw[0].prod().item()
    N2 = grid_thw[1].prod().item()
    
    # Due to 2x2 spatial patch merging in the Qwen VL model,
    # the number of visual tokens in the LLM sequence is N // 4.
    M1 = N1 // 4
    M2 = N2 // 4
    
    image_indices = torch.where(is_image_token)[0]
    if len(image_indices) != M1 + M2:
        raise ValueError(f"Found {len(image_indices)} image padding tokens, but expected {M1} + {M2} = {M1+M2}")
        
    cam1_indices = image_indices[:M1]
    cam2_indices = image_indices[M1 : M1 + M2]
    
    cam1_tokens = hidden[cam1_indices] # [M1, 2560]
    cam2_tokens = hidden[cam2_indices] # [M2, 2560]
    
    # Reshape and adaptive pool each camera to 8x8 = 64 tokens.
    # The feature grid size is halved in both dimensions: (H // 2, W // 2)
    H1, W1 = grid_thw[0, 1].item() // 2, grid_thw[0, 2].item() // 2
    cam1_tokens_reshaped = cam1_tokens.view(1, H1, W1, -1).permute(0, 3, 1, 2)
    cam1_pooled = F.adaptive_avg_pool2d(cam1_tokens_reshaped.float(), (8, 8))
    cam1_pooled = cam1_pooled.permute(0, 2, 3, 1).view(64, -1).to(hidden.dtype)
    
    H2, W2 = grid_thw[1, 1].item() // 2, grid_thw[1, 2].item() // 2
    cam2_tokens_reshaped = cam2_tokens.view(1, H2, W2, -1).permute(0, 3, 1, 2)
    cam2_pooled = F.adaptive_avg_pool2d(cam2_tokens_reshaped.float(), (8, 8))
    cam2_pooled = cam2_pooled.permute(0, 2, 3, 1).view(64, -1).to(hidden.dtype)
    
    img_tokens = torch.cat([cam1_pooled, cam2_pooled], dim=0) # [128, 2560]
    
    # Locate instruction text tokens
    vision_end_indices = torch.where(input_ids == vision_end_token_id)[0]
    if len(vision_end_indices) < 2:
         raise ValueError("Could not locate both images in the prompt.")
    instruction_start_idx = vision_end_indices[-1].item() + 1
    
    im_end_indices = torch.where(input_ids == im_end_token_id)[0]
    valid_im_ends = im_end_indices[im_end_indices > instruction_start_idx]
    if len(valid_im_ends) == 0:
         raise ValueError("Could not find the end of instruction user block.")
    instruction_end_idx = valid_im_ends[0].item()
    
    lang_indices = torch.arange(instruction_start_idx, instruction_end_idx)
    lang_tokens = hidden[lang_indices] # [L, 2560]
    
    print(f"Extracted lang_tokens of shape: {list(lang_tokens.shape)}")
    print(f"Extracted img_tokens of shape: {list(img_tokens.shape)}")
    
    # Save the features
    output_path = "one_real_qwen_probe.pt"
    torch.save(
        {
            "lang_tokens": lang_tokens.cpu(),
            "img_tokens": img_tokens.cpu(),
            "state": torch.zeros(7, dtype=torch.float32),
            "actions": torch.zeros(64, 7, dtype=torch.float32),
            "ctrl_freq": 10.0,
        },
        output_path
    )
    print(f"Successfully saved probe features to {output_path}")

if __name__ == "__main__":
    main()
