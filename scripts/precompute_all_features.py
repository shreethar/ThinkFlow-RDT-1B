#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText, 
    T5EncoderModel, 
    T5Tokenizer,
    SiglipVisionModel,
    SiglipImageProcessor
)
from PIL import Image

from thinkflow_rdt.config import load_config

# Import dataset and collation from online script
from train_b0_online import DroidOnlineDataset, droid_online_collate_fn, find_subsequence

def extract_qwen_kv(batch, processor, vlm, device="cuda"):
    instructions = batch["instructions"]
    primary_images_np = batch["primary_images"]
    wrist_images_np = batch["wrist_images"]
    
    images_list = []
    texts_list = []
    
    for p_img_np, w_img_np, inst in zip(primary_images_np, wrist_images_np, instructions):
        primary_img = Image.fromarray(p_img_np)
        wrist_img = Image.fromarray(w_img_np)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": primary_img},
                    {"type": "image", "image": wrist_img},
                    {"type": "text", "text": inst},
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images_list.append([primary_img, wrist_img])
        texts_list.append(text)
        
    inputs = processor(
        text=texts_list,
        images=images_list,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    tokenizer = processor.tokenizer
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    out = vlm.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=None,
        top_p=None,
        use_cache=True,
        return_dict_in_generate=True,
        output_hidden_states=False,
        eos_token_id=im_end_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    generated_ids = out.sequences
    past_key_values = out.past_key_values
    
    if isinstance(past_key_values, tuple):
        K = past_key_values[7][0]
        V = past_key_values[7][1]
    elif hasattr(past_key_values, "key_cache"):
        K = past_key_values.key_cache[7]
        V = past_key_values.value_cache[7]
    else:
        K = past_key_values.layers[7].keys
        V = past_key_values.layers[7].values
        
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    
    qwen_kv_list = []
    
    B = generated_ids.shape[0]
    for b in range(B):
        ids = generated_ids[b]
        
        think_end_pos = find_subsequence(ids, think_end_ids)
        if think_end_pos != -1:
            target_idx = think_end_pos - 1
        else:
            im_end_positions = torch.where(ids == im_end_id)[0]
            if len(im_end_positions) > 0:
                target_idx = im_end_positions[-1].item() - 1
            else:
                target_idx = len(ids) - 1
                
        k_vec = K[b, :, target_idx, :]
        v_vec = V[b, :, target_idx, :]
        
        k_flat = k_vec.reshape(-1)
        v_flat = v_vec.reshape(-1)
        kv_concat = torch.cat([k_flat, v_flat], dim=-1)
        
        qwen_kv_list.append(kv_concat.unsqueeze(0))
        
    qwen_kv = torch.cat(qwen_kv_list, dim=0).to(torch.bfloat16)
    return qwen_kv

def extract_t5_features(batch, tokenizer, encoder, max_lang_tokens=128, device="cuda"):
    instructions = batch["instructions"]
    
    tokenized_res = tokenizer(
        instructions, return_tensors="pt",
        padding="longest",
        truncation=True
    )
    tokens = tokenized_res["input_ids"].to(device)
    attn_mask = tokenized_res["attention_mask"].to(device)
    
    with torch.no_grad():
        text_embeds = encoder(
            input_ids=tokens,
            attention_mask=attn_mask
        )["last_hidden_state"]
        
    B = text_embeds.shape[0]
    lang_tokens_list = []
    lang_mask_list = []
    
    for i in range(B):
        mask = attn_mask[i].bool()
        embed = text_embeds[i][mask]
        
        if embed.shape[0] > max_lang_tokens:
            embed = embed[:max_lang_tokens]
            mask_out = torch.ones(max_lang_tokens, dtype=torch.bool, device=device)
        else:
            pad_len = max_lang_tokens - embed.shape[0]
            embed = torch.cat([embed, torch.zeros(pad_len, embed.shape[1], device=device, dtype=embed.dtype)], dim=0)
            mask_out = torch.cat([torch.ones(embed.shape[0] - pad_len, dtype=torch.bool, device=device), 
                                  torch.zeros(pad_len, dtype=torch.bool, device=device)], dim=0)
            
        lang_tokens_list.append(embed.unsqueeze(0))
        lang_mask_list.append(mask_out.unsqueeze(0))
        
    lang_tokens = torch.cat(lang_tokens_list, dim=0).to(torch.bfloat16)
    lang_mask = torch.cat(lang_mask_list, dim=0)
    return lang_tokens, lang_mask

def extract_siglip_features(batch, processor, encoder, max_img_tokens=1458, device="cuda"):
    primary_images_np = batch["primary_images"]
    wrist_images_np = batch["wrist_images"]
    
    images = []
    for p_img_np, w_img_np in zip(primary_images_np, wrist_images_np):
        images.append(Image.fromarray(p_img_np))
        images.append(Image.fromarray(w_img_np))
        
    inputs = processor(images=images, return_tensors="pt").to(device)
    
    with torch.no_grad():
        vision_outputs = encoder(**inputs)
        image_features = vision_outputs.last_hidden_state
        
    B = len(primary_images_np)
    img_tokens_list = []
    img_mask_list = []
    
    for i in range(B):
        primary_feat = image_features[2 * i]
        wrist_feat = image_features[2 * i + 1]
        
        concat_feat = torch.cat([primary_feat, wrist_feat], dim=0)
        
        if concat_feat.shape[0] > max_img_tokens:
            concat_feat = concat_feat[:max_img_tokens]
            mask_out = torch.ones(max_img_tokens, dtype=torch.bool, device=device)
        else:
            pad_len = max_img_tokens - concat_feat.shape[0]
            concat_feat = torch.cat([concat_feat, torch.zeros(pad_len, concat_feat.shape[1], device=device, dtype=concat_feat.dtype)], dim=0)
            mask_out = torch.cat([torch.ones(concat_feat.shape[0] - pad_len, dtype=torch.bool, device=device),
                                  torch.zeros(pad_len, dtype=torch.bool, device=device)], dim=0)
                                  
        img_tokens_list.append(concat_feat.unsqueeze(0))
        img_mask_list.append(mask_out.unsqueeze(0))
        
    img_tokens = torch.cat(img_tokens_list, dim=0).to(torch.bfloat16)
    img_mask = torch.cat(img_mask_list, dim=0)
    return img_tokens, img_mask


def main():
    parser = argparse.ArgumentParser(description="Precompute Qwen KV, T5, and SigLIP features simultaneously.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-dir", default="dataset/droid_100/1.0.0")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", default="dataset/precomputed_features.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset from {args.dataset_dir}...")
    dataset = DroidOnlineDataset(args.dataset_dir, num_episodes=args.num_episodes, pred_horizon=cfg.model.pred_horizon)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=droid_online_collate_fn,
        drop_last=False
    )

    print("Loading VLM (Qwen)...")
    vlm_model_id = "shreethar/stage1_unsloth"
    qwen_processor = AutoProcessor.from_pretrained(vlm_model_id)
    qwen_processor.tokenizer.padding_side = "left"
    qwen_vlm = AutoModelForImageTextToText.from_pretrained(
        vlm_model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    qwen_vlm.eval()
    qwen_vlm.requires_grad_(False)

    print("Loading T5-XXL...")
    t5_model_id = "/home/ubuntu/RoboticsDiffusionTransformer/google/t5-v1_1-xxl"
    if not os.path.exists(t5_model_id):
        t5_model_id = "google/t5-v1_1-xxl"
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_id)
    t5_encoder = T5EncoderModel.from_pretrained(t5_model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    t5_encoder.eval()
    t5_encoder.requires_grad_(False)

    print("Loading SigLIP...")
    siglip_model_id = "/home/ubuntu/RoboticsDiffusionTransformer/google/siglip-so400m-patch14-384"
    if not os.path.exists(siglip_model_id):
        siglip_model_id = "google/siglip-so400m-patch14-384"
    siglip_processor = SiglipImageProcessor.from_pretrained(siglip_model_id)
    siglip_encoder = SiglipVisionModel.from_pretrained(siglip_model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    siglip_encoder.eval()
    siglip_encoder.requires_grad_(False)

    all_qwen_kv = []
    all_lang_tokens = []
    all_lang_mask = []
    all_img_tokens = []
    all_img_mask = []

    print(f"Starting precomputation for {len(dataset)} windows...")
    
    # RDT limits
    max_lang_tokens = cfg.model.max_lang_tokens
    max_img_tokens = cfg.model.image_tokens

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # Extract features
        qwen_kv = extract_qwen_kv(batch, qwen_processor, qwen_vlm, device=device)
        lang_tokens, lang_mask = extract_t5_features(batch, t5_tokenizer, t5_encoder, max_lang_tokens, device=device)
        img_tokens, img_mask = extract_siglip_features(batch, siglip_processor, siglip_encoder, max_img_tokens, device=device)
        
        # Move to CPU
        all_qwen_kv.append(qwen_kv.cpu())
        all_lang_tokens.append(lang_tokens.cpu())
        all_lang_mask.append(lang_mask.cpu())
        all_img_tokens.append(img_tokens.cpu())
        all_img_mask.append(img_mask.cpu())
        
    print("Concatenating all features...")
    final_dict = {
        "qwen_kv": torch.cat(all_qwen_kv, dim=0),
        "lang_tokens": torch.cat(all_lang_tokens, dim=0),
        "lang_mask": torch.cat(all_lang_mask, dim=0),
        "img_tokens": torch.cat(all_img_tokens, dim=0),
        "img_mask": torch.cat(all_img_mask, dim=0),
    }
    
    print(f"Final qwen_kv shape: {final_dict['qwen_kv'].shape}")
    print(f"Final lang_tokens shape: {final_dict['lang_tokens'].shape}")
    print(f"Final img_tokens shape: {final_dict['img_tokens'].shape}")
    
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving to {args.output}...")
    torch.save(final_dict, args.output)
    print("Precomputation finished successfully!")

if __name__ == "__main__":
    main()
