#!/usr/bin/env python
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

# Import from the existing online script to reuse dataset loading and extraction logic
from train_b0_online import DroidOnlineDataset, droid_online_collate_fn, extract_b0_features

def main():
    parser = argparse.ArgumentParser(description="Precompute Qwen KV features for the DROID dataset.")
    parser.add_argument("--dataset-dir", default="dataset/droid_100/1.0.0")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="dataset/qwen_features.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.dataset_dir}...")
    dataset = DroidOnlineDataset(args.dataset_dir, num_episodes=args.num_episodes, pred_horizon=64)
    
    # DO NOT shuffle so the indices match perfectly
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=droid_online_collate_fn,
        drop_last=False
    )

    # Load VLM
    vlm_model_id = "shreethar/stage1_unsloth"
    print(f"Loading VLM from {vlm_model_id}...")
    processor = AutoProcessor.from_pretrained(vlm_model_id)
    processor.tokenizer.padding_side = "left"
    
    vlm = AutoModelForImageTextToText.from_pretrained(
        vlm_model_id,
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="sdpa"
    )
    vlm.eval()
    vlm.requires_grad_(False)

    all_features = []
    print(f"Starting precomputation for {len(dataset)} windows...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # Extract features (returns [B, 1, 2048] tensor)
        qwen_kv, _ = extract_b0_features(batch, processor, vlm, max_lang_tokens=128, device=device)
        
        # Move to CPU to save RAM before appending
        all_features.append(qwen_kv.cpu())
        
    print("Concatenating all features...")
    final_tensor = torch.cat(all_features, dim=0) # [Total_Windows, 1, 2048]
    
    print(f"Final tensor shape: {final_tensor.shape}")
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving to {args.output}...")
    torch.save(final_tensor, args.output)
    print("Precomputation finished successfully!")

if __name__ == "__main__":
    main()
