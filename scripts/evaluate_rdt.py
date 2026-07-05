#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import SFTConditionedRDT
from train_b0_online import DroidOnlineDataset, droid_online_collate_fn

def main():
    parser = argparse.ArgumentParser(description="Evaluate RDT trajectory predictions.")
    parser.add_argument("--config", default="configs/b0_rdt1b_lora.yaml")
    parser.add_argument("--dataset-dir", default="dataset/droid_100/1.0.0")
    parser.add_argument("--precomputed-path", required=True, help="Path to precomputed Qwen KV tensor.")
    parser.add_argument("--model-path", default="outputs/b0_sft_rdt1b_lora/model.pt", help="Path to trained model weights.")
    parser.add_argument("--output", default="outputs/trajectory_comparison.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    # 1. Load Dataset and take 1 sample
    print("Loading dataset...")
    dataset = DroidOnlineDataset(args.dataset_dir, num_episodes=1, pred_horizon=cfg.model.pred_horizon, precomputed_path=args.precomputed_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=droid_online_collate_fn)
    
    raw_batch = next(iter(dataloader))
    
    # Prepare batch
    batch = {
        "state": raw_batch["state"].to(device),
        "actions": raw_batch["actions"].to(device),
        "action_time_mask": raw_batch["action_time_mask"].to(device),
        "action_dim_mask": raw_batch["action_dim_mask"].to(device),
        "ctrl_freq": raw_batch["ctrl_freq"].to(device),
        "lang_tokens": torch.zeros(1, 1, 2560, device=device),
        "lang_mask": torch.ones(1, 1, dtype=torch.bool, device=device),
        "img_tokens": torch.zeros(1, 1, 2560, device=device),
        "img_mask": torch.ones(1, 1, dtype=torch.bool, device=device),
        "qwen_kv": raw_batch["qwen_kv"].to(device),
    }

    # 2. Load Baseline Model
    print("Loading baseline RDT model...")
    model = SFTConditionedRDT(cfg, load_pretrained=True)
    model.to(device)
    model.eval()

    print("Generating baseline trajectory...")
    # Use manual seed for deterministic diffusion sampling
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        
    baseline_traj = model.sample_actions(batch)[0].float().cpu().numpy() # [64, 7]

    # 3. Load Trained Model
    print(f"Loading trained weights from {args.model_path}...")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Trained model not found at {args.model_path}. Did you finish training?")
    
    trained_weights = torch.load(args.model_path, map_location=device)
    model.load_state_dict(trained_weights, strict=False)
    
    print("Generating trained trajectory...")
    torch.manual_seed(cfg.seed) # Use same seed so noise schedule is identical
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        
    trained_traj = model.sample_actions(batch)[0].float().cpu().numpy() # [64, 7]
    gt_traj = batch["actions"][0].float().cpu().numpy() # [64, 7]

    # 4. Plot Comparison
    print(f"Plotting results to {args.output}...")
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    coords = ["X (Cartesian)", "Y (Cartesian)", "Z (Cartesian)"]
    
    for i in range(3):
        axs[i].plot(gt_traj[:, i], label="Ground Truth", linestyle="--", color="black", linewidth=2)
        axs[i].plot(baseline_traj[:, i], label="Baseline (Untrained LoRA)", alpha=0.7, color="red")
        axs[i].plot(trained_traj[:, i], label="Trained (LoRA)", alpha=0.9, color="blue", linewidth=2)
        axs[i].set_title(f"Trajectory Dimension: {coords[i]}")
        axs[i].set_xlabel("Timestep")
        axs[i].set_ylabel("Position")
        axs[i].grid(True, alpha=0.3)
        axs[i].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=300)
    print("Evaluation finished!")

if __name__ == "__main__":
    main()
