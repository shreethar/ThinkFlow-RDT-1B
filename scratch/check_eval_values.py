#!/usr/bin/env python
import torch
import numpy as np
from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import SFTConditionedRDT
from train_b0_online import DroidOnlineDataset, droid_online_collate_fn
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("configs/b0_rdt1b_lora.yaml")
    
    dataset = DroidOnlineDataset("dataset/droid_100/1.0.0", num_episodes=1, pred_horizon=64, precomputed_path="dataset/droid_100_qwen_features.pt")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=droid_online_collate_fn)
    raw_batch = next(iter(dataloader))
    
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

    model = SFTConditionedRDT(cfg, load_pretrained=True)
    model.to(device)
    model.eval()

    print("Generating baseline trajectory...")
    torch.manual_seed(42)
    baseline_traj = model.sample_actions(batch)[0].float().cpu()
    print("Baseline Trajectory:")
    print(baseline_traj)
    print(f"Max: {baseline_traj.max()}, Min: {baseline_traj.min()}, Mean: {baseline_traj.mean()}")

if __name__ == "__main__":
    main()
