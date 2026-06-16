#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from thinkflow_rdt.checkpoint import load_trainable_artifact
from thinkflow_rdt.config import load_config
from thinkflow_rdt.data import CachedFeatureDataset, RDTBatchCollator
from thinkflow_rdt.model import SFTConditionedRDT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="sampled_actions.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = args.manifest or cfg.data.val_manifest
    dataset = CachedFeatureDataset(manifest)
    sample = dataset[args.index]
    collator = RDTBatchCollator(
        max_lang_tokens=cfg.model.max_lang_tokens,
        image_tokens=cfg.model.image_tokens,
        pred_horizon=cfg.model.pred_horizon,
        feature_dim=cfg.model.qwen_hidden_size,
        state_dim=cfg.model.state_dim,
        action_dim=cfg.model.action_dim,
    )
    batch = collator([sample])

    model = SFTConditionedRDT(cfg, load_pretrained=True)
    load_trainable_artifact(model, args.artifact, trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    batch = {key: value.to(device) for key, value in batch.items()}
    prediction = model.sample_actions(batch).float().cpu()
    target = batch["actions"].float().cpu()
    output = Path(args.output)
    torch.save(
        {
            "prediction": prediction,
            "target": target,
            "action_time_mask": batch["action_time_mask"].cpu(),
            "source": sample.get("_path"),
        },
        output,
    )
    print(f"Saved sampled actions to {output.resolve()}")
    print("prediction shape:", tuple(prediction.shape))
    print("prediction range:", float(prediction.min()), float(prediction.max()))


if __name__ == "__main__":
    main()
