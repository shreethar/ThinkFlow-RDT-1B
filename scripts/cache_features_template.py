#!/usr/bin/env python
"""
Integration template for your existing post-SFT Qwen feature extractor.

This file intentionally does not guess your Qwen3.5/Unsloth loading code or your
canonical dataset class. Replace `build_dataset()` and `extract_features()` with
the implementations you already validated during SFT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def build_dataset(split: str) -> Dataset:
    raise NotImplementedError(
        "Return your stable FixedIndexDataset here. Each item should provide "
        "instruction, current images, proprioception, future actions and ctrl_freq."
    )


@torch.inference_mode()
def extract_features(sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
      lang_tokens: [L_text, 2560], layer-24 instruction-token hidden states
      img_tokens:  [128, 2560], pooled layer-24 visual hidden states

    Reuse your existing batch-friendly image token masks. Pool each camera's
    visual grid to 8x8, then concatenate the two cameras to get 128 tokens.
    Do not include ground-truth action text in the VLM prompt.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=64)
    args = parser.parse_args()

    dataset = build_dataset(args.split)
    output = Path(args.output) / args.split
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index in range(len(dataset)):
            sample = dataset[index]
            lang_tokens, img_tokens = extract_features(sample)
            actions = torch.as_tensor(sample["actions"], dtype=torch.float32)
            actions = actions[: args.horizon]
            record = {
                "lang_tokens": lang_tokens.cpu().to(torch.float16),
                "img_tokens": img_tokens.cpu().to(torch.float16),
                "state": torch.as_tensor(sample["proprio"], dtype=torch.float32),
                "actions": actions.cpu(),
                "ctrl_freq": float(sample.get("ctrl_freq", 20.0)),
                "episode_id": sample.get("episode_id"),
                "timestep": sample.get("timestep"),
            }
            path = output / f"sample_{index:09d}.pt"
            torch.save(record, path)
            manifest.write(json.dumps({"path": path.name}) + "\n")

    print(f"Saved {len(dataset)} samples to {output}")


if __name__ == "__main__":
    main()
