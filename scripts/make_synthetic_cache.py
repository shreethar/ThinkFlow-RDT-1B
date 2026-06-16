#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def write_split(
    output: Path,
    split: str,
    count: int,
    feature_dim: int,
    lang_tokens: int,
    image_tokens: int,
    horizon: int,
    action_dim: int,
) -> None:
    split_dir = output / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest = split_dir / "manifest.jsonl"
    generator = torch.Generator().manual_seed(123 if split == "train" else 456)
    with manifest.open("w", encoding="utf-8") as handle:
        for index in range(count):
            lang = torch.randn(lang_tokens, feature_dim, generator=generator)
            image = torch.randn(image_tokens, feature_dim, generator=generator)
            state = torch.randn(action_dim, generator=generator) * 0.1
            # Learnable deterministic target tied to state and condition means.
            condition = 0.05 * lang.mean() + 0.05 * image.mean()
            time = torch.linspace(0, 1, horizon).unsqueeze(1)
            actions = state.unsqueeze(0) * (1 - time) + condition * time
            actions = actions + 0.01 * torch.randn(
                horizon, action_dim, generator=generator
            )
            path = split_dir / f"sample_{index:06d}.pt"
            torch.save(
                {
                    "lang_tokens": lang.to(torch.float16),
                    "img_tokens": image.to(torch.float16),
                    "state": state,
                    "actions": actions,
                    "ctrl_freq": 20.0,
                },
                path,
            )
            handle.write(json.dumps({"path": path.name}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cache_synthetic")
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--val-count", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--lang-tokens", type=int, default=12)
    parser.add_argument("--image-tokens", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=7)
    args = parser.parse_args()
    output = Path(args.output)
    write_split(
        output,
        "train",
        args.train_count,
        args.feature_dim,
        args.lang_tokens,
        args.image_tokens,
        args.horizon,
        args.action_dim,
    )
    write_split(
        output,
        "val",
        args.val_count,
        args.feature_dim,
        args.lang_tokens,
        args.image_tokens,
        args.horizon,
        args.action_dim,
    )
    print(f"Wrote synthetic cache to {output.resolve()}")


if __name__ == "__main__":
    main()
