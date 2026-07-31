#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from thinkflow_rdt.adapters.action_stats import compute_action_quantile_stats
from thinkflow_rdt.adapters.libero import iter_libero_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LIBERO HDF5 and compute RDT action statistics.")
    parser.add_argument("--dataset-id", default="libero_spatial")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Usually <root>/<dataset-id>/audit.json")
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()

    parts = []
    episodes = steps = 0
    for episode in iter_libero_episodes(args.data_dir):
        parts.append(episode.actions)
        episodes += 1
        steps += len(episode.actions)
        if args.max_episodes is not None and episodes >= args.max_episodes:
            break
    if not parts:
        raise RuntimeError("No LIBERO demonstrations were found")
    stats = compute_action_quantile_stats(np.concatenate(parts, axis=0))
    payload = {
        "dataset_id": args.dataset_id,
        "episodes": episodes,
        "steps": steps,
        "conventions": {
            "state": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_closed"],
            "action": ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper_closed"],
            "gripper_open": 0,
            "gripper_closed": 1,
        },
        "action_normalization": stats.to_audit_block(source={"dataset_id": args.dataset_id, "steps": steps}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} from {episodes} episodes / {steps} actions")


if __name__ == "__main__":
    main()
