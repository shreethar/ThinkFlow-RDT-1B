#!/usr/bin/env python3
"""Update the demo's discrete initial-state labels from exact singleton rollouts.

Missing task/state pairs are written as ``null`` so the live demo can distinguish
an unfinished evaluation (displayed as 9) from an evaluated failure (0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCES = {
    ("libero_spatial", "b0"): "libero_spatial_b0/episodes.jsonl",
    ("libero_spatial", "b2"): "libero_spatial_b2/episodes.jsonl",
    ("libero_spatial", "b3"): "libero_spatial_b3/episodes.jsonl",
    ("libero_10", "b0"): "libero_10_b0/episodes.jsonl",
    ("libero_10", "b2"): "libero_10_b2/episodes.jsonl",
}


def load_grid(path: Path) -> tuple[dict[str, list[bool | None]], int]:
    rows = (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.is_file()
        else []
    )
    by_key: dict[tuple[int, int], bool] = {}
    for row in rows:
        key = (int(row["task_id"]), int(row["init_state_index"]))
        if key in by_key:
            raise ValueError(f"Duplicate task/state pair {key} in {path}")
        by_key[key] = bool(row["success"])
    expected = {(task, state) for task in range(10) for state in range(5)}
    extra = sorted(by_key.keys() - expected)
    if extra:
        raise ValueError(f"Unexpected task/state pairs in {path}: {extra}")
    grid = {
        str(task): [by_key.get((task, state)) for state in range(5)]
        for task in range(10)
    }
    return grid, len(by_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root", type=Path, default=Path("output_2/demo_seed_exact_250")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/libero_rdt_demo_successes.json")
    )
    args = parser.parse_args()

    payload = json.loads(args.config.read_text())
    loaded = {
        key: load_grid(args.results_root / relative)
        for key, relative in SOURCES.items()
    }
    payload["evaluation_contract"].update(
        {
            "initial_states": [0, 1, 2, 3, 4],
            "diffusion_steps": 5,
            "action_chunk": 10,
            "policy_batch_size": 1,
            "spatial_max_steps": 300,
            "long_max_steps": 600,
            "seed": 42,
            "simulator_seed_formula": "seed + task_id",
            "diffusion_seed_formula": "seed + task_id*100000 + init_state_index*1000 + plan_index",
        }
    )
    payload["evaluation_contract"].pop("environment_batch_size", None)
    for (suite, model), relative in SOURCES.items():
        entry = payload["results"][suite][model]
        entry["source"] = str(args.results_root / relative)
        entry["tasks"] = loaded[(suite, model)][0]

    # Atomic replacement prevents the Gradio process from observing half-written JSON.
    temporary = args.config.with_suffix(args.config.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.config)
    for (suite, model), (grid, completed) in loaded.items():
        successes = sum(result is True for states in grid.values() for result in states)
        rate = successes / completed if completed else 0.0
        print(f"{suite}/{model}: {completed}/50 complete, {successes} successes ({rate:.0%})")
    print(f"Updated {args.config}")


if __name__ == "__main__":
    main()
