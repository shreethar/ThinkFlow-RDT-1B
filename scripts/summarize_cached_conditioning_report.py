#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def get_average_loss(model_report: dict[str, Any], variant: str) -> dict[str, float] | None:
    buckets = model_report.get("loss_by_timestep_bucket", {}).get(variant)
    if not buckets:
        return None
    return buckets.get("all_buckets_average")


def format_delta(base: float, other: float) -> str:
    delta = other - base
    pct = 100.0 * delta / max(abs(base), 1e-12)
    sign = "+" if delta >= 0 else ""
    return f"{other:.6f} ({sign}{delta:.6f}, {sign}{pct:.1f}%)"


def print_model_summary(name: str, report: dict[str, Any]) -> None:
    print(f"\n{name}")
    baseline = get_average_loss(report, "baseline")
    if baseline is None:
        print("  no baseline loss found")
        return
    print(
        "  baseline loss: "
        f"{baseline['loss']:.6f} "
        f"xyz={baseline['loss_xyz']:.6f} "
        f"rot={baseline['loss_rot']:.6f} "
        f"grip={baseline['loss_gripper']:.6f}"
    )
    for variant in (
        "zero_qwen",
        "shuffle_qwen",
        "zero_lang",
        "shuffle_lang",
        "zero_image",
        "shuffle_image",
        "zero_state",
        "zero_ctrl_freq",
        "shuffle_all_context",
    ):
        values = get_average_loss(report, variant)
        if values is None:
            continue
        print(f"  {variant:20s} loss={format_delta(baseline['loss'], values['loss'])}")

    sample = report.get("sample_action_metrics", {}).get("baseline")
    if sample:
        metrics = sample["sample_metrics"]
        gripper = sample["gripper"]
        horizon = sample["horizon"]
        print(
            "  sample_actions baseline: "
            f"mse={metrics['loss']:.6f} "
            f"xyz={metrics['loss_xyz']:.6f} "
            f"rot={metrics['loss_rot']:.6f} "
            f"grip_mse={metrics['loss_gripper']:.6f} "
            f"grip_acc={gripper['accuracy']:.3f} "
            f"transition_acc={gripper['transition_accuracy']:.3f} "
            f"transitions={gripper['transition_valid']:.0f}"
        )
        if horizon["mse"]:
            first = horizon["mse"][0]
            short = sum(horizon["mse"][:8]) / max(len(horizon["mse"][:8]), 1)
            tail = sum(horizon["mse"][-8:]) / max(len(horizon["mse"][-8:]), 1)
            print(f"  horizon mse: h0={first:.6f} first8={short:.6f} last8={tail:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize cached conditioning report JSON.")
    parser.add_argument("report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(f"Report: {args.report}")
    print(f"selected_samples: {report.get('selected_samples')}")
    print(f"loss_variants: {', '.join(report.get('loss_variants', []))}")
    print(f"sample_variants: {', '.join(report.get('sample_variants', []))}")
    for name, model_report in report.get("models", {}).items():
        print_model_summary(name, model_report)


if __name__ == "__main__":
    main()
