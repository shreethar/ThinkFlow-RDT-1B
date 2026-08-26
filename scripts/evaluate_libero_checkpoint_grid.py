#!/usr/bin/env python
"""Run a resumable fixed-grid LIBERO evaluation for one saved checkpoint.

The four suite evaluations run sequentially in this process.  That keeps the
VLM, T5, SigLIP, and simulator memory bounded to one suite at a time while the
underlying evaluator batches fixed initial states of each selected task.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one RDT checkpoint with online Qwen KV extraction on a "
            "fixed LIBERO task/initial-state grid."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/libero_b0_native128_full.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--cache-parent", type=Path, default=Path("cache_features_libero_b0_raw_ortho6d"))
    parser.add_argument("--libero-root", type=Path, default=Path("/home/ubuntu/LIBERO"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--env-batch-size", type=int, default=2)
    parser.add_argument("--action-chunk", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--t5-precision", choices=("bf16", "8bit"), default="bf16")
    parser.add_argument("--siglip-model-id", default="/home/ubuntu/models/siglip-so400m-patch14-384")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--report-to", choices=("wandb", "none"), default="wandb")
    parser.add_argument("--wandb-project", default="ThinkLite B0 LIBERO")
    parser.add_argument("--wandb-run-name", default="libero-b0-native128-from-oxe20k-full-v2")
    parser.add_argument("--suite", action="append", choices=DEFAULT_SUITES)
    parser.add_argument(
        "--task-id",
        type=int,
        action="append",
        choices=range(10),
        help=(
            "Task ID to evaluate in every suite; may be repeated. Defaults to "
            "all 10 tasks."
        ),
    )
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation did not create {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    suites = tuple(args.suite) if args.suite else DEFAULT_SUITES
    task_ids = tuple(sorted(set(args.task_id))) if args.task_id else tuple(range(10))
    if not 1 <= args.episodes_per_task <= 50:
        raise ValueError("--episodes-per-task must be in [1, 50]")
    if args.env_batch_size <= 0:
        raise ValueError("--env-batch-size must be positive")
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    for required in ("rdt_full.pt", "interfaces.pt", "metadata.json"):
        if not (args.checkpoint / required).is_file():
            raise FileNotFoundError(args.checkpoint / required)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    suite_summaries: dict[str, dict[str, Any]] = {}
    for suite in suites:
        cache_root = args.cache_parent / suite
        if not (cache_root / "precompute_metadata.json").is_file():
            raise FileNotFoundError(cache_root / "precompute_metadata.json")
        suite_output = args.output_dir / suite
        command = [
            sys.executable,
            "scripts/evaluate_libero_rdt.py",
            "--config", str(args.config),
            "--benchmark", suite,
            "--checkpoint", str(args.checkpoint),
            "--cache-root", str(cache_root),
            "--libero-root", str(args.libero_root),
            "--output-dir", str(suite_output),
            "--episodes-per-task", str(args.episodes_per_task),
            "--env-batch-size", str(args.env_batch_size),
            "--action-chunk", str(args.action_chunk),
            "--max-steps", str(args.max_steps),
            "--seed", str(args.seed),
            "--qwen-max-new-tokens", str(args.qwen_max_new_tokens),
            "--require-qwen-fusion",
            "--t5-precision", args.t5_precision,
            "--siglip-model-id", args.siglip_model_id,
        ]
        for task_id in task_ids:
            command.extend(["--task-id", str(task_id)])
        if args.save_videos:
            command.extend(
                [
                    "--save-videos",
                    "--video-resolution", str(args.video_resolution),
                    "--video-fps", str(args.video_fps),
                ]
            )
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        suite_summaries[suite] = read_summary(suite_output / "summary.json")

    episodes = sum(int(summary["episodes"]) for summary in suite_summaries.values())
    successes = sum(int(summary["successes"]) for summary in suite_summaries.values())
    expected = len(suites) * len(task_ids) * args.episodes_per_task
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": args.checkpoint_step,
        "episodes": episodes,
        "expected_episodes": expected,
        "successes": successes,
        "success_rate": successes / max(episodes, 1),
        "complete": episodes == expected,
        "action_chunk": args.action_chunk,
        "episodes_per_task": args.episodes_per_task,
        "task_ids": list(task_ids),
        "tasks_per_suite": len(task_ids),
        "qwen_mode": "online_per_replan",
        "suites": suite_summaries,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    if args.report_to == "wandb":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_run_name}-rollout-step{args.checkpoint_step}",
            group=args.wandb_run_name,
            job_type="libero-rollout-eval",
            config={
                "checkpoint": result["checkpoint"],
                "checkpoint_step": args.checkpoint_step,
                "episodes_per_task": args.episodes_per_task,
                "task_ids": list(task_ids),
                "tasks_per_suite": len(task_ids),
                "qwen_mode": "online_per_replan",
                "action_chunk": args.action_chunk,
                "suites": list(suites),
            },
        )
        metrics: dict[str, float] = {
            "rollout/episodes": float(episodes),
            "rollout/successes": float(successes),
            "rollout/success_rate": float(result["success_rate"]),
            "rollout/complete": float(result["complete"]),
            "rollout/elapsed_seconds": float(result["elapsed_seconds"]),
        }
        for suite, summary in suite_summaries.items():
            prefix = f"rollout/{suite}"
            metrics[f"{prefix}/episodes"] = float(summary["episodes"])
            metrics[f"{prefix}/successes"] = float(summary["successes"])
            metrics[f"{prefix}/success_rate"] = float(summary["success_rate"])
        run.log(metrics, step=args.checkpoint_step)
        run.summary.update(metrics)
        run.finish()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
