#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thinkflow-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thinkflow-matplotlib")

import numpy as np
import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    SiglipImageProcessor,
    SiglipVisionModel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from precompute_all_features import (  # noqa: E402
    extract_qwen_kv,
    extract_siglip_features,
    standardized_collate_fn,
)
from rollout_libero_rdt import (  # noqa: E402
    install_robosuite_mujoco_compatibility,
    load_cached_language_features,
    load_feature_metadata,
    rollout_sample,
)
from thinkflow_rdt.adapters.action_stats import load_action_stats  # noqa: E402
from thinkflow_rdt.adapters.libero import rdt_action_to_libero  # noqa: E402
from thinkflow_rdt.checkpoint import load_trainable_artifact  # noqa: E402
from thinkflow_rdt.config import load_config  # noqa: E402
from thinkflow_rdt.model import SFTConditionedRDT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure RDT success rate across LIBERO Object episodes.")
    parser.add_argument("--config", default="configs/b0_rdt1b_lora.yaml")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/libero_object_full/checkpoint-1000"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache_features/libero_object/full"))
    parser.add_argument("--action-stats", type=Path, default=Path("dataset/LIBERO/Object/datasets/libero_object/audit.json"))
    parser.add_argument("--libero-root", type=Path, default=Path("/home/ubuntu/LIBERO"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/libero_object_evaluation/checkpoint-1000"))
    parser.add_argument("--episodes-per-task", type=int, default=20, help="LIBERO official default is 20; each task has 50 available.")
    parser.add_argument("--all-episodes", action="store_true", help="Evaluate all 50 states for each of 10 tasks (500 rollouts).")
    parser.add_argument("--env-batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--action-chunk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    return parser.parse_args()


def existing_result_keys(path: Path) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                keys.add((int(row["task_id"]), int(row["init_state_index"])))
    return keys


def write_summary(results_path: Path, summary_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["task_id"])
        block = tasks.setdefault(key, {"task_id": row["task_id"], "instruction": row["instruction"], "episodes": 0, "successes": 0})
        block["episodes"] += 1
        block["successes"] += int(row["success"])
    for block in tasks.values():
        block["success_rate"] = block["successes"] / max(block["episodes"], 1)
    summary = {
        "episodes": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": sum(int(row["success"]) for row in rows) / max(len(rows), 1),
        "tasks": [tasks[key] for key in sorted(tasks, key=int)],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if str(args.libero_root) not in sys.path:
        sys.path.insert(0, str(args.libero_root))
    if not 1 <= args.episodes_per_task <= 50:
        raise ValueError("--episodes-per-task must be in [1, 50]")
    if args.all_episodes:
        args.episodes_per_task = 50
    if args.env_batch_size <= 0:
        raise ValueError("--env-batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RDT evaluation")

    install_robosuite_mujoco_compatibility()
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

    device = torch.device("cuda")
    cfg = load_config(args.config)
    stats = load_action_stats(args.action_stats)
    metadata = load_feature_metadata(args.cache_root)
    qwen_id = metadata.get("qwen_model_id", "shreethar/stage1_unsloth")
    qwen_processor_id = metadata.get("qwen_processor_id", qwen_id)
    siglip_id = metadata.get("siglip_model_id", "google/siglip-so400m-patch14-384")
    benchmark = get_benchmark("libero_object")(0)

    language_by_task = {
        task_id: load_cached_language_features(
            benchmark.get_task(task_id).name,
            cache_root=args.cache_root,
            cfg=cfg,
        )
        for task_id in range(10)
    }

    print("Loading Qwen and SigLIP encoders...")
    qwen_processor = AutoProcessor.from_pretrained(qwen_processor_id)
    qwen_processor.tokenizer.padding_side = "left"
    qwen = AutoModelForImageTextToText.from_pretrained(
        qwen_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation="sdpa",
    ).eval()
    siglip_processor = SiglipImageProcessor.from_pretrained(siglip_id)
    siglip = SiglipVisionModel.from_pretrained(
        siglip_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    ).eval()

    print(f"Loading RDT artifact {args.checkpoint}...")
    model = SFTConditionedRDT(cfg, load_pretrained=True)
    load_trainable_artifact(model, args.checkpoint, trainable=False)
    model.to(device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "episodes.jsonl"
    summary_path = args.output_dir / "summary.json"
    completed = existing_result_keys(results_path)
    total_requested = 10 * args.episodes_per_task
    print(f"Evaluation target: {total_requested} episodes; resuming after {len(completed)} completed")

    with results_path.open("a", encoding="utf-8") as output:
        for task_id in range(10):
            task = benchmark.get_task(task_id)
            all_init_states = torch.load(
                args.libero_root / "libero" / "libero" / "init_files" / task.problem_folder / task.init_states_file,
                map_location="cpu",
                weights_only=False,
            )
            pending = [index for index in range(args.episodes_per_task) if (task_id, index) not in completed]
            for batch_start in range(0, len(pending), args.env_batch_size):
                indices = pending[batch_start : batch_start + args.env_batch_size]
                if not indices:
                    continue
                env_args = {
                    "bddl_file_name": benchmark.get_task_bddl_file_path(task_id),
                    "camera_heights": 128,
                    "camera_widths": 128,
                    "horizon": args.max_steps + 10,
                }
                env = SubprocVectorEnv([lambda env_args=env_args: OffScreenRenderEnv(**env_args) for _ in indices])
                env.reset()
                observations = list(env.set_init_state(all_init_states[indices]))
                for _ in range(5):
                    observations, _, _, _ = env.step(np.zeros((len(indices), 7), dtype=np.float32))
                    observations = list(observations)

                previous: list[dict[str, Any] | None] = [None] * len(indices)
                done = np.zeros(len(indices), dtype=bool)
                success_step = np.full(len(indices), args.max_steps, dtype=np.int32)
                simulator_step = 0
                plan_index = 0
                batch_started = time.perf_counter()
                while simulator_step < args.max_steps and not bool(done.all()):
                    active = np.flatnonzero(~done).tolist()
                    samples = [
                        rollout_sample(
                            observations[index],
                            previous[index],
                            instruction=task.language,
                            horizon=cfg.model.pred_horizon,
                        )
                        for index in active
                    ]
                    encoded = standardized_collate_fn(
                        samples,
                        max_images_per_sample=6,
                        image_history_size=2,
                        image_jpeg_quality=90,
                        skip_no_image=True,
                        encode_image_slots=False,
                    )
                    assert encoded is not None
                    qwen_kv = extract_qwen_kv(
                        encoded,
                        qwen_processor,
                        qwen,
                        device=device,
                        layer_index=int(metadata.get("qwen_layer_index", 7)),
                        max_new_tokens=args.qwen_max_new_tokens,
                        expected_dim=cfg.model.qwen_kv_dim,
                        stop_at_think_end=bool(metadata.get("qwen_stop_at_think", True)),
                        prompt_template=metadata.get("qwen_trajectory_prompt_template"),
                        enable_thinking=bool(metadata.get("qwen_enable_thinking", False)),
                    )
                    img_tokens, img_mask = extract_siglip_features(
                        encoded,
                        siglip_processor,
                        siglip,
                        max_img_tokens=cfg.model.image_tokens,
                        expected_dim=cfg.model.img_token_dim,
                        device=device,
                    )
                    lang_tokens, lang_mask = language_by_task[task_id]
                    policy_batch = {
                        "state": encoded["state"].to(device),
                        "action_dim_mask": encoded["action_dim_mask"].to(device),
                        "ctrl_freq": encoded["ctrl_freq"].to(device),
                        "lang_tokens": lang_tokens.expand(len(active), -1, -1).to(device),
                        "lang_mask": lang_mask.expand(len(active), -1).to(device),
                        "img_tokens": img_tokens,
                        "img_mask": img_mask,
                        "qwen_kv": qwen_kv,
                    }
                    torch.manual_seed(args.seed + task_id * 100_000 + batch_start * 1_000 + plan_index)
                    normalized = model.sample_actions(policy_batch).float().cpu().numpy()
                    predicted = rdt_action_to_libero(normalized, stats)
                    if not np.isfinite(predicted).all():
                        raise FloatingPointError("RDT produced NaN/Inf actions")

                    chunk = min(args.action_chunk, args.max_steps - simulator_step)
                    for action_offset in range(chunk):
                        actions = np.zeros((len(indices), 7), dtype=np.float32)
                        for active_position, env_index in enumerate(active):
                            actions[env_index] = predicted[active_position, action_offset]
                            previous[env_index] = observations[env_index]
                        next_obs, _, step_done, _ = env.step(actions)
                        observations = list(next_obs)
                        simulator_step += 1
                        newly_done = (~done) & np.asarray(step_done, dtype=bool)
                        success_step[newly_done] = simulator_step
                        done |= np.asarray(step_done, dtype=bool)
                        if bool(done.all()):
                            break
                    plan_index += 1
                    if plan_index % 10 == 0:
                        print(
                            f"task={task_id} states={indices} plan={plan_index} "
                            f"step={simulator_step}/{args.max_steps} done={int(done.sum())}/{len(done)}",
                            flush=True,
                        )

                elapsed = time.perf_counter() - batch_started
                env.close()
                for local_index, init_index in enumerate(indices):
                    row = {
                        "task_id": task_id,
                        "task_name": task.name,
                        "instruction": task.language,
                        "init_state_index": init_index,
                        "success": bool(done[local_index]),
                        "steps": int(success_step[local_index]),
                        "checkpoint": str(args.checkpoint.resolve()),
                    }
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    completed.add((task_id, init_index))
                summary = write_summary(results_path, summary_path)
                print(
                    f"task={task_id} states={indices} batch_successes={int(done.sum())}/{len(done)} "
                    f"overall={summary['successes']}/{summary['episodes']} "
                    f"rate={summary['success_rate']:.3f} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    summary = write_summary(results_path, summary_path)
    summary["requested_episodes"] = total_requested
    summary["complete"] = summary["episodes"] >= total_requested
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
