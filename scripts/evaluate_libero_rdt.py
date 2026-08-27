#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import MethodType
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thinkflow-cache")
# Model snapshots were downloaded to the normal user Hugging Face cache. Keep
# simulation/matplotlib scratch files in /tmp without hiding those snapshots.
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thinkflow-matplotlib")

import imageio.v2 as imageio
import h5py
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
    extract_t5_features,
    extract_qwen_kv,
    extract_siglip_features,
    standardized_collate_fn,
)
from rollout_libero_rdt import (  # noqa: E402
    LIBERO_BENCHMARK_CHOICES,
    LIBERO_DEFAULT_BENCHMARK,
    absolute_target_state_to_libero_action,
    frame_for_video,
    install_robosuite_mujoco_compatibility,
    load_feature_metadata,
    load_t5_encoder,
    resolve_model_id,
    rdt_state_open_from_libero,
    rollout_sample,
    t5_device_from_encoder,
)
from thinkflow_rdt.adapters.action_stats import load_action_stats  # noqa: E402
from thinkflow_rdt.adapters.libero import rdt_action_to_libero  # noqa: E402
from thinkflow_rdt.checkpoint import load_trainable_artifact  # noqa: E402
from thinkflow_rdt.config import load_config  # noqa: E402
from thinkflow_rdt.model import SFTConditionedRDT  # noqa: E402


def native_rdt_action_to_libero_10d(actions: np.ndarray) -> np.ndarray:
    """Extract LIBERO's supervised 10-D command from native RDT output."""
    values = np.asarray(actions)
    if values.shape[-1] != 128:
        raise ValueError(
            f"Expected native RDT action width 128, got {values.shape[-1]}"
        )
    return np.concatenate(
        [values[..., 30:33], values[..., 33:39], values[..., 10:11]],
        axis=-1,
    )


def native_rdt_policy_inputs(
    state: torch.Tensor,
    state_dim_mask: torch.Tensor,
    action_dim_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map live LIBERO 11-D state / 10-D action masks into RDT's 128 slots."""
    if state.ndim != 2 or state.shape[-1] != 11:
        raise ValueError(f"Expected live LIBERO state [B, 11], got {tuple(state.shape)}")
    if state_dim_mask.shape != state.shape:
        raise ValueError("LIBERO state_dim_mask must match state shape")
    if action_dim_mask.ndim != 2 or action_dim_mask.shape != (state.shape[0], 10):
        raise ValueError(
            "Expected live LIBERO action_dim_mask [B, 10], got "
            f"{tuple(action_dim_mask.shape)}"
        )
    native_state = state.new_zeros(state.shape[0], 128)
    native_state_mask = state_dim_mask.new_zeros(state.shape[0], 128)
    native_state[:, 30:39] = state[:, :9] * state_dim_mask[:, :9]
    native_state_mask[:, 30:39] = state_dim_mask[:, :9]
    native_state[:, 10:12] = state[:, 9:11] * state_dim_mask[:, 9:11]
    native_state_mask[:, 10:12] = state_dim_mask[:, 9:11]

    native_action_mask = action_dim_mask.new_zeros(state.shape[0], 128)
    native_action_mask[:, 30:39] = action_dim_mask[:, :9]
    native_action_mask[:, 10] = action_dim_mask[:, 9]
    return native_state, native_state_mask, native_action_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure RDT success rate across LIBERO episodes.")
    parser.add_argument("--config", default="configs/b0_rdt1b_lora.yaml")
    parser.add_argument("--benchmark", choices=LIBERO_BENCHMARK_CHOICES, default=LIBERO_DEFAULT_BENCHMARK)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/libero_spatial_full/checkpoint-1600"))
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=None,
        help=(
            "Optional merged/full RDT base artifact to load before applying the "
            "trainable checkpoint. Use the same base artifact used for training."
        ),
    )
    parser.add_argument(
        "--pretrained-only",
        action="store_true",
        help=(
            "Load only cfg.pretrained_model and skip the trainable checkpoint "
            "artifact. Qwen fusion is disabled so no random Qwen adaptor is used."
        ),
    )
    parser.add_argument(
        "--disable-qwen-fusion",
        action="store_true",
        help="Disable Qwen fusion/extraction for rollout while still loading the checkpoint artifact.",
    )
    parser.add_argument(
        "--require-qwen-fusion",
        action="store_true",
        help="Fail instead of silently running if the resolved config disables Qwen fusion.",
    )
    parser.add_argument("--cache-root", type=Path, default=Path("cache_features/libero_spatial/full"))
    parser.add_argument("--action-stats", type=Path, default=Path("dataset/LIBERO/Spatial/datasets/libero_spatial/audit.json"))
    parser.add_argument(
        "--action-output-mode",
        choices=["raw_delta_ortho6d", "absolute_target_state", "normalized_delta"],
        default="raw_delta_ortho6d",
        help=(
            "Use raw_delta_ortho6d for the current 10D LIBERO command model. "
            "Other modes are retained for legacy checkpoints."
        ),
    )
    parser.add_argument(
        "--target-state-start-index",
        type=int,
        default=1,
        help="First predicted target-state token to execute; 1 skips target[0] == current state.",
    )
    parser.add_argument(
        "--max-delta-pos",
        type=float,
        default=1.0,
        help="Clip absolute-target xyz deltas before sending them to LIBERO; set negative to disable.",
    )
    parser.add_argument(
        "--max-delta-rot",
        type=float,
        default=1.0,
        help="Clip absolute-target rpy deltas before sending them to LIBERO; set negative to disable.",
    )
    parser.add_argument(
        "--pos-scale",
        type=float,
        default=10.0,
        help="Scale absolute-target xyz error into LIBERO controller command space.",
    )
    parser.add_argument(
        "--rot-scale",
        type=float,
        default=10.0,
        help="Scale absolute-target rpy error into LIBERO controller command space.",
    )
    parser.add_argument("--libero-root", type=Path, default=Path("/home/ubuntu/LIBERO"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/libero_spatial_evaluation/checkpoint-1600"))
    parser.add_argument("--episodes-per-task", type=int, default=20, help="LIBERO official default is 20; each task has 50 available.")
    parser.add_argument("--all-episodes", action="store_true", help="Evaluate all 50 states for each of 10 tasks (500 rollouts).")
    parser.add_argument(
        "--demo-hdf5",
        type=Path,
        help=(
            "Use exact initial simulator states from this LIBERO demonstration "
            "file instead of benchmark init-state files. Requires one --task-id."
        ),
    )
    parser.add_argument(
        "--demo-name",
        action="append",
        help=(
            "HDF5 demo group to evaluate; may be repeated. With --demo-hdf5 "
            "and no names, evaluates every demo group."
        ),
    )
    parser.add_argument("--env-batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--action-chunk",
        type=int,
        default=8,
        help=(
            "Number of sampled actions to execute before observing again and "
            "re-planning. The model still predicts cfg.model.pred_horizon actions."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--t5-model-id",
        default=None,
        help="Override T5 XXL model path/id. Defaults to cache metadata, then local RDT model root, then google/t5-v1_1-xxl.",
    )
    parser.add_argument("--t5-fallback-model-id", default="google/t5-v1_1-xxl")
    parser.add_argument("--t5-precision", choices=["bf16", "8bit"], default="bf16")
    parser.add_argument(
        "--qwen-model-id",
        default=None,
        help="Override Qwen model path/id. Defaults to cache metadata, then shreethar/stage1_unsloth.",
    )
    parser.add_argument(
        "--qwen-processor-id",
        default=None,
        help="Override Qwen processor path/id. Defaults to --qwen-model-id.",
    )
    parser.add_argument(
        "--siglip-model-id",
        default=None,
        help="Override SigLIP model path/id. Defaults to cache metadata, then google/siglip-so400m-patch14-384.",
    )
    parser.add_argument(
        "--siglip-fallback-model-id",
        default="google/siglip-so400m-patch14-384",
        help=(
            "Fallback when --siglip-model-id is a local path that does not exist."
        ),
    )
    parser.add_argument(
        "--task-id",
        type=int,
        action="append",
        choices=range(10),
        help="Evaluate only this task ID; may be repeated. Defaults to all tasks.",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save one separately rendered high-resolution MP4 per episode.",
    )
    parser.add_argument("--video-resolution", type=int, default=512)
    parser.add_argument("--video-fps", type=int, default=20)
    return parser.parse_args()


def _high_resolution_render(
    env: Any,
    *,
    width: int,
    height: int,
    camera_name: str,
) -> np.ndarray:
    return env.env.sim.render(
        width=width,
        height=height,
        camera_name=camera_name,
    )


def make_recordable_env(env_args: dict[str, Any]) -> Any:
    """Build an environment with a render method callable through venv."""
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(**env_args)
    env.render = MethodType(_high_resolution_render, env)
    return env


def render_vector_parallel(env: Any, **kwargs: Any) -> list[np.ndarray]:
    """Render all subprocess environments concurrently."""
    for worker in env.workers:
        worker.parent_remote.send(["render", kwargs])
    return [worker.parent_remote.recv() for worker in env.workers]


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


def load_demo_initial_states(
    path: Path,
    requested_names: list[str] | None,
) -> tuple[list[str], np.ndarray]:
    """Load exact initial MuJoCo states from selected HDF5 demonstrations."""
    with h5py.File(path, "r") as handle:
        root = handle["data"] if "data" in handle else handle
        available = sorted(
            name for name in root if hasattr(root[name], "keys") and "states" in root[name]
        )
        names = available if not requested_names else list(dict.fromkeys(requested_names))
        missing = [name for name in names if name not in root]
        if missing:
            raise KeyError(
                f"Demo groups not found in {path}: {missing}; available examples: {available[:10]}"
            )
        if not names:
            raise ValueError(f"No demonstrations with states found in {path}")
        states = [np.asarray(root[name]["states"][0], dtype=np.float64) for name in names]
    shapes = {state.shape for state in states}
    if len(shapes) != 1:
        raise ValueError(f"Selected demo initial states have inconsistent shapes: {shapes}")
    return names, np.stack(states)


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
    task_ids = sorted(set(args.task_id)) if args.task_id else list(range(10))
    demo_names: list[str] | None = None
    demo_initial_states: np.ndarray | None = None
    if args.demo_hdf5 is not None:
        if args.all_episodes:
            raise ValueError("--all-episodes cannot be combined with --demo-hdf5")
        if len(task_ids) != 1:
            raise ValueError("--demo-hdf5 requires exactly one --task-id")
        demo_names, demo_initial_states = load_demo_initial_states(
            args.demo_hdf5.expanduser().resolve(),
            args.demo_name,
        )
        args.episodes_per_task = len(demo_names)
    elif args.demo_name:
        raise ValueError("--demo-name requires --demo-hdf5")
    if args.env_batch_size <= 0:
        raise ValueError("--env-batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RDT evaluation")

    install_robosuite_mujoco_compatibility()
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

    device = torch.device("cuda")
    cfg = load_config(args.config)
    if args.pretrained_only or args.disable_qwen_fusion:
        cfg = replace(cfg, model=replace(cfg.model, qwen_fusion="none"))
    if args.require_qwen_fusion and cfg.model.qwen_fusion == "none":
        raise ValueError(
            "--require-qwen-fusion was requested, but the resolved model config "
            "has qwen_fusion='none'"
        )
    if args.action_chunk <= 0:
        raise ValueError("--action-chunk must be positive")
    if args.action_chunk > cfg.model.pred_horizon:
        raise ValueError(
            f"--action-chunk ({args.action_chunk}) cannot exceed "
            f"cfg.model.pred_horizon ({cfg.model.pred_horizon})"
        )
    if args.target_state_start_index < 0:
        raise ValueError("--target-state-start-index must be non-negative")
    max_delta_pos = None if args.max_delta_pos < 0 else args.max_delta_pos
    max_delta_rot = None if args.max_delta_rot < 0 else args.max_delta_rot
    stats = load_action_stats(args.action_stats) if args.action_output_mode == "normalized_delta" else None
    if (
        args.action_output_mode != "raw_delta_ortho6d"
        and cfg.model.action_encoder_layout == "libero_ortho6d"
    ):
        raise ValueError(
            "libero_ortho6d checkpoints must use --action-output-mode "
            "raw_delta_ortho6d"
        )
    metadata = load_feature_metadata(args.cache_root)
    qwen_id = args.qwen_model_id or metadata.get("qwen_model_id", "shreethar/stage1_unsloth")
    qwen_processor_id = args.qwen_processor_id or metadata.get("qwen_processor_id", qwen_id)
    siglip_id = resolve_model_id(
        args.siglip_model_id
        or metadata.get(
            "siglip_model_id", "google/siglip-so400m-patch14-384"
        ),
        args.siglip_fallback_model_id,
    )
    t5_id = (
        args.t5_model_id
        or metadata.get("t5_model_id")
        or "/home/ubuntu/RoboticsDiffusionTransformer/google/t5-v1_1-xxl"
    )
    benchmark = get_benchmark(args.benchmark)(0)

    use_qwen = cfg.model.qwen_fusion != "none"
    print("Loading T5, SigLIP, and optional Qwen encoders...")
    t5_tokenizer, t5 = load_t5_encoder(
        model_id=t5_id,
        fallback_model_id=args.t5_fallback_model_id,
        precision=args.t5_precision,
        device_map=args.device_map,
        cfg=cfg,
    )
    qwen_processor = None
    qwen = None
    if use_qwen:
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
    language_by_task = {
        task_id: extract_t5_features(
            {"instructions": [benchmark.get_task(task_id).language]},
            t5_tokenizer,
            t5,
            max_lang_tokens=cfg.model.max_lang_tokens,
            expected_dim=cfg.model.lang_token_dim,
            device=t5_device_from_encoder(t5, device),
        )
        for task_id in task_ids
    }

    if args.pretrained_only:
        print(f"Loading pretrained RDT baseline {cfg.pretrained_model}...")
    else:
        print(f"Loading RDT artifact {args.checkpoint}...")
    model = SFTConditionedRDT(
        cfg,
        load_pretrained=True,
        base_artifact=(
            None if args.base_artifact is None else str(args.base_artifact)
        ),
    )
    if not args.pretrained_only:
        load_trainable_artifact(model, args.checkpoint, trainable=False)
    model.to(device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "episodes.jsonl"
    summary_path = args.output_dir / "summary.json"
    completed = existing_result_keys(results_path)
    total_requested = len(task_ids) * args.episodes_per_task
    print(f"Evaluation target: {total_requested} episodes; resuming after {len(completed)} completed")

    with results_path.open("a", encoding="utf-8") as output:
        for task_id in task_ids:
            task = benchmark.get_task(task_id)
            if demo_initial_states is not None:
                all_init_states = demo_initial_states
                assert demo_names is not None
                state_labels = demo_names
                settle_steps = 0
            else:
                all_init_states = torch.load(
                    args.libero_root / "libero" / "libero" / "init_files" / task.problem_folder / task.init_states_file,
                    map_location="cpu",
                    weights_only=False,
                )
                state_labels = [f"init{index:02d}" for index in range(args.episodes_per_task)]
                settle_steps = 5
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
                if args.save_videos:
                    env_fns = [
                        lambda env_args=env_args: make_recordable_env(env_args)
                        for _ in indices
                    ]
                else:
                    env_fns = [
                        lambda env_args=env_args: OffScreenRenderEnv(**env_args)
                        for _ in indices
                    ]
                env = SubprocVectorEnv(env_fns)
                env.reset()
                observations = list(env.set_init_state(all_init_states[indices]))
                for _ in range(settle_steps):
                    observations, _, _, _ = env.step(np.zeros((len(indices), 7), dtype=np.float32))
                    observations = list(observations)

                previous: list[dict[str, Any] | None] = [None] * len(indices)
                done = np.zeros(len(indices), dtype=bool)
                success_step = np.full(len(indices), args.max_steps, dtype=np.int32)
                simulator_step = 0
                plan_index = 0
                batch_started = time.perf_counter()
                video_paths: list[Path | None] = [None] * len(indices)
                writers: list[Any | None] = [None] * len(indices)
                if args.save_videos:
                    video_dir = args.output_dir / "videos"
                    video_dir.mkdir(parents=True, exist_ok=True)
                    for local_index, init_index in enumerate(indices):
                        state_label = state_labels[init_index]
                        video_path = video_dir / f"task{task_id:02d}_{state_label}.mp4"
                        video_paths[local_index] = video_path
                        writers[local_index] = imageio.get_writer(
                            video_path,
                            format="FFMPEG",
                            fps=args.video_fps,
                            codec="libx264",
                            quality=8,
                        )
                while simulator_step < args.max_steps and not bool(done.all()):
                    active = np.flatnonzero(~done).tolist()
                    samples = [
                        rollout_sample(
                            observations[index],
                            previous[index],
                            dataset_id=args.benchmark,
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
                    qwen_kv = None
                    if use_qwen:
                        assert qwen_processor is not None and qwen is not None
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
                    state = encoded["state"]
                    state_dim_mask = encoded["state_dim_mask"]
                    action_dim_mask = encoded["action_dim_mask"]
                    if cfg.model.state_encoder_layout == "rdt_native_128":
                        state, state_dim_mask, action_dim_mask = (
                            native_rdt_policy_inputs(
                                state,
                                state_dim_mask,
                                action_dim_mask,
                            )
                        )
                    policy_batch = {
                        "state": state.to(device),
                        "state_dim_mask": state_dim_mask.to(device),
                        "action_dim_mask": action_dim_mask.to(device),
                        "ctrl_freq": encoded["ctrl_freq"].to(device),
                        "lang_tokens": lang_tokens.expand(len(active), -1, -1).to(device),
                        "lang_mask": lang_mask.expand(len(active), -1).to(device),
                        "img_tokens": img_tokens,
                        "img_mask": img_mask,
                    }
                    if use_qwen:
                        assert qwen_kv is not None
                        policy_batch["qwen_kv"] = qwen_kv
                    torch.manual_seed(args.seed + task_id * 100_000 + batch_start * 1_000 + plan_index)
                    model_output = model.sample_actions(policy_batch).float().cpu().numpy()
                    predicted = None
                    if args.action_output_mode == "raw_delta_ortho6d":
                        rdt_commands = (
                            native_rdt_action_to_libero_10d(model_output)
                            if cfg.model.action_encoder_layout == "rdt_native_128"
                            else model_output
                        )
                        predicted = rdt_action_to_libero(rdt_commands)
                        finite_actions = predicted
                    elif args.action_output_mode == "normalized_delta":
                        assert stats is not None
                        rdt_commands = (
                            native_rdt_action_to_libero_10d(model_output)
                            if cfg.model.action_encoder_layout == "rdt_native_128"
                            else model_output
                        )
                        predicted = rdt_action_to_libero(rdt_commands, stats)
                        finite_actions = predicted
                    else:
                        finite_actions = model_output
                    if not np.isfinite(finite_actions).all():
                        raise FloatingPointError("RDT produced NaN/Inf actions")

                    chunk = min(args.action_chunk, args.max_steps - simulator_step)
                    for action_offset in range(chunk):
                        done_before_step = done.copy()
                        actions = np.zeros((len(indices), 7), dtype=np.float32)
                        for active_position, env_index in enumerate(active):
                            if args.action_output_mode == "absolute_target_state":
                                target_index = min(
                                    action_offset + args.target_state_start_index,
                                    model_output.shape[1] - 1,
                                )
                                current_state = rdt_state_open_from_libero(observations[env_index])
                                actions[env_index] = absolute_target_state_to_libero_action(
                                    model_output[active_position, target_index],
                                    current_state,
                                    pos_scale=args.pos_scale,
                                    rot_scale=args.rot_scale,
                                    max_delta_pos=max_delta_pos,
                                    max_delta_rot=max_delta_rot,
                                )
                            else:
                                assert predicted is not None
                                actions[env_index] = predicted[active_position, action_offset]
                            previous[env_index] = observations[env_index]
                        next_obs, _, step_done, _ = env.step(actions)
                        observations = list(next_obs)
                        simulator_step += 1
                        newly_done = (~done) & np.asarray(step_done, dtype=bool)
                        success_step[newly_done] = simulator_step
                        done |= np.asarray(step_done, dtype=bool)
                        if args.save_videos:
                            rendered = render_vector_parallel(
                                env,
                                width=args.video_resolution,
                                height=args.video_resolution,
                                camera_name="agentview",
                            )
                            for local_index, writer in enumerate(writers):
                                if writer is None or done_before_step[local_index]:
                                    continue
                                label = (
                                    f"task={task_id} state={state_labels[indices[local_index]]} "
                                    f"step={simulator_step} success={int(done[local_index])}"
                                )
                                writer.append_data(frame_for_video(rendered[local_index], label))
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
                for writer in writers:
                    if writer is not None:
                        writer.close()
                env.close()
                for local_index, init_index in enumerate(indices):
                    row = {
                        "benchmark": args.benchmark,
                        "task_id": task_id,
                        "task_name": task.name,
                        "instruction": task.language,
                        "init_state_index": init_index,
                        "initial_state_label": state_labels[init_index],
                        "initial_state_source": (
                            "demo_hdf5" if demo_initial_states is not None else "benchmark"
                        ),
                        "success": bool(done[local_index]),
                        "steps": int(success_step[local_index]),
                        "checkpoint": "pretrained-only" if args.pretrained_only else str(args.checkpoint.resolve()),
                        "pretrained_only": bool(args.pretrained_only),
                    }
                    if demo_initial_states is not None:
                        row["demo_hdf5"] = str(args.demo_hdf5.resolve())
                        row["demo_name"] = state_labels[init_index]
                    if video_paths[local_index] is not None:
                        row["video"] = str(video_paths[local_index].resolve())
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
