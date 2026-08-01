#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thinkflow-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/thinkflow-matplotlib")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
    T5Tokenizer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
LIBERO_ROOT_DEFAULT = Path("/home/ubuntu/LIBERO")
LIBERO_BENCHMARK_CHOICES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)
LIBERO_DEFAULT_BENCHMARK = "libero_spatial"
for path in (SRC_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from precompute_all_features import (  # noqa: E402
    extract_t5_features,
    extract_qwen_kv,
    extract_siglip_features,
    resolve_model_id,
    standardized_collate_fn,
)
from thinkflow_rdt.adapters.action_stats import (  # noqa: E402
    ACTION_DIM_NAMES,
    denormalize_action_array,
    load_action_stats,
)
from thinkflow_rdt.adapters.libero import (  # noqa: E402
    libero_observation_to_rdt,
    rdt_action_to_libero,
)
from thinkflow_rdt.checkpoint import load_trainable_artifact  # noqa: E402
from thinkflow_rdt.config import load_config  # noqa: E402
from thinkflow_rdt.model import SFTConditionedRDT  # noqa: E402


def install_robosuite_mujoco_compatibility() -> None:
    """Adapt robosuite 1.4's sole old-style MuJoCo mass-matrix call."""
    try:
        from robosuite.controllers.base_controller import Controller
    except ModuleNotFoundError:
        # Newer robosuite releases moved/removed this old controller module.
        # In that case the legacy mj_fullM patch is not applicable.
        return

    def update(controller, force: bool = False) -> None:
        if not (controller.new_update or force):
            return
        sim = controller.sim
        sim.forward()
        site_id = sim.model.site_name2id(controller.eef_name)
        controller.ee_pos = np.asarray(sim.data.site_xpos[site_id]).copy()
        controller.ee_ori_mat = np.asarray(sim.data.site_xmat[site_id]).reshape(3, 3).copy()
        controller.ee_pos_vel = np.asarray(sim.data.get_site_xvelp(controller.eef_name)).copy()
        controller.ee_ori_vel = np.asarray(sim.data.get_site_xvelr(controller.eef_name)).copy()
        controller.joint_pos = np.asarray(sim.data.qpos[controller.qpos_index]).copy()
        controller.joint_vel = np.asarray(sim.data.qvel[controller.qvel_index]).copy()
        controller.J_pos = np.asarray(
            sim.data.get_site_jacp(controller.eef_name).reshape((3, -1))[:, controller.qvel_index]
        ).copy()
        controller.J_ori = np.asarray(
            sim.data.get_site_jacr(controller.eef_name).reshape((3, -1))[:, controller.qvel_index]
        ).copy()
        controller.J_full = np.vstack([controller.J_pos, controller.J_ori])
        mass_matrix = np.empty((sim.model.nv, sim.model.nv), dtype=np.float64, order="C")
        try:
            mujoco.mj_fullM(sim.model._model, sim.data._data, mass_matrix)
        except TypeError:
            mujoco.mj_fullM(sim.model._model, mass_matrix, sim.data.qM)
        controller.mass_matrix = mass_matrix[controller.qvel_index, :][:, controller.qvel_index]
        controller.new_update = False

    Controller.update = update


def load_feature_metadata(cache_root: Path) -> dict[str, Any]:
    path = cache_root / "precompute_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def free_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_t5_encoder(
    *,
    model_id: str,
    fallback_model_id: str | None,
    precision: str,
    device_map: str,
    cfg: Any,
) -> tuple[T5Tokenizer, T5EncoderModel]:
    resolved_model_id = resolve_model_id(model_id, fallback_model_id)
    tokenizer = T5Tokenizer.from_pretrained(resolved_model_id)
    if precision == "8bit":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "--t5-precision 8bit requires transformers BitsAndBytesConfig "
                "and a bitsandbytes-capable environment."
            ) from exc
        encoder = T5EncoderModel.from_pretrained(
            resolved_model_id,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map=device_map,
        )
    elif precision == "bf16":
        encoder = T5EncoderModel.from_pretrained(
            resolved_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    else:
        raise ValueError("--t5-precision must be 'bf16' or '8bit'")
    encoder.eval()
    encoder.requires_grad_(False)
    if getattr(encoder.config, "d_model", cfg.model.lang_token_dim) != cfg.model.lang_token_dim:
        raise ValueError(
            f"T5 d_model {encoder.config.d_model} != "
            f"cfg.model.lang_token_dim {cfg.model.lang_token_dim}"
        )
    return tokenizer, encoder


def t5_device_from_encoder(encoder: Any, fallback: torch.device) -> torch.device:
    try:
        return next(encoder.parameters()).device
    except StopIteration:
        return fallback


def array_to_nested_float_list(values: np.ndarray) -> list:
    return np.asarray(values, dtype=np.float32).astype(float).tolist()


def action_debug_stats(values: np.ndarray) -> dict[str, list[float]]:
    action = np.asarray(values, dtype=np.float32)
    return {
        "min": action.min(axis=0).astype(float).tolist(),
        "max": action.max(axis=0).astype(float).tolist(),
        "mean": action.mean(axis=0).astype(float).tolist(),
    }


def rollout_sample(
    observation: dict[str, Any],
    previous_observation: dict[str, Any] | None,
    *,
    dataset_id: str,
    instruction: str,
    horizon: int,
) -> dict[str, Any]:
    converted = libero_observation_to_rdt(observation)
    current = {
        "primary": Image.fromarray(converted["primary"]).convert("RGB"),
        "wrist": None if converted["wrist"] is None else Image.fromarray(converted["wrist"]).convert("RGB"),
        "secondary": None,
    }
    if previous_observation is None:
        previous = current
        previous_mask = {"primary": 0, "wrist": 0, "secondary": 0}
    else:
        old = libero_observation_to_rdt(previous_observation)
        previous = {
            "primary": Image.fromarray(old["primary"]).convert("RGB"),
            "wrist": None if old["wrist"] is None else Image.fromarray(old["wrist"]).convert("RGB"),
            "secondary": None,
        }
        previous_mask = {
            "primary": 1,
            "wrist": int(previous["wrist"] is not None),
            "secondary": 0,
        }
    current_mask = {
        "primary": 1,
        "wrist": int(current["wrist"] is not None),
        "secondary": 0,
    }
    return {
        "dataset_id": dataset_id,
        "episode_id": "rollout",
        "step_idx": "0",
        "instruction": instruction,
        "images": current,
        "image_mask": current_mask,
        "image_history": [previous, current],
        "image_history_mask": [previous_mask, current_mask],
        "state": converted["state"],
        "state_mask": np.ones(7, dtype=np.float32),
        "actions": np.zeros((horizon, 7), dtype=np.float32),
        "actions_mask": np.ones(horizon, dtype=np.float32),
        "ctrl_freq": 20.0,
    }


def frame_for_video(frame: np.ndarray, text: str) -> np.ndarray:
    # LIBERO camera arrays use OpenGL's bottom-up convention.
    frame = np.asarray(frame)[::-1].copy()
    width = int(frame.shape[1])
    cv2.rectangle(frame, (0, 0), (width, 34), (0, 0, 0), thickness=-1)
    cv2.putText(frame, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out a trained RDT artifact in LIBERO and record MP4.")
    parser.add_argument("--config", default="configs/b0_rdt1b_lora.yaml")
    parser.add_argument("--benchmark", choices=LIBERO_BENCHMARK_CHOICES, default=LIBERO_DEFAULT_BENCHMARK)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/libero_spatial_full/checkpoint-1600"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache_features/libero_spatial/full"))
    parser.add_argument("--action-stats", type=Path, default=Path("dataset/LIBERO/Spatial/datasets/libero_spatial/audit.json"))
    parser.add_argument("--libero-root", type=Path, default=LIBERO_ROOT_DEFAULT)
    parser.add_argument("--task-id", type=int, default=0, choices=range(10))
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument(
        "--demo-hdf5",
        type=Path,
        help="Start from the first recorded simulator state of this HDF5 demo instead of a benchmark init state.",
    )
    parser.add_argument("--demo-name", default="demo_0")
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
    parser.add_argument("--qwen-refresh-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--video-resolution",
        type=int,
        default=512,
        help="True simulator render resolution for the MP4; policy observations remain 128x128.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/libero_spatial_rollout/task0_checkpoint1600.mp4"))
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
        "--action-debug-jsonl",
        type=Path,
        help=(
            "Write per-replan normalized, denormalized, and final LIBERO "
            "actions to this JSONL file."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.libero_root) not in sys.path:
        sys.path.insert(0, str(args.libero_root))
    install_robosuite_mujoco_compatibility()
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RDT rollout")
    device = torch.device("cuda")
    cfg = load_config(args.config)
    if args.action_chunk <= 0:
        raise ValueError("--action-chunk must be positive")
    if args.action_chunk > cfg.model.pred_horizon:
        raise ValueError(
            f"--action-chunk ({args.action_chunk}) cannot exceed "
            f"cfg.model.pred_horizon ({cfg.model.pred_horizon})"
        )
    stats = load_action_stats(args.action_stats)
    print("Action denormalization stats:")
    print(json.dumps({
        "dim_names": ACTION_DIM_NAMES[: len(stats.q01)],
        "q01": stats.q01.astype(float).tolist(),
        "q99": stats.q99.astype(float).tolist(),
        "eps": stats.eps,
    }, indent=2))
    metadata = load_feature_metadata(args.cache_root)
    qwen_id = args.qwen_model_id or metadata.get("qwen_model_id", "shreethar/stage1_unsloth")
    qwen_processor_id = args.qwen_processor_id or metadata.get("qwen_processor_id", qwen_id)
    siglip_id = args.siglip_model_id or metadata.get("siglip_model_id", "google/siglip-so400m-patch14-384")
    t5_id = (
        args.t5_model_id
        or metadata.get("t5_model_id")
        or "/home/ubuntu/RoboticsDiffusionTransformer/google/t5-v1_1-xxl"
    )

    benchmark = get_benchmark(args.benchmark)(0)
    task = benchmark.get_task(args.task_id)
    instruction = task.language
    print("Loading T5, Qwen, and SigLIP encoders...")
    t5_tokenizer, t5 = load_t5_encoder(
        model_id=t5_id,
        fallback_model_id=args.t5_fallback_model_id,
        precision=args.t5_precision,
        device_map=args.device_map,
        cfg=cfg,
    )
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
    lang_tokens, lang_mask = extract_t5_features(
        {"instructions": [instruction]},
        t5_tokenizer,
        t5,
        max_lang_tokens=cfg.model.max_lang_tokens,
        expected_dim=cfg.model.lang_token_dim,
        device=t5_device_from_encoder(t5, device),
    )

    print(f"Loading RDT artifact {args.checkpoint}...")
    model = SFTConditionedRDT(cfg, load_pretrained=True)
    load_trainable_artifact(model, args.checkpoint, trainable=False)
    model.to(device).eval()

    env = OffScreenRenderEnv(
        bddl_file_name=benchmark.get_task_bddl_file_path(args.task_id),
        camera_heights=128,
        camera_widths=128,
        horizon=args.max_steps + 10,
    )
    observation = env.reset()
    if args.demo_hdf5 is not None:
        import h5py

        with h5py.File(args.demo_hdf5, "r") as handle:
            demo = handle["data"][args.demo_name]
            recorded_state = np.asarray(demo["states"][0], dtype=np.float64)
        state_index = -1
        observation = env.set_init_state(recorded_state)
    else:
        init_states = torch.load(
            args.libero_root / "libero" / "libero" / "init_files" / task.problem_folder / task.init_states_file,
            map_location="cpu",
            weights_only=False,
        )
        state_index = args.init_state_index % len(init_states)
        observation = env.set_init_state(init_states[state_index])
        for _ in range(5):
            observation, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output,
        format="FFMPEG",
        fps=args.fps,
        codec="libx264",
        quality=8,
    )
    previous_observation = None
    cached_qwen: torch.Tensor | None = None
    success = False
    simulator_step = 0
    plan_index = 0
    start = time.perf_counter()
    action_debug_handle = None
    if args.action_debug_jsonl is not None:
        args.action_debug_jsonl.parent.mkdir(parents=True, exist_ok=True)
        action_debug_handle = args.action_debug_jsonl.open("w", encoding="utf-8")
    try:
        while simulator_step < args.max_steps and not success:
            sample = rollout_sample(
                observation,
                previous_observation,
                dataset_id=args.benchmark,
                instruction=instruction,
                horizon=cfg.model.pred_horizon,
            )
            encoded = standardized_collate_fn(
                [sample],
                max_images_per_sample=6,
                image_history_size=2,
                image_jpeg_quality=90,
                skip_no_image=True,
                encode_image_slots=False,
            )
            assert encoded is not None
            if cached_qwen is None or plan_index % args.qwen_refresh_every == 0:
                cached_qwen = extract_qwen_kv(
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
            batch = {
                "state": encoded["state"].to(device),
                "action_dim_mask": encoded["action_dim_mask"].to(device),
                "ctrl_freq": encoded["ctrl_freq"].to(device),
                "lang_tokens": lang_tokens.to(device),
                "lang_mask": lang_mask.to(device),
                "img_tokens": img_tokens,
                "img_mask": img_mask,
                "qwen_kv": cached_qwen,
            }
            torch.manual_seed(args.seed + plan_index)
            normalized = model.sample_actions(batch)[0].float().cpu().numpy()
            denormalized = denormalize_action_array(normalized, stats)
            actions = rdt_action_to_libero(normalized, stats)
            if not np.isfinite(actions).all():
                raise FloatingPointError("RDT produced NaN/Inf actions")

            chunk = min(args.action_chunk, args.max_steps - simulator_step)
            if action_debug_handle is not None:
                action_debug_handle.write(
                    json.dumps(
                        {
                            "plan_index": plan_index,
                            "simulator_step_start": simulator_step,
                            "pred_horizon": int(cfg.model.pred_horizon),
                            "executed_steps": int(chunk),
                            "dim_names": ACTION_DIM_NAMES[: normalized.shape[-1]],
                            "normalization": {
                                "q01": stats.q01.astype(float).tolist(),
                                "q99": stats.q99.astype(float).tolist(),
                                "formula": "denormalized = (clip(normalized,-1,1)+1)*0.5*(q99-q01)+q01",
                            },
                            "normalized_stats": action_debug_stats(normalized),
                            "denormalized_delta_stats": action_debug_stats(denormalized),
                            "libero_action_stats": action_debug_stats(actions),
                            "normalized_actions": array_to_nested_float_list(normalized),
                            "denormalized_delta_actions": array_to_nested_float_list(denormalized),
                            "libero_actions": array_to_nested_float_list(actions),
                            "executed_libero_actions": array_to_nested_float_list(actions[:chunk]),
                        }
                    )
                    + "\n"
                )
                action_debug_handle.flush()
            for action_index in range(chunk):
                # Keep adjacent t-1/t frames for the next SigLIP history, matching
                # the feature-precomputation contract even when actions are chunked.
                previous_observation = observation
                observation, reward, done, _ = env.step(actions[action_index])
                simulator_step += 1
                success = bool(done) or bool(env.check_success())
                label = f"task={args.task_id} step={simulator_step} plan={plan_index} success={int(success)}"
                video_frame = env.env.sim.render(
                    width=args.video_resolution,
                    height=args.video_resolution,
                    camera_name="agentview",
                )
                writer.append_data(frame_for_video(video_frame, label))
                if success:
                    break
            plan_index += 1
            print(
                f"plan={plan_index} simulator_step={simulator_step} "
                f"success={success} elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    finally:
        if action_debug_handle is not None:
            action_debug_handle.close()
        writer.close()
        env.close()

    summary = {
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "instruction": instruction,
        "checkpoint": str(args.checkpoint.resolve()),
        "init_state_index": state_index,
        "steps": simulator_step,
        "plans": plan_index,
        "success": success,
        "video": str(args.output.resolve()),
    }
    if args.action_debug_jsonl is not None:
        summary["action_debug_jsonl"] = str(args.action_debug_jsonl.resolve())
    if args.demo_hdf5 is not None:
        summary["demo_hdf5"] = str(args.demo_hdf5.resolve())
        summary["demo_name"] = args.demo_name
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
