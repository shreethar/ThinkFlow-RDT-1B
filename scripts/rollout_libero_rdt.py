#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thinkflow-cache")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
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
B2_TRAJECTORY_PROMPT = (
    "You are a robot manipulation assistant. Given an observation image and a "
    "task instruction, predict the end-effector's 2D trajectory as 5 waypoints. "
    "Output ONLY the coordinate list in this exact format: "
    "[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]\n\n"
    "Task: The task is {task}. What is the trajectory that the end effector should take?"
)
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
    LIBERO_ACTION_DIM,
    LIBERO_STATE_DIM,
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


def resolve_qwen_extraction_mode(
    requested: str,
    *,
    checkpoint: str | Path,
    config: str | Path,
) -> str:
    """Select B0 or the shared B2/B3 LatentStudent extraction contract."""
    if requested in {"b0", "b2", "b3"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported Qwen extraction mode: {requested!r}")
    names = f"{Path(checkpoint)} {Path(config)}".lower()
    if "b3" in names:
        return "b3"
    return "b2" if "b2" in names else "b0"


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
    sample = {
        "min": action.min(axis=0).astype(float).tolist(),
        "max": action.max(axis=0).astype(float).tolist(),
        "mean": action.mean(axis=0).astype(float).tolist(),
    }


def wrap_rpy_delta(delta: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(delta), np.cos(delta)).astype(np.float32)


def rdt_state_open_from_libero(observation: dict[str, Any]) -> np.ndarray:
    """Return the current raw-finger LIBERO state (legacy name)."""
    return libero_observation_to_rdt(observation)["state"].copy()


def absolute_target_state_to_libero_action(
    target_state: np.ndarray,
    current_state: np.ndarray,
    *,
    pos_scale: float,
    rot_scale: float,
    max_delta_pos: float | None,
    max_delta_rot: float | None,
) -> np.ndarray:
    """Convert predicted absolute target state to LIBERO delta-controller action."""
    target = np.asarray(target_state, dtype=np.float32)
    current = np.asarray(current_state, dtype=np.float32)
    action = np.zeros((7,), dtype=np.float32)
    action[:3] = (target[:3] - current[:3]) * float(pos_scale)
    action[3:6] = wrap_rpy_delta(target[3:6] - current[3:6]) * float(rot_scale)
    if max_delta_pos is not None:
        action[:3] = np.clip(action[:3], -float(max_delta_pos), float(max_delta_pos))
    if max_delta_rot is not None:
        action[3:6] = np.clip(action[3:6], -float(max_delta_rot), float(max_delta_rot))
    # Legacy absolute-target thresholding only. The current raw-command path
    # bypasses this function and preserves the demonstrated action sign.
    action[6] = 1.0 if float(target[6]) >= 0.5 else -1.0
    return action


def rollout_sample(
    observation: dict[str, Any],
    previous_observation: dict[str, Any] | None,
    *,
    dataset_id: str,
    instruction: str,
    horizon: int,
) -> dict[str, Any]:
    converted = libero_observation_to_rdt(observation)
    state = converted["state"].copy()
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
    sample = {
        "dataset_id": dataset_id,
        "episode_id": "rollout",
        "step_idx": "0",
        "instruction": instruction,
        "images": current,
        "image_mask": current_mask,
        "image_history": [previous, current],
        "image_history_mask": [previous_mask, current_mask],
        "state": state,
        "state_mask": np.ones(LIBERO_STATE_DIM, dtype=np.float32),
        "actions": np.zeros((horizon, LIBERO_ACTION_DIM), dtype=np.float32),
        "actions_mask": np.ones(horizon, dtype=np.float32),
        "action_dim_mask": np.ones(LIBERO_ACTION_DIM, dtype=np.float32),
        "ctrl_freq": 20.0,
    }
    if "joint_state" in converted:
        sample["joint_state"] = converted["joint_state"].copy()
    return sample


def native_rdt_policy_inputs(
    state: torch.Tensor,
    state_dim_mask: torch.Tensor,
    action_dim_mask: torch.Tensor,
    *,
    mapping: str = "eef_pose_ortho6d",
    joint_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack live LIBERO state/action masks into native RDT's 128 slots."""
    if state.ndim != 2 or state.shape[-1] != LIBERO_STATE_DIM:
        raise ValueError(
            f"Expected live LIBERO state [B, {LIBERO_STATE_DIM}], got "
            f"{tuple(state.shape)}"
        )
    if state_dim_mask.shape != state.shape:
        raise ValueError("LIBERO state_dim_mask must match state shape")
    expected_action_dim = 7 if mapping == "libero_joint_eef_delta" else LIBERO_ACTION_DIM
    if action_dim_mask.ndim != 2 or action_dim_mask.shape != (
        state.shape[0],
        expected_action_dim,
    ):
        raise ValueError(
            f"Expected live LIBERO action_dim_mask [B, {expected_action_dim}], "
            f"got {tuple(action_dim_mask.shape)}"
        )

    native_state = state.new_zeros(state.shape[0], 128)
    native_state_mask = state_dim_mask.new_zeros(state.shape[0], 128)
    native_action_mask = action_dim_mask.new_zeros(state.shape[0], 128)
    if mapping == "libero_joint_eef_delta":
        if joint_state is None or joint_state.ndim != 2 or joint_state.shape[1] < 7:
            raise ValueError(
                "libero_joint_eef_delta rollout requires joint_state [B,>=7]"
            )
        native_state[:, :7] = joint_state[:, :7].to(native_state)
        native_state_mask[:, :7] = 1
        gripper_min = -0.04245
        gripper_max = 0.05185
        native_state[:, 10:12] = (
            state[:, 9:11] - gripper_min
        ) / (gripper_max - gripper_min)
        native_state_mask[:, 10:12] = state_dim_mask[:, 9:11]
        native_action_mask[:, 39:45] = action_dim_mask[:, :6]
        native_action_mask[:, 10] = action_dim_mask[:, 6]
        return native_state, native_state_mask, native_action_mask
    if mapping != "eef_pose_ortho6d":
        raise ValueError(f"Unsupported native RDT mapping: {mapping!r}")
    # Compact live state: xyz + absolute ortho6D + two finger qpos values.
    native_state[:, 30:39] = state[:, :9] * state_dim_mask[:, :9]
    native_state_mask[:, 30:39] = state_dim_mask[:, :9]
    native_state[:, 10:12] = state[:, 9:11] * state_dim_mask[:, 9:11]
    native_state_mask[:, 10:12] = state_dim_mask[:, 9:11]

    # Compact action: dxyz + relative ortho6D + raw gripper command.
    native_action_mask[:, 30:39] = action_dim_mask[:, :9]
    native_action_mask[:, 10] = action_dim_mask[:, 9]
    return native_state, native_state_mask, native_action_mask


def native_rdt_action_to_libero_10d(actions: np.ndarray) -> np.ndarray:
    """Extract the supervised LIBERO 10D command from native RDT output."""
    values = np.asarray(actions)
    if values.shape[-1] != 128:
        raise ValueError(
            f"Expected native RDT action width 128, got {values.shape[-1]}"
        )
    return np.concatenate(
        [values[..., 30:33], values[..., 33:39], values[..., 10:11]],
        axis=-1,
    )


def native_rdt_action_to_libero_7d(actions: np.ndarray) -> np.ndarray:
    """Extract Libero_RDT's EEF-delta and binary gripper action slots."""
    values = np.asarray(actions)
    if values.shape[-1] != 128:
        raise ValueError(
            f"Expected native RDT action width 128, got {values.shape[-1]}"
        )
    result = np.concatenate(
        [values[..., 39:45], values[..., 10:11]],
        axis=-1,
    ).astype(np.float32)
    result[..., :6] = np.clip(result[..., :6], -1.0, 1.0)
    result[..., 6] = np.where(result[..., 6] < 0.0, -1.0, 1.0)
    return result


def apply_demo_action_override(
    model_action: np.ndarray,
    demo_action: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Replace only motion or gripper with the aligned demonstration command."""
    predicted = np.asarray(model_action, dtype=np.float32)
    reference = np.asarray(demo_action, dtype=np.float32)
    if predicted.shape != (7,) or reference.shape[0] < 7:
        raise ValueError(
            "Causal LIBERO action override expects model [7] and demo [>=7], "
            f"got {predicted.shape} and {reference.shape}"
        )
    result = predicted.copy()
    if mode == "none":
        return result
    if mode == "gripper":
        result[6] = reference[6]
        return result
    if mode == "motion":
        result[:6] = reference[:6]
        return result
    raise ValueError(f"Unknown demonstration action override mode: {mode!r}")


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
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=None,
        help="Optional merged/full RDT base loaded before the trainable artifact.",
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
    parser.add_argument("--cache-root", type=Path, default=Path("cache_features/libero_spatial/full"))
    parser.add_argument("--action-stats", type=Path, default=Path("dataset/LIBERO/Spatial/datasets/libero_spatial/audit.json"))
    parser.add_argument(
        "--action-output-mode",
        choices=["raw_delta_ortho6d", "absolute_target_state", "normalized_delta"],
        default="raw_delta_ortho6d",
        help=(
            "Use raw_delta_ortho6d for the 10D LIBERO command model. The other "
            "modes are retained for legacy checkpoints."
        ),
    )
    parser.add_argument(
        "--target-state-start-index",
        type=int,
        default=1,
        help=(
            "First predicted target-state token to execute. Training target[0] "
            "is the current state, so 1 avoids a deliberate no-op."
        ),
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
    parser.add_argument("--libero-root", type=Path, default=LIBERO_ROOT_DEFAULT)
    parser.add_argument("--task-id", type=int, default=0, choices=range(10))
    parser.add_argument(
        "--instruction",
        default=None,
        help="Override the benchmark task language while keeping the same environment.",
    )
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument(
        "--demo-hdf5",
        type=Path,
        help="Start from the first recorded simulator state of this HDF5 demo instead of a benchmark init state.",
    )
    parser.add_argument("--demo-name", default="demo_0")
    parser.add_argument(
        "--demo-action-override",
        choices=["none", "gripper", "motion"],
        default="none",
        help=(
            "Closed-loop causal ablation. Replace only the executed gripper "
            "command or the six motion commands with the aligned HDF5 demo "
            "action while the model continues observing and replanning."
        ),
    )
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
    parser.add_argument(
        "--clean-x0-gripper",
        action="store_true",
        help=(
            "Use dimension 9 from the final clean-x0 model prediction for the "
            "LIBERO gripper while retaining DPM-Solver output for motion."
        ),
    )
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
        "--qwen-extraction",
        choices=("auto", "b0", "b2", "b3"),
        default="auto",
        help=(
            "Use B0's single </think> KV token or the shared B2/B3 "
            "LatentStudent five-token KV/hidden/waypoint extraction. Auto "
            "detects b3, then b2, from checkpoint/config paths."
        ),
    )
    parser.add_argument("--qwen-layer-index", type=int, default=7)
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
        "--student-model-id",
        default=None,
        help="B2/B3 LatentStudent checkpoint or Hub id.",
    )
    parser.add_argument(
        "--processor-id",
        default=None,
        help="B2/B3 processor path/id; defaults to --student-model-id.",
    )
    parser.add_argument(
        "--latent-student-code-dir",
        type=Path,
        default=Path("/home/ubuntu/VLA-FYP/train/stage2"),
    )
    parser.add_argument("--spatial-parameters-path", type=Path, default=None)
    parser.add_argument("--latent-count", type=int, default=6)
    parser.add_argument("--spatial-token-count", type=int, default=5)
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--student-precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--b2-prompt-template", default=B2_TRAJECTORY_PROMPT)
    parser.add_argument(
        "--siglip-model-id",
        default=None,
        help="Override SigLIP model path/id. Defaults to cache metadata, then google/siglip-so400m-patch14-384.",
    )
    parser.add_argument(
        "--action-debug-jsonl",
        type=Path,
        help=(
            "Write per-replan model outputs and final LIBERO actions to this JSONL file."
        ),
    )
    parser.add_argument(
        "--observation-debug-npz",
        type=Path,
        help=(
            "Save synchronized policy agent-view/wrist frames and observed "
            "11D policy states plus full MuJoCo simulator states for notebook "
            "diagnostics."
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
    if args.pretrained_only or args.disable_qwen_fusion:
        cfg = replace(cfg, model=replace(cfg.model, qwen_fusion="none"))
    if args.action_chunk <= 0:
        raise ValueError("--action-chunk must be positive")
    if args.action_chunk > cfg.model.pred_horizon:
        raise ValueError(
            f"--action-chunk ({args.action_chunk}) cannot exceed "
            f"cfg.model.pred_horizon ({cfg.model.pred_horizon})"
        )
    if (
        cfg.model.native_rdt_128_mapping == "libero_joint_eef_delta"
        and args.action_output_mode != "raw_delta_ortho6d"
    ):
        raise ValueError(
            "libero_joint_eef_delta outputs raw 7-D controller commands and "
            "requires --action-output-mode raw_delta_ortho6d"
        )
    if args.target_state_start_index < 0:
        raise ValueError("--target-state-start-index must be non-negative")
    if args.demo_action_override != "none" and args.demo_hdf5 is None:
        raise ValueError(
            "--demo-action-override requires --demo-hdf5 and --demo-name"
        )
    max_delta_pos = None if args.max_delta_pos < 0 else args.max_delta_pos
    max_delta_rot = None if args.max_delta_rot < 0 else args.max_delta_rot
    stats = None
    if args.action_output_mode == "normalized_delta":
        stats = load_action_stats(args.action_stats)
        print("Action denormalization stats:")
        print(json.dumps({
            "dim_names": ACTION_DIM_NAMES[: len(stats.q01)],
            "q01": stats.q01.astype(float).tolist(),
            "q99": stats.q99.astype(float).tolist(),
            "eps": stats.eps,
        }, indent=2))
    if (
        args.action_output_mode != "raw_delta_ortho6d"
        and cfg.model.action_encoder_layout == "libero_ortho6d"
    ):
        raise ValueError(
            "libero_ortho6d checkpoints must use --action-output-mode "
            "raw_delta_ortho6d"
        )
    metadata = load_feature_metadata(args.cache_root)
    qwen_extraction_mode = resolve_qwen_extraction_mode(
        args.qwen_extraction,
        checkpoint=args.checkpoint,
        config=args.config,
    )
    if (
        cfg.model.qwen_fusion == "hidden_waypoint_cross_attention"
        and qwen_extraction_mode == "b0"
    ):
        raise ValueError(
            "hidden_waypoint_cross_attention requires --qwen-extraction b2 or "
            "b3 plus the matching --student-model-id"
        )
    if (
        cfg.model.qwen_fusion == "hidden_cross_attention"
        and qwen_extraction_mode != "b0"
    ):
        raise ValueError("hidden_cross_attention requires --qwen-extraction b0")
    qwen_id = args.qwen_model_id or metadata.get("qwen_model_id", "shreethar/stage1_unsloth")
    qwen_processor_id = args.qwen_processor_id or metadata.get("qwen_processor_id", qwen_id)
    student_model_id = args.student_model_id or metadata.get("student_model_id")
    student_processor_id = (
        args.processor_id
        or metadata.get("processor_id")
        or student_model_id
    )
    siglip_id = args.siglip_model_id or metadata.get("siglip_model_id", "google/siglip-so400m-patch14-384")
    t5_id = (
        args.t5_model_id
        or metadata.get("t5_model_id")
        or "/home/ubuntu/RoboticsDiffusionTransformer/google/t5-v1_1-xxl"
    )

    benchmark = get_benchmark(args.benchmark)(0)
    task = benchmark.get_task(args.task_id)
    instruction = args.instruction or task.language
    use_qwen = cfg.model.qwen_fusion != "none"
    print(
        "Loading T5, SigLIP, and optional "
        f"Qwen-{qwen_extraction_mode.upper()} encoders..."
    )
    t5_tokenizer, t5 = load_t5_encoder(
        model_id=t5_id,
        fallback_model_id=args.t5_fallback_model_id,
        precision=args.t5_precision,
        device_map=args.device_map,
        cfg=cfg,
    )
    qwen_processor = None
    qwen = None
    latent_student = None
    extract_latent_student_spatial_kv = None
    if use_qwen:
        if qwen_extraction_mode in {"b2", "b3"}:
            if student_model_id is None:
                raise ValueError(
                    f"--qwen-extraction {qwen_extraction_mode} requires "
                    "--student-model-id (or student_model_id in cache metadata)"
                )
            from precompute_latent_student_kv import (
                extract_latent_student_spatial_kv as extract_spatial_features,
                load_student_and_processor,
            )
            from run_precompute_32frame_episode_packs_latent_student_kv import (
                validate_student_runtime_contract,
            )

            args.student_model_id = student_model_id
            args.processor_id = student_processor_id
            args.layer_index = args.qwen_layer_index
            latent_student, qwen_processor = load_student_and_processor(args, device)
            validate_student_runtime_contract(
                latent_student,
                qwen_processor,
                args=args,
                cfg=cfg,
            )
            extract_latent_student_spatial_kv = extract_spatial_features
        else:
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
    model.decode_clean_x0_gripper = bool(args.clean_x0_gripper)
    print(
        "Gripper decoding:",
        "final clean x0" if model.decode_clean_x0_gripper else "diffusion solver",
    )
    model.to(device).eval()

    env = OffScreenRenderEnv(
        bddl_file_name=benchmark.get_task_bddl_file_path(args.task_id),
        camera_heights=128,
        camera_widths=128,
        horizon=args.max_steps + 10,
    )
    observation = env.reset()
    demo_actions: np.ndarray | None = None
    if args.demo_hdf5 is not None:
        import h5py

        with h5py.File(args.demo_hdf5, "r") as handle:
            demo = handle["data"][args.demo_name]
            recorded_state = np.asarray(demo["states"][0], dtype=np.float64)
            if args.demo_action_override != "none":
                if "actions" not in demo:
                    raise KeyError(
                        f"{demo.name} has no actions for causal override"
                    )
                demo_actions = np.asarray(demo["actions"], dtype=np.float32)[:, :7]
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
    cached_qwen_hidden_states: torch.Tensor | None = None
    cached_latent_waypoints: torch.Tensor | None = None
    success = False
    simulator_step = 0
    plan_index = 0
    start = time.perf_counter()
    action_debug_handle = None
    initial_converted = libero_observation_to_rdt(observation)
    debug_states = [initial_converted["state"].copy()]
    # Keep the complete MuJoCo state alongside the compact 11D policy state.
    # This is only persisted when --observation-debug-npz is requested and is
    # useful for exact-state, full-episode trajectory diagnostics.
    debug_sim_states = [np.asarray(env.get_sim_state()).copy()]
    debug_agent_frames = [np.asarray(initial_converted["primary"]).copy()]
    debug_wrist_frames = (
        []
        if initial_converted["wrist"] is None
        else [np.asarray(initial_converted["wrist"]).copy()]
    )
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
            if use_qwen and (cached_qwen is None or plan_index % args.qwen_refresh_every == 0):
                assert qwen_processor is not None
                if qwen_extraction_mode in {"b2", "b3"}:
                    if (
                        latent_student is None
                        or extract_latent_student_spatial_kv is None
                    ):
                        raise RuntimeError("LatentStudent extractor was not loaded")
                    (
                        cached_qwen,
                        cached_qwen_hidden_states,
                        cached_latent_waypoints,
                    ) = extract_latent_student_spatial_kv(
                        encoded,
                        student=latent_student,
                        processor=qwen_processor,
                        device=device,
                        layer_index=args.qwen_layer_index,
                        expected_dim=cfg.model.qwen_kv_dim,
                        spatial_token_count=args.spatial_token_count,
                        prompt_template=args.b2_prompt_template,
                    )
                else:
                    assert qwen is not None
                    b0_features = extract_qwen_kv(
                        encoded,
                        qwen_processor,
                        qwen,
                        device=device,
                        layer_index=args.qwen_layer_index,
                        max_new_tokens=args.qwen_max_new_tokens,
                        expected_dim=cfg.model.qwen_kv_dim,
                        stop_at_think_end=bool(metadata.get("qwen_stop_at_think", True)),
                        prompt_template=metadata.get("qwen_trajectory_prompt_template"),
                        enable_thinking=bool(metadata.get("qwen_enable_thinking", False)),
                        return_hidden_state=(
                            cfg.model.qwen_fusion == "hidden_cross_attention"
                        ),
                        think_token_selector="think_end",
                    )
                    if isinstance(b0_features, tuple):
                        cached_qwen, cached_qwen_hidden_states = b0_features
                    else:
                        cached_qwen = b0_features
            img_tokens, img_mask = extract_siglip_features(
                encoded,
                siglip_processor,
                siglip,
                max_img_tokens=cfg.model.image_tokens,
                expected_dim=cfg.model.img_token_dim,
                device=device,
                encode_invalid_slots=(
                    cfg.model.state_encoder_layout == "rdt_native_128"
                    and cfg.model.native_rdt_128_mapping
                    == "libero_joint_eef_delta"
                ),
            )
            state = encoded["state"]
            state_dim_mask = encoded["state_dim_mask"]
            action_dim_mask = encoded["action_dim_mask"]
            if cfg.model.state_encoder_layout == "rdt_native_128":
                if (
                    cfg.model.native_rdt_128_mapping
                    == "libero_joint_eef_delta"
                ):
                    action_dim_mask = torch.ones(
                        state.shape[0],
                        7,
                        dtype=action_dim_mask.dtype,
                    )
                state, state_dim_mask, action_dim_mask = native_rdt_policy_inputs(
                    state,
                    state_dim_mask,
                    action_dim_mask,
                    mapping=cfg.model.native_rdt_128_mapping,
                    joint_state=encoded.get("joint_state"),
                )
            batch = {
                "state": state.to(device),
                "state_dim_mask": state_dim_mask.to(device),
                "action_dim_mask": action_dim_mask.to(device),
                "ctrl_freq": encoded["ctrl_freq"].to(device),
                "lang_tokens": lang_tokens.to(device),
                "lang_mask": lang_mask.to(device),
                "img_tokens": img_tokens,
                "img_mask": img_mask,
            }
            if use_qwen:
                assert cached_qwen is not None
                batch["qwen_kv"] = cached_qwen
                if cfg.model.qwen_fusion in {
                    "hidden_cross_attention",
                    "hidden_waypoint_cross_attention",
                }:
                    if cached_qwen_hidden_states is None:
                        raise RuntimeError(
                            "Hidden fusion is enabled but online hidden states "
                            "are unavailable"
                        )
                    expected_hidden_shape = (
                        1,
                        cfg.model.spatial_token_count,
                        cfg.model.qwen_hidden_size,
                    )
                    if tuple(cached_qwen_hidden_states.shape) != expected_hidden_shape:
                        raise ValueError(
                            "Online spatial hidden states have shape "
                            f"{tuple(cached_qwen_hidden_states.shape)}, expected "
                            f"{expected_hidden_shape}"
                        )
                    batch["qwen_hidden_states"] = cached_qwen_hidden_states
                    if cfg.model.qwen_fusion == "hidden_waypoint_cross_attention":
                        if cached_latent_waypoints is None:
                            raise RuntimeError("Online latent waypoints are unavailable")
                        expected_waypoint_shape = (
                            1,
                            cfg.model.spatial_token_count,
                            cfg.model.waypoint_dim,
                        )
                        if tuple(cached_latent_waypoints.shape) != expected_waypoint_shape:
                            raise ValueError(
                                "Online latent waypoints have shape "
                                f"{tuple(cached_latent_waypoints.shape)}, expected "
                                f"{expected_waypoint_shape}"
                            )
                        batch["latent_waypoints"] = cached_latent_waypoints
                    batch["plan_mask"] = torch.ones(
                        cached_qwen_hidden_states.shape[:2],
                        dtype=torch.bool,
                        device=cached_qwen_hidden_states.device,
                    )
            torch.manual_seed(args.seed + plan_index)
            native_model_output = model.sample_actions(batch)[0].float().cpu().numpy()
            if cfg.model.action_encoder_layout == "rdt_native_128":
                if (
                    cfg.model.native_rdt_128_mapping
                    == "libero_joint_eef_delta"
                ):
                    model_output = native_rdt_action_to_libero_7d(
                        native_model_output
                    )
                else:
                    model_output = native_rdt_action_to_libero_10d(
                        native_model_output
                    )
            else:
                model_output = native_model_output
            if args.action_output_mode == "raw_delta_ortho6d":
                denormalized = None
                if (
                    cfg.model.native_rdt_128_mapping
                    == "libero_joint_eef_delta"
                ):
                    actions = model_output
                else:
                    actions = rdt_action_to_libero(model_output)
            elif args.action_output_mode == "normalized_delta":
                assert stats is not None
                denormalized = denormalize_action_array(model_output, stats)
                actions = rdt_action_to_libero(model_output, stats)
            else:
                denormalized = None
                plan_start_state = rdt_state_open_from_libero(observation)
                actions = np.stack(
                    [
                        absolute_target_state_to_libero_action(
                            model_output[min(index + args.target_state_start_index, len(model_output) - 1)],
                            plan_start_state,
                            pos_scale=args.pos_scale,
                            rot_scale=args.rot_scale,
                            max_delta_pos=max_delta_pos,
                            max_delta_rot=max_delta_rot,
                        )
                        for index in range(len(model_output))
                    ],
                    axis=0,
                )
            if not np.isfinite(actions).all():
                raise FloatingPointError("RDT produced NaN/Inf actions")

            chunk = min(args.action_chunk, args.max_steps - simulator_step)
            debug_row: dict[str, Any] | None = None
            if action_debug_handle is not None:
                debug_row = {
                    "plan_index": plan_index,
                    "simulator_step_start": simulator_step,
                    "pred_horizon": int(cfg.model.pred_horizon),
                    "executed_steps": int(chunk),
                    "dim_names": (
                        [
                            "dx", "dy", "dz",
                            "rot6d_0", "rot6d_1", "rot6d_2",
                            "rot6d_3", "rot6d_4", "rot6d_5",
                            "raw_gripper_command",
                        ]
                        if model_output.shape[-1] == LIBERO_ACTION_DIM
                        else ACTION_DIM_NAMES[: model_output.shape[-1]]
                    ),
                    "action_output_mode": args.action_output_mode,
                    "model_output_stats": action_debug_stats(model_output),
                    "model_outputs": array_to_nested_float_list(model_output),
                    "planned_libero_action_stats": action_debug_stats(actions),
                    "planned_libero_actions_at_replan_start": array_to_nested_float_list(actions),
                }
                if cfg.model.action_encoder_layout == "rdt_native_128":
                    debug_row["native_128_model_output_stats"] = (
                        action_debug_stats(native_model_output)
                    )
                if args.action_output_mode == "normalized_delta":
                    assert stats is not None and denormalized is not None
                    debug_row["normalization"] = {
                        "q01": stats.q01.astype(float).tolist(),
                        "q99": stats.q99.astype(float).tolist(),
                        "formula": "denormalized = (clip(model_output,-1,1)+1)*0.5*(q99-q01)+q01",
                    }
                    debug_row["denormalized_delta_stats"] = action_debug_stats(denormalized)
                    debug_row["denormalized_delta_actions"] = array_to_nested_float_list(denormalized)
                elif args.action_output_mode == "absolute_target_state":
                    debug_row["absolute_target_state_conversion"] = {
                        "target_state_start_index": int(args.target_state_start_index),
                        "pos_scale": float(args.pos_scale),
                        "rot_scale": float(args.rot_scale),
                        "max_delta_pos": max_delta_pos,
                        "max_delta_rot": max_delta_rot,
                        "gripper": (
                            "legacy absolute-state binary conversion at dim 6; "
                            "not used by raw_delta_ortho6d"
                        ),
                    }
                elif (
                    cfg.model.native_rdt_128_mapping
                    == "libero_joint_eef_delta"
                ):
                    debug_row["raw_eef_delta_conversion"] = {
                        "motion": "native slots [39:45], clipped to [-1,1]",
                        "gripper": (
                            "native slot [10], thresholded to -1 below zero "
                            "and +1 otherwise"
                        ),
                    }
                else:
                    debug_row["raw_delta_ortho6d_conversion"] = {
                        "translation": "dims 0:3 copied and clipped to [-1,1]",
                        "rotation": "dims 3:9 Gram-Schmidt -> rotvec / 0.5",
                        "gripper": "dim 9 copied unchanged apart from [-1,1] clipping",
                    }
            executed_actions: list[np.ndarray] = []
            override_reference_actions: list[np.ndarray] = []
            observed_states = [
                libero_observation_to_rdt(observation)["state"].copy()
            ]
            for action_index in range(chunk):
                if args.action_output_mode == "absolute_target_state":
                    target_index = min(action_index + args.target_state_start_index, len(model_output) - 1)
                    current_state = rdt_state_open_from_libero(observation)
                    action = absolute_target_state_to_libero_action(
                        model_output[target_index],
                        current_state,
                        pos_scale=args.pos_scale,
                        rot_scale=args.rot_scale,
                        max_delta_pos=max_delta_pos,
                        max_delta_rot=max_delta_rot,
                    )
                else:
                    action = actions[action_index]
                if args.demo_action_override != "none":
                    assert demo_actions is not None
                    if simulator_step >= len(demo_actions):
                        raise IndexError(
                            "Policy rollout exceeded the available aligned "
                            f"demonstration actions ({len(demo_actions)})"
                        )
                    reference_action = demo_actions[simulator_step]
                    action = apply_demo_action_override(
                        action,
                        reference_action,
                        args.demo_action_override,
                    )
                    override_reference_actions.append(reference_action.copy())
                executed_actions.append(action.copy())
                # Keep adjacent t-1/t frames for the next SigLIP history, matching
                # the feature-precomputation contract even when actions are chunked.
                previous_observation = observation
                observation, reward, done, _ = env.step(action)
                converted_observation = libero_observation_to_rdt(observation)
                observed_states.append(converted_observation["state"].copy())
                debug_states.append(converted_observation["state"].copy())
                debug_sim_states.append(np.asarray(env.get_sim_state()).copy())
                debug_agent_frames.append(
                    np.asarray(converted_observation["primary"]).copy()
                )
                if converted_observation["wrist"] is not None:
                    debug_wrist_frames.append(
                        np.asarray(converted_observation["wrist"]).copy()
                    )
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
            if action_debug_handle is not None and debug_row is not None:
                debug_row["executed_libero_actions"] = array_to_nested_float_list(np.stack(executed_actions, axis=0))
                debug_row["observed_rdt_states"] = array_to_nested_float_list(
                    np.stack(observed_states, axis=0)
                )
                debug_row["demo_action_override"] = args.demo_action_override
                if override_reference_actions:
                    debug_row["override_reference_actions"] = (
                        array_to_nested_float_list(
                            np.stack(override_reference_actions, axis=0)
                        )
                    )
                action_debug_handle.write(json.dumps(debug_row) + "\n")
                action_debug_handle.flush()
            plan_index += 1
            print(
                f"plan={plan_index} simulator_step={simulator_step} "
                f"success={success} elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    finally:
        if action_debug_handle is not None:
            action_debug_handle.close()
        if args.observation_debug_npz is not None:
            args.observation_debug_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.observation_debug_npz,
                states=np.stack(debug_states),
                sim_states=np.stack(debug_sim_states),
                agent_frames=np.stack(debug_agent_frames),
                wrist_frames=(
                    np.stack(debug_wrist_frames)
                    if debug_wrist_frames
                    else np.empty((0, 0, 0, 3), dtype=np.uint8)
                ),
            )
        writer.close()
        env.close()

    summary = {
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "instruction": instruction,
        "checkpoint": "pretrained-only" if args.pretrained_only else str(args.checkpoint.resolve()),
        "pretrained_only": bool(args.pretrained_only),
        "init_state_index": state_index,
        "steps": simulator_step,
        "plans": plan_index,
        "success": success,
        "demo_action_override": args.demo_action_override,
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
