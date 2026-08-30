#!/usr/bin/env python3
"""Evaluate a B0 or B2 OXE RDT checkpoint in SimplerEnv.

The policy and simulator intentionally run in separate processes.  ThinkFlow
needs Python 3.12 and its PyTorch/Transformers stack, while the official
SimplerEnv main branch uses Python 3.10/3.11, SAPIEN 2, and ManiSkill2.

Modes:
  contract  Check state packing and action decoding without loading models.
  probe     Run one policy inference on a supplied/static SimplerEnv image.
  rollout   Run one closed-loop SimplerEnv episode through the isolated worker.
  suite     Evaluate the five canonical Google/Fractal task families.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _path in (SRC_ROOT, REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

RDT_GRIPPER_INDEX = 10
RDT_XYZ_SLICE = slice(30, 33)
RDT_ORTHO6D_SLICE = slice(33, 39)
DEFAULT_SIMPLER_ROOT = Path("/home/ubuntu/SimplerEnv")
DEFAULT_SIMPLER_PYTHON = DEFAULT_SIMPLER_ROOT / ".venv/bin/python"
DEFAULT_CHECKPOINT = REPO_ROOT / "output_2/checkpoint-20000"
DEFAULT_CONFIG = REPO_ROOT / "configs/part3_rdt1b.yaml"
DEFAULT_QWEN = REPO_ROOT / "model/model/stage1_unsloth"
DEFAULT_B2_STUDENT = REPO_ROOT / "model/LatentStudent-ckpt-400-fixed"
DEFAULT_B2_CODE_DIR = Path("/home/ubuntu/VLA-FYP/train/stage2")
B2_TRAJECTORY_PROMPT = (
    "You are a robot manipulation assistant. Given an observation image and a "
    "task instruction, predict the end-effector's 2D trajectory as 5 waypoints. "
    "Output ONLY the coordinate list in this exact format: "
    "[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]\n\n"
    "Task: The task is {task}. What is the trajectory that the end effector should take?"
)
AF_UNIX_SAFE_PATH_BYTES = 103
FRACTAL_SUITE_TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
)


def resolve_qwen_extraction_mode(
    requested: str,
    *,
    checkpoint: str | Path,
    config: str | Path,
) -> str:
    """Resolve B0/B2 extraction while keeping an explicit override available."""
    if requested in {"b0", "b2"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported Qwen extraction mode: {requested!r}")
    names = f"{Path(checkpoint)} {Path(config)}".lower()
    return "b2" if "b2" in names else "b0"


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload)) + "\n")


def euler_xyz_to_ortho6d_numpy(euler: np.ndarray) -> np.ndarray:
    values = np.asarray(euler, dtype=np.float32)
    matrix = Rotation.from_euler("xyz", values.reshape(-1, 3)).as_matrix()
    encoded = np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)
    return encoded.reshape(*values.shape[:-1], 6).astype(np.float32)


def ortho6d_to_matrix(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    first = values[..., :3]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    default_first = np.zeros_like(first)
    default_first[..., 0] = 1.0
    first = np.where(first_norm > eps, first, default_first)
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), eps)

    second = values[..., 3:6]
    second -= np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    fallback_index = np.argmin(np.abs(first), axis=-1)
    fallback = np.eye(3, dtype=np.float64)[fallback_index]
    fallback -= np.sum(first * fallback, axis=-1, keepdims=True) * first
    fallback /= np.maximum(np.linalg.norm(fallback, axis=-1, keepdims=True), eps)
    second = np.where(second_norm > eps, second, fallback)
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), eps)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-1).astype(np.float32)


def pack_oxe_state(state_7d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack [xyz, Euler XYZ, gripper_closed] into native RDT-128."""
    state = np.asarray(state_7d, dtype=np.float32)
    if state.shape != (7,):
        raise ValueError(f"Expected state [7], got {state.shape}")
    packed = np.zeros(128, dtype=np.float32)
    mask = np.zeros(128, dtype=np.float32)
    packed[RDT_XYZ_SLICE] = state[:3]
    packed[RDT_ORTHO6D_SLICE] = euler_xyz_to_ortho6d_numpy(state[3:6])
    packed[RDT_GRIPPER_INDEX] = 1.0 - state[6]
    mask[RDT_XYZ_SLICE] = 1.0
    mask[RDT_ORTHO6D_SLICE] = 1.0
    mask[RDT_GRIPPER_INDEX] = 1.0
    return packed, mask


@dataclass(frozen=True)
class ActionStats:
    q01: np.ndarray
    q99: np.ndarray


@dataclass
class GoogleGripperTargetAdapter:
    """Convert an absolute open target into Google's relative command.

    OXE preprocessing turns Fractal's command stream into a persistent binary
    target. SimplerEnv's Google controller instead consumes ``+1`` to close,
    ``-1`` to open, and ``0`` to hold. A sticky transition gives the physical
    gripper enough controller steps to finish moving if the prediction chatters.
    """

    sticky_steps: int
    assumed_open: bool
    active_command: float = 0.0
    remaining_steps: int = 0

    def command(self, desired_open: bool) -> float:
        desired_open = bool(desired_open)
        if self.sticky_steps <= 0:
            # Preserve the original stateless evaluator behavior when disabled.
            return -1.0 if desired_open else 1.0

        if self.remaining_steps > 0:
            command = self.active_command
            self.remaining_steps -= 1
            if self.remaining_steps == 0:
                self.assumed_open = command < 0.0
                self.active_command = 0.0
            return command

        if desired_open == self.assumed_open:
            return 0.0

        self.active_command = -1.0 if desired_open else 1.0
        self.remaining_steps = self.sticky_steps - 1
        command = self.active_command
        if self.remaining_steps == 0:
            self.assumed_open = desired_open
            self.active_command = 0.0
        return command


def load_stats(path: Path) -> ActionStats:
    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload.get("action_normalization", payload)
    q01 = np.asarray(block["q01"], dtype=np.float32)
    q99 = np.asarray(block["q99"], dtype=np.float32)
    if q01.shape != (7,) or q99.shape != (7,):
        raise ValueError(f"Expected 7-D action statistics in {path}")
    return ActionStats(q01=q01, q99=q99)


def denormalize_xyz(normalized_xyz: np.ndarray, stats: ActionStats) -> np.ndarray:
    values = np.clip(np.asarray(normalized_xyz, dtype=np.float32), -1.0, 1.0)
    return ((values + 1.0) * 0.5 * (stats.q99[:3] - stats.q01[:3]) + stats.q01[:3]).astype(np.float32)


def robot_family(task: str) -> str:
    lowered = task.lower()
    if "google_robot" in lowered:
        return "google_robot"
    if "widowx" in lowered or "bridge" in lowered:
        return "widowx"
    raise ValueError(
        "Cannot infer robot family from task. Use a standard google_robot_* or "
        "widowx/Bridge SimplerEnv task."
    )


def default_dataset(task: str) -> str:
    return "fractal" if robot_family(task) == "google_robot" else "bridge"


def default_stats_path(dataset: str) -> Path:
    return REPO_ROOT / f"dataset/mock_dataset/{dataset}_dataset/audit.json"


def decode_native_actions(
    native_actions: np.ndarray,
    *,
    dataset: str,
    task: str,
    stats: ActionStats,
    rotation_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Decode RDT-128 output into SimplerEnv's 7-D controller action.

    The OXE model learned normalized XYZ, physical Euler rotations encoded as
    orthogonal-6D, and a gripper-open score.  SimplerEnv expects XYZ delta,
    axis-angle delta, and an embodiment-specific binary gripper command.
    """
    values = np.asarray(native_actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 128:
        raise ValueError(f"Expected [horizon, 128], got {values.shape}")
    normalized_xyz = values[:, RDT_XYZ_SLICE]
    xyz = denormalize_xyz(normalized_xyz, stats)
    rotation_6d = values[:, RDT_ORTHO6D_SLICE]
    matrices = ortho6d_to_matrix(rotation_6d)
    euler_xyz = Rotation.from_matrix(matrices).as_euler("xyz").astype(np.float32)
    family = robot_family(task)
    if family == "google_robot":
        # Fractal/RT-1 supplies the three learned rotation-delta numbers
        # directly to this controller.  This deliberately inverts the exact
        # numerical training transform rather than reinterpreting it.
        environment_rotation = euler_xyz.copy()
    else:
        # Bridge/WidowX's controller consumes a true axis-angle (rotvec).
        environment_rotation = Rotation.from_euler("xyz", euler_xyz).as_rotvec().astype(np.float32)
    environment_rotation = (environment_rotation * float(rotation_scale)).astype(np.float32)

    gripper_open_score = values[:, RDT_GRIPPER_INDEX]
    gripper_open = gripper_open_score >= 0.0
    if family == "google_robot":
        # Google Robot: +1 closes, -1 opens.
        environment_gripper = np.where(gripper_open, -1.0, 1.0).astype(np.float32)
    else:
        # WidowX: +1 opens, -1 closes.
        environment_gripper = np.where(gripper_open, 1.0, -1.0).astype(np.float32)
    environment_action = np.concatenate(
        [xyz, environment_rotation, environment_gripper[:, None]], axis=-1
    ).astype(np.float32)
    return {
        "native_relevant_10d": np.concatenate(
            [normalized_xyz, rotation_6d, gripper_open_score[:, None]], axis=-1
        ),
        "normalized_xyz": normalized_xyz,
        "denormalized_xyz": xyz,
        "rotation_6d": rotation_6d,
        "decoded_euler_xyz": euler_xyz,
        "environment_rotation": environment_rotation,
        "gripper_open_score": gripper_open_score,
        "gripper_open_binary": gripper_open.astype(np.float32),
        "environment_action": environment_action,
    }


def build_policy_sample(
    *,
    image: np.ndarray,
    previous_image: np.ndarray | None,
    state_7d: np.ndarray,
    instruction: str,
    dataset: str,
    horizon: int,
    control_frequency: float,
) -> dict[str, Any]:
    current = {
        "primary": Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB"),
        "wrist": None,
        "secondary": None,
    }
    if previous_image is None:
        previous = current
        previous_mask = {"primary": 0, "wrist": 0, "secondary": 0}
    else:
        previous = {
            "primary": Image.fromarray(np.asarray(previous_image, dtype=np.uint8)).convert("RGB"),
            "wrist": None,
            "secondary": None,
        }
        previous_mask = {"primary": 1, "wrist": 0, "secondary": 0}
    current_mask = {"primary": 1, "wrist": 0, "secondary": 0}
    return {
        "dataset_id": dataset,
        "episode_id": "simpler_rollout",
        "step_idx": "0",
        "instruction": instruction,
        "images": current,
        "image_mask": current_mask,
        "image_history": [previous, current],
        "image_history_mask": [previous_mask, current_mask],
        "state": np.asarray(state_7d, dtype=np.float32),
        "state_mask": np.ones(7, dtype=np.float32),
        "actions": np.zeros((horizon, 7), dtype=np.float32),
        "actions_mask": np.ones(horizon, dtype=np.float32),
        "action_dim_mask": np.ones(7, dtype=np.float32),
        "ctrl_freq": float(control_frequency),
    }


class PolicyEngine:
    def __init__(self, args: argparse.Namespace, instruction: str, dataset: str):
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            SiglipImageProcessor,
            SiglipVisionModel,
        )

        from precompute_all_features import (
            extract_qwen_kv,
            extract_siglip_features,
            extract_t5_features,
            standardized_collate_fn,
        )
        from rollout_libero_rdt import load_t5_encoder, t5_device_from_encoder
        from thinkflow_rdt.checkpoint import load_trainable_artifact
        from thinkflow_rdt.config import load_config
        from thinkflow_rdt.model import SFTConditionedRDT

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for B0 OXE inference")
        self.torch = torch
        self.device = torch.device("cuda")
        self.args = args
        self.dataset = dataset
        self.instruction = instruction
        self.qwen_extraction_mode = resolve_qwen_extraction_mode(
            args.qwen_extraction,
            checkpoint=args.checkpoint,
            config=args.config,
        )
        self.extract_qwen_kv = extract_qwen_kv
        self.extract_siglip_features = extract_siglip_features
        self.standardized_collate_fn = standardized_collate_fn
        self.cfg = load_config(args.config)
        if self.cfg.model.qwen_fusion == "none":
            raise ValueError("The selected config disables Qwen fusion; this is not B0 evaluation")

        print(
            "Loading frozen T5, "
            f"Qwen-{self.qwen_extraction_mode.upper()}, and SigLIP encoders...",
            flush=True,
        )
        self.t5_tokenizer, self.t5 = load_t5_encoder(
            model_id=args.t5_model_id,
            fallback_model_id=None,
            precision=args.t5_precision,
            device_map=args.device_map,
            cfg=self.cfg,
        )
        self.extract_t5_features = extract_t5_features
        self.t5_device = t5_device_from_encoder(self.t5, self.device)
        self.set_instruction(instruction)
        self.qwen = None
        self.latent_student = None
        self.extract_latent_student_spatial_kv = None
        if self.qwen_extraction_mode == "b2":
            from precompute_latent_student_kv import (
                extract_latent_student_spatial_kv,
                load_student_and_processor,
            )
            from run_precompute_32frame_episode_packs_latent_student_kv import (
                validate_student_runtime_contract,
            )

            # The shared precompute validator names this field ``layer_index``.
            args.layer_index = args.qwen_layer_index
            self.latent_student, self.qwen_processor = load_student_and_processor(
                args,
                self.device,
            )
            validate_student_runtime_contract(
                self.latent_student,
                self.qwen_processor,
                args=args,
                cfg=self.cfg,
            )
            self.extract_latent_student_spatial_kv = extract_latent_student_spatial_kv
        else:
            self.qwen_processor = AutoProcessor.from_pretrained(args.qwen_model_id)
            self.qwen_processor.tokenizer.padding_side = "left"
            self.qwen = AutoModelForImageTextToText.from_pretrained(
                args.qwen_model_id,
                torch_dtype=torch.bfloat16,
                device_map=args.device_map,
                attn_implementation="sdpa",
            ).eval()
        self.siglip_processor = SiglipImageProcessor.from_pretrained(args.siglip_model_id)
        self.siglip = SiglipVisionModel.from_pretrained(
            args.siglip_model_id,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
        ).eval()

        print(
            f"Loading {self.qwen_extraction_mode.upper()} OXE artifact "
            f"{args.checkpoint}...",
            flush=True,
        )
        self.model = SFTConditionedRDT(self.cfg, load_pretrained=True)
        load_trainable_artifact(self.model, args.checkpoint, trainable=False)
        self.model.to(self.device).eval()

    def set_instruction(self, instruction: str) -> None:
        """Refresh only language conditioning while retaining loaded models."""
        if instruction == getattr(self, "instruction", None) and hasattr(
            self, "lang_tokens"
        ):
            return
        self.instruction = instruction
        self.lang_tokens, self.lang_mask = self.extract_t5_features(
            {"instructions": [instruction]},
            self.t5_tokenizer,
            self.t5,
            max_lang_tokens=self.cfg.model.max_lang_tokens,
            expected_dim=self.cfg.model.lang_token_dim,
            device=self.t5_device,
        )

    def infer(
        self,
        *,
        image: np.ndarray,
        previous_image: np.ndarray | None,
        state_7d: np.ndarray,
        control_frequency: float,
        seed: int,
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        torch = self.torch
        started = time.perf_counter()
        sample = build_policy_sample(
            image=image,
            previous_image=previous_image,
            state_7d=state_7d,
            instruction=self.instruction,
            dataset=self.dataset,
            horizon=self.cfg.model.pred_horizon,
            control_frequency=control_frequency,
        )
        encoded = self.standardized_collate_fn(
            [sample],
            max_images_per_sample=6,
            image_history_size=2,
            image_jpeg_quality=90,
            skip_no_image=True,
            encode_image_slots=False,
        )
        if encoded is None:
            raise RuntimeError("The SimplerEnv observation did not contain a valid image")

        qwen_started = time.perf_counter()
        if self.qwen_extraction_mode == "b2":
            assert self.extract_latent_student_spatial_kv is not None
            assert self.latent_student is not None
            qwen_kv, _latent_waypoints = self.extract_latent_student_spatial_kv(
                encoded,
                student=self.latent_student,
                processor=self.qwen_processor,
                device=self.device,
                layer_index=self.args.qwen_layer_index,
                expected_dim=self.cfg.model.qwen_kv_dim,
                spatial_token_count=self.args.spatial_token_count,
                prompt_template=self.args.b2_prompt_template,
            )
        else:
            assert self.qwen is not None
            qwen_kv = self.extract_qwen_kv(
                encoded,
                self.qwen_processor,
                self.qwen,
                device=self.device,
                layer_index=self.args.qwen_layer_index,
                max_new_tokens=self.args.qwen_max_new_tokens,
                expected_dim=self.cfg.model.qwen_kv_dim,
                stop_at_think_end=True,
                enable_thinking=False,
            )
        expected_qwen_tokens = 5 if self.qwen_extraction_mode == "b2" else 1
        expected_qwen_shape = (
            1,
            expected_qwen_tokens,
            self.cfg.model.qwen_kv_dim,
        )
        if tuple(qwen_kv.shape) != expected_qwen_shape:
            raise ValueError(
                f"{self.qwen_extraction_mode.upper()} extraction returned "
                f"{tuple(qwen_kv.shape)}, expected {expected_qwen_shape}"
            )
        qwen_seconds = time.perf_counter() - qwen_started
        siglip_started = time.perf_counter()
        img_tokens, img_mask = self.extract_siglip_features(
            encoded,
            self.siglip_processor,
            self.siglip,
            max_img_tokens=self.cfg.model.image_tokens,
            expected_dim=self.cfg.model.img_token_dim,
            device=self.device,
        )
        siglip_seconds = time.perf_counter() - siglip_started
        native_state, native_state_mask = pack_oxe_state(state_7d)
        action_mask = np.zeros(128, dtype=np.float32)
        action_mask[RDT_XYZ_SLICE] = 1.0
        action_mask[RDT_ORTHO6D_SLICE] = 1.0
        action_mask[RDT_GRIPPER_INDEX] = 1.0
        batch = {
            "state": torch.from_numpy(native_state[None]).to(self.device),
            "state_dim_mask": torch.from_numpy(native_state_mask[None]).to(self.device),
            "action_dim_mask": torch.from_numpy(action_mask[None]).to(self.device),
            "ctrl_freq": torch.tensor([control_frequency], dtype=torch.float32, device=self.device),
            "lang_tokens": self.lang_tokens.to(self.device),
            "lang_mask": self.lang_mask.to(self.device),
            "img_tokens": img_tokens,
            "img_mask": img_mask,
            "qwen_kv": qwen_kv,
        }
        torch.manual_seed(seed)
        rdt_started = time.perf_counter()
        output = self.model.sample_actions(batch).float().cpu().numpy()[0]
        rdt_seconds = time.perf_counter() - rdt_started
        if output.shape != (self.cfg.model.pred_horizon, 128):
            raise ValueError(f"Unexpected RDT output shape {output.shape}")
        if not np.isfinite(output).all():
            raise FloatingPointError("RDT emitted NaN/Inf")
        return output, {
            "qwen_seconds": qwen_seconds,
            "siglip_seconds": siglip_seconds,
            "rdt_diffusion_seconds": rdt_seconds,
            "total_policy_seconds": time.perf_counter() - started,
            "qwen_tokens": float(qwen_kv.shape[1]),
            "qwen_extraction_mode": self.qwen_extraction_mode,
        }


def worker_socket_path() -> Path:
    """Return a short temporary path below Linux's AF_UNIX path limit."""
    filename = f"tfse_{os.getpid()}_{secrets.token_hex(4)}.sock"
    configured = os.environ.get("THINKFLOW_SIMPLER_SOCKET_DIR")
    candidate_dirs = []
    if configured:
        candidate_dirs.append(Path(configured).expanduser())
    candidate_dirs.extend((Path(tempfile.gettempdir()), Path("/tmp")))
    for directory in candidate_dirs:
        candidate = directory.resolve() / filename
        if len(os.fsencode(candidate)) <= AF_UNIX_SAFE_PATH_BYTES:
            return candidate
    raise OSError(
        "Could not construct a SimplerEnv worker socket path shorter than "
        f"{AF_UNIX_SAFE_PATH_BYTES + 1} bytes"
    )


def start_worker(args: argparse.Namespace) -> tuple[subprocess.Popen[str], Any, Path]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    socket_path = worker_socket_path()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    authkey = secrets.token_bytes(32)
    command = [
        str(args.simpler_python),
        str(REPO_ROOT / "scripts/simpler_env_worker.py"),
        "--socket",
        str(socket_path),
        "--authkey-hex",
        authkey.hex(),
        "--task",
        args.task,
        "--seed",
        str(args.seed),
    ]
    if args.renderer_offscreen:
        command.append("--renderer-offscreen")
    log_handle = (output_dir / "simpler_worker.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(args.simpler_root),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + args.worker_timeout
    while not socket_path.exists():
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(
                f"SimplerEnv worker exited with code {process.returncode}; see {output_dir / 'simpler_worker.log'}"
            )
        if time.monotonic() >= deadline:
            process.terminate()
            log_handle.close()
            raise TimeoutError("Timed out waiting for the SimplerEnv worker socket")
        time.sleep(0.05)
    connection = Client(str(socket_path), family="AF_UNIX", authkey=authkey)
    # Keep the log descriptor alive through the child lifetime.
    process._simpler_log_handle = log_handle  # type: ignore[attr-defined]
    return process, connection, socket_path


def stop_worker(process: subprocess.Popen[str], connection: Any, socket_path: Path) -> None:
    try:
        if process.poll() is None:
            connection.send({"command": "close"})
            if connection.poll(2.0):
                connection.recv()
    except (BrokenPipeError, EOFError, OSError):
        pass
    try:
        connection.close()
    except OSError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    handle = getattr(process, "_simpler_log_handle", None)
    if handle is not None:
        handle.close()
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass


def annotated_frame(image: np.ndarray, text: str) -> np.ndarray:
    frame = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, frame.width, 20), fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255))
    return np.asarray(frame)


def save_probe_plot(path: Path, decoded: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    actions = decoded["environment_action"]
    horizon = np.arange(actions.shape[0])
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for index, label in enumerate(("dx", "dy", "dz")):
        axes[0].plot(horizon, actions[:, index], label=label)
    axes[0].set_ylabel("translation delta (m)")
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=0.25)
    for index, label in enumerate(("rx", "ry", "rz"), start=3):
        axes[1].plot(horizon, actions[:, index], label=label)
    axes[1].set_ylabel("rotation delta (rad)")
    axes[1].legend(ncol=3)
    axes[1].grid(alpha=0.25)
    axes[2].plot(
        horizon,
        decoded["gripper_open_score"],
        label="RDT gripper-open score",
    )
    axes[2].step(
        horizon,
        decoded["gripper_open_binary"],
        where="post",
        label="open decision",
    )
    axes[2].axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[2].set_ylabel("gripper")
    axes[2].set_xlabel("predicted horizon offset")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    figure.suptitle("B0 OXE action decoded for SimplerEnv")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_contract(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset or default_dataset(args.task)
    stats = load_stats(args.action_stats or default_stats_path(dataset))
    state = np.asarray(args.probe_state, dtype=np.float32)
    packed, mask = pack_oxe_state(state)
    native = np.zeros((2, 128), dtype=np.float32)
    native[0, RDT_XYZ_SLICE] = [-1.0, 0.0, 1.0]
    native[1, RDT_XYZ_SLICE] = [1.0, -1.0, 0.0]
    identity_6d = euler_xyz_to_ortho6d_numpy(np.zeros(3, dtype=np.float32))
    native[:, RDT_ORTHO6D_SLICE] = identity_6d
    native[:, RDT_GRIPPER_INDEX] = [0.25, -0.25]
    decoded = decode_native_actions(
        native,
        dataset=dataset,
        task=args.task,
        stats=stats,
        rotation_scale=args.rotation_scale,
    )

    expected_first_xyz = np.asarray(
        [stats.q01[0], (stats.q01[1] + stats.q99[1]) * 0.5, stats.q99[2]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(decoded["denormalized_xyz"][0], expected_first_xyz, atol=1e-6)
    np.testing.assert_allclose(decoded["environment_rotation"], 0.0, atol=1e-6)
    family = robot_family(args.task)
    expected_gripper = [-1.0, 1.0] if family == "google_robot" else [1.0, -1.0]
    np.testing.assert_array_equal(decoded["environment_action"][:, 6], expected_gripper)
    assert packed.shape == (128,) and mask.sum() == 10
    assert packed[RDT_GRIPPER_INDEX] == np.float32(1.0 - state[6])
    result = {
        "status": "passed",
        "task": args.task,
        "robot_family": family,
        "dataset": dataset,
        "state_7d": state,
        "packed_active_indices": np.flatnonzero(mask),
        "packed_active_values": packed[mask.astype(bool)],
        "synthetic_native_output": native,
        "decoded": decoded,
    }
    write_json(args.output_dir / "contract_test.json", result)
    print(json.dumps(jsonable(result), indent=2))
    return result


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset or default_dataset(args.task)
    stats = load_stats(args.action_stats or default_stats_path(dataset))
    image_path = args.probe_image
    if image_path is None:
        filename = (
            "google_robot_coke_can_visual_matching.png"
            if robot_family(args.task) == "google_robot"
            else "widowx_put_eggplant_in_basket_visual_matching.png"
        )
        image_path = args.simpler_root / "images/example_visualization" / filename
    image = np.asarray(Image.open(image_path).convert("RGB"))
    state = np.asarray(args.probe_state, dtype=np.float32)
    instruction = args.instruction or "pick up the coke can"
    control_frequency = args.control_frequency or (3.0 if robot_family(args.task) == "google_robot" else 5.0)
    engine = PolicyEngine(args, instruction, dataset)
    native, timing = engine.infer(
        image=image,
        previous_image=None,
        state_7d=state,
        control_frequency=control_frequency,
        seed=args.seed,
    )
    decoded = decode_native_actions(
        native,
        dataset=dataset,
        task=args.task,
        stats=stats,
        rotation_scale=args.rotation_scale,
    )
    result = {
        "status": "completed",
        "warning": "Static probe: the image is official SimplerEnv output, but the state is user/synthetic because this host cannot create a Vulkan simulator.",
        "task": args.task,
        "dataset": dataset,
        "instruction": instruction,
        "image": str(image_path.resolve()),
        "image_shape": list(image.shape),
        "state_7d": state,
        "control_frequency": control_frequency,
        "native_output_shape": list(native.shape),
        "first_10_horizon_steps": {key: value[:10] for key, value in decoded.items()},
        "timing": timing,
    }
    write_json(args.output_dir / "static_policy_probe.json", result)
    np.savez_compressed(args.output_dir / "static_policy_probe_arrays.npz", native=native, **decoded)
    save_probe_plot(args.output_dir / "static_policy_probe.png", decoded)
    print(json.dumps(jsonable(result), indent=2))
    return result


def run_rollout(
    args: argparse.Namespace,
    *,
    engine: PolicyEngine | None = None,
) -> dict[str, Any]:
    dataset = args.dataset or default_dataset(args.task)
    stats = load_stats(args.action_stats or default_stats_path(dataset))
    process, connection, socket_path = start_worker(args)
    video_writer = None
    try:
        if not connection.poll(args.worker_timeout):
            raise TimeoutError("SimplerEnv worker did not finish environment initialization")
        packet = connection.recv()
        if packet.get("kind") == "error":
            write_json(args.output_dir / "environment_error.json", packet)
            raise RuntimeError(
                f"SimplerEnv failed during initialization: {packet['error_type']}: {packet['message']}"
            )
        if packet.get("kind") != "ready":
            raise RuntimeError(f"Unexpected worker message: {packet.get('kind')!r}")
        instruction = args.instruction or str(packet["instruction"])
        control_frequency = args.control_frequency or float(packet["control_frequency"])
        write_json(
            args.output_dir / "environment_contract.json",
            {
                "task": args.task,
                "robot_uid": packet["robot_uid"],
                "instruction": instruction,
                "control_frequency": control_frequency,
                "action_low": packet["action_low"],
                "action_high": packet["action_high"],
                "initial_state_7d": packet["state_7d"],
                "initial_robot_qpos": packet["robot_qpos"],
                "initial_object_states": packet["object_states"],
                "rotation_scale": args.rotation_scale,
                "google_gripper_sticky_steps": args.google_gripper_sticky_steps,
            },
        )
        if engine is None:
            # Load the large policy stack only after the simulator proves it can
            # initialize. This makes Vulkan/asset failures fast and unambiguous.
            engine = PolicyEngine(args, instruction, dataset)
        else:
            if engine.dataset != dataset:
                raise ValueError(
                    f"Reusable policy engine dataset {engine.dataset!r} does not "
                    f"match rollout dataset {dataset!r}"
                )
            engine.set_instruction(instruction)
        video_path = args.output_dir / "rollout.mp4"
        video_writer = imageio.get_writer(
            video_path, format="FFMPEG", fps=args.video_fps, codec="libx264", quality=8
        )
        previous_image: np.ndarray | None = None
        google_gripper = None
        if robot_family(args.task) == "google_robot":
            google_gripper = GoogleGripperTargetAdapter(
                sticky_steps=args.google_gripper_sticky_steps,
                assumed_open=float(packet["state_7d"][6]) < 0.5,
            )
        step = 0
        plan_index = 0
        success = False
        started = time.perf_counter()
        while step < args.max_steps:
            native, timing = engine.infer(
                image=packet["image"],
                previous_image=previous_image,
                state_7d=packet["state_7d"],
                control_frequency=control_frequency,
                seed=args.seed + plan_index,
            )
            decoded = decode_native_actions(
                native,
                dataset=dataset,
                task=args.task,
                stats=stats,
                rotation_scale=args.rotation_scale,
            )
            chunk = min(args.action_chunk, native.shape[0], args.max_steps - step)
            instruction_changed = False
            for offset in range(chunk):
                before = packet
                requested = decoded["environment_action"][offset].copy()
                raw_gripper_command = float(requested[6])
                if google_gripper is not None:
                    requested[6] = google_gripper.command(
                        bool(decoded["gripper_open_binary"][offset])
                    )
                connection.send({"command": "step", "action": requested})
                if not connection.poll(args.worker_timeout):
                    raise TimeoutError("SimplerEnv worker did not return from env.step")
                packet = connection.recv()
                if packet.get("kind") == "error":
                    write_json(args.output_dir / "environment_error.json", packet)
                    raise RuntimeError(
                        f"SimplerEnv failed in env.step: {packet['error_type']}: {packet['message']}"
                    )
                achieved_delta = np.asarray(packet["state_7d"][:6]) - np.asarray(before["state_7d"][:6])
                row = {
                    "step": step,
                    "plan_index": plan_index,
                    "action_offset": offset,
                    "instruction": instruction,
                    "state_before_7d": before["state_7d"],
                    "native_relevant_10d": decoded["native_relevant_10d"][offset],
                    "normalized_xyz": decoded["normalized_xyz"][offset],
                    "denormalized_xyz": decoded["denormalized_xyz"][offset],
                    "rotation_6d": decoded["rotation_6d"][offset],
                    "decoded_euler_xyz": decoded["decoded_euler_xyz"][offset],
                    "environment_rotation": decoded["environment_rotation"][offset],
                    "gripper_open_score": decoded["gripper_open_score"][offset],
                    "gripper_open_binary": decoded["gripper_open_binary"][offset],
                    "raw_environment_gripper_command": raw_gripper_command,
                    "postprocessed_gripper_command": float(requested[6]),
                    "google_gripper_sticky_remaining": (
                        google_gripper.remaining_steps if google_gripper is not None else 0
                    ),
                    "requested_action": requested,
                    "executed_action": packet["executed_action"],
                    "state_after_7d": packet["state_7d"],
                    "achieved_tcp_delta_xyz_rpy": achieved_delta,
                    "robot_qpos_after": packet["robot_qpos"],
                    "object_states_after": packet["object_states"],
                    "reward": packet["reward"],
                    "terminated": packet["terminated"],
                    "truncated": packet["truncated"],
                    "info": packet["info"],
                    "plan_timing": timing if offset == 0 else None,
                }
                append_jsonl(args.output_dir / "trajectory.jsonl", row)
                success = success or bool(packet["info"].get("success", False)) or bool(packet["terminated"])
                video_writer.append_data(
                    annotated_frame(packet["image"], f"step={step} grip_open={float(decoded['gripper_open_binary'][offset]):.0f}")
                )
                step += 1
                next_instruction = str(packet.get("instruction", instruction))
                if next_instruction != instruction:
                    instruction = next_instruction
                    engine.set_instruction(instruction)
                    instruction_changed = True
                if packet["terminated"] or packet["truncated"]:
                    break
                if instruction_changed:
                    break
            previous_image = np.asarray(packet["image"], dtype=np.uint8).copy()
            plan_index += 1
            if packet["terminated"] or packet["truncated"]:
                break
            if instruction_changed:
                continue
        summary = {
            "status": "completed",
            "success": success,
            "task": args.task,
            "seed": args.seed,
            "dataset": dataset,
            "instruction": instruction,
            "steps": step,
            "plans": plan_index,
            "action_chunk": args.action_chunk,
            "rotation_scale": args.rotation_scale,
            "google_gripper_sticky_steps": args.google_gripper_sticky_steps,
            "qwen_extraction_mode": engine.qwen_extraction_mode,
            "elapsed_seconds": time.perf_counter() - started,
            "video": str(video_path.resolve()),
            "trajectory": str((args.output_dir / "trajectory.jsonl").resolve()),
        }
        write_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        if video_writer is not None:
            video_writer.close()
        stop_worker(process, connection, socket_path)


def run_fractal_suite(args: argparse.Namespace) -> dict[str, Any]:
    """Run the canonical five-task Google Robot benchmark with one model load."""
    if args.dataset not in (None, "fractal"):
        raise ValueError("Fractal suite evaluation requires --dataset fractal")
    args.dataset = "fractal"
    if args.instruction is not None:
        raise ValueError("Suite evaluation obtains each instruction from SimplerEnv")

    engine = PolicyEngine(args, "initialize Fractal evaluation", "fractal")
    results: list[dict[str, Any]] = []
    for task in args.suite_tasks:
        for episode_index in range(args.episodes_per_task):
            seed = args.seed + episode_index
            episode_root = args.output_dir / task / f"seed_{seed:06d}"
            summary_path = episode_root / "summary.json"
            if summary_path.is_file():
                with summary_path.open("r", encoding="utf-8") as handle:
                    result = json.load(handle)
                result["resumed_from_existing"] = True
                results.append(result)
                print(
                    f"[suite] reuse {task} seed={seed}: "
                    f"success={bool(result.get('success', False))}",
                    flush=True,
                )
                continue

            episode_dir = episode_root
            if episode_root.is_dir() and any(episode_root.iterdir()):
                attempt_index = 2
                while (episode_root / f"attempt_{attempt_index:02d}").exists():
                    attempt_index += 1
                episode_dir = episode_root / f"attempt_{attempt_index:02d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            episode_args = copy.copy(args)
            episode_args.mode = "rollout"
            episode_args.task = task
            episode_args.seed = seed
            episode_args.output_dir = episode_dir
            write_json(episode_dir / "arguments.json", vars(episode_args))
            try:
                result = run_rollout(episode_args, engine=engine)
                result["episode_index"] = episode_index
                result["error"] = None
                write_json(summary_path, result)
            except Exception as error:
                result = {
                    "status": "error",
                    "success": False,
                    "task": task,
                    "seed": seed,
                    "episode_index": episode_index,
                    "error": f"{type(error).__name__}: {error}",
                }
                write_json(episode_dir / "suite_error.json", result)
                print(f"[suite] ERROR {task} seed={seed}: {result['error']}", flush=True)
                if args.suite_fail_fast:
                    raise
            results.append(result)
            write_json(args.output_dir / "suite_results.json", results)

    per_task: dict[str, dict[str, Any]] = {}
    for task in args.suite_tasks:
        task_results = [row for row in results if row.get("task") == task]
        successes = sum(bool(row.get("success", False)) for row in task_results)
        per_task[task] = {
            "successes": successes,
            "episodes": len(task_results),
            "success_rate": successes / len(task_results) if task_results else 0.0,
            "errors": sum(row.get("status") == "error" for row in task_results),
        }
    total_successes = sum(bool(row.get("success", False)) for row in results)
    completed = len(results)
    task_macro_success_rate = (
        sum(row["success_rate"] for row in per_task.values()) / len(per_task)
        if per_task
        else 0.0
    )
    summary = {
        "status": "completed",
        "checkpoint": args.checkpoint,
        "config": args.config,
        "tasks": list(args.suite_tasks),
        "episodes_per_task": args.episodes_per_task,
        "total_successes": total_successes,
        "total_episodes": completed,
        "micro_success_rate": total_successes / completed if completed else 0.0,
        "macro_success_rate": task_macro_success_rate,
        "errors": sum(row.get("status") == "error" for row in results),
        "per_task": per_task,
    }
    write_json(args.output_dir / "suite_results.json", results)
    write_json(args.output_dir / "suite_summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("contract", "probe", "rollout", "suite"), default="rollout"
    )
    parser.add_argument("--task", default="google_robot_pick_coke_can")
    parser.add_argument("--dataset", choices=("fractal", "bridge"), default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--action-stats", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output_2/simpler_b0_oxe")
    parser.add_argument("--simpler-root", type=Path, default=DEFAULT_SIMPLER_ROOT)
    parser.add_argument("--simpler-python", type=Path, default=DEFAULT_SIMPLER_PYTHON)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--suite-tasks",
        nargs="+",
        default=list(FRACTAL_SUITE_TASKS),
        help="SimplerEnv task names used by --mode suite.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=10,
        help="Number of deterministic seeds per task in suite mode.",
    )
    parser.add_argument("--suite-fail-fast", action="store_true")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--action-chunk", type=int, default=1)
    parser.add_argument(
        "--rotation-scale",
        type=float,
        default=1.0,
        help="Multiply decoded rotation commands by this value; 0 disables rotation.",
    )
    parser.add_argument(
        "--google-gripper-sticky-steps",
        type=int,
        default=0,
        help=(
            "Repeat Google Robot open/close transitions for this many control "
            "steps. Use 15 to match SimplerEnv's Google Octo post-processing."
        ),
    )
    parser.add_argument("--control-frequency", type=float, default=None)
    parser.add_argument("--worker-timeout", type=float, default=120.0)
    parser.add_argument("--renderer-offscreen", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--qwen-extraction",
        choices=("auto", "b0", "b2"),
        default="auto",
        help=(
            "Select one </think> KV token (B0) or the LatentStudent five-spatial-"
            "token KV path (B2). Auto selects B2 when the checkpoint/config path "
            "contains 'b2'."
        ),
    )
    parser.add_argument("--qwen-model-id", default=str(DEFAULT_QWEN))
    parser.add_argument("--qwen-layer-index", type=int, default=7)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--student-model-id", default=str(DEFAULT_B2_STUDENT))
    parser.add_argument("--processor-id", default=str(DEFAULT_QWEN))
    parser.add_argument(
        "--latent-student-code-dir",
        type=Path,
        default=DEFAULT_B2_CODE_DIR,
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
    parser.add_argument("--siglip-model-id", default="google/siglip-so400m-patch14-384")
    parser.add_argument("--t5-model-id", default="google/t5-v1_1-xxl")
    parser.add_argument("--t5-precision", choices=("bf16", "8bit"), default="bf16")
    parser.add_argument("--probe-image", type=Path, default=None)
    parser.add_argument(
        "--probe-state",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW", "GRIPPER_CLOSED"),
        default=[0.35, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0],
    )
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.latent_student_code_dir = args.latent_student_code_dir.expanduser().resolve()
    if args.spatial_parameters_path is not None:
        args.spatial_parameters_path = args.spatial_parameters_path.expanduser().resolve()
    for name in ("student_model_id", "processor_id"):
        value = Path(getattr(args, name)).expanduser()
        if value.exists():
            setattr(args, name, str(value.resolve()))
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.action_chunk <= 0:
        parser.error("--action-chunk must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.episodes_per_task <= 0:
        parser.error("--episodes-per-task must be positive")
    if args.rotation_scale < 0.0:
        parser.error("--rotation-scale must be non-negative")
    if args.google_gripper_sticky_steps < 0:
        parser.error("--google-gripper-sticky-steps must be non-negative")
    if args.latent_count <= 0:
        parser.error("--latent-count must be positive")
    if args.spatial_token_count != 5:
        parser.error("B2 rollout requires --spatial-token-count 5")
    if "{task}" not in args.b2_prompt_template:
        parser.error("--b2-prompt-template must contain {task}")
    return args


def main() -> None:
    args = parse_args()
    write_json(args.output_dir / "arguments.json", vars(args))
    if args.mode == "contract":
        run_contract(args)
    elif args.mode == "probe":
        run_probe(args)
    elif args.mode == "suite":
        run_fractal_suite(args)
    else:
        run_rollout(args)


if __name__ == "__main__":
    main()
