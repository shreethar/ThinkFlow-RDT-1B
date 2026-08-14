"""Streaming reader for LIBERO feature shards.

The reader intentionally lives outside ``thinkflow_rdt``. It consumes only the
on-disk shard contracts: native 8D/7D or legacy 11D/10D proprioception,
Qwen KV, instructions, and pooled raw arrays or lossless image bytes.
"""

from __future__ import annotations

import io
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


LIBERO_ROT_COMMAND_SCALE = 0.5


@dataclass(frozen=True)
class _CachedTaskIndex:
    """Small dataframe-like task index used only by LeRobot model cards."""

    index: tuple[str, ...] = ("cached LIBERO Qwen-KV",)


def ortho6d_to_rotation_matrix(ortho6d: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Project cached first-two-column rotations onto SO(3)."""

    values = np.asarray(ortho6d, dtype=np.float64)
    if values.shape[-1] != 6:
        raise ValueError(f"Expected ortho6D [...,6], got {values.shape}")
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
    return np.stack([first, second, third], axis=-1)


def cached_state_to_libero_state(state_11d: Tensor | np.ndarray) -> Tensor:
    """Convert `[xyz, absolute ortho6D, finger0, finger1]` to native 8D LIBERO state."""

    values = np.asarray(torch.as_tensor(state_11d).float().cpu(), dtype=np.float64)
    if values.shape[-1] != 11:
        raise ValueError(f"Expected cached state [...,11], got {values.shape}")
    matrices = ortho6d_to_rotation_matrix(values[..., 3:9])
    rotvec = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec()
    rotvec = rotvec.reshape(*values.shape[:-1], 3)
    native = np.concatenate([values[..., :3], rotvec, values[..., 9:11]], axis=-1)
    return torch.from_numpy(native.astype(np.float32, copy=False))


def cached_action_to_libero_action(action_10d: Tensor | np.ndarray) -> Tensor:
    """Convert `[dxyz, relative ortho6D, gripper]` back to raw 7D LIBERO command."""

    values = np.asarray(torch.as_tensor(action_10d).float().cpu(), dtype=np.float64)
    if values.shape[-1] != 10:
        raise ValueError(f"Expected cached action [...,10], got {values.shape}")
    matrices = ortho6d_to_rotation_matrix(values[..., 3:9])
    delta_rotvec = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec()
    rotation_command = delta_rotvec.reshape(*values.shape[:-1], 3) / LIBERO_ROT_COMMAND_SCALE
    rotation_command = np.clip(rotation_command, -1.0, 1.0)
    native = np.concatenate(
        [values[..., :3], rotation_command, values[..., 9:10]], axis=-1
    )
    return torch.from_numpy(native.astype(np.float32, copy=False))


def add_native_libero_tensors(pack: dict) -> dict:
    """Expose native tensors, decoding legacy RDT caches once per shard."""

    if "_smolvla_state" not in pack:
        state = torch.as_tensor(pack["state"], dtype=torch.float32)
        if state.shape[-1] == 8:
            pack["_smolvla_state"] = state
        elif state.shape[-1] == 11:
            pack["_smolvla_state"] = cached_state_to_libero_state(state)
        else:
            raise ValueError(
                f"Expected cached state dimension 8 or 11, got {state.shape[-1]}"
            )
    if "_smolvla_actions" not in pack:
        actions = torch.as_tensor(pack["actions"], dtype=torch.float32)
        if actions.shape[-1] == 7:
            pack["_smolvla_actions"] = actions
        elif actions.shape[-1] == 10:
            pack["_smolvla_actions"] = cached_action_to_libero_action(actions)
        else:
            raise ValueError(
                f"Expected cached action dimension 7 or 10, got {actions.shape[-1]}"
            )
    return pack


def list_shards(cache_root: str | Path, suite: str, split: str = "train") -> list[Path]:
    directory = Path(cache_root).expanduser().resolve() / suite / split
    paths = sorted(directory.glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt files under {directory}")
    return paths


def decode_rgb_image(encoded: bytes) -> Tensor:
    """Decode a cached PNG/JPEG byte string as float CHW in [0, 1]."""

    with Image.open(io.BytesIO(encoded)) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def cached_rgb_image(value: bytes | Tensor | np.ndarray) -> Tensor:
    """Read either an encoded image or an uncompressed uint8 HWC/CHW array."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        return decode_rgb_image(bytes(value))
    image = torch.as_tensor(value)
    if image.ndim != 3:
        raise ValueError(
            f"Expected cached RGB image with 3 dimensions, got {tuple(image.shape)}"
        )
    if image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    elif image.shape[0] != 3:
        raise ValueError(f"Expected cached HWC or CHW RGB image, got {tuple(image.shape)}")
    if image.dtype == torch.uint8:
        return image.float().div_(255.0)
    return image.float()


def _pad_action_chunk(actions: Tensor, valid: Tensor, chunk_size: int) -> tuple[Tensor, Tensor]:
    actions = torch.as_tensor(actions, dtype=torch.float32)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    if actions.ndim != 2 or valid.ndim != 1 or actions.shape[0] != valid.shape[0]:
        raise ValueError(
            f"Invalid action/time-mask shapes: actions={tuple(actions.shape)}, valid={tuple(valid.shape)}"
        )
    if actions.shape[0] >= chunk_size:
        return actions[:chunk_size], valid[:chunk_size]
    missing = chunk_size - actions.shape[0]
    actions = torch.cat(
        [actions, torch.zeros(missing, actions.shape[1], dtype=actions.dtype)], dim=0
    )
    valid = torch.cat([valid, torch.zeros(missing, dtype=torch.bool)], dim=0)
    return actions, valid


def sample_from_pack(
    pack: dict,
    sample_index: int,
    *,
    chunk_size: int,
    camera_slots: Sequence[int] = (3, 4),
    expected_qwen_tokens: int | None = None,
) -> dict:
    """Materialize one SmolVLA training example from a loaded cache shard.

    Cache slots 3 and 4 are the current agent-view and wrist-view frames. Slots
    0--2 are prior-frame history and slot 5 is the absent LIBERO third camera.
    """

    qwen_kv = torch.as_tensor(pack["qwen_kv"])[sample_index]
    if qwen_kv.ndim == 1:
        qwen_kv = qwen_kv.unsqueeze(0)
    if qwen_kv.ndim != 2 or qwen_kv.shape[-1] != 2048:
        raise ValueError(f"Expected cached Qwen KV [T,2048], got {tuple(qwen_kv.shape)}")
    if expected_qwen_tokens is not None and qwen_kv.shape[0] != expected_qwen_tokens:
        raise ValueError(
            f"Expected {expected_qwen_tokens} cached Qwen KV tokens, got {qwen_kv.shape[0]}"
        )

    add_native_libero_tensors(pack)
    state = torch.as_tensor(pack["_smolvla_state"], dtype=torch.float32)[sample_index]
    if state.shape != (8,):
        raise ValueError(f"Expected native LIBERO state [8], got {tuple(state.shape)}")
    actions, valid = _pad_action_chunk(
        torch.as_tensor(pack["_smolvla_actions"])[sample_index],
        torch.as_tensor(pack["action_time_mask"])[sample_index],
        chunk_size,
    )
    if actions.shape[-1] != 7:
        raise ValueError(f"Expected native LIBERO actions [T,7], got {tuple(actions.shape)}")

    indices = torch.as_tensor(pack["sample_image_indices"])[sample_index]
    masks = torch.as_tensor(pack["sample_image_mask"], dtype=torch.bool)[sample_index]
    images = []
    for slot in camera_slots:
        if slot >= len(indices) or not bool(masks[slot]):
            raise ValueError(
                f"Required current camera slot {slot} is invalid for sample {sample_index}"
            )
        pool_index = int(indices[slot])
        image_pool = pack.get("image_arrays", pack.get("image_jpegs"))
        if image_pool is None:
            raise KeyError("Shard has neither image_arrays nor image_jpegs")
        images.append(cached_rgb_image(image_pool[pool_index]))

    instruction = str(pack["instructions"][sample_index])
    # Preserve a stable task identity through LeRobot's processor stack. The
    # fusion ranking objective uses it to select a genuinely different-task KV
    # donor instead of merely rotating samples from the same demonstration.
    group_digest = hashlib.blake2b(instruction.encode("utf-8"), digest_size=8).digest()
    qwen_group_id = int.from_bytes(group_digest, "little") & ((1 << 63) - 1)
    return {
        OBS_STATE: state,
        f"{OBS_IMAGES}.image": images[0],
        f"{OBS_IMAGES}.image2": images[1],
        ACTION: actions,
        "action_is_pad": ~valid,
        "task": instruction,
        "qwen_kv": qwen_kv,
        "qwen_group_id": qwen_group_id,
    }


class CachedLiberoIterableDataset(IterableDataset):
    """Shard-local shuffle that reads each large shard once per epoch.

    Globally shuffling individual indices would repeatedly deserialize unrelated
    shards and make training I/O-bound. This dataset shuffles shard order and then
    sample order inside each loaded shard. With DataLoader workers, shards are
    partitioned between workers before shuffling.
    """

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        *,
        chunk_size: int = 50,
        seed: int = 42,
        repeat: bool = True,
        camera_slots: Sequence[int] = (3, 4),
        expected_qwen_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.shard_paths = tuple(Path(path) for path in shard_paths)
        if not self.shard_paths:
            raise ValueError("At least one shard path is required")
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self.repeat = bool(repeat)
        self.camera_slots = tuple(int(slot) for slot in camera_slots)
        if expected_qwen_tokens is not None and expected_qwen_tokens <= 0:
            raise ValueError("expected_qwen_tokens must be positive")
        self.expected_qwen_tokens = expected_qwen_tokens

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        worker_paths = list(self.shard_paths[worker_id::worker_count])
        if not worker_paths:
            return

        epoch = 0
        while True:
            rng = random.Random(self.seed + 1_000_003 * epoch + worker_id)
            rng.shuffle(worker_paths)
            for path in worker_paths:
                pack = torch.load(path, map_location="cpu", weights_only=False)
                add_native_libero_tensors(pack)
                count = int(pack.get("num_samples", len(pack["state"])))
                order = list(range(count))
                rng.shuffle(order)
                for sample_index in order:
                    yield sample_from_pack(
                        pack,
                        sample_index,
                        chunk_size=self.chunk_size,
                        camera_slots=self.camera_slots,
                        expected_qwen_tokens=self.expected_qwen_tokens,
                    )
                del pack
            epoch += 1
            if not self.repeat:
                break


@dataclass
class CachedLiberoMetadata:
    """The subset of LeRobotDatasetMetadata consumed by LeRobot's trainer.

    Cached feature shards are already materialized and therefore do not have a
    LeRobot v3 parquet metadata tree.  This adapter exposes the same feature,
    statistics, camera, and cardinality contract without copying the cache.
    """

    repo_id: str
    root: Path
    stats: dict
    total_frames: int
    total_episodes: int
    fps: int = 20
    robot_type: str = "libero_panda"

    def __post_init__(self) -> None:
        self.features = {
            OBS_STATE: {
                "dtype": "float32",
                "shape": (8,),
                "names": [
                    "x",
                    "y",
                    "z",
                    "ee_rotvec_x",
                    "ee_rotvec_y",
                    "ee_rotvec_z",
                    "finger_0",
                    "finger_1",
                ],
            },
            f"{OBS_IMAGES}.image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channels"],
            },
            f"{OBS_IMAGES}.image2": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channels"],
            },
            ACTION: {
                "dtype": "float32",
                "shape": (7,),
                "names": ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"],
            },
        }
        self.camera_keys = [f"{OBS_IMAGES}.image", f"{OBS_IMAGES}.image2"]
        self.depth_keys: list[str] = []
        self.has_language_columns = False
        # Streaming training never constructs EpisodeAwareSampler, but these
        # fields make diagnostics and third-party utilities safe to call.
        self.episodes = {
            "dataset_from_index": np.array([0], dtype=np.int64),
            "dataset_to_index": np.array([self.total_frames], dtype=np.int64),
            "tasks": [["cached LIBERO Qwen-KV"]],
        }
        self.tasks = _CachedTaskIndex()


class LeRobotCachedLiberoDataset(CachedLiberoIterableDataset):
    """Cached iterable with the metadata expected by ``lerobot-train``."""

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        *,
        cache_root: str | Path,
        repo_id: str,
        stats: dict,
        num_samples: int,
        chunk_size: int = 50,
        seed: int = 42,
        repeat: bool = True,
        approximate_episodes: int = 0,
        expected_qwen_tokens: int | None = None,
    ) -> None:
        super().__init__(
            shard_paths,
            chunk_size=chunk_size,
            seed=seed,
            repeat=repeat,
            expected_qwen_tokens=expected_qwen_tokens,
        )
        self.num_frames = int(num_samples)
        self.num_episodes = int(approximate_episodes)
        self.meta = CachedLiberoMetadata(
            repo_id=repo_id,
            root=Path(cache_root).expanduser().resolve(),
            stats=stats,
            total_frames=self.num_frames,
            total_episodes=self.num_episodes,
        )
        self.episodes = None

    def __len__(self) -> int:
        return self.num_frames
