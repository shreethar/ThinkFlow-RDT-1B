from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = {
    "qwen_kv",
    "lang_tokens",
    "img_tokens",
    "state",
    "actions",
    "ctrl_freq",
}
ONLINE_SIGLIP_REQUIRED_KEYS = {
    "qwen_kv",
    "lang_tokens",
    "image_slot_jpegs",
    "image_slot_mask",
    "state",
    "actions",
    "ctrl_freq",
}


class CachedFeatureDataset(Dataset[dict[str, Any]]):
    """
    Stable indexed dataset backed by cached feature .pt files.

    Each manifest line can be either:
      {"path": "relative/or/absolute/sample.pt"}
    or a plain JSON string containing the path.

    Newer manifests may point at one episode pack per line:
      {"path": "episode_000000000.pt", "cache_layout": "episode_pack", "num_samples": 64}
    In that case this dataset expands the episode pack into sample-level items.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        required_keys: set[str] | frozenset[str] | None = None,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.required_keys = set(REQUIRED_KEYS if required_keys is None else required_keys)
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        self.base_dir = self.manifest_path.parent
        self.entries: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                path_value = item if isinstance(item, str) else item.get("path")
                if not path_value:
                    raise ValueError(
                        f"Manifest line {line_number} has no path: {self.manifest_path}"
                    )
                path = Path(path_value)
                if not path.is_absolute():
                    path = (self.base_dir / path).resolve()
                if isinstance(item, dict) and item.get("cache_layout") == "episode_pack":
                    num_samples = int(item.get("num_samples", 0))
                    if num_samples <= 0:
                        raise ValueError(
                            f"Episode-pack manifest line {line_number} has invalid "
                            f"num_samples={num_samples}: {self.manifest_path}"
                        )
                    for sample_index in range(num_samples):
                        self.entries.append(
                            {
                                "path": path,
                                "cache_layout": "episode_pack",
                                "sample_index": sample_index,
                            }
                        )
                else:
                    self.entries.append(
                        {
                            "path": path,
                            "cache_layout": "sample",
                            "sample_index": None,
                        }
                    )
        if not self.entries:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        self.paths = [entry["path"] for entry in self.entries]
        self._pack_cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()
        self._pack_cache_size = 8

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        path = entry["path"]
        if entry["cache_layout"] == "episode_pack":
            pack = self._load_episode_pack(path)
            sample = self._sample_from_episode_pack(
                pack,
                int(entry["sample_index"]),
                path=path,
            )
        else:
            sample = torch.load(path, map_location="cpu", weights_only=False)
        missing = self.required_keys.difference(sample)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")
        sample["_path"] = str(path)
        return sample

    def _load_episode_pack(self, path: Path) -> dict[str, Any]:
        cached = self._pack_cache.get(path)
        if cached is not None:
            self._pack_cache.move_to_end(path)
            return cached
        pack = torch.load(path, map_location="cpu", weights_only=False)
        if pack.get("cache_layout") != "episode_pack":
            raise ValueError(f"{path} is not an episode_pack cache file")
        self._pack_cache[path] = pack
        self._pack_cache.move_to_end(path)
        while len(self._pack_cache) > self._pack_cache_size:
            self._pack_cache.popitem(last=False)
        return pack

    def _sample_from_episode_pack(
        self,
        pack: dict[str, Any],
        sample_index: int,
        *,
        path: Path,
    ) -> dict[str, Any]:
        num_samples = int(pack["num_samples"])
        if sample_index < 0 or sample_index >= num_samples:
            raise IndexError(f"sample_index {sample_index} out of range for {path}")

        anchor_indices = torch.as_tensor(pack["sample_anchor_index"], dtype=torch.long)
        anchor_index = int(anchor_indices[sample_index].item())
        qwen_anchor_kv = torch.as_tensor(pack["qwen_anchor_kv"])
        if anchor_index < 0 or anchor_index >= int(qwen_anchor_kv.shape[0]):
            raise IndexError(f"anchor_index {anchor_index} out of range for {path}")

        image_pool = list(pack.get("image_jpegs", []))
        sample_image_indices = torch.as_tensor(pack["sample_image_indices"], dtype=torch.long)
        image_indices = sample_image_indices[sample_index].flatten().tolist()
        image_slot_jpegs = []
        for image_index in image_indices:
            if image_index < 0 or image_index >= len(image_pool):
                raise IndexError(f"image index {image_index} out of range for {path}")
            image_slot_jpegs.append(image_pool[image_index])

        step_idx_values = pack.get("sample_step_idx")
        step_idx = (
            str(step_idx_values[sample_index])
            if step_idx_values is not None
            else str(sample_index)
        )
        ctrl_freq = pack.get("ctrl_freq", 0.0)
        if isinstance(ctrl_freq, torch.Tensor) and ctrl_freq.ndim > 0:
            ctrl_freq = float(ctrl_freq[sample_index].item())
        else:
            ctrl_freq = float(ctrl_freq)

        sample = {
            "qwen_kv": qwen_anchor_kv[anchor_index],
            "lang_tokens": pack["lang_tokens"],
            "lang_mask": pack["lang_mask"],
            "state": pack["state"][sample_index],
            "actions": pack["actions"][sample_index],
            "action_time_mask": pack["action_time_mask"][sample_index],
            "action_dim_mask": pack["action_dim_mask"][sample_index],
            "ctrl_freq": ctrl_freq,
            "image_slot_jpegs": image_slot_jpegs,
            "image_slot_mask": pack["sample_image_mask"][sample_index],
            "dataset_id": pack.get("dataset_id"),
            "episode_id": pack.get("episode_id"),
            "step_idx": step_idx,
            "qwen_cache_scope": pack.get("qwen_cache_scope", "episode_anchors"),
            "qwen_anchor_step_idx": str(pack["qwen_anchor_step_idx"][anchor_index]),
            "qwen_anchor_kind": str(pack["qwen_anchor_kind"][anchor_index]),
            "qwen_anchor_count": int(qwen_anchor_kv.shape[0]),
        }
        return sample


@dataclass
class RDTBatchCollator:
    max_lang_tokens: int
    image_tokens: int
    pred_horizon: int
    feature_dim: int
    state_dim: int
    action_dim: int
    lang_token_dim: int | None = None
    img_token_dim: int | None = None

    def __post_init__(self) -> None:
        if self.lang_token_dim is None:
            self.lang_token_dim = self.feature_dim
        if self.img_token_dim is None:
            self.img_token_dim = self.feature_dim

    def _pad_sequence(
        self,
        tensor: torch.Tensor,
        length: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = torch.as_tensor(tensor)
        if tensor.ndim != 2 or tensor.shape[1] != width:
            raise ValueError(
                f"Expected [tokens, {width}], got {tuple(tensor.shape)}"
            )
        tensor = tensor[:length]
        valid = tensor.shape[0]
        output = torch.zeros(length, width, dtype=tensor.dtype)
        output[:valid] = tensor
        mask = torch.zeros(length, dtype=torch.bool)
        mask[:valid] = True
        return output, mask

    def _pad_actions(
        self,
        actions: torch.Tensor,
        provided_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = torch.as_tensor(actions, dtype=torch.float32)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(
                f"Expected actions [T, {self.action_dim}], got {tuple(actions.shape)}"
            )
        actions = actions[: self.pred_horizon]
        valid = actions.shape[0]
        output = torch.zeros(
            self.pred_horizon, self.action_dim, dtype=actions.dtype
        )
        output[:valid] = actions
        mask = torch.zeros(self.pred_horizon, dtype=torch.bool)
        mask[:valid] = True
        if provided_mask is not None:
            supplied = torch.as_tensor(provided_mask, dtype=torch.bool)
            supplied = supplied[: self.pred_horizon]
            mask[: supplied.shape[0]] &= supplied
        return output, mask

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch: dict[str, list[torch.Tensor]] = {
            "qwen_kv": [],
            "lang_tokens": [],
            "lang_mask": [],
            "img_tokens": [],
            "img_mask": [],
            "state": [],
            "actions": [],
            "action_time_mask": [],
            "action_dim_mask": [],
            "ctrl_freq": [],
        }

        for sample in samples:
            lang, default_lang_mask = self._pad_sequence(
                sample["lang_tokens"], self.max_lang_tokens, self.lang_token_dim
            )
            image, default_img_mask = self._pad_sequence(
                sample["img_tokens"], self.image_tokens, self.img_token_dim
            )

            qwen_kv = torch.as_tensor(sample["qwen_kv"], dtype=torch.float32)
            if qwen_kv.ndim == 1:
                qwen_kv = qwen_kv.unsqueeze(0)
            if qwen_kv.ndim != 2:
                raise ValueError(
                    f"Expected qwen_kv [tokens, dim] or [dim], got {tuple(qwen_kv.shape)}"
                )

            if "lang_mask" in sample:
                supplied = torch.as_tensor(sample["lang_mask"], dtype=torch.bool)
                supplied = supplied[: self.max_lang_tokens]
                default_lang_mask[: supplied.shape[0]] &= supplied
            if "img_mask" in sample:
                supplied = torch.as_tensor(sample["img_mask"], dtype=torch.bool)
                supplied = supplied[: self.image_tokens]
                default_img_mask[: supplied.shape[0]] &= supplied

            state = torch.as_tensor(sample["state"], dtype=torch.float32).flatten()
            if state.numel() != self.state_dim:
                raise ValueError(
                    f"Expected state dim {self.state_dim}, got {state.numel()} "
                    f"in {sample.get('_path', '<memory>')}"
                )

            actions, action_time_mask = self._pad_actions(
                sample["actions"], sample.get("action_time_mask")
            )
            action_dim_mask = torch.as_tensor(
                sample.get("action_dim_mask", torch.ones(self.action_dim)),
                dtype=torch.float32,
            ).flatten()
            if action_dim_mask.numel() != self.action_dim:
                raise ValueError("action_dim_mask has the wrong width")

            batch["qwen_kv"].append(qwen_kv)
            batch["lang_tokens"].append(lang.to(torch.float32))
            batch["lang_mask"].append(default_lang_mask)
            batch["img_tokens"].append(image.to(torch.float32))
            batch["img_mask"].append(default_img_mask)
            batch["state"].append(state)
            batch["actions"].append(actions)
            batch["action_time_mask"].append(action_time_mask)
            batch["action_dim_mask"].append(action_dim_mask)
            batch["ctrl_freq"].append(
                torch.tensor(float(sample["ctrl_freq"]), dtype=torch.float32)
            )

        return {key: torch.stack(values, dim=0) for key, values in batch.items()}


@dataclass
class RDTOnlineSiglipBatchCollator:
    max_lang_tokens: int
    pred_horizon: int
    feature_dim: int
    state_dim: int
    action_dim: int
    lang_token_dim: int | None = None

    def __post_init__(self) -> None:
        if self.lang_token_dim is None:
            self.lang_token_dim = self.feature_dim
        self._base = RDTBatchCollator(
            max_lang_tokens=self.max_lang_tokens,
            image_tokens=1,
            pred_horizon=self.pred_horizon,
            feature_dim=self.feature_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lang_token_dim=self.lang_token_dim,
            img_token_dim=1,
        )

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        tensor_batch: dict[str, list[torch.Tensor]] = {
            "qwen_kv": [],
            "lang_tokens": [],
            "lang_mask": [],
            "state": [],
            "actions": [],
            "action_time_mask": [],
            "action_dim_mask": [],
            "ctrl_freq": [],
            "image_slot_mask": [],
        }
        image_slot_jpegs: list[list[bytes]] = []

        for sample in samples:
            lang, default_lang_mask = self._base._pad_sequence(
                sample["lang_tokens"], self.max_lang_tokens, self.lang_token_dim
            )
            if "lang_mask" in sample:
                supplied = torch.as_tensor(sample["lang_mask"], dtype=torch.bool)
                supplied = supplied[: self.max_lang_tokens]
                default_lang_mask[: supplied.shape[0]] &= supplied

            qwen_kv = torch.as_tensor(sample["qwen_kv"], dtype=torch.float32)
            if qwen_kv.ndim == 1:
                qwen_kv = qwen_kv.unsqueeze(0)
            if qwen_kv.ndim != 2:
                raise ValueError(
                    f"Expected qwen_kv [tokens, dim] or [dim], got {tuple(qwen_kv.shape)}"
                )

            state = torch.as_tensor(sample["state"], dtype=torch.float32).flatten()
            if state.numel() != self.state_dim:
                raise ValueError(
                    f"Expected state dim {self.state_dim}, got {state.numel()} "
                    f"in {sample.get('_path', '<memory>')}"
                )

            actions, action_time_mask = self._base._pad_actions(
                sample["actions"], sample.get("action_time_mask")
            )
            action_dim_mask = torch.as_tensor(
                sample.get("action_dim_mask", torch.ones(self.action_dim)),
                dtype=torch.float32,
            ).flatten()
            if action_dim_mask.numel() != self.action_dim:
                raise ValueError("action_dim_mask has the wrong width")

            slot_mask = torch.as_tensor(sample["image_slot_mask"], dtype=torch.bool).flatten()
            image_slots = list(sample["image_slot_jpegs"])
            if len(image_slots) != int(slot_mask.numel()):
                raise ValueError(
                    f"image_slot_jpegs length {len(image_slots)} != mask length {slot_mask.numel()}"
                )

            tensor_batch["qwen_kv"].append(qwen_kv)
            tensor_batch["lang_tokens"].append(lang.to(torch.float32))
            tensor_batch["lang_mask"].append(default_lang_mask)
            tensor_batch["state"].append(state)
            tensor_batch["actions"].append(actions)
            tensor_batch["action_time_mask"].append(action_time_mask)
            tensor_batch["action_dim_mask"].append(action_dim_mask)
            tensor_batch["ctrl_freq"].append(
                torch.tensor(float(sample["ctrl_freq"]), dtype=torch.float32)
            )
            tensor_batch["image_slot_mask"].append(slot_mask)
            image_slot_jpegs.append(image_slots)

        batch: dict[str, Any] = {
            key: torch.stack(values, dim=0) for key, values in tensor_batch.items()
        }
        batch["image_slot_jpegs"] = image_slot_jpegs
        return batch
