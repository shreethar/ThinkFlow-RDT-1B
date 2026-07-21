from __future__ import annotations

import json
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


class CachedFeatureDataset(Dataset[dict[str, Any]]):
    """
    Stable indexed dataset backed by one .pt file per timestep.

    Each manifest line can be either:
      {"path": "relative/or/absolute/sample.pt"}
    or a plain JSON string containing the path.
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        self.base_dir = self.manifest_path.parent
        self.paths: list[Path] = []
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
                self.paths.append(path)
        if not self.paths:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        sample = torch.load(path, map_location="cpu", weights_only=False)
        missing = REQUIRED_KEYS.difference(sample)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")
        sample["_path"] = str(path)
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
