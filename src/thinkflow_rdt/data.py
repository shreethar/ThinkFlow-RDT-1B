from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Iterator

import torch
from torch.utils.data import Dataset, Sampler


RDT_GRIPPER_INDEX = 10
RDT_XYZ_SLICE = slice(30, 33)
RDT_ORTHO6D_SLICE = slice(33, 39)
RDT_EEF_DELTA_SLICE = slice(39, 45)
LIBERO_GRIPPER_QPOS_MIN = -0.04245
LIBERO_GRIPPER_QPOS_MAX = 0.05185


def euler_xyz_to_ortho6d(euler: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles to first-two-columns orthogonal 6-D."""
    roll, pitch, yaw = euler.unbind(dim=-1)
    cr, sr = roll.cos(), roll.sin()
    cp, sp = pitch.cos(), pitch.sin()
    cy, sy = yaw.cos(), yaw.sin()
    first_column = torch.stack([cy * cp, sy * cp, -sp], dim=-1)
    second_column = torch.stack(
        [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr],
        dim=-1,
    )
    return torch.cat([first_column, second_column], dim=-1)


def _load_action_stats_paths(
    paths: dict[str, str] | None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for dataset_id, value in (paths or {}).items():
        path = Path(value).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        block = payload.get("action_normalization", payload)
        q01 = torch.as_tensor(block["q01"], dtype=torch.float32)
        q99 = torch.as_tensor(block["q99"], dtype=torch.float32)
        if q01.shape != (7,) or q99.shape != (7,):
            raise ValueError(
                f"Expected seven-dimensional action stats in {path}, got "
                f"{tuple(q01.shape)} and {tuple(q99.shape)}"
            )
        result[str(dataset_id)] = (q01, q99)
    return result


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
PLAN_FEATURE_REQUIRED_KEYS = {
    "qwen_hidden_states",
    "latent_waypoints",
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

    Batched shard manifests are also supported:
      {"path": "shard_000000000.pt", "cache_layout": "sample_shard", "num_samples": 64}
    Shards are likewise expanded into sample-level items while retaining file-local
    ranges for efficient sampling.

    ``excluded_dataset_ids`` filters entries using manifest metadata, without
    opening their cache files. Plain-string entries have no dataset metadata and
    therefore cannot be filtered this way.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        required_keys: set[str] | frozenset[str] | None = None,
        excluded_dataset_ids: Collection[str] | None = None,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.required_keys = set(REQUIRED_KEYS if required_keys is None else required_keys)
        if isinstance(excluded_dataset_ids, str):
            excluded_dataset_ids = (excluded_dataset_ids,)
        self.excluded_dataset_ids = frozenset(
            str(dataset_id) for dataset_id in (excluded_dataset_ids or ())
        )
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        self.base_dir = self.manifest_path.parent
        self.entries: list[dict[str, Any]] = []
        manifest_entry_ranges: list[range] = []
        manifest_entry_dataset_ids: list[str | None] = []
        manifest_entry_episode_ids: list[str | None] = []
        episode_pack_ranges: list[range] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                manifest_dataset_id = (
                    item.get("dataset_id") or item.get("first_dataset_id")
                    if isinstance(item, dict)
                    else None
                )
                if (
                    manifest_dataset_id is not None
                    and str(manifest_dataset_id) in self.excluded_dataset_ids
                ):
                    continue
                path_value = item if isinstance(item, str) else item.get("path")
                if not path_value:
                    raise ValueError(
                        f"Manifest line {line_number} has no path: {self.manifest_path}"
                    )
                path = Path(path_value)
                if not path.is_absolute():
                    path = (self.base_dir / path).resolve()
                range_start = len(self.entries)
                cache_layout = item.get("cache_layout") if isinstance(item, dict) else None
                if cache_layout in {"episode_pack", "sample_shard"}:
                    num_samples = int(item.get("num_samples", 0))
                    if num_samples <= 0:
                        raise ValueError(
                            f"{cache_layout} manifest line {line_number} has invalid "
                            f"num_samples={num_samples}: {self.manifest_path}"
                        )
                    for sample_index in range(num_samples):
                        self.entries.append(
                            {
                                "path": path,
                                "cache_layout": cache_layout,
                                "sample_index": sample_index,
                            }
                        )
                    if cache_layout == "episode_pack":
                        episode_pack_ranges.append(range(range_start, len(self.entries)))
                else:
                    self.entries.append(
                        {
                            "path": path,
                            "cache_layout": "sample",
                            "sample_index": None,
                        }
                    )
                manifest_entry_ranges.append(range(range_start, len(self.entries)))
                manifest_entry_dataset_ids.append(
                    str(manifest_dataset_id)
                    if manifest_dataset_id is not None
                    else None
                )
                manifest_entry_episode_ids.append(
                    str(item.get("first_episode_id"))
                    if isinstance(item, dict) and item.get("first_episode_id")
                    else None
                )
        if not self.entries:
            if self.excluded_dataset_ids:
                raise ValueError(
                    f"Manifest has no entries after filtering: {self.manifest_path}"
                )
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        # Each manifest line expands into exactly one contiguous index range. These
        # immutable ranges let samplers preserve episode-pack I/O locality without
        # exposing or depending on the internal entry dictionaries.
        self.contiguous_ranges = tuple(manifest_entry_ranges)
        self.contiguous_range_dataset_ids = tuple(manifest_entry_dataset_ids)
        self.contiguous_range_episode_ids = tuple(manifest_entry_episode_ids)
        self.episode_pack_ranges = tuple(episode_pack_ranges)
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
        elif entry["cache_layout"] == "sample_shard":
            pack = self._load_sample_shard(path)
            sample = self._sample_from_sample_shard(
                pack,
                int(entry["sample_index"]),
                path=path,
            )
        else:
            sample = torch.load(path, map_location="cpu", weights_only=True)
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
        pack = torch.load(path, map_location="cpu", weights_only=True)
        if pack.get("cache_layout") != "episode_pack":
            raise ValueError(f"{path} is not an episode_pack cache file")
        self._pack_cache[path] = pack
        self._pack_cache.move_to_end(path)
        while len(self._pack_cache) > self._pack_cache_size:
            self._pack_cache.popitem(last=False)
        return pack

    def _load_sample_shard(self, path: Path) -> dict[str, Any]:
        cached = self._pack_cache.get(path)
        if cached is not None:
            self._pack_cache.move_to_end(path)
            return cached
        pack = torch.load(path, map_location="cpu", weights_only=True)
        if pack.get("cache_layout") != "sample_shard":
            raise ValueError(f"{path} is not a sample_shard cache file")
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
        anchor_kinds = list(pack.get("qwen_anchor_kind", []))
        original_anchor_kind = (
            str(anchor_kinds[anchor_index])
            if anchor_index < len(anchor_kinds)
            else "unknown"
        )
        # Older Part 3 packs always stored a second, uniformly sampled anchor
        # even when no gripper change existed. Treat it as the first-step anchor
        # without rewriting the large cache files.
        if original_anchor_kind in {"uniform", "uniform_fallback"}:
            anchor_index = 0

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

        raw_instructions = pack.get("instructions")
        if isinstance(raw_instructions, (list, tuple)) and sample_index < len(
            raw_instructions
        ):
            instruction = str(raw_instructions[sample_index])
        else:
            raw_instruction = pack.get("instruction")
            instruction = None if raw_instruction is None else str(raw_instruction)

        sample = {
            "qwen_kv": qwen_anchor_kv[anchor_index],
            "lang_tokens": pack["lang_tokens"],
            "lang_mask": pack["lang_mask"],
            "state": pack["state"][sample_index],
            "state_dim_mask": (
                pack["state_dim_mask"][sample_index]
                if "state_dim_mask" in pack
                else torch.ones_like(pack["state"][sample_index])
            ),
            "actions": pack["actions"][sample_index],
            "action_time_mask": pack["action_time_mask"][sample_index],
            "action_dim_mask": pack["action_dim_mask"][sample_index],
            "ctrl_freq": ctrl_freq,
            "image_slot_jpegs": image_slot_jpegs,
            "image_slot_mask": pack["sample_image_mask"][sample_index],
            "dataset_id": pack.get("dataset_id"),
            "episode_id": pack.get("episode_id"),
            "step_idx": step_idx,
            "instruction": instruction,
            "actions_normalized": bool(pack.get("actions_normalized", False)),
            "qwen_cache_scope": pack.get("qwen_cache_scope", "episode_anchors"),
            "qwen_anchor_step_idx": str(pack["qwen_anchor_step_idx"][anchor_index]),
            "qwen_anchor_kind": str(pack["qwen_anchor_kind"][anchor_index]),
            "qwen_anchor_original_kind": original_anchor_kind,
            "qwen_anchor_count": sum(
                str(kind) not in {"uniform", "uniform_fallback"}
                for kind in pack.get("qwen_anchor_kind", [])
            )
            or int(qwen_anchor_kv.shape[0]),
        }
        hidden_states = pack.get("qwen_anchor_hidden_states")
        if hidden_states is None:
            hidden_states = pack.get("qwen_hidden_states")
        if hidden_states is not None:
            sample["qwen_hidden_states"] = torch.as_tensor(hidden_states)[
                anchor_index
            ]
        if "latent_waypoints" in pack:
            sample["latent_waypoints"] = torch.as_tensor(
                pack["latent_waypoints"]
            )[sample_index]
        if "joint_state" in pack:
            sample["joint_state"] = torch.as_tensor(pack["joint_state"])[
                sample_index
            ]
        return sample

    def _sample_from_sample_shard(
        self,
        pack: dict[str, Any],
        sample_index: int,
        *,
        path: Path,
    ) -> dict[str, Any]:
        num_samples = int(pack["num_samples"])
        if sample_index < 0 or sample_index >= num_samples:
            raise IndexError(f"sample_index {sample_index} out of range for {path}")

        metadata_values = pack.get("metadata", [{} for _ in range(num_samples)])
        metadata = metadata_values[sample_index] if sample_index < len(metadata_values) else {}

        lang_tokens = pack.get("lang_tokens")
        lang_mask = pack.get("lang_mask")
        if lang_tokens is not None:
            sample_lang_index = pack.get("sample_lang_index")
            lang_index = (
                int(torch.as_tensor(sample_lang_index)[sample_index].item())
                if sample_lang_index is not None
                else sample_index
            )
            if isinstance(lang_tokens, list):
                lang_tokens = lang_tokens[lang_index]
                lang_mask = lang_mask[lang_index] if isinstance(lang_mask, list) else lang_mask
            else:
                lang_tokens = lang_tokens[lang_index]
                lang_mask = lang_mask[lang_index] if lang_mask is not None else None

        image_slot_jpegs = pack.get("image_slot_jpegs")
        if image_slot_jpegs is not None:
            image_slot_jpegs = list(image_slot_jpegs[sample_index])
        elif "image_jpegs" in pack and "sample_image_indices" in pack:
            image_pool = list(pack["image_jpegs"])
            image_indices = torch.as_tensor(
                pack["sample_image_indices"], dtype=torch.long
            )[sample_index].flatten().tolist()
            image_slot_jpegs = []
            for image_index in image_indices:
                if image_index < 0 or image_index >= len(image_pool):
                    raise IndexError(f"image index {image_index} out of range for {path}")
                image_slot_jpegs.append(image_pool[image_index])

        image_slot_mask = pack.get("image_slot_mask")
        if image_slot_mask is None:
            image_slot_mask = pack.get("sample_image_mask")
        if image_slot_mask is not None:
            image_slot_mask = image_slot_mask[sample_index]

        ctrl_freq = pack.get("ctrl_freq", 0.0)
        if isinstance(ctrl_freq, torch.Tensor) and ctrl_freq.ndim > 0:
            ctrl_freq = float(ctrl_freq[sample_index].item())
        else:
            ctrl_freq = float(ctrl_freq)

        sample = {
            "qwen_kv": torch.as_tensor(pack["qwen_kv"])[sample_index],
            "state": pack["state"][sample_index],
            "state_dim_mask": (
                pack["state_dim_mask"][sample_index]
                if "state_dim_mask" in pack
                else torch.ones_like(pack["state"][sample_index])
            ),
            "actions": pack["actions"][sample_index],
            "action_time_mask": pack["action_time_mask"][sample_index],
            "action_dim_mask": pack["action_dim_mask"][sample_index],
            "ctrl_freq": ctrl_freq,
            "dataset_id": metadata.get("dataset_id"),
            "episode_id": metadata.get("episode_id"),
            "step_idx": str(metadata.get("step_idx", sample_index)),
            "cache_layout": "sample_shard",
        }
        instructions = pack.get("instructions")
        if instructions is not None:
            if isinstance(instructions, (list, tuple)):
                instruction_index = (
                    sample_index
                    if len(instructions) == num_samples
                    else lang_index
                )
                if instruction_index < len(instructions):
                    sample["instruction"] = str(instructions[instruction_index])
            else:
                sample["instruction"] = str(instructions)
        if lang_tokens is not None:
            sample["lang_tokens"] = lang_tokens
        if lang_mask is not None:
            sample["lang_mask"] = lang_mask
        if image_slot_jpegs is not None:
            sample["image_slot_jpegs"] = image_slot_jpegs
        if image_slot_mask is not None:
            sample["image_slot_mask"] = image_slot_mask
        if "latent_waypoints" in pack:
            sample["latent_waypoints"] = pack["latent_waypoints"][sample_index]
        hidden_states = pack.get("qwen_hidden_states")
        if hidden_states is None:
            hidden_states = pack.get("qwen_anchor_hidden_states")
        if hidden_states is not None:
            sample["qwen_hidden_states"] = torch.as_tensor(hidden_states)[
                sample_index
            ]
        if "joint_state" in pack:
            sample["joint_state"] = torch.as_tensor(pack["joint_state"])[
                sample_index
            ]
        return sample


class EpisodePackSampler(Sampler[int]):
    """
    Shuffle manifest entries and their samples while keeping each pack contiguous.

    Sample-record manifest entries are treated as singleton ranges, so the sampler
    remains usable with legacy and mixed-layout manifests. Call ``set_epoch`` at
    each epoch boundary to get a deterministic new order.
    """

    def __init__(
        self,
        data_source: CachedFeatureDataset,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.data_source = data_source
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        ranges = self.data_source.contiguous_ranges
        if not self.shuffle:
            for index_range in ranges:
                yield from index_range
            return

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        range_order = torch.randperm(len(ranges), generator=generator).tolist()
        for range_index in range_order:
            index_range = ranges[range_index]
            offsets = torch.randperm(len(index_range), generator=generator).tolist()
            for offset in offsets:
                yield index_range.start + offset

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class FixedStratifiedSampler(Sampler[int]):
    """Fixed suite/task/demo-diverse validation order.

    The first pass emits one shuffled sample from each shuffled manifest range,
    round-robin across tasks and suites. For 32-sample validation batches over
    four LIBERO suites, this yields eight suites/tasks/demos representatives per
    suite instead of 32 adjacent steps from one shard. The order is regenerated
    from the same seed on every iteration, so checkpoint comparisons remain
    sample-for-sample fixed.
    """

    def __init__(self, data_source: CachedFeatureDataset, *, seed: int = 0) -> None:
        self.data_source = data_source
        self.seed = int(seed)

    def __iter__(self) -> Iterator[int]:
        grouped_ranges: dict[str, dict[str, list[range]]] = {}
        for range_index, (index_range, dataset_id) in enumerate(zip(
            self.data_source.contiguous_ranges,
            self.data_source.contiguous_range_dataset_ids,
        )):
            episode_id = self.data_source.contiguous_range_episode_ids[range_index]
            # LIBERO episode IDs are ``<task>_demo:demo_<n>``. Grouping by the
            # prefix makes the initial selection cover different tasks before
            # taking another demonstration/shard from the same task.
            task_id = (
                episode_id.rsplit("_demo:", 1)[0]
                if episode_id and "_demo:" in episode_id
                else episode_id or f"<range-{range_index}>"
            )
            grouped_ranges.setdefault(dataset_id or "<unknown>", {}).setdefault(
                task_id, []
            ).append(index_range)

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        streams: list[list[int]] = []
        for dataset_id in sorted(grouped_ranges):
            by_task = grouped_ranges[dataset_id]
            task_names = sorted(by_task)
            task_order = torch.randperm(
                len(task_names), generator=generator
            ).tolist()
            shuffled_by_task: list[list[range]] = []
            for task_index in task_order:
                ranges = by_task[task_names[task_index]]
                order = torch.randperm(len(ranges), generator=generator).tolist()
                shuffled_by_task.append([ranges[index] for index in order])

            # Interleave task range lists: first one range per task, then the
            # second range per task, etc. This is what gives the first batch its
            # task diversity.
            range_order: list[range] = []
            depth = 0
            while True:
                emitted = False
                for ranges in shuffled_by_task:
                    if depth < len(ranges):
                        range_order.append(ranges[depth])
                        emitted = True
                if not emitted:
                    break
                depth += 1

            randomized_ranges: list[list[int]] = []
            for index_range in range_order:
                # The merged manifest identifies the first episode in each
                # shard. Keep that representative first so its suite/task/demo
                # label is exact; shuffle the remaining shard samples.
                offsets = [0]
                if len(index_range) > 1:
                    offsets.extend(
                        1 + offset
                        for offset in torch.randperm(
                            len(index_range) - 1,
                            generator=generator,
                        ).tolist()
                    )
                randomized_ranges.append(
                    [index_range.start + offset for offset in offsets]
                )

            # Emit one sample per range before returning to the next sample in
            # that range. This prevents shard-local adjacent trajectories from
            # occupying an entire qualitative batch.
            stream = []
            depth = 0
            while True:
                emitted = False
                for indices in randomized_ranges:
                    if depth < len(indices):
                        stream.append(indices[depth])
                        emitted = True
                if not emitted:
                    break
                depth += 1
            streams.append(stream)

        positions = [0] * len(streams)
        remaining = sum(len(stream) for stream in streams)
        while remaining:
            for stream_index, stream in enumerate(streams):
                position = positions[stream_index]
                if position >= len(stream):
                    continue
                yield stream[position]
                positions[stream_index] += 1
                remaining -= 1

    def __len__(self) -> int:
        return len(self.data_source)


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
    qwen_kv_dim: int | None = None
    plan_hidden_dim: int | None = None
    spatial_token_count: int = 5
    waypoint_dim: int = 2
    require_plan_features: bool = False
    convert_cached_gripper_closed_to_open: bool = True
    cache_state_dim: int | None = None
    cache_action_dim: int | None = None
    native_rdt_128: bool = False
    native_rdt_128_mapping: str = "eef_pose_ortho6d"
    action_stats_paths: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.lang_token_dim is None:
            self.lang_token_dim = self.feature_dim
        if self.img_token_dim is None:
            self.img_token_dim = self.feature_dim
        if self.cache_state_dim is None:
            self.cache_state_dim = self.state_dim
        if self.cache_action_dim is None:
            self.cache_action_dim = self.action_dim
        if self.plan_hidden_dim is None:
            self.plan_hidden_dim = self.feature_dim
        if self.spatial_token_count <= 0:
            raise ValueError("spatial_token_count must be positive")
        if self.waypoint_dim <= 0:
            raise ValueError("waypoint_dim must be positive")
        self._action_stats = _load_action_stats_paths(self.action_stats_paths)
        if self.native_rdt_128:
            if self.state_dim != 128 or self.action_dim != 128:
                raise ValueError("native_rdt_128 requires state_dim=action_dim=128")
            if self.native_rdt_128_mapping not in {
                "eef_pose_ortho6d",
                "libero_joint_eef_delta",
            }:
                raise ValueError("Unsupported native_rdt_128_mapping")
            allowed = (
                {(8, 7)}
                if self.native_rdt_128_mapping == "libero_joint_eef_delta"
                else {(7, 7), (11, 10)}
            )
            if (self.cache_state_dim, self.cache_action_dim) not in allowed:
                raise ValueError(
                    "native_rdt_128 cache dimensions do not match mapping "
                    f"{self.native_rdt_128_mapping!r}"
                )

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
        if actions.ndim != 2 or actions.shape[1] != self.cache_action_dim:
            raise ValueError(
                f"Expected cached actions [T, {self.cache_action_dim}], got "
                f"{tuple(actions.shape)}"
            )
        actions = actions[: self.pred_horizon]
        valid = actions.shape[0]
        output = torch.zeros(
            self.pred_horizon, self.cache_action_dim, dtype=actions.dtype
        )
        output[:valid] = actions
        mask = torch.zeros(self.pred_horizon, dtype=torch.bool)
        mask[:valid] = True
        if provided_mask is not None:
            supplied = torch.as_tensor(provided_mask, dtype=torch.bool)
            supplied = supplied[: self.pred_horizon]
            mask[: supplied.shape[0]] &= supplied
        return output, mask

    def _to_native_rdt_state(
        self,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        *,
        joint_state: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = state.new_zeros(128)
        mask = state_mask.new_zeros(128)
        if self.native_rdt_128_mapping == "libero_joint_eef_delta":
            if joint_state is None:
                raise KeyError(
                    "libero_joint_eef_delta requires joint_state in every cache sample"
                )
            joints = torch.as_tensor(
                joint_state,
                dtype=state.dtype,
                device=state.device,
            ).flatten()
            if joints.numel() < 7:
                raise ValueError(
                    f"Expected at least seven LIBERO joints, got {joints.numel()}"
                )
            if state.numel() != 8:
                raise ValueError(
                    "libero_joint_eef_delta requires cached native LIBERO state8"
                )
            if not bool(torch.isfinite(joints[:7]).all()):
                raise ValueError("joint_state contains NaN or Inf")
            values[:7] = joints[:7]
            mask[:7] = 1
            gripper = (
                state[6:8] - LIBERO_GRIPPER_QPOS_MIN
            ) / (LIBERO_GRIPPER_QPOS_MAX - LIBERO_GRIPPER_QPOS_MIN)
            values[10:12] = gripper * state_mask[6:8]
            mask[10:12] = state_mask[6:8]
            return values, mask
        values[RDT_XYZ_SLICE] = state[:3] * state_mask[:3]
        mask[RDT_XYZ_SLICE] = state_mask[:3]
        if self.cache_state_dim == 11:
            # LIBERO cache: xyz + absolute ortho6D + two raw finger states.
            values[RDT_ORTHO6D_SLICE] = state[3:9] * state_mask[3:9]
            mask[RDT_ORTHO6D_SLICE] = state_mask[3:9]
            values[10:12] = state[9:11] * state_mask[9:11]
            mask[10:12] = state_mask[9:11]
            return values, mask
        rotation_valid = state_mask[3:6].amin()
        rotation = euler_xyz_to_ortho6d(state[3:6])
        values[RDT_ORTHO6D_SLICE] = rotation * rotation_valid
        mask[RDT_ORTHO6D_SLICE] = rotation_valid
        # Cached proprioception is gripper_closed in [0,1]; native RDT uses
        # gripper_open in the same range.
        values[RDT_GRIPPER_INDEX] = (1.0 - state[6]) * state_mask[6]
        mask[RDT_GRIPPER_INDEX] = state_mask[6]
        return values, mask

    def _to_native_rdt_actions(
        self,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        dataset_id: str,
        actions_normalized: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = actions.new_zeros(actions.shape[0], 128)
        mask = action_mask.new_zeros(128)

        if self.native_rdt_128_mapping == "libero_joint_eef_delta":
            if self.cache_action_dim != 7 or actions_normalized:
                raise ValueError(
                    "libero_joint_eef_delta requires unnormalized raw 7-D LIBERO actions"
                )
            values[:, RDT_EEF_DELTA_SLICE] = (
                actions[:, :6] * action_mask[:6]
            )
            mask[RDT_EEF_DELTA_SLICE] = action_mask[:6]
            values[:, RDT_GRIPPER_INDEX] = actions[:, 6] * action_mask[6]
            mask[RDT_GRIPPER_INDEX] = action_mask[6]
            return values, mask

        if self.cache_action_dim == 10:
            if actions_normalized:
                raise ValueError(
                    "LIBERO 10-D ortho6D actions must be cached without "
                    "normalization"
                )
            # LIBERO cache: dxyz + relative ortho6D + raw gripper command.
            values[:, RDT_XYZ_SLICE] = actions[:, :3] * action_mask[:3]
            mask[RDT_XYZ_SLICE] = action_mask[:3]
            values[:, RDT_ORTHO6D_SLICE] = actions[:, 3:9] * action_mask[3:9]
            mask[RDT_ORTHO6D_SLICE] = action_mask[3:9]
            values[:, RDT_GRIPPER_INDEX] = actions[:, 9] * action_mask[9]
            mask[RDT_GRIPPER_INDEX] = action_mask[9]
            return values, mask

        # Keep the already normalized XYZ commands. Euler angles must first be
        # restored to radians; applying trigonometry to standardized values is
        # not a valid rotation conversion.
        values[:, RDT_XYZ_SLICE] = actions[:, :3] * action_mask[:3]
        mask[RDT_XYZ_SLICE] = action_mask[:3]
        rotation_euler = actions[:, 3:6]
        if actions_normalized:
            stats = self._action_stats.get(dataset_id)
            if stats is None:
                raise KeyError(
                    "Native 128-D conversion needs q01/q99 action stats for "
                    f"dataset {dataset_id!r}; configure data.action_stats_paths"
                )
            q01, q99 = stats
            rotation_euler = (
                (rotation_euler.clamp(-1.0, 1.0) + 1.0)
                * 0.5
                * (q99[3:6] - q01[3:6])
                + q01[3:6]
            )
        rotation_valid = action_mask[3:6].amin()
        rotation = euler_xyz_to_ortho6d(rotation_euler)
        values[:, RDT_ORTHO6D_SLICE] = rotation * rotation_valid
        mask[RDT_ORTHO6D_SLICE] = rotation_valid

        if actions_normalized:
            # q01/q99 maps cached gripper_closed to [-1,+1]. Native RDT's
            # gripper_open convention is the exact sign inverse.
            gripper_open = -actions[:, 6]
        else:
            gripper_open = 1.0 - actions[:, 6]
        values[:, RDT_GRIPPER_INDEX] = gripper_open * action_mask[6]
        mask[RDT_GRIPPER_INDEX] = action_mask[6]
        return values, mask

    def _prepare_qwen_kv(self, value: Any) -> torch.Tensor:
        qwen_kv = torch.as_tensor(value)
        if qwen_kv.ndim == 1:
            qwen_kv = qwen_kv.unsqueeze(0)
        if qwen_kv.ndim != 2:
            raise ValueError(
                f"Expected qwen_kv [tokens, dim] or [dim], got {tuple(qwen_kv.shape)}"
            )
        if self.qwen_kv_dim is not None and qwen_kv.shape[1] != self.qwen_kv_dim:
            raise ValueError(
                f"Expected qwen_kv width {self.qwen_kv_dim}, "
                f"got {qwen_kv.shape[1]}"
            )
        return qwen_kv

    def _prepare_plan_features(
        self,
        hidden_value: Any,
        waypoint_value: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.as_tensor(hidden_value)
        waypoints = torch.as_tensor(waypoint_value, dtype=torch.float32)
        expected_hidden = (self.spatial_token_count, self.plan_hidden_dim)
        expected_waypoints = (self.spatial_token_count, self.waypoint_dim)
        if tuple(hidden.shape) != expected_hidden:
            raise ValueError(
                "Expected qwen_hidden_states "
                f"{expected_hidden}, got {tuple(hidden.shape)}"
            )
        if tuple(waypoints.shape) != expected_waypoints:
            raise ValueError(
                f"Expected latent_waypoints {expected_waypoints}, got "
                f"{tuple(waypoints.shape)}"
            )
        if not bool(torch.isfinite(hidden.float()).all()):
            raise ValueError("qwen_hidden_states contains NaN or Inf")
        if not bool(torch.isfinite(waypoints).all()):
            raise ValueError("latent_waypoints contains NaN or Inf")
        return hidden, waypoints

    def _convert_cached_gripper_to_rdt_open(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        *,
        actions_normalized: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # IMPORTANT: cached shards keep the project/data convention in dim 6:
        # gripper_closed, where 1=closed and 0=open. The pretrained RDT action
        # model uses the opposite binary convention in dim 6: gripper_open,
        # where 0=closed and 1=open. Flip at load/collate time so old caches
        # remain usable while all model inputs and targets match RDT.
        state = state.clone()
        actions = actions.clone()
        state[..., 6] = 1.0 - state[..., 6]
        if actions_normalized:
            # q01/q99 normalization maps binary gripper_closed 0/1 to -1/+1.
            # Switching to gripper_open in the same normalized range is a sign
            # inversion, not ``1 - x`` (which would incorrectly yield 2/0).
            actions[..., 6] = -actions[..., 6]
        else:
            actions[..., 6] = 1.0 - actions[..., 6]
        return state, actions

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, list[torch.Tensor]] = {
            "qwen_kv": [],
            "lang_tokens": [],
            "lang_mask": [],
            "img_tokens": [],
            "img_mask": [],
            "state": [],
            "state_dim_mask": [],
            "actions": [],
            "action_time_mask": [],
            "action_dim_mask": [],
            "ctrl_freq": [],
        }
        if self.require_plan_features:
            batch["qwen_hidden_states"] = []
            batch["latent_waypoints"] = []
            batch["plan_mask"] = []
        dataset_ids: list[str] = []
        instructions: list[str] = []
        episode_ids: list[str] = []
        step_indices: list[str] = []

        for sample in samples:
            lang, default_lang_mask = self._pad_sequence(
                sample["lang_tokens"], self.max_lang_tokens, self.lang_token_dim
            )
            image, default_img_mask = self._pad_sequence(
                sample["img_tokens"], self.image_tokens, self.img_token_dim
            )

            qwen_kv = self._prepare_qwen_kv(sample["qwen_kv"])
            plan_features: tuple[torch.Tensor, torch.Tensor] | None = None
            if self.require_plan_features:
                missing = PLAN_FEATURE_REQUIRED_KEYS.difference(sample)
                if missing:
                    raise KeyError(
                        "Hidden/waypoint fusion requires cache keys: "
                        + ", ".join(sorted(missing))
                    )
                plan_features = self._prepare_plan_features(
                    sample["qwen_hidden_states"],
                    sample["latent_waypoints"],
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
            if state.numel() != self.cache_state_dim:
                raise ValueError(
                    f"Expected cached state dim {self.cache_state_dim}, got "
                    f"{state.numel()} "
                    f"in {sample.get('_path', '<memory>')}"
                )

            actions, action_time_mask = self._pad_actions(
                sample["actions"], sample.get("action_time_mask")
            )
            if self.convert_cached_gripper_closed_to_open and not self.native_rdt_128:
                state, actions = self._convert_cached_gripper_to_rdt_open(
                    state,
                    actions,
                    actions_normalized=bool(sample.get("actions_normalized", False)),
                )
            state_dim_mask = torch.as_tensor(
                sample.get("state_dim_mask", torch.ones(self.cache_state_dim)),
                dtype=torch.float32,
            ).flatten()
            if state_dim_mask.numel() != self.cache_state_dim:
                raise ValueError("state_dim_mask has the wrong width")
            action_dim_mask = torch.as_tensor(
                sample.get("action_dim_mask", torch.ones(self.cache_action_dim)),
                dtype=torch.float32,
            ).flatten()
            if action_dim_mask.numel() != self.cache_action_dim:
                raise ValueError("action_dim_mask has the wrong width")
            dataset_id = str(sample.get("dataset_id") or "unknown")
            if self.native_rdt_128:
                state, state_dim_mask = self._to_native_rdt_state(
                    state,
                    state_dim_mask,
                    joint_state=sample.get("joint_state"),
                )
                actions, action_dim_mask = self._to_native_rdt_actions(
                    actions,
                    action_dim_mask,
                    dataset_id=dataset_id,
                    actions_normalized=bool(sample.get("actions_normalized", False)),
                )

            batch["qwen_kv"].append(qwen_kv)
            if plan_features is not None:
                hidden_states, latent_waypoints = plan_features
                batch["qwen_hidden_states"].append(hidden_states)
                batch["latent_waypoints"].append(latent_waypoints)
                batch["plan_mask"].append(
                    torch.ones(self.spatial_token_count, dtype=torch.bool)
                )
            batch["lang_tokens"].append(lang)
            batch["lang_mask"].append(default_lang_mask)
            # Preserve the cached feature dtype (normally BF16). Expanding every
            # image token to FP32 here doubles host/pinned-memory use, and the
            # model casts it back to its compute dtype immediately anyway.
            batch["img_tokens"].append(image)
            batch["img_mask"].append(default_img_mask)
            batch["state"].append(state)
            batch["state_dim_mask"].append(state_dim_mask)
            batch["actions"].append(actions)
            batch["action_time_mask"].append(action_time_mask)
            batch["action_dim_mask"].append(action_dim_mask)
            batch["ctrl_freq"].append(
                torch.tensor(float(sample["ctrl_freq"]), dtype=torch.float32)
            )
            dataset_ids.append(dataset_id)
            instructions.append(str(sample.get("instruction") or ""))
            episode_ids.append(str(sample.get("episode_id") or ""))
            step_indices.append(str(sample.get("step_idx") or ""))

        output: dict[str, Any] = {
            key: torch.stack(values, dim=0) for key, values in batch.items()
        }
        output["dataset_id"] = dataset_ids
        output["instruction"] = instructions
        output["episode_id"] = episode_ids
        output["step_idx"] = step_indices
        return output


@dataclass
class RDTOnlineSiglipBatchCollator:
    max_lang_tokens: int
    pred_horizon: int
    feature_dim: int
    state_dim: int
    action_dim: int
    lang_token_dim: int | None = None
    qwen_kv_dim: int | None = None
    plan_hidden_dim: int | None = None
    spatial_token_count: int = 5
    waypoint_dim: int = 2
    require_plan_features: bool = False
    convert_cached_gripper_closed_to_open: bool = True
    cache_state_dim: int | None = None
    cache_action_dim: int | None = None
    native_rdt_128: bool = False
    native_rdt_128_mapping: str = "eef_pose_ortho6d"
    action_stats_paths: dict[str, str] | None = None

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
            qwen_kv_dim=self.qwen_kv_dim,
            plan_hidden_dim=self.plan_hidden_dim,
            spatial_token_count=self.spatial_token_count,
            waypoint_dim=self.waypoint_dim,
            require_plan_features=self.require_plan_features,
            convert_cached_gripper_closed_to_open=(
                self.convert_cached_gripper_closed_to_open
            ),
            cache_state_dim=self.cache_state_dim,
            cache_action_dim=self.cache_action_dim,
            native_rdt_128=self.native_rdt_128,
            native_rdt_128_mapping=self.native_rdt_128_mapping,
            action_stats_paths=self.action_stats_paths,
        )

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        tensor_batch: dict[str, list[torch.Tensor]] = {
            "qwen_kv": [],
            "lang_tokens": [],
            "lang_mask": [],
            "state": [],
            "state_dim_mask": [],
            "actions": [],
            "action_time_mask": [],
            "action_dim_mask": [],
            "ctrl_freq": [],
            "image_slot_mask": [],
        }
        if self.require_plan_features:
            tensor_batch["qwen_hidden_states"] = []
            tensor_batch["latent_waypoints"] = []
            tensor_batch["plan_mask"] = []
        image_slot_jpegs: list[list[bytes]] = []
        dataset_ids: list[str] = []
        instructions: list[str] = []
        episode_ids: list[str] = []
        step_indices: list[str] = []

        for sample in samples:
            lang, default_lang_mask = self._base._pad_sequence(
                sample["lang_tokens"], self.max_lang_tokens, self.lang_token_dim
            )
            if "lang_mask" in sample:
                supplied = torch.as_tensor(sample["lang_mask"], dtype=torch.bool)
                supplied = supplied[: self.max_lang_tokens]
                default_lang_mask[: supplied.shape[0]] &= supplied

            qwen_kv = self._base._prepare_qwen_kv(sample["qwen_kv"])
            plan_features: tuple[torch.Tensor, torch.Tensor] | None = None
            if self.require_plan_features:
                missing = PLAN_FEATURE_REQUIRED_KEYS.difference(sample)
                if missing:
                    raise KeyError(
                        "Hidden/waypoint fusion requires cache keys: "
                        + ", ".join(sorted(missing))
                    )
                plan_features = self._base._prepare_plan_features(
                    sample["qwen_hidden_states"],
                    sample["latent_waypoints"],
                )

            state = torch.as_tensor(sample["state"], dtype=torch.float32).flatten()
            if state.numel() != self._base.cache_state_dim:
                raise ValueError(
                    f"Expected cached state dim {self._base.cache_state_dim}, got "
                    f"{state.numel()} "
                    f"in {sample.get('_path', '<memory>')}"
                )

            actions, action_time_mask = self._base._pad_actions(
                sample["actions"], sample.get("action_time_mask")
            )
            if self.convert_cached_gripper_closed_to_open and not self.native_rdt_128:
                state, actions = self._base._convert_cached_gripper_to_rdt_open(
                    state,
                    actions,
                    actions_normalized=bool(sample.get("actions_normalized", False)),
                )
            state_dim_mask = torch.as_tensor(
                sample.get(
                    "state_dim_mask", torch.ones(self._base.cache_state_dim)
                ),
                dtype=torch.float32,
            ).flatten()
            if state_dim_mask.numel() != self._base.cache_state_dim:
                raise ValueError("state_dim_mask has the wrong width")
            action_dim_mask = torch.as_tensor(
                sample.get(
                    "action_dim_mask", torch.ones(self._base.cache_action_dim)
                ),
                dtype=torch.float32,
            ).flatten()
            if action_dim_mask.numel() != self._base.cache_action_dim:
                raise ValueError("action_dim_mask has the wrong width")
            dataset_id = str(sample.get("dataset_id") or "unknown")
            if self.native_rdt_128:
                state, state_dim_mask = self._base._to_native_rdt_state(
                    state, state_dim_mask
                )
                actions, action_dim_mask = self._base._to_native_rdt_actions(
                    actions,
                    action_dim_mask,
                    dataset_id=dataset_id,
                    actions_normalized=bool(sample.get("actions_normalized", False)),
                )

            slot_mask = torch.as_tensor(sample["image_slot_mask"], dtype=torch.bool).flatten()
            image_slots = list(sample["image_slot_jpegs"])
            if len(image_slots) != int(slot_mask.numel()):
                raise ValueError(
                    f"image_slot_jpegs length {len(image_slots)} != mask length {slot_mask.numel()}"
                )

            tensor_batch["qwen_kv"].append(qwen_kv)
            if plan_features is not None:
                hidden_states, latent_waypoints = plan_features
                tensor_batch["qwen_hidden_states"].append(hidden_states)
                tensor_batch["latent_waypoints"].append(latent_waypoints)
                tensor_batch["plan_mask"].append(
                    torch.ones(self.spatial_token_count, dtype=torch.bool)
                )
            tensor_batch["lang_tokens"].append(lang)
            tensor_batch["lang_mask"].append(default_lang_mask)
            tensor_batch["state"].append(state)
            tensor_batch["state_dim_mask"].append(state_dim_mask)
            tensor_batch["actions"].append(actions)
            tensor_batch["action_time_mask"].append(action_time_mask)
            tensor_batch["action_dim_mask"].append(action_dim_mask)
            tensor_batch["ctrl_freq"].append(
                torch.tensor(float(sample["ctrl_freq"]), dtype=torch.float32)
            )
            tensor_batch["image_slot_mask"].append(slot_mask)
            image_slot_jpegs.append(image_slots)
            dataset_ids.append(dataset_id)
            instructions.append(str(sample.get("instruction") or ""))
            episode_ids.append(str(sample.get("episode_id") or ""))
            step_indices.append(str(sample.get("step_idx") or ""))

        batch: dict[str, Any] = {
            key: torch.stack(values, dim=0) for key, values in tensor_batch.items()
        }
        batch["image_slot_jpegs"] = image_slot_jpegs
        batch["dataset_id"] = dataset_ids
        batch["instruction"] = instructions
        batch["episode_id"] = episode_ids
        batch["step_idx"] = step_indices
        return batch
