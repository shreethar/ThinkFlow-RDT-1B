#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    T5EncoderModel,
    T5Tokenizer,
)
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from precompute_all_features import (  # noqa: E402
    IMAGE_KEYS,
    QWEN_TRAJECTORY_PROMPT_TEMPLATE,
    SPLIT_NAMES,
    apply_qwen_chat_template,
    build_lazy_configs,
    compact_tokens,
    format_qwen_trajectory_prompt,
    image_to_lossless_png_bytes,
    layer_key_values_from_past,
    prepare_split_output,
    resolve_model_id,
    standardized_collate_fn,
)
from thinkflow_rdt.adapters.combined_lazy import (  # noqa: E402
    LIBERO_DATASET_IDS,
    build_combined_standardized_splits,
)
from thinkflow_rdt.adapters.sample_filtering import (  # noqa: E402
    DEFAULT_CLOSE_TO_OPEN_AFTER,
    DEFAULT_CLOSE_TO_OPEN_BEFORE,
    DEFAULT_GRIPPER_WINDOW_AFTER,
    DEFAULT_GRIPPER_WINDOW_BEFORE,
    DEFAULT_MAX_SAMPLES_PER_EPISODE,
    DEFAULT_OPEN_TO_CLOSE_AFTER,
    DEFAULT_OPEN_TO_CLOSE_BEFORE,
)
from thinkflow_rdt.config import load_config  # noqa: E402


def import_latent_student(code_dir: Path | None) -> type[Any]:
    if code_dir is not None:
        resolved = code_dir.expanduser().resolve()
        if str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
    try:
        from models.latent_student import LatentStudent
    except ImportError as exc:
        raise ImportError(
            "Could not import models.latent_student.LatentStudent. "
            "Pass --latent-student-code-dir /path/to/VLA-FYP/train/stage2."
        ) from exc
    return LatentStudent


def tokenizer_end_think_id(tokenizer: Any) -> int:
    token_id = tokenizer.convert_tokens_to_ids("</think>")
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unk_id:
        encoded = tokenizer.encode("</think>", add_special_tokens=False)
        if not encoded:
            raise ValueError("Tokenizer could not encode </think>")
        token_id = encoded[-1]
    return int(token_id)


def load_spatial_parameters(student: Any, spatial_path: str | Path) -> None:
    spatial_path = Path(spatial_path).expanduser()
    if not spatial_path.exists():
        raise FileNotFoundError(spatial_path)
    state = torch.load(spatial_path, map_location="cpu", weights_only=False)
    if "spatial_tokens" in state:
        student.spatial_tokens.data.copy_(state["spatial_tokens"])
    else:
        raise KeyError(f"{spatial_path} is missing spatial_tokens")
    if "spatial_mlp" in state:
        student.spatial_mlp.load_state_dict(state["spatial_mlp"])
    else:
        raise KeyError(f"{spatial_path} is missing spatial_mlp")
    print(f"Loaded spatial parameters from {spatial_path}")


def load_local_spatial_parameters_if_present(student: Any, model_id: str) -> None:
    model_path = Path(model_id).expanduser()
    if not model_path.exists():
        return
    for filename in ("spatial_parameters.pt", "training_state.pt"):
        spatial_path = model_path / filename
        if spatial_path.exists():
            load_spatial_parameters(student, spatial_path)
            return


@contextmanager
def override_image_text_attention(implementation: str | None):
    """Temporarily override nested VLM loading for rollout-only compatibility."""

    if implementation is None:
        yield
        return
    original = AutoModelForImageTextToText.from_pretrained

    def from_pretrained(*model_args: Any, **model_kwargs: Any):
        # LatentStudent checkpoints can serialize flash_attention_2 in their
        # base config. Simulation environments need not compile FlashAttention,
        # so explicitly override that saved preference with SDPA when requested.
        model_kwargs["attn_implementation"] = implementation
        return original(*model_args, **model_kwargs)

    AutoModelForImageTextToText.from_pretrained = from_pretrained
    try:
        yield
    finally:
        AutoModelForImageTextToText.from_pretrained = original


def load_student_and_processor(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    processor_id = args.processor_id or args.student_model_id
    processor = AutoProcessor.from_pretrained(processor_id, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    end_think_id = tokenizer_end_think_id(processor.tokenizer)

    LatentStudent = import_latent_student(args.latent_student_code_dir)
    with override_image_text_attention(getattr(args, "attn_implementation", None)):
        student = LatentStudent.from_pretrained(
            args.student_model_id,
            end_think_token_id=end_think_id,
            M=args.latent_count,
            K=args.spatial_token_count,
        )
    if args.spatial_parameters_path is not None:
        load_spatial_parameters(student, args.spatial_parameters_path)
    else:
        load_local_spatial_parameters_if_present(student, args.student_model_id)
    spatial_shape = tuple(student.spatial_tokens.shape)
    if spatial_shape[0] != args.spatial_token_count:
        raise ValueError(
            f"Loaded student spatial token count {spatial_shape[0]} does not match "
            f"--spatial-token-count {args.spatial_token_count}"
        )
    print(f"Using latent student spatial_tokens shape={spatial_shape}")
    student.eval()
    student.requires_grad_(False)
    student.to(device)
    return student, processor


def load_t5(args: argparse.Namespace, cfg: Any) -> tuple[Any, Any]:
    t5_model_id = resolve_model_id(args.t5_model_id, args.t5_fallback_model_id)
    tokenizer = T5Tokenizer.from_pretrained(t5_model_id)
    if args.t5_precision == "8bit":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "--t5-precision 8bit requires transformers BitsAndBytesConfig "
                "and a bitsandbytes-capable environment."
            ) from exc
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        encoder = T5EncoderModel.from_pretrained(
            t5_model_id,
            quantization_config=quantization_config,
            device_map=args.device_map,
        )
    else:
        encoder = T5EncoderModel.from_pretrained(
            t5_model_id,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
        )
    encoder.eval()
    encoder.requires_grad_(False)
    if encoder.config.d_model != cfg.model.lang_token_dim:
        raise ValueError(
            f"T5 d_model {encoder.config.d_model} != cfg.model.lang_token_dim {cfg.model.lang_token_dim}"
        )
    return tokenizer, encoder


def t5_device_from_encoder(encoder: Any, fallback: torch.device) -> torch.device:
    try:
        return next(encoder.parameters()).device
    except StopIteration:
        return fallback


def batch_to_latent_student_inputs(
    batch: dict[str, Any],
    processor: Any,
    *,
    prompt_template: str,
    device: torch.device,
) -> dict[str, Any]:
    texts: list[str] = []
    images: list[list[Image.Image]] = []
    for instruction, image_group in zip(batch["instructions"], batch["qwen_images"]):
        qwen_instruction = format_qwen_trajectory_prompt(instruction, prompt_template)
        content = [{"type": "image", "image": image} for image in image_group]
        content.append({"type": "text", "text": qwen_instruction})
        messages = [{"role": "user", "content": content}]
        texts.append(
            apply_qwen_chat_template(
                processor,
                messages,
                enable_thinking=True,
            )
        )
        images.append(image_group)

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def flatten_spatial_layer_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    spatial_token_count: int,
    expected_dim: int,
) -> torch.Tensor:
    if keys.ndim != 4 or values.ndim != 4:
        raise ValueError(
            f"Expected K/V caches [B, H, T, D], got {tuple(keys.shape)} and {tuple(values.shape)}"
        )
    if keys.shape != values.shape:
        raise ValueError(f"Key/value cache shapes differ: {tuple(keys.shape)} vs {tuple(values.shape)}")
    if keys.shape[2] < spatial_token_count:
        raise ValueError(
            f"Cache length {keys.shape[2]} is smaller than spatial_token_count={spatial_token_count}"
        )

    key_tokens = keys[:, :, -spatial_token_count:, :].permute(0, 2, 1, 3).contiguous()
    value_tokens = values[:, :, -spatial_token_count:, :].permute(0, 2, 1, 3).contiguous()
    flat = torch.cat(
        [key_tokens.flatten(start_dim=2), value_tokens.flatten(start_dim=2)],
        dim=-1,
    ).to(torch.bfloat16)
    if flat.shape[-1] != expected_dim:
        raise ValueError(f"Expected latent KV dim {expected_dim}, got {flat.shape[-1]}")
    return flat


@torch.inference_mode()
def extract_latent_student_spatial_kv(
    batch: dict[str, Any],
    *,
    student: Any,
    processor: Any,
    device: torch.device,
    layer_index: int,
    expected_dim: int,
    spatial_token_count: int,
    prompt_template: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch_to_latent_student_inputs(
        batch,
        processor,
        prompt_template=prompt_template,
        device=device,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    pixel_values_videos = inputs.get("pixel_values_videos")
    video_grid_thw = inputs.get("video_grid_thw")

    prefix_embeds, seed_hidden = student.encode_prefix(
        input_ids,
        pixel_values,
        image_grid_thw,
        attention_mask,
        pixel_values_videos,
        video_grid_thw,
    )

    batch_size = int(input_ids.shape[0])
    current_embeds = prefix_embeds
    current_mask = attention_mask
    current_token = seed_hidden

    for _ in range(int(student.M)):
        current_embeds = torch.cat([current_embeds, current_token.unsqueeze(1)], dim=1)
        current_mask = torch.cat(
            [
                current_mask,
                torch.ones(batch_size, 1, device=device, dtype=current_mask.dtype),
            ],
            dim=1,
        )
        out = student._language_model(
            inputs_embeds=current_embeds,
            attention_mask=current_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        current_token = out.last_hidden_state[:, -1, :]

    end_think_embed = student._embed_tokens(
        torch.full(
            (batch_size, 1),
            int(student.end_think_token_id),
            device=device,
            dtype=torch.long,
        )
    ).to(dtype=current_embeds.dtype)
    spatial_embeds = (
        student.spatial_tokens.unsqueeze(0)
        .expand(batch_size, -1, -1)
        .to(device=device, dtype=current_embeds.dtype)
    )
    if spatial_embeds.shape[1] != spatial_token_count:
        raise ValueError(
            f"Student has {spatial_embeds.shape[1]} spatial tokens, expected {spatial_token_count}"
        )

    tail_embeds = torch.cat([end_think_embed, spatial_embeds], dim=1)
    tail_mask = torch.ones(
        batch_size,
        spatial_token_count + 1,
        device=device,
        dtype=current_mask.dtype,
    )
    full_embeds = torch.cat([current_embeds, tail_embeds], dim=1)
    full_mask = torch.cat([current_mask, tail_mask], dim=1)

    final_out = student._language_model(
        inputs_embeds=full_embeds,
        attention_mask=full_mask,
        use_cache=True,
        output_hidden_states=False,
        return_dict=True,
    )
    keys, values = layer_key_values_from_past(
        final_out.past_key_values,
        layer_index=layer_index,
    )
    spatial_kv = flatten_spatial_layer_kv(
        keys,
        values,
        spatial_token_count=spatial_token_count,
        expected_dim=expected_dim,
    )
    spatial_hidden = final_out.last_hidden_state[:, -spatial_token_count:, :]
    waypoints = student.spatial_mlp(spatial_hidden).to(torch.float32)
    return spatial_kv, waypoints


def unique_instruction_indices(instructions: list[str]) -> tuple[list[str], torch.Tensor]:
    index_by_instruction: dict[str, int] = {}
    unique: list[str] = []
    sample_indices: list[int] = []
    for instruction in instructions:
        key = str(instruction)
        if key not in index_by_instruction:
            index_by_instruction[key] = len(unique)
            unique.append(key)
        sample_indices.append(index_by_instruction[key])
    return unique, torch.as_tensor(sample_indices, dtype=torch.long)


def batch_dataset_label(batch: dict[str, Any]) -> str:
    ids = [str(item.get("dataset_id", "unknown")) for item in batch.get("metadata", [])]
    unique_ids = list(dict.fromkeys(ids))
    if not unique_ids:
        return "unknown"
    if len(unique_ids) <= 3:
        return "+".join(unique_ids)
    return "+".join(unique_ids[:3]) + f"+{len(unique_ids) - 3}more"


def build_shard_image_pool(
    batch: dict[str, Any],
    *,
    image_history_size: int,
    image_jpeg_quality: int,
    image_storage: str,
) -> tuple[list[bytes] | list[torch.Tensor], torch.Tensor]:
    key_to_index: dict[tuple[str, ...], int] = {}
    image_pool: list[bytes] | list[torch.Tensor] = []
    sample_image_indices: list[list[int]] = []
    blank_image = Image.new("RGB", (384, 384), color=(0, 0, 0))
    blank_payload = (
        image_to_lossless_png_bytes(blank_image)
        if image_storage == "lossless_png"
        else torch.from_numpy(np.array(blank_image, dtype=np.uint8, copy=True))
    )

    def serialize_image(image: Image.Image) -> bytes | torch.Tensor:
        if image_storage == "lossless_png":
            return image_to_lossless_png_bytes(image)
        if image_storage == "raw_uint8":
            return torch.from_numpy(np.array(image.convert("RGB"), dtype=np.uint8, copy=True))
        raise ValueError(f"Unsupported image storage mode: {image_storage}")

    for sample_index, (metadata, sample_slots) in enumerate(
        zip(batch["metadata"], batch["siglip_image_slots"])
    ):
        slot_mask = batch["siglip_slot_mask"][sample_index]
        dataset_id = str(metadata["dataset_id"])
        episode_id = str(metadata["episode_id"])
        try:
            step_idx = int(metadata["step_idx"])
        except (TypeError, ValueError):
            step_idx = sample_index

        slot_indices: list[int] = []
        for slot_index, image in enumerate(sample_slots):
            valid = bool(slot_mask[slot_index])
            if valid:
                frame_pos = slot_index // len(IMAGE_KEYS)
                image_key = IMAGE_KEYS[slot_index % len(IMAGE_KEYS)]
                relative_offset = frame_pos - (image_history_size - 1)
                logical_step_idx = max(0, step_idx + relative_offset)
                pool_key = (
                    dataset_id,
                    episode_id,
                    str(logical_step_idx),
                    image_key,
                )
            else:
                pool_key = ("blank",)

            image_index = key_to_index.get(pool_key)
            if image_index is None:
                image_index = len(image_pool)
                key_to_index[pool_key] = image_index
                image_pool.append(
                    serialize_image(image) if valid else blank_payload
                )
            slot_indices.append(image_index)
        sample_image_indices.append(slot_indices)

    return image_pool, torch.as_tensor(sample_image_indices, dtype=torch.long)


def compact_lang_pool(
    lang_tokens: torch.Tensor,
    lang_mask: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    tokens_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    for index in range(lang_tokens.shape[0]):
        compact = compact_tokens(lang_tokens[index], lang_mask[index])
        tokens_list.append(compact.cpu())
        mask_list.append(torch.ones(compact.shape[0], dtype=torch.bool))
    return tokens_list, mask_list


def save_sample_shard(
    *,
    split_dir: Path,
    shard_index: int,
    sample_start_index: int,
    batch: dict[str, Any],
    latent_kv: torch.Tensor,
    waypoints: torch.Tensor,
    lang_tokens: torch.Tensor | None,
    lang_mask: torch.Tensor | None,
    sample_lang_index: torch.Tensor | None,
    image_history_size: int,
    image_jpeg_quality: int,
    cache_image_slots: bool,
    save_padded_features: bool,
    cache_proprioception_schema: str,
    image_storage: str,
) -> tuple[int, str]:
    batch_size = int(latent_kv.shape[0])
    filename = f"shard_{shard_index:09d}.pt"
    path = split_dir / filename

    if cache_proprioception_schema == "libero_native":
        if "libero_native_state" not in batch or "libero_native_actions" not in batch:
            raise ValueError(
                "--cache-proprioception-schema libero_native requires a LIBERO dataset"
            )
        cached_state = batch["libero_native_state"]
        cached_actions = batch["libero_native_actions"]
        state_dim_mask = torch.ones_like(cached_state)
        action_dim_mask = torch.ones(
            cached_actions.shape[0], cached_actions.shape[-1], dtype=torch.float32
        )
        proprioception_schema = "libero_native_state8_action7_v1"
    else:
        cached_state = batch["state"]
        cached_actions = batch["actions"]
        state_dim_mask = batch["state_dim_mask"]
        action_dim_mask = batch["action_dim_mask"]
        proprioception_schema = "rdt_encoded"

    metadata = [dict(item) for item in batch["metadata"]]
    for item in metadata:
        item["state_dim"] = int(cached_state.shape[-1])
        item["action_dim"] = int(cached_actions.shape[-1])
        item["proprioception_schema"] = proprioception_schema

    record: dict[str, Any] = {
        "cache_layout": "sample_shard",
        "feature_type": "latent_student_spatial_kv",
        "num_samples": batch_size,
        "sample_start_index": sample_start_index,
        "sample_stop_index": sample_start_index + batch_size,
        "qwen_kv": latent_kv.cpu(),
        "latent_waypoints": waypoints.cpu(),
        "proprioception_schema": proprioception_schema,
        "state": cached_state.cpu(),
        "state_dim_mask": state_dim_mask.cpu(),
        "actions": cached_actions.cpu(),
        "action_time_mask": batch["action_time_mask"].cpu(),
        "action_dim_mask": action_dim_mask.cpu(),
        "ctrl_freq": batch["ctrl_freq"].cpu(),
        "metadata": metadata,
        "instructions": [str(instruction) for instruction in batch["instructions"]],
    }
    if "joint_state" in batch:
        record["joint_state"] = batch["joint_state"].cpu()
    if "joint_states" in batch:
        record["joint_states"] = batch["joint_states"].cpu()
    if "joint_states_mask" in batch:
        record["joint_states_mask"] = batch["joint_states_mask"].cpu()

    if lang_tokens is not None and lang_mask is not None and sample_lang_index is not None:
        if save_padded_features:
            record["lang_tokens"] = lang_tokens.cpu()
            record["lang_mask"] = lang_mask.cpu()
        else:
            token_list, mask_list = compact_lang_pool(lang_tokens, lang_mask)
            record["lang_tokens"] = token_list
            record["lang_mask"] = mask_list
        record["sample_lang_index"] = sample_lang_index.cpu()

    if cache_image_slots:
        image_pool, sample_image_indices = build_shard_image_pool(
            batch,
            image_history_size=image_history_size,
            image_jpeg_quality=image_jpeg_quality,
            image_storage=image_storage,
        )
        image_pool_key = "image_arrays" if image_storage == "raw_uint8" else "image_jpegs"
        record[image_pool_key] = image_pool
        record["image_storage"] = image_storage
        record["sample_image_indices"] = sample_image_indices.cpu()
        record["sample_image_mask"] = batch["siglip_slot_mask"].cpu()
        record["image_slot_count"] = int(batch["siglip_slot_mask"].shape[1])

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(record, tmp_path)
    os.replace(tmp_path, path)

    first_metadata = batch["metadata"][0]
    last_metadata = batch["metadata"][-1]
    manifest_line = (
        json.dumps(
            {
                "path": filename,
                "cache_layout": "sample_shard",
                "feature_type": "latent_student_spatial_kv",
                "num_samples": batch_size,
                "sample_start_index": sample_start_index,
                "sample_stop_index": sample_start_index + batch_size,
                "first_dataset_id": first_metadata["dataset_id"],
                "first_episode_id": first_metadata["episode_id"],
                "first_step_idx": first_metadata["step_idx"],
                "last_dataset_id": last_metadata["dataset_id"],
                "last_episode_id": last_metadata["episode_id"],
                "last_step_idx": last_metadata["step_idx"],
                "qwen_token_count": int(latent_kv.shape[1]),
                "qwen_kv_dim": int(latent_kv.shape[2]),
                "has_lang_tokens": lang_tokens is not None,
                "has_image_slots": cache_image_slots,
                "has_joint_states": "joint_states" in record,
                "proprioception_schema": proprioception_schema,
                "state_dim": int(cached_state.shape[-1]),
                "action_dim": int(cached_actions.shape[-1]),
                "image_storage": image_storage if cache_image_slots else None,
            }
        )
        + "\n"
    )
    return batch_size, manifest_line


def precompute_split(
    *,
    split_name: str,
    dataset: Any,
    output_dir: Path,
    cfg: Any,
    args: argparse.Namespace,
    student: Any,
    processor: Any,
    t5_tokenizer: Any | None,
    t5_encoder: Any | None,
    device: torch.device,
) -> None:
    split_dir = output_dir / split_name
    manifest_path = prepare_split_output(split_dir, overwrite=args.overwrite)
    tmp_manifest_path = split_dir / "manifest.jsonl.tmp"

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=lambda samples: standardized_collate_fn(
            samples,
            max_images_per_sample=args.max_images_per_sample,
            image_history_size=args.image_history_size,
            image_jpeg_quality=args.image_jpeg_quality,
            skip_no_image=not args.keep_no_image,
            encode_image_slots=False,
        ),
    )

    sample_count = 0
    skipped_no_image = 0
    shard_count = 0
    start_time = time.perf_counter()
    with tmp_manifest_path.open("w", encoding="utf-8") as manifest:
        progress = tqdm(dataloader, desc=f"latent-kv {split_name}", unit="shard")
        for batch_index, batch in enumerate(progress):
            if args.max_batches_per_split is not None and batch_index >= args.max_batches_per_split:
                break
            if batch is None:
                continue
            if args.max_samples_per_split is not None and sample_count >= args.max_samples_per_split:
                break
            if args.max_samples_per_split is not None:
                keep = args.max_samples_per_split - sample_count
                if keep <= 0:
                    break
                if keep < len(batch["metadata"]):
                    for key in (
                        "state",
                        "actions",
                        "action_time_mask",
                        "action_dim_mask",
                        "ctrl_freq",
                        "joint_state",
                        "joint_states",
                        "joint_states_mask",
                        "libero_native_state",
                        "libero_native_actions",
                    ):
                        if key in batch:
                            batch[key] = batch[key][:keep]
                    batch["metadata"] = batch["metadata"][:keep]
                    batch["instructions"] = batch["instructions"][:keep]
                    batch["qwen_images"] = batch["qwen_images"][:keep]
                    batch["siglip_image_slots"] = batch["siglip_image_slots"][:keep]
                    batch["siglip_slot_mask"] = batch["siglip_slot_mask"][:keep]
                    batch["kept_samples"] = batch["kept_samples"][:keep]

            dataset_label = batch_dataset_label(batch)
            skipped_no_image += int(batch.get("skipped_no_image", 0))
            kv_start = time.perf_counter()
            latent_kv, waypoints = extract_latent_student_spatial_kv(
                batch,
                student=student,
                processor=processor,
                device=device,
                layer_index=args.layer_index,
                expected_dim=cfg.model.qwen_kv_dim,
                spatial_token_count=args.spatial_token_count,
                prompt_template=args.prompt_template,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            kv_seconds = time.perf_counter() - kv_start

            lang_tokens = None
            lang_mask = None
            sample_lang_index = None
            t5_seconds = 0.0
            if args.include_t5:
                unique_instructions, sample_lang_index = unique_instruction_indices(batch["instructions"])
                t5_start = time.perf_counter()
                lang_tokens, lang_mask = precompute_t5_features_chunked(
                    unique_instructions,
                    tokenizer=t5_tokenizer,
                    encoder=t5_encoder,
                    max_lang_tokens=cfg.model.max_lang_tokens,
                    expected_dim=cfg.model.lang_token_dim,
                    device=device,
                    batch_size=args.t5_batch_size,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t5_seconds = time.perf_counter() - t5_start

            save_start = time.perf_counter()
            saved, manifest_line = save_sample_shard(
                split_dir=split_dir,
                shard_index=shard_count,
                sample_start_index=sample_count,
                batch=batch,
                latent_kv=latent_kv,
                waypoints=waypoints,
                lang_tokens=lang_tokens,
                lang_mask=lang_mask,
                sample_lang_index=sample_lang_index,
                image_history_size=args.image_history_size,
                image_jpeg_quality=args.image_jpeg_quality,
                cache_image_slots=args.cache_image_slots,
                save_padded_features=args.save_padded_features,
                cache_proprioception_schema=args.cache_proprioception_schema,
                image_storage=args.image_storage,
            )
            save_seconds = time.perf_counter() - save_start
            manifest.write(manifest_line)
            sample_count += saved
            shard_count += 1
            progress.set_postfix(
                dataset=dataset_label,
                samples=sample_count,
                kv=f"{kv_seconds:.2f}s",
                t5=f"{t5_seconds:.2f}s",
                save=f"{save_seconds:.2f}s",
            )

            if args.empty_cache_every > 0 and shard_count % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

    shutil.move(str(tmp_manifest_path), str(manifest_path))
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    print(
        f"[{split_name}] wrote {sample_count} samples in {shard_count} shards "
        f"to {split_dir} ({sample_count / elapsed:.1f} samples/s)"
    )
    if skipped_no_image:
        print(f"[{split_name}] skipped {skipped_no_image} samples with no available images")


def precompute_t5_features(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    from precompute_all_features import extract_t5_features

    return extract_t5_features(*args, **kwargs)


def precompute_t5_features_chunked(
    instructions: list[str],
    *,
    tokenizer: Any,
    encoder: Any,
    max_lang_tokens: int,
    expected_dim: int,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("--t5-batch-size must be positive")
    token_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    encoder_device = t5_device_from_encoder(encoder, device)
    for start in range(0, len(instructions), batch_size):
        chunk = instructions[start : start + batch_size]
        tokens, mask = precompute_t5_features(
            {"instructions": chunk},
            tokenizer,
            encoder,
            max_lang_tokens=max_lang_tokens,
            expected_dim=expected_dim,
            device=encoder_device,
        )
        token_chunks.append(tokens.cpu())
        mask_chunks.append(mask.cpu())
    if not token_chunks:
        return (
            torch.zeros(0, max_lang_tokens, expected_dim, dtype=torch.bfloat16),
            torch.zeros(0, max_lang_tokens, dtype=torch.bool),
        )
    return torch.cat(token_chunks, dim=0), torch.cat(mask_chunks, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute b2 LatentStudent spatial-token KV caches as batched shards."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "dataset" / "mock_dataset")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "cache_latent_student")
    parser.add_argument("--split", action="append", choices=SPLIT_NAMES)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["bc_z", "bridge", "droid", "fractal", "kuka", *LIBERO_DATASET_IDS],
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--stage-count", type=int, default=3)
    parser.add_argument("--droid-stage-count", type=int, default=2)
    parser.add_argument("--no-stage-subdir", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--max-batches-per-split", type=int, default=None)
    parser.add_argument(
        "--max-samples-per-episode",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_EPISODE,
    )
    parser.add_argument(
        "--all-samples-per-episode",
        action="store_true",
        help=(
            "Disable per-episode timestep subsampling and emit every valid "
            "timestep in each episode."
        ),
    )
    parser.add_argument(
        "--gripper-window-before",
        type=int,
        default=DEFAULT_GRIPPER_WINDOW_BEFORE,
    )
    parser.add_argument(
        "--gripper-window-after",
        type=int,
        default=DEFAULT_GRIPPER_WINDOW_AFTER,
        help=(
            "Number of selected steps from the gripper-change step onward. "
            "Use 11 to keep the change step plus 10 after it."
        ),
    )
    parser.add_argument(
        "--gripper-change-scope",
        choices=["all", "first", "directional"],
        default="all",
        help=(
            "Which gripper transitions receive priority sampling windows. "
            "Use 'directional' for open-to-close and close-to-open windows with "
            "separate before/after counts."
        ),
    )
    parser.add_argument("--open-to-close-before", type=int, default=DEFAULT_OPEN_TO_CLOSE_BEFORE)
    parser.add_argument("--open-to-close-after", type=int, default=DEFAULT_OPEN_TO_CLOSE_AFTER)
    parser.add_argument("--close-to-open-before", type=int, default=DEFAULT_CLOSE_TO_OPEN_BEFORE)
    parser.add_argument("--close-to-open-after", type=int, default=DEFAULT_CLOSE_TO_OPEN_AFTER)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-normalize-actions", action="store_true")
    parser.add_argument(
        "--action-target-mode",
        choices=["delta", "absolute_state"],
        default="delta",
        help=(
            "delta uses each adapter's command targets (raw 10D command-space "
            "targets for LIBERO). absolute_state is a legacy 7D mode for other "
            "adapters."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--empty-cache-every", type=int, default=25)

    parser.add_argument("--student-model-id", default="shreethar/LatentStudent-ckpt-240")
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--latent-student-code-dir", type=Path, default=None)
    parser.add_argument(
        "--spatial-parameters-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit spatial_parameters.pt or training_state.pt. "
            "Use this for local checkpoints whose spatial tokens are not bundled "
            "inside LatentStudent.from_pretrained()."
        ),
    )
    parser.add_argument("--latent-count", type=int, default=6)
    parser.add_argument("--spatial-token-count", type=int, default=5)
    parser.add_argument("--layer-index", type=int, default=7)
    parser.add_argument("--prompt-template", default=QWEN_TRAJECTORY_PROMPT_TEMPLATE)
    parser.add_argument("--device-map", default="auto")

    parser.add_argument(
        "--include-t5",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also encode and cache T5 XXL language tokens. Disabled by default; "
            "instruction strings are always stored."
        ),
    )
    parser.add_argument(
        "--t5-model-id",
        default="/home/ubuntu/RoboticsDiffusionTransformer/google/t5-v1_1-xxl",
    )
    parser.add_argument("--t5-fallback-model-id", default="google/t5-v1_1-xxl")
    parser.add_argument(
        "--t5-precision",
        choices=["bf16", "8bit"],
        default="bf16",
        help="Load T5 XXL in bf16 or bitsandbytes 8-bit.",
    )
    parser.add_argument(
        "--t5-batch-size",
        type=int,
        default=32,
        help=(
            "Number of unique raw instructions per T5 forward pass. This is "
            "independent from --batch-size, which controls LatentStudent KV shard size."
        ),
    )
    parser.add_argument("--save-padded-features", action="store_true")
    parser.add_argument(
        "--cache-proprioception-schema",
        choices=["rdt", "libero_native"],
        default="rdt",
        help=(
            "Store the normal adapter tensors, or exact LIBERO-native 8D state "
            "and 7D command tensors. libero_native is valid only for LIBERO."
        ),
    )

    parser.add_argument("--image-history-size", type=int, default=2)
    parser.add_argument("--max-images-per-sample", type=int, default=6)
    parser.add_argument(
        "--image-jpeg-quality",
        type=int,
        default=100,
        help="Deprecated compatibility option; image slots are always lossless PNG.",
    )
    parser.add_argument("--keep-no-image", action="store_true")
    parser.add_argument("--cache-image-slots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--image-storage",
        choices=["lossless_png", "raw_uint8"],
        default="lossless_png",
        help="Store pooled images as lossless PNG bytes or uncompressed uint8 HWC tensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device_map == "cuda" and device.type != "cuda":
        args.device_map = "cpu"
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.t5_batch_size <= 0:
        raise ValueError("--t5-batch-size must be positive")
    if args.spatial_token_count <= 0:
        raise ValueError("--spatial-token-count must be positive")
    if "{task}" not in args.prompt_template:
        raise ValueError("--prompt-template must contain {task}")
    max_samples_per_episode = (
        None if args.all_samples_per_episode else args.max_samples_per_episode
    )
    if max_samples_per_episode is not None and max_samples_per_episode <= 0:
        raise ValueError("--max-samples-per-episode must be positive")
    normalize_actions = not args.no_normalize_actions
    if args.action_target_mode == "absolute_state":
        if normalize_actions:
            print(
                "Using action_target_mode=absolute_state; disabling q01/q99 action "
                "normalization so RDT targets stay in physical units."
            )
        normalize_actions = False
    print(f"Using latent-student extraction device: {device}")
    if args.include_t5:
        print(f"Using transformers device_map for T5: {args.device_map}")
    else:
        print("T5 language embedding extraction is disabled; storing instruction text only")

    configs = build_lazy_configs(
        root=args.root.expanduser().resolve(),
        dataset_ids=args.dataset,
        max_episodes=args.max_episodes,
    )
    if args.cache_proprioception_schema == "libero_native" and (
        not configs
        or any(config.dataset_id not in LIBERO_DATASET_IDS for config in configs)
    ):
        raise ValueError(
            "--cache-proprioception-schema libero_native can only be used when "
            "all selected datasets are LIBERO suites"
        )
    if configs and all(config.dataset_id in LIBERO_DATASET_IDS for config in configs):
        if args.action_target_mode != "delta":
            raise ValueError(
                "The 11D-state/10D-action LIBERO schema requires "
                "--action-target-mode delta"
            )
        if normalize_actions:
            print(
                "Using raw LIBERO 10D command targets; disabling legacy q01/q99 "
                "action normalization."
            )
        normalize_actions = False
    splits = build_combined_standardized_splits(
        configs=configs,
        seed=seed,
        stage=args.stage,
        stage_count=args.stage_count,
        droid_stage_count=args.droid_stage_count,
        horizon=cfg.model.pred_horizon,
        normalize_actions=normalize_actions,
        action_target_mode=args.action_target_mode,
        max_samples_per_episode=max_samples_per_episode,
        gripper_window_before=args.gripper_window_before,
        gripper_window_after=args.gripper_window_after,
        gripper_change_scope=args.gripper_change_scope,
        open_to_close_before=args.open_to_close_before,
        open_to_close_after=args.open_to_close_after,
        close_to_open_before=args.close_to_open_before,
        close_to_open_after=args.close_to_open_after,
    )

    student, processor = load_student_and_processor(args, device)
    t5_tokenizer = None
    t5_encoder = None
    if args.include_t5:
        print("Loading T5 encoder...")
        t5_tokenizer, t5_encoder = load_t5(args, cfg)

    output_dir = args.output_dir
    if args.stage is not None and not args.no_stage_subdir:
        output_dir = output_dir / f"stage_{args.stage}"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = args.split or list(SPLIT_NAMES)
    metadata = {
        "feature_type": "latent_student_spatial_kv",
        "config": args.config,
        "root": str(args.root),
        "splits": split_names,
        "datasets": [config.dataset_id for config in configs],
        "seed": seed,
        "stage": args.stage,
        "normalize_actions": normalize_actions,
        "action_target_mode": args.action_target_mode,
        "state_dim": (
            8 if args.cache_proprioception_schema == "libero_native" else cfg.model.state_dim
        ),
        "action_dim": (
            7 if args.cache_proprioception_schema == "libero_native" else cfg.model.action_dim
        ),
        "proprioception_schema": (
            "libero_native_state8_action7_v1"
            if args.cache_proprioception_schema == "libero_native"
            else "rdt_encoded"
        ),
        "state_encoder_layout": cfg.model.state_encoder_layout,
        "action_encoder_layout": cfg.model.action_encoder_layout,
        "gripper_processing": (
            "raw two-finger qpos state; raw HDF5 action command"
            if all(config.dataset_id in LIBERO_DATASET_IDS for config in configs)
            else "dataset-specific"
        ),
        "max_samples_per_episode": max_samples_per_episode,
        "all_samples_per_episode": bool(args.all_samples_per_episode),
        "gripper_window_before": args.gripper_window_before,
        "gripper_window_after": args.gripper_window_after,
        "gripper_change_scope": args.gripper_change_scope,
        "open_to_close_before": args.open_to_close_before,
        "open_to_close_after": args.open_to_close_after,
        "close_to_open_before": args.close_to_open_before,
        "close_to_open_after": args.close_to_open_after,
        "student_model_id": args.student_model_id,
        "processor_id": args.processor_id or args.student_model_id,
        "spatial_parameters_path": (
            str(args.spatial_parameters_path) if args.spatial_parameters_path is not None else None
        ),
        "latent_count": args.latent_count,
        "spatial_token_count": args.spatial_token_count,
        "layer_index": args.layer_index,
        "prompt_template": args.prompt_template,
        "qwen_kv_dim": cfg.model.qwen_kv_dim,
        "include_t5": args.include_t5,
        "t5_precision": args.t5_precision,
        "t5_batch_size": args.t5_batch_size,
        "cache_image_slots": args.cache_image_slots,
        "image_history_size": args.image_history_size,
        "max_images_per_sample": args.max_images_per_sample,
        "image_storage_codec": args.image_storage,
        "image_storage_lossless": True,
        "image_storage_compressed": args.image_storage != "raw_uint8",
        "image_jpeg_quality": None,
        "cache_layout": "sample_shard",
        "batch_size": args.batch_size,
    }
    (output_dir / "precompute_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    for split_name in split_names:
        precompute_split(
            split_name=split_name,
            dataset=splits[split_name],
            output_dir=output_dir,
            cfg=cfg,
            args=args,
            student=student,
            processor=processor,
            t5_tokenizer=t5_tokenizer,
            t5_encoder=t5_encoder,
            device=device,
        )

    print("Latent-student KV precomputation finished successfully.")


if __name__ == "__main__":
    main()
