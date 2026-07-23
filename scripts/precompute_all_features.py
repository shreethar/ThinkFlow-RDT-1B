#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    SiglipImageProcessor,
    SiglipVisionModel,
    StoppingCriteria,
    StoppingCriteriaList,
    T5EncoderModel,
    T5Tokenizer,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thinkflow_rdt.adapters.combined_lazy import (  # noqa: E402
    LazyStandardizedDatasetConfig,
    build_combined_standardized_splits,
    default_lazy_standardized_dataset_configs,
)
from thinkflow_rdt.config import load_config  # noqa: E402


CTRL_FREQ_BY_DATASET = {
    "bc_z": 10.0,
    "bridge": 10.0,
    "droid": 15.0,
    "fractal": 3.0,
    "kuka": 3.0,
}
IMAGE_KEYS = ("primary", "wrist", "secondary")
SPLIT_NAMES = ("train", "validation", "test")
QWEN_TRAJECTORY_PROMPT_TEMPLATE = (
    "You are a robot manipulation assistant. Given an observation image and a "
    "task instruction, predict the end-effector's 2D trajectory as 5 waypoints. "
    "Output ONLY the coordinate list in this exact format: "
    "[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]\n\n"
    "Task: The task is {task}. What is the trajectory that the end effector should take?"
)


class StopAfterSubsequence(StoppingCriteria):
    def __init__(self, subsequence_ids: list[int]) -> None:
        self.subsequence_ids = list(subsequence_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if not self.subsequence_ids:
            return False
        return all(find_subsequence(sequence, self.subsequence_ids) != -1 for sequence in input_ids)


def find_subsequence(sequence: torch.Tensor | list[int], subsequence: list[int]) -> int:
    seq_list = sequence.tolist() if isinstance(sequence, torch.Tensor) else list(sequence)
    sub_len = len(subsequence)
    if sub_len == 0:
        return -1
    for index in range(len(seq_list) - sub_len + 1):
        if seq_list[index : index + sub_len] == subsequence:
            return index + sub_len
    return -1


def map_sequence_index_to_cache_index(
    *,
    sequence_index: int,
    sequence_length: int,
    prompt_length: int,
    cache_length: int,
) -> int:
    if cache_length == sequence_length:
        cache_index = sequence_index
    elif cache_length == max(0, sequence_length - prompt_length):
        cache_index = sequence_index - prompt_length
    elif cache_length == max(0, sequence_length - 1):
        cache_index = sequence_index - 1
    else:
        cache_index = sequence_index
    return max(0, min(int(cache_index), cache_length - 1))


def layer_key_values_from_past(
    past_key_values: Any,
    *,
    layer_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(past_key_values, tuple):
        keys = past_key_values[layer_index][0]
        values = past_key_values[layer_index][1]
    elif hasattr(past_key_values, "key_cache"):
        keys = past_key_values.key_cache[layer_index]
        values = past_key_values.value_cache[layer_index]
    else:
        keys = past_key_values.layers[layer_index].keys
        values = past_key_values.layers[layer_index].values
    return keys, values


def kv_vectors_from_layer_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    target_indices: list[int],
    expected_dim: int | None,
) -> torch.Tensor:
    kv_vectors: list[torch.Tensor] = []
    cache_len = int(keys.shape[2])
    for batch_index, target_index in enumerate(target_indices):
        target_index = max(0, min(int(target_index), cache_len - 1))
        key_vec = keys[batch_index, :, target_index, :]
        value_vec = values[batch_index, :, target_index, :]
        kv_vectors.append(torch.cat([key_vec.reshape(-1), value_vec.reshape(-1)], dim=-1))

    qwen_kv = torch.stack(kv_vectors, dim=0).unsqueeze(1).to(torch.bfloat16)
    if expected_dim is not None and qwen_kv.shape[-1] != expected_dim:
        raise ValueError(
            f"Qwen KV dim mismatch: expected {expected_dim}, got {qwen_kv.shape[-1]}"
        )
    return qwen_kv


def as_rgb_pil(image: Any) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.size == 0 or not np.any(array):
        return None
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


def blank_rgb_image(size: int = 384) -> Image.Image:
    return Image.new("RGB", (size, size), color=(0, 0, 0))


def format_qwen_trajectory_prompt(task: str, template: str = QWEN_TRAJECTORY_PROMPT_TEMPLATE) -> str:
    return template.format(task=str(task).strip())


def apply_qwen_chat_template(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    enable_thinking: bool,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return processor.apply_chat_template(
            messages,
            **kwargs,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def image_to_jpeg_bytes(image: Image.Image, *, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def jpeg_bytes_to_image(payload: bytes) -> Image.Image:
    return Image.open(io.BytesIO(payload)).convert("RGB")


def selected_current_images_for_sample(
    sample: dict[str, Any],
    *,
    max_images_per_sample: int,
) -> list[Image.Image]:
    images = sample.get("images", {})
    selected: list[Image.Image] = []
    for key in IMAGE_KEYS:
        image = as_rgb_pil(images.get(key))
        if image is not None:
            selected.append(image)
        if len(selected) >= max_images_per_sample:
            break
    return selected


def image_history_for_sample(
    sample: dict[str, Any],
    *,
    image_history_size: int,
) -> tuple[list[dict[str, Image.Image | None]], list[dict[str, int]]]:
    if "image_history" in sample:
        raw_history = list(sample["image_history"])[-image_history_size:]
        raw_masks = list(sample.get("image_history_mask", []))[-image_history_size:]
    else:
        raw_history = [sample.get("images", {})]
        raw_masks = [sample.get("image_mask", {})]

    if not raw_history:
        raw_history = [sample.get("images", {})]
        raw_masks = [sample.get("image_mask", {})]

    while len(raw_history) < image_history_size:
        raw_history.insert(0, raw_history[0])
        raw_masks.insert(0, {key: 0 for key in IMAGE_KEYS})

    history: list[dict[str, Image.Image | None]] = []
    masks: list[dict[str, int]] = []
    for frame_index, raw_frame in enumerate(raw_history[-image_history_size:]):
        raw_mask = raw_masks[frame_index] if frame_index < len(raw_masks) else {}
        frame: dict[str, Image.Image | None] = {}
        mask: dict[str, int] = {}
        for key in IMAGE_KEYS:
            image = as_rgb_pil(raw_frame.get(key))
            frame[key] = image
            mask[key] = int(bool(raw_mask.get(key, image is not None)) and image is not None)
        history.append(frame)
        masks.append(mask)
    return history, masks


def image_slots_for_sample(
    sample: dict[str, Any],
    *,
    image_history_size: int,
    max_images_per_sample: int,
) -> tuple[list[Image.Image], list[bool], list[Image.Image]]:
    history, masks = image_history_for_sample(
        sample,
        image_history_size=image_history_size,
    )
    slots: list[Image.Image] = []
    slot_mask: list[bool] = []
    qwen_images: list[Image.Image] = []
    for frame, mask in zip(history, masks):
        for key in IMAGE_KEYS:
            image = frame.get(key)
            valid = bool(mask.get(key, 0)) and image is not None
            slots.append(image if image is not None else blank_rgb_image())
            slot_mask.append(valid)
            if valid and image is not None:
                qwen_images.append(image)
            if len(slots) >= max_images_per_sample:
                return slots, slot_mask, qwen_images
    return slots, slot_mask, qwen_images


def primary_image_for_qwen(sample: dict[str, Any]) -> Image.Image:
    image = as_rgb_pil(sample.get("images", {}).get("primary"))
    return image if image is not None else blank_rgb_image()


def standardized_collate_fn(
    samples: list[dict[str, Any]],
    *,
    max_images_per_sample: int,
    image_history_size: int,
    image_jpeg_quality: int,
    skip_no_image: bool,
) -> dict[str, Any] | None:
    kept: list[dict[str, Any]] = []
    qwen_image_groups: list[list[Image.Image]] = []
    siglip_image_slots: list[list[Image.Image]] = []
    siglip_slot_masks: list[list[bool]] = []
    skipped_no_image = 0

    for sample in samples:
        slots, slot_mask, _qwen_images = image_slots_for_sample(
            sample,
            image_history_size=image_history_size,
            max_images_per_sample=max_images_per_sample,
        )
        if not any(slot_mask) and skip_no_image:
            skipped_no_image += 1
            continue
        kept.append(sample)
        qwen_image_groups.append([primary_image_for_qwen(sample)])
        siglip_image_slots.append(slots)
        siglip_slot_masks.append(slot_mask)

    if not kept:
        return None

    dataset_ids = [str(sample["dataset_id"]) for sample in kept]
    return {
        "instructions": [str(sample["instruction"]) for sample in kept],
        "qwen_images": qwen_image_groups,
        "siglip_image_slots": siglip_image_slots,
        "siglip_slot_mask": torch.as_tensor(siglip_slot_masks, dtype=torch.bool),
        "image_slot_jpegs": [
            [image_to_jpeg_bytes(image, quality=image_jpeg_quality) for image in slots]
            for slots in siglip_image_slots
        ],
        "state": torch.stack(
            [torch.as_tensor(sample["state"], dtype=torch.float32) for sample in kept]
        ),
        "actions": torch.stack(
            [torch.as_tensor(sample["actions"], dtype=torch.float32) for sample in kept]
        ),
        "action_time_mask": torch.stack(
            [
                torch.as_tensor(sample["actions_mask"], dtype=torch.bool)
                for sample in kept
            ]
        ),
        "action_dim_mask": torch.stack(
            [
                torch.as_tensor(
                    sample.get("action_dim_mask", np.ones((7,), dtype=np.float32)),
                    dtype=torch.float32,
                )
                for sample in kept
            ]
        ),
        "ctrl_freq": torch.tensor(
            [float(sample.get("ctrl_freq", CTRL_FREQ_BY_DATASET[dataset_id])) for sample, dataset_id in zip(kept, dataset_ids)],
            dtype=torch.float32,
        ),
        "metadata": [
            {
                "dataset_id": dataset_id,
                "episode_id": str(sample["episode_id"]),
                "step_idx": str(sample["step_idx"]),
                "image_count": int(sum(slot_mask)),
                "image_slot_count": len(slot_mask),
            }
            for sample, dataset_id, slot_mask in zip(kept, dataset_ids, siglip_slot_masks)
        ],
        "kept_samples": kept,
        "skipped_no_image": skipped_no_image,
    }


def nested_images_to_flat(
    image_groups: list[list[Image.Image]],
) -> tuple[list[Image.Image], list[tuple[int, int]]]:
    flat_images: list[Image.Image] = []
    spans: list[tuple[int, int]] = []
    for images in image_groups:
        start = len(flat_images)
        flat_images.extend(images)
        spans.append((start, len(flat_images)))
    return flat_images, spans


@torch.inference_mode()
def extract_qwen_kv(
    batch: dict[str, Any],
    processor: Any,
    vlm: Any,
    *,
    device: torch.device,
    layer_index: int,
    max_new_tokens: int,
    expected_dim: int | None,
    stop_at_think_end: bool = True,
    prompt_template: str | None = QWEN_TRAJECTORY_PROMPT_TEMPLATE,
    enable_thinking: bool = False,
) -> torch.Tensor:
    texts_list: list[str] = []
    images_list: list[list[Image.Image]] = []

    for instruction, images in zip(batch["instructions"], batch["qwen_images"]):
        qwen_instruction = (
            format_qwen_trajectory_prompt(instruction, prompt_template)
            if prompt_template is not None
            else str(instruction)
        )
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": qwen_instruction})
        messages = [{"role": "user", "content": content}]
        texts_list.append(
            apply_qwen_chat_template(
                processor,
                messages,
                enable_thinking=enable_thinking,
            )
        )
        images_list.append(images)

    inputs = processor(
        text=texts_list,
        images=images_list,
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }

    tokenizer = processor.tokenizer
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_token_id = im_end_id if im_end_id is not None else tokenizer.eos_token_id
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)

    prompt_target_indices: list[int] = []
    if think_end_ids and "input_ids" in inputs:
        for ids in inputs["input_ids"]:
            think_end_pos = find_subsequence(ids, think_end_ids)
            prompt_target_indices.append(think_end_pos - 1 if think_end_pos != -1 else -1)

    if prompt_target_indices and all(index >= 0 for index in prompt_target_indices):
        outputs = vlm(
            **inputs,
            use_cache=True,
            return_dict=True,
        )
        keys, values = layer_key_values_from_past(
            outputs.past_key_values,
            layer_index=layer_index,
        )
        return kv_vectors_from_layer_cache(
            keys,
            values,
            target_indices=prompt_target_indices,
            expected_dim=expected_dim,
        )

    if prompt_target_indices and any(index >= 0 for index in prompt_target_indices):
        raise ValueError(
            "Mixed Qwen prompts where only some samples contain </think> are not supported. "
            "Use a consistent --qwen-enable-thinking setting for the whole batch."
        )

    stopping_criteria = None
    if stop_at_think_end and think_end_ids:
        stopping_criteria = StoppingCriteriaList([StopAfterSubsequence(think_end_ids)])

    generated = vlm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        use_cache=True,
        return_dict_in_generate=True,
        output_hidden_states=False,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stopping_criteria,
    )

    generated_ids = generated.sequences
    keys, values = layer_key_values_from_past(
        generated.past_key_values,
        layer_index=layer_index,
    )
    cache_len = keys.shape[2]
    target_indices: list[int] = []
    for batch_index in range(generated_ids.shape[0]):
        ids = generated_ids[batch_index]
        prompt_length = int(inputs["attention_mask"][batch_index].sum().item())
        think_end_pos = find_subsequence(ids, think_end_ids)
        if think_end_pos != -1:
            target_index = think_end_pos - 1
        else:
            im_end_positions = torch.where(ids == eos_token_id)[0]
            target_index = (
                int(im_end_positions[-1].item()) - 1
                if len(im_end_positions) > 0
                else len(ids) - 1
            )
        target_index = map_sequence_index_to_cache_index(
            sequence_index=int(target_index),
            sequence_length=int(ids.shape[0]),
            prompt_length=prompt_length,
            cache_length=int(cache_len),
        )
        target_indices.append(target_index)

    return kv_vectors_from_layer_cache(
        keys,
        values,
        target_indices=target_indices,
        expected_dim=expected_dim,
    )


@torch.inference_mode()
def extract_t5_features(
    batch: dict[str, Any],
    tokenizer: T5Tokenizer,
    encoder: T5EncoderModel,
    *,
    max_lang_tokens: int,
    expected_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenized = tokenizer(
        batch["instructions"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_lang_tokens,
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    text_embeds = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    if text_embeds.shape[-1] != expected_dim:
        raise ValueError(
            f"T5 hidden dim mismatch: expected {expected_dim}, got {text_embeds.shape[-1]}"
        )

    batch_size, _, width = text_embeds.shape
    output = torch.zeros(
        batch_size,
        max_lang_tokens,
        width,
        device=device,
        dtype=text_embeds.dtype,
    )
    mask_out = torch.zeros(batch_size, max_lang_tokens, dtype=torch.bool, device=device)
    valid_lengths = attention_mask.bool().sum(dim=1).clamp(max=max_lang_tokens)

    for index, valid_length_tensor in enumerate(valid_lengths):
        valid_length = int(valid_length_tensor.item())
        output[index, :valid_length] = text_embeds[index, :valid_length]
        mask_out[index, :valid_length] = True

    return output.to(torch.bfloat16), mask_out


@torch.inference_mode()
def extract_siglip_features(
    batch: dict[str, Any],
    processor: SiglipImageProcessor,
    encoder: SiglipVisionModel,
    *,
    max_img_tokens: int,
    expected_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_slots: list[list[Image.Image]] = batch["siglip_image_slots"]
    slot_mask = torch.as_tensor(batch["siglip_slot_mask"], dtype=torch.bool, device=device)
    flat_images = [image for slots in image_slots for image in slots]
    batch_size = len(image_slots)
    slots_per_sample = max((len(slots) for slots in image_slots), default=0)
    if not flat_images:
        output = torch.zeros(
            batch_size,
            max_img_tokens,
            expected_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        mask = torch.zeros(batch_size, max_img_tokens, dtype=torch.bool, device=device)
        return output, mask

    inputs = processor(images=flat_images, return_tensors="pt")
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    image_features = encoder(**inputs).last_hidden_state
    if image_features.shape[-1] != expected_dim:
        raise ValueError(
            f"SigLIP hidden dim mismatch: expected {expected_dim}, got {image_features.shape[-1]}"
        )
    tokens_per_image = int(image_features.shape[1])
    required_tokens = slots_per_sample * tokens_per_image
    if required_tokens > max_img_tokens:
        raise ValueError(
            "SigLIP image token budget is too small for the selected image count: "
            f"{slots_per_sample} image slots x {tokens_per_image} tokens/image = "
            f"{required_tokens}, but cfg.model.image_tokens={max_img_tokens}. "
            "Increase cfg.model.image_tokens or lower --max-images-per-sample."
        )

    image_features = image_features.reshape(
        batch_size,
        slots_per_sample,
        tokens_per_image,
        image_features.shape[-1],
    )
    output = torch.zeros(
        batch_size,
        max_img_tokens,
        expected_dim,
        device=device,
        dtype=image_features.dtype,
    )
    mask_out = torch.zeros(batch_size, max_img_tokens, dtype=torch.bool, device=device)

    for sample_index in range(batch_size):
        for slot_index in range(slots_per_sample):
            token_start = slot_index * tokens_per_image
            token_stop = token_start + tokens_per_image
            if bool(slot_mask[sample_index, slot_index]):
                output[sample_index, token_start:token_stop] = image_features[
                    sample_index, slot_index
                ]
                mask_out[sample_index, token_start:token_stop] = True

    return output.to(torch.bfloat16), mask_out


def resolve_model_id(primary: str, fallback: str | None = None) -> str:
    if os.path.exists(primary):
        return primary
    if fallback is not None:
        return fallback
    return primary


def build_lazy_configs(
    *,
    root: Path,
    dataset_ids: list[str] | None,
    max_episodes: int | None,
) -> list[LazyStandardizedDatasetConfig]:
    configs = default_lazy_standardized_dataset_configs(
        dataset_ids=dataset_ids,
        root=root,
    )
    if max_episodes is None:
        return configs
    return [
        LazyStandardizedDatasetConfig(
            dataset_id=config.dataset_id,
            data_dir=config.data_dir,
            source_split=config.source_split,
            max_episodes=max_episodes,
            shard_pattern=config.shard_pattern,
            adapter_kwargs=dict(config.adapter_kwargs),
        )
        for config in configs
    ]


def prepare_split_output(split_dir: Path, *, overwrite: bool) -> Path:
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.jsonl"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists. Pass --overwrite to replace it."
        )
    if overwrite:
        manifest_path.unlink(missing_ok=True)
        tmp_manifest = split_dir / "manifest.jsonl.tmp"
        tmp_manifest.unlink(missing_ok=True)
    return manifest_path


def compact_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=tokens.device, dtype=torch.bool)
    return tokens[mask].contiguous()


def save_batch_records(
    *,
    split_dir: Path,
    manifest_handle: Any,
    start_index: int,
    batch: dict[str, Any],
    qwen_kv: torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_mask: torch.Tensor,
    img_tokens: torch.Tensor | None,
    img_mask: torch.Tensor | None,
    save_padded_features: bool,
    cache_image_slots: bool,
) -> int:
    batch_size = qwen_kv.shape[0]
    for batch_index in range(batch_size):
        global_index = start_index + batch_index
        metadata = batch["metadata"][batch_index]
        filename = f"sample_{global_index:09d}.pt"
        path = split_dir / filename
        sample_lang_tokens = lang_tokens[batch_index]
        sample_lang_mask = lang_mask[batch_index]
        sample_img_tokens = img_tokens[batch_index] if img_tokens is not None else None
        sample_img_mask = img_mask[batch_index] if img_mask is not None else None
        if not save_padded_features:
            sample_lang_tokens = compact_tokens(sample_lang_tokens, sample_lang_mask)
            if sample_img_tokens is not None and sample_img_mask is not None:
                sample_img_tokens = compact_tokens(sample_img_tokens, sample_img_mask)
            sample_lang_mask = torch.ones(
                sample_lang_tokens.shape[0], dtype=torch.bool, device=sample_lang_tokens.device
            )
            if sample_img_tokens is not None:
                sample_img_mask = torch.ones(
                    sample_img_tokens.shape[0],
                    dtype=torch.bool,
                    device=sample_img_tokens.device,
                )
        record = {
            "qwen_kv": qwen_kv[batch_index].cpu(),
            "lang_tokens": sample_lang_tokens.cpu(),
            "lang_mask": sample_lang_mask.cpu(),
            "state": batch["state"][batch_index].cpu(),
            "actions": batch["actions"][batch_index].cpu(),
            "action_time_mask": batch["action_time_mask"][batch_index].cpu(),
            "action_dim_mask": batch["action_dim_mask"][batch_index].cpu(),
            "ctrl_freq": float(batch["ctrl_freq"][batch_index].item()),
            **metadata,
        }
        if sample_img_tokens is not None and sample_img_mask is not None:
            record["img_tokens"] = sample_img_tokens.cpu()
            record["img_mask"] = sample_img_mask.cpu()
        if cache_image_slots:
            record["image_slot_jpegs"] = batch["image_slot_jpegs"][batch_index]
            record["image_slot_mask"] = batch["siglip_slot_mask"][batch_index].cpu()
        torch.save(record, path)
        manifest_handle.write(
            json.dumps(
                {
                    "path": filename,
                    "dataset_id": metadata["dataset_id"],
                    "episode_id": metadata["episode_id"],
                    "step_idx": metadata["step_idx"],
                    "image_count": metadata["image_count"],
                    "image_slot_count": metadata["image_slot_count"],
                    "lang_token_count": int(sample_lang_tokens.shape[0]),
                    "img_token_count": (
                        int(sample_img_tokens.shape[0])
                        if sample_img_tokens is not None
                        else None
                    ),
                    "has_img_tokens": sample_img_tokens is not None,
                    "has_image_slots": cache_image_slots,
                }
            )
            + "\n"
        )
    return start_index + batch_size


def iter_episode_sample_groups(dataset: Any) -> Iterable[list[dict[str, Any]]]:
    current_key: tuple[str, str] | None = None
    current_samples: list[dict[str, Any]] = []
    for sample in dataset:
        key = (str(sample["dataset_id"]), str(sample["episode_id"]))
        if current_key is None:
            current_key = key
        if key != current_key:
            if current_samples:
                yield current_samples
            current_key = key
            current_samples = []
        current_samples.append(sample)
    if current_samples:
        yield current_samples


def sample_gripper_binary(sample: dict[str, Any], *, normalized_actions: bool) -> int:
    actions = np.asarray(sample["actions"], dtype=np.float32)
    value = float(actions[0, 6])
    threshold = 0.0 if normalized_actions else 0.5
    return int(value >= threshold)


def select_episode_qwen_anchors(
    samples: list[dict[str, Any]],
    *,
    normalized_actions: bool,
) -> list[dict[str, Any]]:
    if not samples:
        return []
    anchors = [samples[0]]
    previous = sample_gripper_binary(samples[0], normalized_actions=normalized_actions)
    for sample in samples[1:]:
        current = sample_gripper_binary(sample, normalized_actions=normalized_actions)
        if current != previous:
            if str(sample["step_idx"]) != str(samples[0]["step_idx"]):
                anchors.append(sample)
            break
        previous = current
    return anchors


def qwen_anchor_batch(
    anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "instructions": [str(sample["instruction"]) for sample in anchors],
        "qwen_images": [[primary_image_for_qwen(sample)] for sample in anchors],
    }


def t5_episode_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {"instructions": [str(samples[0]["instruction"])]}


def anchor_index_for_step(
    step_idx: int,
    anchors: list[dict[str, Any]],
) -> int:
    anchor_steps = [int(anchor["step_idx"]) for anchor in anchors]
    selected = 0
    for index, anchor_step in enumerate(anchor_steps):
        if anchor_step <= step_idx:
            selected = index
        else:
            break
    return selected


def save_episode_anchor_records(
    *,
    split_dir: Path,
    manifest_handle: Any,
    start_index: int,
    batch: dict[str, Any],
    anchors: list[dict[str, Any]],
    qwen_kv_by_anchor: torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_mask: torch.Tensor,
    save_padded_features: bool,
    cache_image_slots: bool,
) -> int:
    batch_size = len(batch["metadata"])
    episode_lang_tokens = lang_tokens[0]
    episode_lang_mask = lang_mask[0]
    if not save_padded_features:
        episode_lang_tokens = compact_tokens(episode_lang_tokens, episode_lang_mask)
        episode_lang_mask = torch.ones(
            episode_lang_tokens.shape[0],
            dtype=torch.bool,
            device=episode_lang_tokens.device,
        )

    for batch_index in range(batch_size):
        global_index = start_index + batch_index
        metadata = batch["metadata"][batch_index]
        step_idx = int(metadata["step_idx"])
        anchor_index = anchor_index_for_step(step_idx, anchors)
        anchor = anchors[anchor_index]
        filename = f"sample_{global_index:09d}.pt"
        path = split_dir / filename
        record = {
            "qwen_kv": qwen_kv_by_anchor[anchor_index].cpu(),
            "lang_tokens": episode_lang_tokens.cpu(),
            "lang_mask": episode_lang_mask.cpu(),
            "state": batch["state"][batch_index].cpu(),
            "actions": batch["actions"][batch_index].cpu(),
            "action_time_mask": batch["action_time_mask"][batch_index].cpu(),
            "action_dim_mask": batch["action_dim_mask"][batch_index].cpu(),
            "ctrl_freq": float(batch["ctrl_freq"][batch_index].item()),
            "qwen_cache_scope": "episode_anchors",
            "qwen_anchor_step_idx": str(anchor["step_idx"]),
            "qwen_anchor_kind": "first_step" if anchor_index == 0 else "first_gripper_change",
            **metadata,
        }
        if cache_image_slots:
            record["image_slot_jpegs"] = batch["image_slot_jpegs"][batch_index]
            record["image_slot_mask"] = batch["siglip_slot_mask"][batch_index].cpu()
        torch.save(record, path)
        manifest_handle.write(
            json.dumps(
                {
                    "path": filename,
                    "dataset_id": metadata["dataset_id"],
                    "episode_id": metadata["episode_id"],
                    "step_idx": metadata["step_idx"],
                    "image_count": metadata["image_count"],
                    "image_slot_count": metadata["image_slot_count"],
                    "lang_token_count": int(episode_lang_tokens.shape[0]),
                    "img_token_count": None,
                    "has_img_tokens": False,
                    "has_image_slots": cache_image_slots,
                    "qwen_cache_scope": "episode_anchors",
                    "qwen_anchor_step_idx": str(anchor["step_idx"]),
                    "qwen_anchor_kind": "first_step" if anchor_index == 0 else "first_gripper_change",
                }
            )
            + "\n"
        )
    return start_index + batch_size


def precompute_split_episode_anchors(
    *,
    split_name: str,
    dataset: Any,
    output_dir: Path,
    cfg: Any,
    args: argparse.Namespace,
    models: dict[str, Any],
    device: torch.device,
) -> None:
    if args.feature_set != "qwen_t5":
        raise ValueError("episode_anchors currently supports --feature-set qwen_t5 only")

    split_dir = output_dir / split_name
    manifest_path = prepare_split_output(split_dir, overwrite=args.overwrite)
    tmp_manifest_path = split_dir / "manifest.jsonl.tmp"

    sample_count = 0
    episode_count = 0
    skipped_no_image = 0
    with tmp_manifest_path.open("w", encoding="utf-8") as manifest:
        progress = tqdm(
            iter_episode_sample_groups(dataset),
            desc=f"precompute {split_name} episodes",
            unit="episode",
        )
        for episode_samples in progress:
            if args.max_batches_per_split is not None and episode_count >= args.max_batches_per_split:
                break
            if args.max_samples_per_split is not None and sample_count >= args.max_samples_per_split:
                break

            batch = standardized_collate_fn(
                episode_samples,
                max_images_per_sample=args.max_images_per_sample,
                image_history_size=args.image_history_size,
                image_jpeg_quality=args.image_jpeg_quality,
                skip_no_image=not args.keep_no_image,
            )
            if batch is None:
                skipped_no_image += len(episode_samples)
                continue

            kept_samples = list(batch["kept_samples"])

            if args.max_samples_per_split is not None:
                keep = min(args.max_samples_per_split - sample_count, len(batch["metadata"]))
                if keep <= 0:
                    break
                for key in ("state", "actions", "action_time_mask", "action_dim_mask", "ctrl_freq"):
                    batch[key] = batch[key][:keep]
                batch["metadata"] = batch["metadata"][:keep]
                batch["image_slot_jpegs"] = batch["image_slot_jpegs"][:keep]
                batch["siglip_slot_mask"] = batch["siglip_slot_mask"][:keep]
                kept_samples = kept_samples[:keep]

            anchors = select_episode_qwen_anchors(
                kept_samples,
                normalized_actions=not args.no_normalize_actions,
            )
            qwen_kv_by_anchor = extract_qwen_kv(
                qwen_anchor_batch(anchors),
                models["qwen_processor"],
                models["qwen_vlm"],
                device=device,
                layer_index=args.qwen_layer_index,
                max_new_tokens=args.qwen_max_new_tokens,
                expected_dim=cfg.model.qwen_kv_dim,
                stop_at_think_end=args.qwen_stop_at_think,
                prompt_template=args.qwen_trajectory_prompt_template,
                enable_thinking=args.qwen_enable_thinking,
            )
            lang_tokens, lang_mask = extract_t5_features(
                t5_episode_batch(kept_samples),
                models["t5_tokenizer"],
                models["t5_encoder"],
                max_lang_tokens=cfg.model.max_lang_tokens,
                expected_dim=cfg.model.lang_token_dim,
                device=device,
            )

            sample_count = save_episode_anchor_records(
                split_dir=split_dir,
                manifest_handle=manifest,
                start_index=sample_count,
                batch=batch,
                anchors=anchors,
                qwen_kv_by_anchor=qwen_kv_by_anchor,
                lang_tokens=lang_tokens,
                lang_mask=lang_mask,
                save_padded_features=args.save_padded_features,
                cache_image_slots=True,
            )
            episode_count += 1
            progress.set_postfix(samples=sample_count, episodes=episode_count)

            if args.empty_cache_every > 0 and episode_count % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

    shutil.move(str(tmp_manifest_path), str(manifest_path))
    print(f"[{split_name}] wrote {sample_count} samples from {episode_count} episodes to {split_dir}")
    if skipped_no_image:
        print(f"[{split_name}] skipped {skipped_no_image} samples with no available images")


def precompute_split(
    *,
    split_name: str,
    dataset: Any,
    output_dir: Path,
    cfg: Any,
    args: argparse.Namespace,
    models: dict[str, Any],
    device: torch.device,
) -> None:
    if args.qwen_cache_scope == "episode_anchors":
        precompute_split_episode_anchors(
            split_name=split_name,
            dataset=dataset,
            output_dir=output_dir,
            cfg=cfg,
            args=args,
            models=models,
            device=device,
        )
        return

    split_dir = output_dir / split_name
    manifest_path = prepare_split_output(split_dir, overwrite=args.overwrite)
    tmp_manifest_path = split_dir / "manifest.jsonl.tmp"

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=partial(
            standardized_collate_fn,
            max_images_per_sample=args.max_images_per_sample,
            image_history_size=args.image_history_size,
            image_jpeg_quality=args.image_jpeg_quality,
            skip_no_image=not args.keep_no_image,
        ),
    )

    sample_count = 0
    skipped_no_image = 0
    with tmp_manifest_path.open("w", encoding="utf-8") as manifest:
        progress = tqdm(dataloader, desc=f"precompute {split_name}", unit="batch")
        for batch_index, batch in enumerate(progress):
            if args.max_batches_per_split is not None and batch_index >= args.max_batches_per_split:
                break
            if batch is None:
                continue
            if args.max_samples_per_split is not None and sample_count >= args.max_samples_per_split:
                break

            skipped_no_image += int(batch.get("skipped_no_image", 0))
            qwen_kv = extract_qwen_kv(
                batch,
                models["qwen_processor"],
                models["qwen_vlm"],
                device=device,
                layer_index=args.qwen_layer_index,
                max_new_tokens=args.qwen_max_new_tokens,
                expected_dim=cfg.model.qwen_kv_dim,
                stop_at_think_end=args.qwen_stop_at_think,
                prompt_template=args.qwen_trajectory_prompt_template,
                enable_thinking=args.qwen_enable_thinking,
            )
            lang_tokens, lang_mask = extract_t5_features(
                batch,
                models["t5_tokenizer"],
                models["t5_encoder"],
                max_lang_tokens=cfg.model.max_lang_tokens,
                expected_dim=cfg.model.lang_token_dim,
                device=device,
            )
            if args.feature_set == "all":
                img_tokens, img_mask = extract_siglip_features(
                    batch,
                    models["siglip_processor"],
                    models["siglip_encoder"],
                    max_img_tokens=cfg.model.image_tokens,
                    expected_dim=cfg.model.img_token_dim,
                    device=device,
                )
            else:
                img_tokens = None
                img_mask = None

            if args.max_samples_per_split is not None:
                keep = min(args.max_samples_per_split - sample_count, qwen_kv.shape[0])
                qwen_kv = qwen_kv[:keep]
                lang_tokens = lang_tokens[:keep]
                lang_mask = lang_mask[:keep]
                if img_tokens is not None:
                    img_tokens = img_tokens[:keep]
                if img_mask is not None:
                    img_mask = img_mask[:keep]
                for key in ("state", "actions", "action_time_mask", "action_dim_mask", "ctrl_freq"):
                    batch[key] = batch[key][:keep]
                batch["metadata"] = batch["metadata"][:keep]
                batch["image_slot_jpegs"] = batch["image_slot_jpegs"][:keep]
                batch["siglip_slot_mask"] = batch["siglip_slot_mask"][:keep]

            sample_count = save_batch_records(
                split_dir=split_dir,
                manifest_handle=manifest,
                start_index=sample_count,
                batch=batch,
                qwen_kv=qwen_kv,
                lang_tokens=lang_tokens,
                lang_mask=lang_mask,
                img_tokens=img_tokens,
                img_mask=img_mask,
                save_padded_features=args.save_padded_features,
                cache_image_slots=args.feature_set == "qwen_t5",
            )
            progress.set_postfix(samples=sample_count, skipped_no_image=skipped_no_image)

            if args.empty_cache_every > 0 and (batch_index + 1) % args.empty_cache_every == 0:
                torch.cuda.empty_cache()

    shutil.move(str(tmp_manifest_path), str(manifest_path))
    print(f"[{split_name}] wrote {sample_count} samples to {split_dir}")
    if skipped_no_image:
        print(f"[{split_name}] skipped {skipped_no_image} samples with no available images")


def load_models(args: argparse.Namespace, cfg: Any, device: torch.device) -> dict[str, Any]:
    print("Loading Qwen VLM...")
    qwen_processor = AutoProcessor.from_pretrained(args.qwen_model_id)
    qwen_processor.tokenizer.padding_side = "left"
    qwen_vlm = AutoModelForImageTextToText.from_pretrained(
        args.qwen_model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    qwen_vlm.eval()
    qwen_vlm.requires_grad_(False)

    print("Loading T5 encoder...")
    t5_model_id = resolve_model_id(args.t5_model_id, args.t5_fallback_model_id)
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_id)
    t5_encoder = T5EncoderModel.from_pretrained(
        t5_model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    )
    t5_encoder.eval()
    t5_encoder.requires_grad_(False)
    if getattr(t5_encoder.config, "d_model", cfg.model.lang_token_dim) != cfg.model.lang_token_dim:
        raise ValueError(
            f"T5 d_model {t5_encoder.config.d_model} != cfg.model.lang_token_dim {cfg.model.lang_token_dim}"
        )

    siglip_processor = None
    siglip_encoder = None
    if args.feature_set == "all":
        print("Loading SigLIP vision encoder...")
        siglip_model_id = resolve_model_id(args.siglip_model_id, args.siglip_fallback_model_id)
        siglip_processor = SiglipImageProcessor.from_pretrained(siglip_model_id)
        siglip_encoder = SiglipVisionModel.from_pretrained(
            siglip_model_id,
            torch_dtype=torch.bfloat16,
            device_map=args.device_map,
        )
        siglip_encoder.eval()
        siglip_encoder.requires_grad_(False)
        if getattr(siglip_encoder.config, "hidden_size", cfg.model.img_token_dim) != cfg.model.img_token_dim:
            raise ValueError(
                "SigLIP hidden_size "
                f"{siglip_encoder.config.hidden_size} != cfg.model.img_token_dim {cfg.model.img_token_dim}"
            )

    return {
        "qwen_processor": qwen_processor,
        "qwen_vlm": qwen_vlm,
        "t5_tokenizer": t5_tokenizer,
        "t5_encoder": t5_encoder,
        "siglip_processor": siglip_processor,
        "siglip_encoder": siglip_encoder,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batched precomputation of Qwen KV, T5 language tokens, and SigLIP "
            "vision tokens for the lazy combined standardized dataset splits."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "dataset" / "mock_dataset",
        help="Repo root, mock_dataset root, or HF dataset layout root.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "cache_features")
    parser.add_argument(
        "--split",
        action="append",
        choices=SPLIT_NAMES,
        help="Split to process. Repeat for multiple. Defaults to train/validation/test.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["bc_z", "bridge", "droid", "fractal", "kuka"],
        help="Dataset id to include. Repeat for multiple. Defaults to all.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help=(
            "Optional curriculum/precompute stage. Sampled timesteps from non-Droid "
            "datasets are split across stages 1/2/3; Droid samples are split across "
            "stages 1/2 and excluded from stage 3."
        ),
    )
    parser.add_argument("--stage-count", type=int, default=3)
    parser.add_argument("--droid-stage-count", type=int, default=2)
    parser.add_argument(
        "--no-stage-subdir",
        action="store_true",
        help="When --stage is set, write directly to --output-dir instead of output-dir/stage_N.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--max-batches-per-split", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-normalize-actions", action="store_true")
    parser.add_argument(
        "--feature-set",
        choices=["all", "qwen_t5"],
        default="all",
        help=(
            "all caches Qwen, T5, and SigLIP features; qwen_t5 caches Qwen/T5 "
            "plus compressed image slots for online SigLIP during training."
        ),
    )
    parser.add_argument("--image-history-size", type=int, default=2)
    parser.add_argument("--max-images-per-sample", type=int, default=6)
    parser.add_argument("--image-jpeg-quality", type=int, default=90)
    parser.add_argument("--keep-no-image", action="store_true")
    parser.add_argument(
        "--save-padded-features",
        action="store_true",
        help=(
            "Save full padded T5/SigLIP tensors. By default, only valid tokens are "
            "saved and the training collator pads them back in memory."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--empty-cache-every", type=int, default=25)

    parser.add_argument("--qwen-model-id", default="shreethar/stage1_unsloth")
    parser.add_argument("--qwen-layer-index", type=int, default=7)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--qwen-cache-scope",
        choices=["auto", "per_sample", "episode_anchors"],
        default="auto",
        help=(
            "Qwen KV caching granularity. auto uses episode_anchors for --feature-set "
            "qwen_t5 and per_sample for --feature-set all."
        ),
    )
    parser.add_argument(
        "--qwen-stop-at-think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop Qwen generation as soon as every sequence has emitted </think>.",
    )
    parser.add_argument(
        "--qwen-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request Qwen thinking mode in the chat template when the processor supports it.",
    )
    parser.add_argument(
        "--qwen-trajectory-prompt-template",
        default=QWEN_TRAJECTORY_PROMPT_TEMPLATE,
        help=(
            "Prompt template for Qwen KV extraction. Must contain {task}."
        ),
    )
    parser.add_argument("--attn-implementation", default="sdpa")

    parser.add_argument(
        "--t5-model-id",
        default="/home/ubuntu/RoboticsDiffusionTransformer/google/t5-v1_1-xxl",
    )
    parser.add_argument("--t5-fallback-model-id", default="google/t5-v1_1-xxl")
    parser.add_argument(
        "--siglip-model-id",
        default="/home/ubuntu/RoboticsDiffusionTransformer/google/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--siglip-fallback-model-id",
        default="google/siglip-so400m-patch14-384",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map for all frozen encoders. Use cuda, auto, or cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.qwen_cache_scope == "auto":
        args.qwen_cache_scope = (
            "episode_anchors" if args.feature_set == "qwen_t5" else "per_sample"
        )
    if args.qwen_cache_scope == "episode_anchors" and args.feature_set != "qwen_t5":
        raise ValueError("--qwen-cache-scope episode_anchors requires --feature-set qwen_t5")
    if "{task}" not in args.qwen_trajectory_prompt_template:
        raise ValueError("--qwen-trajectory-prompt-template must contain {task}")
    seed = cfg.seed if args.seed is None else args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device_map == "cuda" and device.type != "cuda":
        args.device_map = "cpu"
    print(f"Using feature extraction device: {device}")
    print(f"Using transformers device_map: {args.device_map}")

    configs = build_lazy_configs(
        root=args.root.expanduser().resolve(),
        dataset_ids=args.dataset,
        max_episodes=args.max_episodes,
    )
    splits = build_combined_standardized_splits(
        configs=configs,
        seed=seed,
        stage=args.stage,
        stage_count=args.stage_count,
        droid_stage_count=args.droid_stage_count,
        horizon=cfg.model.pred_horizon,
        normalize_actions=not args.no_normalize_actions,
    )
    split_names = args.split or list(SPLIT_NAMES)

    models = load_models(args, cfg, device)
    output_dir = args.output_dir
    if args.stage is not None and not args.no_stage_subdir:
        output_dir = output_dir / f"stage_{args.stage}"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "config": args.config,
        "root": str(args.root),
        "splits": split_names,
        "datasets": [config.dataset_id for config in configs],
        "seed": seed,
        "stage": args.stage,
        "stage_unit": "sampled_timestep",
        "stage_count": args.stage_count,
        "droid_stage_count": args.droid_stage_count,
        "batch_size": args.batch_size,
        "normalize_actions": not args.no_normalize_actions,
        "feature_set": args.feature_set,
        "qwen_cache_scope": args.qwen_cache_scope,
        "qwen_stop_at_think": args.qwen_stop_at_think,
        "qwen_enable_thinking": args.qwen_enable_thinking,
        "qwen_image_source": "primary_current_frame",
        "qwen_trajectory_prompt_template": args.qwen_trajectory_prompt_template,
        "image_history_size": args.image_history_size,
        "max_images_per_sample": args.max_images_per_sample,
        "image_jpeg_quality": args.image_jpeg_quality,
        "feature_storage": "padded" if args.save_padded_features else "compact_valid_tokens",
        "qwen_model_id": args.qwen_model_id,
        "qwen_layer_index": args.qwen_layer_index,
        "t5_model_id": resolve_model_id(args.t5_model_id, args.t5_fallback_model_id),
        "siglip_model_id": resolve_model_id(
            args.siglip_model_id,
            args.siglip_fallback_model_id,
        ),
    }
    (output_dir / "precompute_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    for split_name in split_names:
        precompute_split(
            split_name=split_name,
            dataset=splits[split_name],
            output_dir=output_dir,
            cfg=cfg,
            args=args,
            models=models,
            device=device,
        )

    print("Precomputation finished successfully.")


if __name__ == "__main__":
    main()
