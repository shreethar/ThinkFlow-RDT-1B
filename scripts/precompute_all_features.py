#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Any

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


def as_rgb_pil(image: Any) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.size == 0 or not np.any(array):
        return None
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


def selected_images_for_sample(
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


def blank_rgb_image(size: int = 384) -> Image.Image:
    return Image.new("RGB", (size, size), color=(0, 0, 0))


def standardized_collate_fn(
    samples: list[dict[str, Any]],
    *,
    max_images_per_sample: int,
    skip_no_image: bool,
) -> dict[str, Any] | None:
    kept: list[dict[str, Any]] = []
    image_groups: list[list[Image.Image]] = []
    skipped_no_image = 0

    for sample in samples:
        selected_images = selected_images_for_sample(
            sample,
            max_images_per_sample=max_images_per_sample,
        )
        if not selected_images and skip_no_image:
            skipped_no_image += 1
            continue
        if not selected_images:
            selected_images = [blank_rgb_image()]
        kept.append(sample)
        image_groups.append(selected_images)

    if not kept:
        return None

    dataset_ids = [str(sample["dataset_id"]) for sample in kept]
    return {
        "instructions": [str(sample["instruction"]) for sample in kept],
        "images": image_groups,
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
                "image_count": len(images),
            }
            for sample, dataset_id, images in zip(kept, dataset_ids, image_groups)
        ],
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
) -> torch.Tensor:
    texts_list: list[str] = []
    images_list: list[list[Image.Image]] = []

    for instruction, images in zip(batch["instructions"], batch["images"]):
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]
        texts_list.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
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
    )

    generated_ids = generated.sequences
    past_key_values = generated.past_key_values
    if isinstance(past_key_values, tuple):
        keys = past_key_values[layer_index][0]
        values = past_key_values[layer_index][1]
    elif hasattr(past_key_values, "key_cache"):
        keys = past_key_values.key_cache[layer_index]
        values = past_key_values.value_cache[layer_index]
    else:
        keys = past_key_values.layers[layer_index].keys
        values = past_key_values.layers[layer_index].values

    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    kv_vectors: list[torch.Tensor] = []
    cache_len = keys.shape[2]

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

        key_vec = keys[batch_index, :, target_index, :]
        value_vec = values[batch_index, :, target_index, :]
        kv_vectors.append(torch.cat([key_vec.reshape(-1), value_vec.reshape(-1)], dim=-1))

    qwen_kv = torch.stack(kv_vectors, dim=0).unsqueeze(1).to(torch.bfloat16)
    if expected_dim is not None and qwen_kv.shape[-1] != expected_dim:
        raise ValueError(
            f"Qwen KV dim mismatch: expected {expected_dim}, got {qwen_kv.shape[-1]}"
        )
    return qwen_kv


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
    flat_images, spans = nested_images_to_flat(batch["images"])
    if not flat_images:
        output = torch.zeros(
            len(spans),
            max_img_tokens,
            expected_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        mask = torch.zeros(len(spans), max_img_tokens, dtype=torch.bool, device=device)
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
    max_images_in_batch = max((stop - start for start, stop in spans), default=0)
    required_tokens = max_images_in_batch * tokens_per_image
    if required_tokens > max_img_tokens:
        raise ValueError(
            "SigLIP image token budget is too small for the selected image count: "
            f"{max_images_in_batch} images x {tokens_per_image} tokens/image = "
            f"{required_tokens}, but cfg.model.image_tokens={max_img_tokens}. "
            "Increase cfg.model.image_tokens or lower --max-images-per-sample."
        )

    batch_size = len(spans)
    output = torch.zeros(
        batch_size,
        max_img_tokens,
        expected_dim,
        device=device,
        dtype=image_features.dtype,
    )
    mask_out = torch.zeros(batch_size, max_img_tokens, dtype=torch.bool, device=device)

    for sample_index, (start, stop) in enumerate(spans):
        if start == stop:
            continue
        sample_features = image_features[start:stop].reshape(-1, image_features.shape[-1])
        valid_tokens = int(sample_features.shape[0])
        output[sample_index, :valid_tokens] = sample_features
        mask_out[sample_index, :valid_tokens] = True

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


def save_batch_records(
    *,
    split_dir: Path,
    manifest_handle: Any,
    start_index: int,
    batch: dict[str, Any],
    qwen_kv: torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_mask: torch.Tensor,
    img_tokens: torch.Tensor,
    img_mask: torch.Tensor,
) -> int:
    batch_size = qwen_kv.shape[0]
    for batch_index in range(batch_size):
        global_index = start_index + batch_index
        metadata = batch["metadata"][batch_index]
        filename = f"sample_{global_index:09d}.pt"
        path = split_dir / filename
        record = {
            "qwen_kv": qwen_kv[batch_index].cpu(),
            "lang_tokens": lang_tokens[batch_index].cpu(),
            "lang_mask": lang_mask[batch_index].cpu(),
            "img_tokens": img_tokens[batch_index].cpu(),
            "img_mask": img_mask[batch_index].cpu(),
            "state": batch["state"][batch_index].cpu(),
            "actions": batch["actions"][batch_index].cpu(),
            "action_time_mask": batch["action_time_mask"][batch_index].cpu(),
            "action_dim_mask": batch["action_dim_mask"][batch_index].cpu(),
            "ctrl_freq": float(batch["ctrl_freq"][batch_index].item()),
            **metadata,
        }
        torch.save(record, path)
        manifest_handle.write(
            json.dumps(
                {
                    "path": filename,
                    "dataset_id": metadata["dataset_id"],
                    "episode_id": metadata["episode_id"],
                    "step_idx": metadata["step_idx"],
                }
            )
            + "\n"
        )
    return start_index + batch_size


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
            )
            lang_tokens, lang_mask = extract_t5_features(
                batch,
                models["t5_tokenizer"],
                models["t5_encoder"],
                max_lang_tokens=cfg.model.max_lang_tokens,
                expected_dim=cfg.model.lang_token_dim,
                device=device,
            )
            img_tokens, img_mask = extract_siglip_features(
                batch,
                models["siglip_processor"],
                models["siglip_encoder"],
                max_img_tokens=cfg.model.image_tokens,
                expected_dim=cfg.model.img_token_dim,
                device=device,
            )

            if args.max_samples_per_split is not None:
                keep = min(args.max_samples_per_split - sample_count, qwen_kv.shape[0])
                qwen_kv = qwen_kv[:keep]
                lang_tokens = lang_tokens[:keep]
                lang_mask = lang_mask[:keep]
                img_tokens = img_tokens[:keep]
                img_mask = img_mask[:keep]
                for key in ("state", "actions", "action_time_mask", "action_dim_mask", "ctrl_freq"):
                    batch[key] = batch[key][:keep]
                batch["metadata"] = batch["metadata"][:keep]

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
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--max-batches-per-split", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-normalize-actions", action="store_true")
    parser.add_argument("--max-images-per-sample", type=int, default=3)
    parser.add_argument("--keep-no-image", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--empty-cache-every", type=int, default=25)

    parser.add_argument("--qwen-model-id", default="shreethar/stage1_unsloth")
    parser.add_argument("--qwen-layer-index", type=int, default=7)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
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
        horizon=cfg.model.pred_horizon,
        normalize_actions=not args.no_normalize_actions,
    )
    split_names = args.split or list(SPLIT_NAMES)

    models = load_models(args, cfg, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "config": args.config,
        "root": str(args.root),
        "splits": split_names,
        "datasets": [config.dataset_id for config in configs],
        "seed": seed,
        "batch_size": args.batch_size,
        "normalize_actions": not args.no_normalize_actions,
        "qwen_model_id": args.qwen_model_id,
        "qwen_layer_index": args.qwen_layer_index,
        "t5_model_id": resolve_model_id(args.t5_model_id, args.t5_fallback_model_id),
        "siglip_model_id": resolve_model_id(
            args.siglip_model_id,
            args.siglip_fallback_model_id,
        ),
    }
    (args.output_dir / "precompute_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    for split_name in split_names:
        precompute_split(
            split_name=split_name,
            dataset=splits[split_name],
            output_dir=args.output_dir,
            cfg=cfg,
            args=args,
            models=models,
            device=device,
        )

    print("Precomputation finished successfully.")


if __name__ == "__main__":
    main()
