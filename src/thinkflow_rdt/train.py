from __future__ import annotations

import json
import math
import random
import io
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from PIL import Image
from torch.utils.data import DataLoader
from transformers import get_scheduler
from transformers import SiglipImageProcessor, SiglipVisionModel

from .checkpoint import load_trainable_artifact, save_trainable_artifact
from .config import ExperimentConfig
from .data import (
    ONLINE_SIGLIP_REQUIRED_KEYS,
    CachedFeatureDataset,
    EpisodePackSampler,
    FixedStratifiedSampler,
    RDTBatchCollator,
    RDTOnlineSiglipBatchCollator,
)
from .model import SFTConditionedRDT, resolve_dtype


LIBERO_SUITE_IDS = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensorboard_hparams(
    values: dict[str, object],
    *,
    prefix: str = "",
) -> dict[str, int | float | str | bool | torch.Tensor]:
    """Flatten an experiment config into TensorBoard-compatible hparams."""
    flattened: dict[str, int | float | str | bool | torch.Tensor] = {}
    for key, value in values.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_tensorboard_hparams(value, prefix=full_key))
        elif isinstance(value, (int, float, str, bool, torch.Tensor)):
            flattened[full_key] = value
        elif value is None:
            flattened[full_key] = "null"
        else:
            # TensorBoard rejects lists, tuples, and other structured values.
            # JSON retains their content while satisfying add_hparams' scalar API.
            flattened[full_key] = json.dumps(value, sort_keys=True, default=str)
    return flattened


def create_dataloader(
    manifest: str,
    cfg: ExperimentConfig,
    shuffle: bool,
    *,
    online_siglip: bool = False,
    stratified: bool = False,
) -> DataLoader:
    if online_siglip:
        dataset = CachedFeatureDataset(
            manifest,
            required_keys=ONLINE_SIGLIP_REQUIRED_KEYS,
            excluded_dataset_ids=cfg.data.excluded_dataset_ids,
        )
        collator = RDTOnlineSiglipBatchCollator(
            max_lang_tokens=cfg.model.max_lang_tokens,
            pred_horizon=cfg.model.pred_horizon,
            feature_dim=cfg.model.qwen_hidden_size,
            state_dim=cfg.model.state_dim,
            action_dim=cfg.model.action_dim,
            lang_token_dim=cfg.model.lang_token_dim,
            qwen_kv_dim=cfg.model.qwen_kv_dim,
            convert_cached_gripper_closed_to_open=(
                cfg.model.convert_cached_gripper_closed_to_open
            ),
        )
    else:
        dataset = CachedFeatureDataset(
            manifest,
            excluded_dataset_ids=cfg.data.excluded_dataset_ids,
        )
        collator = RDTBatchCollator(
            max_lang_tokens=cfg.model.max_lang_tokens,
            image_tokens=cfg.model.image_tokens,
            pred_horizon=cfg.model.pred_horizon,
            feature_dim=cfg.model.qwen_hidden_size,
            state_dim=cfg.model.state_dim,
            action_dim=cfg.model.action_dim,
            lang_token_dim=cfg.model.lang_token_dim,
            img_token_dim=cfg.model.img_token_dim,
            qwen_kv_dim=cfg.model.qwen_kv_dim,
            convert_cached_gripper_closed_to_open=(
                cfg.model.convert_cached_gripper_closed_to_open
            ),
        )
    persistent = cfg.data.persistent_workers and cfg.data.num_workers > 0
    sampler = None
    if stratified:
        if shuffle:
            raise ValueError("Stratified validation cannot also request shuffle")
        sampler = FixedStratifiedSampler(
            dataset,
            seed=cfg.training.validation_seed,
        )
    elif cfg.data.episode_aware_shuffle:
        sampler = EpisodePackSampler(
            dataset,
            shuffle=shuffle,
            seed=cfg.seed,
        )
    return DataLoader(
        dataset,
        batch_size=cfg.training.micro_batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=persistent,
        collate_fn=collator,
        drop_last=shuffle,
    )


def resolve_model_id(primary: str, fallback: str | None = None) -> str:
    if Path(primary).expanduser().exists():
        return str(Path(primary).expanduser().resolve())
    if fallback is not None:
        return fallback
    return primary


def decode_jpeg_image(payload: bytes) -> Image.Image:
    return Image.open(io.BytesIO(payload)).convert("RGB")


def load_online_siglip(
    *,
    model_id: str,
    fallback_model_id: str | None,
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[SiglipImageProcessor, SiglipVisionModel]:
    resolved_model_id = resolve_model_id(model_id, fallback_model_id)
    processor = SiglipImageProcessor.from_pretrained(resolved_model_id)
    encoder = SiglipVisionModel.from_pretrained(
        resolved_model_id,
        torch_dtype=resolve_dtype(cfg.model.dtype),
    )
    encoder.eval()
    encoder.requires_grad_(False)
    encoder.to(device)
    if getattr(encoder.config, "hidden_size", cfg.model.img_token_dim) != cfg.model.img_token_dim:
        raise ValueError(
            "SigLIP hidden_size "
            f"{encoder.config.hidden_size} != cfg.model.img_token_dim {cfg.model.img_token_dim}"
        )
    return processor, encoder


@torch.no_grad()
def add_online_siglip_features(
    batch: dict[str, torch.Tensor],
    *,
    processor: SiglipImageProcessor,
    encoder: SiglipVisionModel,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    image_slot_jpegs = batch.get("image_slot_jpegs")
    if image_slot_jpegs is None:
        return batch
    batch_size = len(image_slot_jpegs)
    slots_per_sample = max((len(slots) for slots in image_slot_jpegs), default=0)
    if batch_size == 0 or slots_per_sample == 0:
        batch["img_tokens"] = torch.zeros(
            batch_size,
            cfg.model.image_tokens,
            cfg.model.img_token_dim,
            device=device,
            dtype=resolve_dtype(cfg.model.dtype),
        )
        batch["img_mask"] = torch.zeros(
            batch_size,
            cfg.model.image_tokens,
            device=device,
            dtype=torch.bool,
        )
        return batch

    slot_mask = batch["image_slot_mask"].to(dtype=torch.bool)
    valid_slots: list[tuple[int, int]] = []
    flat_images: list[Image.Image] = []
    for sample_index, slots in enumerate(image_slot_jpegs):
        if len(slots) != slots_per_sample:
            raise ValueError(
                "Every sample must expose the same number of image slots; "
                f"sample 0 has {slots_per_sample}, sample {sample_index} has {len(slots)}"
            )
        for slot_index, payload in enumerate(slots):
            if bool(slot_mask[sample_index, slot_index].item()):
                valid_slots.append((sample_index, slot_index))
                flat_images.append(decode_jpeg_image(payload))

    if not flat_images:
        batch["img_tokens"] = torch.zeros(
            batch_size,
            cfg.model.image_tokens,
            cfg.model.img_token_dim,
            device=device,
            dtype=resolve_dtype(cfg.model.dtype),
        )
        batch["img_mask"] = torch.zeros(
            batch_size,
            cfg.model.image_tokens,
            device=device,
            dtype=torch.bool,
        )
        return batch

    inputs = processor(images=flat_images, return_tensors="pt")
    encoder_dtype = next(encoder.parameters()).dtype
    inputs = {
        key: (
            value.to(
                device=device,
                dtype=encoder_dtype if value.is_floating_point() else value.dtype,
            )
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in inputs.items()
    }
    image_features = encoder(**inputs).last_hidden_state
    if image_features.shape[-1] != cfg.model.img_token_dim:
        raise ValueError(
            f"SigLIP hidden dim mismatch: expected {cfg.model.img_token_dim}, "
            f"got {image_features.shape[-1]}"
        )

    tokens_per_image = int(image_features.shape[1])
    required_tokens = slots_per_sample * tokens_per_image
    if required_tokens > cfg.model.image_tokens:
        raise ValueError(
            "SigLIP image token budget is too small: "
            f"{slots_per_sample} slots x {tokens_per_image} tokens = {required_tokens}, "
            f"but cfg.model.image_tokens={cfg.model.image_tokens}"
        )

    output = torch.zeros(
        batch_size,
        cfg.model.image_tokens,
        cfg.model.img_token_dim,
        device=device,
        dtype=image_features.dtype,
    )
    mask = torch.zeros(
        batch_size,
        cfg.model.image_tokens,
        device=device,
        dtype=torch.bool,
    )
    for feature_index, (sample_index, slot_index) in enumerate(valid_slots):
        token_start = slot_index * tokens_per_image
        token_stop = token_start + tokens_per_image
        output[sample_index, token_start:token_stop] = image_features[feature_index]
        mask[sample_index, token_start:token_stop] = True

    batch["img_tokens"] = output.to(resolve_dtype(cfg.model.dtype))
    batch["img_mask"] = mask
    return batch


def create_optimizer(model: SFTConditionedRDT, cfg: ExperimentConfig):
    if cfg.model.freeze_state_adaptor and any(
        parameter.requires_grad
        for parameter in model.runner.state_adaptor.parameters()
    ):
        raise RuntimeError("The configured frozen state adaptor is trainable")

    if cfg.model.finetune_mode == "full":
        learning_rate = (
            cfg.training.learning_rate
            if cfg.training.learning_rate is not None
            else cfg.training.learning_rate_interfaces
        )
        rdt_parameters: list[torch.nn.Parameter] = []
        projector_parameters: list[torch.nn.Parameter] = []
        other_parameters: list[torch.nn.Parameter] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("runner.model."):
                rdt_parameters.append(parameter)
            elif name.startswith("qwen_adaptor."):
                projector_parameters.append(parameter)
            else:
                other_parameters.append(parameter)
        if not rdt_parameters:
            raise RuntimeError("No full-RDT parameters are trainable")
        if not projector_parameters:
            raise RuntimeError("No Qwen KV projector parameters are trainable")
        parameter_groups = [
            {
                "params": rdt_parameters,
                "lr": learning_rate,
                "weight_decay": cfg.training.weight_decay_interfaces,
                "name": "rdt",
            },
            {
                "params": projector_parameters,
                "lr": learning_rate,
                "weight_decay": cfg.training.weight_decay_interfaces,
                "name": "qwen_projector",
            },
        ]
        if other_parameters:
            parameter_groups.append(
                {
                    "params": other_parameters,
                    "lr": learning_rate,
                    "weight_decay": cfg.training.weight_decay_interfaces,
                    "name": "interfaces",
                }
            )
        return torch.optim.AdamW(
            parameter_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    lora_parameters: list[torch.nn.Parameter] = []
    interface_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            lora_parameters.append(parameter)
        else:
            interface_parameters.append(parameter)
    if not lora_parameters:
        raise RuntimeError("No LoRA parameters are trainable")
    if not interface_parameters:
        raise RuntimeError("No interface/final-layer parameters are trainable")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": cfg.training.learning_rate_lora,
                "weight_decay": 0.0,
                "name": "lora",
            },
            {
                "params": interface_parameters,
                "lr": cfg.training.learning_rate_interfaces,
                "weight_decay": cfg.training.weight_decay_interfaces,
                "name": "interfaces",
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return optimizer


def resolve_gradient_accumulation_steps(cfg: ExperimentConfig) -> tuple[int, int]:
    """Return (accumulation steps, effective global batch size)."""
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    per_microstep = cfg.training.micro_batch_size * world_size
    requested = cfg.training.global_batch_size
    if requested is None:
        accumulation = cfg.training.gradient_accumulation_steps
        return accumulation, per_microstep * accumulation
    if requested % per_microstep != 0:
        raise ValueError(
            "training.global_batch_size must be divisible by "
            "micro_batch_size * world_size; got "
            f"{requested} vs {cfg.training.micro_batch_size} * {world_size}"
        )
    accumulation = requested // per_microstep
    if accumulation <= 0:
        raise ValueError("Resolved gradient accumulation must be positive")
    return accumulation, requested


def resolve_validation_batch_limit(
    cfg: ExperimentConfig,
    accelerator: Accelerator,
) -> int:
    """Resolve a fixed global validation sample budget to local batch rounds."""
    requested_samples = cfg.training.validation_samples
    if requested_samples is None:
        return cfg.training.validation_batches
    global_examples_per_round = (
        cfg.training.micro_batch_size * accelerator.num_processes
    )
    if requested_samples % global_examples_per_round != 0:
        raise ValueError(
            "training.validation_samples must be divisible by "
            "micro_batch_size * world_size for an exact distributed subset; got "
            f"{requested_samples} vs {cfg.training.micro_batch_size} * "
            f"{accelerator.num_processes}"
        )
    return requested_samples // global_examples_per_round


def unwrap_model_without_optional_deepspeed(
    accelerator: Accelerator,
    model: torch.nn.Module,
) -> torch.nn.Module:
    """Unwrap DDP/FSDP while tolerating an unused, incompatible DeepSpeed install."""
    if str(accelerator.distributed_type).upper().split(".")[-1] != "DEEPSPEED":
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        return unwrapped
    try:
        return accelerator.unwrap_model(model)
    except ImportError as exc:
        if "deepspeed" not in str(exc).lower() and "_get_socket_with_port" not in str(exc):
            raise
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        return unwrapped


def model_state_dict_for_save(
    accelerator: Accelerator,
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Gather a complete state dict when the distributed backend shards weights."""
    distributed_type = str(accelerator.distributed_type).upper().split(".")[-1]
    if distributed_type in {"FSDP", "DEEPSPEED"}:
        # These backends require every rank to participate in state gathering.
        # FSDP returns the populated full state only on rank zero.
        return accelerator.get_state_dict(model)
    unwrapped = unwrap_model_without_optional_deepspeed(accelerator, model)
    return unwrapped.state_dict()


def log_metrics(
    accelerator: Accelerator,
    values: dict[str, float | int],
    *,
    step: int,
    report_to: str,
) -> None:
    """Log metrics, using the active W&B run directly for reliable history."""
    if report_to.lower() == "wandb":
        if accelerator.is_main_process:
            run = accelerator.get_tracker("wandb", unwrap=True)
            run.log(values, step=step, commit=True)
        return
    accelerator.log(values, step=step)


@torch.no_grad()
def validate(
    model: SFTConditionedRDT,
    dataloader: DataLoader,
    accelerator: Accelerator,
    cfg: ExperimentConfig,
    *,
    online_siglip: tuple[SiglipImageProcessor, SiglipVisionModel] | None = None,
) -> dict[str, float]:
    model.eval()
    validation_batch_limit = resolve_validation_batch_limit(cfg, accelerator)
    loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    mae_sum = torch.zeros_like(loss_sum)
    valid_count = torch.zeros_like(loss_sum)
    xyz_loss_sum = torch.zeros_like(loss_sum)
    xyz_valid_count = torch.zeros_like(loss_sum)
    rot_loss_sum = torch.zeros_like(loss_sum)
    rot_valid_count = torch.zeros_like(loss_sum)
    gripper_loss_sum = torch.zeros_like(loss_sum)
    gripper_valid_count = torch.zeros_like(loss_sum)
    sample_error_sum = torch.zeros_like(loss_sum)
    sample_valid_count = torch.zeros_like(loss_sum)
    horizon_loss_sum = torch.zeros(
        cfg.model.pred_horizon,
        device=accelerator.device,
        dtype=torch.float64,
    )
    horizon_valid_count = torch.zeros_like(horizon_loss_sum)
    # Columns: loss, MAE, examples, XYZ loss/count, rotation loss/count,
    # gripper loss/count, sampled-action MSE/count.
    suite_stats = torch.zeros(
        len(LIBERO_SUITE_IDS),
        11,
        device=accelerator.device,
        dtype=torch.float64,
    )
    devices = (
        [
            accelerator.device.index
            if accelerator.device.index is not None
            else torch.cuda.current_device()
        ]
        if accelerator.device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.training.validation_seed + accelerator.process_index)
        if accelerator.device.type == "cuda":
            torch.cuda.manual_seed_all(
                cfg.training.validation_seed + accelerator.process_index
            )
        for index, batch in enumerate(dataloader):
            if (
                validation_batch_limit > 0
                and index >= validation_batch_limit
            ):
                break
            if online_siglip is not None:
                batch = add_online_siglip_features(
                    batch,
                    processor=online_siglip[0],
                    encoder=online_siglip[1],
                    cfg=cfg,
                    device=accelerator.device,
                )
            metrics = model(batch)
            horizon_loss_sum += metrics["horizon_loss_sum"].double()
            horizon_valid_count += metrics["horizon_valid_count"].double()
            dataset_ids = list(batch.get("dataset_id", []))
            if len(dataset_ids) != int(metrics["sample_is_valid"].shape[0]):
                raise ValueError(
                    "Validation batches must carry one dataset_id per example; "
                    f"got {len(dataset_ids)} ids for "
                    f"{metrics['sample_is_valid'].shape[0]} examples"
                )
            for suite_index, suite_id in enumerate(LIBERO_SUITE_IDS):
                selected = torch.as_tensor(
                    [dataset_id == suite_id for dataset_id in dataset_ids],
                    device=accelerator.device,
                    dtype=torch.float64,
                )
                sample_valid = metrics["sample_is_valid"].double() * selected
                suite_stats[suite_index, 0] += (
                    metrics["sample_imitation_loss"].double() * sample_valid
                ).sum()
                suite_stats[suite_index, 1] += (
                    metrics["sample_target_mae"].double() * sample_valid
                ).sum()
                suite_stats[suite_index, 2] += sample_valid.sum()

                xyz_valid = metrics["sample_xyz_valid"].double() * selected
                suite_stats[suite_index, 3] += (
                    metrics["sample_xyz_loss"].double() * xyz_valid
                ).sum()
                suite_stats[suite_index, 4] += xyz_valid.sum()
                rot_valid = metrics["sample_rot_valid"].double() * selected
                suite_stats[suite_index, 5] += (
                    metrics["sample_rot_loss"].double() * rot_valid
                ).sum()
                suite_stats[suite_index, 6] += rot_valid.sum()
                gripper_valid = (
                    metrics["sample_gripper_valid"].double() * selected
                )
                suite_stats[suite_index, 7] += (
                    metrics["sample_gripper_loss"].double() * gripper_valid
                ).sum()
                suite_stats[suite_index, 8] += gripper_valid.sum()
            loss_sum += accelerator.reduce(
                metrics["loss_sum"].double(), reduction="sum"
            )
            mae_sum += accelerator.reduce(
                metrics["mae_sum"].double(), reduction="sum"
            )
            valid_count += accelerator.reduce(
                metrics["valid_count"].double(), reduction="sum"
            )
            xyz_loss_sum += accelerator.reduce(
                metrics["xyz_loss_sum"].double(), reduction="sum"
            )
            xyz_valid_count += accelerator.reduce(
                metrics["xyz_valid_count"].double(), reduction="sum"
            )
            rot_loss_sum += accelerator.reduce(
                metrics["rot_loss_sum"].double(), reduction="sum"
            )
            rot_valid_count += accelerator.reduce(
                metrics["rot_valid_count"].double(), reduction="sum"
            )
            gripper_loss_sum += accelerator.reduce(
                metrics["gripper_loss_sum"].double(), reduction="sum"
            )
            gripper_valid_count += accelerator.reduce(
                metrics["gripper_valid_count"].double(), reduction="sum"
            )
            if index < cfg.training.sample_validation_batches:
                # Route sampling through DDP/FSDP so parameter-gathering hooks
                # run just as they do for the training forward.
                prediction = model(batch, sample=True)
                target = batch["actions"].to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                time_mask = batch["action_time_mask"].unsqueeze(-1).to(
                    prediction.dtype
                )
                dim_mask = batch["action_dim_mask"].unsqueeze(1).to(
                    prediction.dtype
                )
                valid = time_mask * dim_mask
                per_sample_count = valid.sum(dim=(1, 2))
                per_sample_valid = (per_sample_count > 0).to(
                    prediction.dtype
                )
                diff = prediction - target
                if cfg.model.action_encoder_layout == "rdt_eef":
                    diff = torch.cat(
                        [
                            diff[..., :3],
                            torch.atan2(
                                torch.sin(diff[..., 3:6]),
                                torch.cos(diff[..., 3:6]),
                            ),
                            diff[..., 6:],
                        ],
                        dim=-1,
                    )
                per_sample_error = (
                    (diff.pow(2) * valid).sum(dim=(1, 2))
                    / per_sample_count.clamp_min(1.0)
                )
                error = (per_sample_error * per_sample_valid).sum()
                sample_error_sum += accelerator.reduce(
                    error.double(), reduction="sum"
                )
                sample_valid_count += accelerator.reduce(
                    per_sample_valid.sum().double(), reduction="sum"
                )
                for suite_index, suite_id in enumerate(LIBERO_SUITE_IDS):
                    selected = torch.as_tensor(
                        [dataset_id == suite_id for dataset_id in dataset_ids],
                        device=accelerator.device,
                        dtype=per_sample_error.dtype,
                    )
                    selected_valid = per_sample_valid * selected
                    suite_stats[suite_index, 9] += (
                        per_sample_error * selected_valid
                    ).sum().double()
                    suite_stats[suite_index, 10] += selected_valid.sum().double()
    model.train()
    horizon_loss_sum = accelerator.reduce(horizon_loss_sum, reduction="sum")
    horizon_valid_count = accelerator.reduce(
        horizon_valid_count, reduction="sum"
    )
    suite_stats = accelerator.reduce(suite_stats, reduction="sum")
    loss_denominator = valid_count.clamp_min(1.0)
    sample_denominator = sample_valid_count.clamp_min(1.0)
    result = {
        "val/loss": (
            float((loss_sum / loss_denominator).cpu())
            if valid_count.item() > 0
            else math.nan
        ),
        "val/imitation_loss": (
            float((loss_sum / loss_denominator).cpu())
            if valid_count.item() > 0
            else math.nan
        ),
        "val/loss_xyz": (
            float((xyz_loss_sum / xyz_valid_count.clamp_min(1.0)).cpu())
            if xyz_valid_count.item() > 0
            else math.nan
        ),
        "val/loss_rot": (
            float((rot_loss_sum / rot_valid_count.clamp_min(1.0)).cpu())
            if rot_valid_count.item() > 0
            else math.nan
        ),
        "val/loss_gripper": (
            float((gripper_loss_sum / gripper_valid_count.clamp_min(1.0)).cpu())
            if gripper_valid_count.item() > 0
            else math.nan
        ),
        "val/target_mae": (
            float((mae_sum / loss_denominator).cpu())
            if valid_count.item() > 0
            else math.nan
        ),
        "val/examples": float(valid_count.cpu()),
        "val/sample_mse": (
            float((sample_error_sum / sample_denominator).cpu())
            if sample_valid_count.item() > 0
            else math.nan
        ),
    }
    for horizon_index in range(cfg.model.pred_horizon):
        count = horizon_valid_count[horizon_index]
        result[f"val/horizon_mse/step_{horizon_index:02d}"] = (
            float((horizon_loss_sum[horizon_index] / count.clamp_min(1.0)).cpu())
            if count.item() > 0
            else math.nan
        )
    for suite_index, suite_id in enumerate(LIBERO_SUITE_IDS):
        stats = suite_stats[suite_index]
        example_count = stats[2]
        if example_count.item() <= 0:
            continue
        prefix = f"val/{suite_id}"
        result[f"{prefix}/loss"] = float((stats[0] / example_count).cpu())
        result[f"{prefix}/imitation_loss"] = result[f"{prefix}/loss"]
        result[f"{prefix}/target_mae"] = float((stats[1] / example_count).cpu())
        result[f"{prefix}/examples"] = float(example_count.cpu())
        result[f"{prefix}/loss_xyz"] = float(
            (stats[3] / stats[4].clamp_min(1.0)).cpu()
        )
        result[f"{prefix}/loss_rot"] = float(
            (stats[5] / stats[6].clamp_min(1.0)).cpu()
        )
        result[f"{prefix}/loss_gripper"] = float(
            (stats[7] / stats[8].clamp_min(1.0)).cpu()
        )
        result[f"{prefix}/sample_mse"] = (
            float((stats[9] / stats[10].clamp_min(1.0)).cpu())
            if stats[10].item() > 0
            else math.nan
        )
    return result


def train(
    cfg: ExperimentConfig,
    load_pretrained: bool = True,
    *,
    online_siglip_model_id: str | None = None,
    online_siglip_fallback_model_id: str | None = "google/siglip-so400m-patch14-384",
    base_artifact: str | Path | None = None,
    init_artifact: str | Path | None = None,
) -> None:
    seed_everything(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulation_steps, effective_global_batch = (
        resolve_gradient_accumulation_steps(cfg)
    )
    report_to = cfg.training.report_to
    accelerator_log_with = (
        None if report_to.lower() in {"", "none", "no"} else report_to
    )
    accumulation_plugin = GradientAccumulationPlugin(
        num_steps=accumulation_steps,
        # Carry an incomplete accumulation window into the next epoch. This
        # keeps every optimizer step at the configured global batch size.
        sync_with_dataloader=False,
    )
    accelerator = Accelerator(
        gradient_accumulation_plugin=accumulation_plugin,
        mixed_precision=cfg.training.mixed_precision,
        log_with=accelerator_log_with,
        project_dir=str(output_dir / "logs"),
        # Step the scheduler explicitly once per optimizer update. Accelerate's
        # default otherwise advances it once per process in multi-GPU runs.
        step_scheduler_with_optimizer=False,
    )
    actual_global_batch = (
        cfg.training.micro_batch_size
        * accumulation_steps
        * accelerator.num_processes
    )
    if (
        cfg.training.global_batch_size is not None
        and actual_global_batch != cfg.training.global_batch_size
    ):
        raise ValueError(
            "Accelerate world size differs from the launch environment used to "
            "resolve global batch size: expected "
            f"{cfg.training.global_batch_size}, got {actual_global_batch}"
        )
    effective_global_batch = actual_global_batch
    if accelerator_log_with is not None:
        init_kwargs: dict[str, dict[str, str]] | None = None
        if cfg.training.wandb_run_name is not None and report_to.lower() == "wandb":
            init_kwargs = {
                "wandb": {"name": cfg.training.wandb_run_name}
            }
        tracker_config = asdict(cfg)
        if report_to.lower() == "tensorboard":
            tracker_config = _tensorboard_hparams(tracker_config)
        accelerator.init_trackers(
            cfg.training.wandb_project,
            config=tracker_config,
            init_kwargs=init_kwargs,
        )

    use_online_siglip = online_siglip_model_id is not None
    train_loader = create_dataloader(
        cfg.data.train_manifest,
        cfg,
        shuffle=True,
        online_siglip=use_online_siglip,
    )
    val_loader = create_dataloader(
        cfg.data.val_manifest,
        cfg,
        shuffle=cfg.data.shuffle_validation,
        online_siglip=use_online_siglip,
        stratified=cfg.data.stratified_validation,
    )
    model = SFTConditionedRDT(
        cfg,
        load_pretrained=load_pretrained,
        base_artifact=base_artifact,
    )
    if init_artifact is not None:
        load_trainable_artifact(model, init_artifact, trainable=True)
    online_siglip = None
    if use_online_siglip:
        online_siglip = load_online_siglip(
            model_id=online_siglip_model_id,
            fallback_model_id=online_siglip_fallback_model_id,
            cfg=cfg,
            device=accelerator.device,
        )
    model_report = model.trainable_parameter_report()
    if accelerator.is_main_process:
        print(json.dumps(model_report, indent=2))
        print(
            "Batch configuration: "
            f"micro={cfg.training.micro_batch_size}, "
            f"accumulation={accumulation_steps}, "
            f"world={accelerator.num_processes}, "
            f"effective_global={effective_global_batch}"
        )
        if model.lora_targets:
            print("First LoRA targets:")
            for target in model.lora_targets[:14]:
                print("  ", target)

    optimizer = create_optimizer(model, cfg)
    scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=cfg.training.max_steps,
    )
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    # DeepSpeed may replace the accumulation value while preparing its engine.
    # Recheck after wrapping so a launcher config cannot silently change the
    # user-requested effective batch size.
    prepared_accumulation = int(accelerator.gradient_accumulation_steps)
    prepared_global_batch = (
        cfg.training.micro_batch_size
        * prepared_accumulation
        * accelerator.num_processes
    )
    if (
        cfg.training.global_batch_size is not None
        and prepared_global_batch != cfg.training.global_batch_size
    ):
        raise RuntimeError(
            "The distributed backend changed gradient accumulation during "
            "prepare: expected global batch "
            f"{cfg.training.global_batch_size}, got {prepared_global_batch}. "
            "For DeepSpeed, set gradient_accumulation_steps to 'auto'."
        )
    accumulation_steps = prepared_accumulation
    effective_global_batch = prepared_global_batch

    global_step = 0
    running_loss = 0.0
    running_mae = 0.0
    running_xyz_loss = 0.0
    running_rot_loss = 0.0
    running_gripper_loss = 0.0
    running_step_time = 0.0
    running_steps = 0
    pending_loss = 0.0
    pending_mae = 0.0
    pending_xyz_loss_sum = 0.0
    pending_xyz_valid_count = 0.0
    pending_rot_loss_sum = 0.0
    pending_rot_valid_count = 0.0
    pending_gripper_loss_sum = 0.0
    pending_gripper_valid_count = 0.0
    pending_microbatches = 0
    update_started_at = time.perf_counter()
    epoch = 0
    model.train()
    while global_step < cfg.training.max_steps:
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        saw_batch = False
        for batch in train_loader:
            saw_batch = True
            if online_siglip is not None:
                batch = add_online_siglip_features(
                    batch,
                    processor=online_siglip[0],
                    encoder=online_siglip[1],
                    cfg=cfg,
                    device=accelerator.device,
                )
            with accelerator.accumulate(model):
                metrics = model(batch)
                loss = metrics["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), cfg.training.max_grad_norm
                    )
                optimizer.step()
                optimizer_update_succeeded = (
                    accelerator.sync_gradients
                    and not accelerator.optimizer_step_was_skipped
                )
                if optimizer_update_succeeded:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            pending_loss += float(loss.detach())
            pending_mae += float(metrics["train_target_mae"].detach())
            pending_xyz_loss_sum += float(metrics["xyz_loss_sum"].detach())
            pending_xyz_valid_count += float(metrics["xyz_valid_count"].detach())
            pending_rot_loss_sum += float(metrics["rot_loss_sum"].detach())
            pending_rot_valid_count += float(metrics["rot_valid_count"].detach())
            pending_gripper_loss_sum += float(
                metrics["gripper_loss_sum"].detach()
            )
            pending_gripper_valid_count += float(
                metrics["gripper_valid_count"].detach()
            )
            pending_microbatches += 1
            if not accelerator.sync_gradients:
                continue
            if not optimizer_update_succeeded:
                pending_loss = 0.0
                pending_mae = 0.0
                pending_xyz_loss_sum = 0.0
                pending_xyz_valid_count = 0.0
                pending_rot_loss_sum = 0.0
                pending_rot_valid_count = 0.0
                pending_gripper_loss_sum = 0.0
                pending_gripper_valid_count = 0.0
                pending_microbatches = 0
                update_started_at = time.perf_counter()
                continue
            global_step += 1
            local_step_time = time.perf_counter() - update_started_at
            step_time_tensor = torch.tensor(
                local_step_time,
                device=accelerator.device,
                dtype=torch.float64,
            )
            # The slowest rank determines distributed optimizer-step latency.
            step_time = float(
                accelerator.reduce(step_time_tensor, reduction="max").cpu()
            )
            step_metrics = torch.tensor(
                [
                    pending_loss / max(pending_microbatches, 1),
                    pending_mae / max(pending_microbatches, 1),
                    pending_xyz_loss_sum / max(pending_xyz_valid_count, 1.0),
                    pending_rot_loss_sum / max(pending_rot_valid_count, 1.0),
                    pending_gripper_loss_sum
                    / max(pending_gripper_valid_count, 1.0),
                ],
                device=accelerator.device,
                dtype=torch.float64,
            )
            step_metrics = accelerator.reduce(step_metrics, reduction="mean")
            running_loss += float(step_metrics[0].cpu())
            running_mae += float(step_metrics[1].cpu())
            running_xyz_loss += float(step_metrics[2].cpu())
            running_rot_loss += float(step_metrics[3].cpu())
            running_gripper_loss += float(step_metrics[4].cpu())
            running_step_time += step_time
            running_steps += 1
            pending_loss = 0.0
            pending_mae = 0.0
            pending_xyz_loss_sum = 0.0
            pending_xyz_valid_count = 0.0
            pending_rot_loss_sum = 0.0
            pending_rot_valid_count = 0.0
            pending_gripper_loss_sum = 0.0
            pending_gripper_valid_count = 0.0
            pending_microbatches = 0

            step_log_data: dict[str, float | int] = {}
            if cfg.training.log_every > 0 and global_step % cfg.training.log_every == 0:
                average_step_time = running_step_time / max(running_steps, 1)
                training_log_data = {
                    "train/loss": running_loss / max(running_steps, 1),
                    "train/imitation_loss": running_loss / max(running_steps, 1),
                    "train/loss_xyz": running_xyz_loss / max(running_steps, 1),
                    "train/loss_rot": running_rot_loss / max(running_steps, 1),
                    "train/loss_gripper": running_gripper_loss
                    / max(running_steps, 1),
                    "train/target_mae": running_mae / max(running_steps, 1),
                    "train/step": global_step,
                    "train/effective_global_batch": effective_global_batch,
                    "train/step_time_sec": average_step_time,
                    "train/samples_per_sec": (
                        effective_global_batch / max(average_step_time, 1e-12)
                    ),
                }
                for index, group in enumerate(optimizer.param_groups):
                    group_name = group.get("name", str(index))
                    training_log_data[f"train/lr_{group_name}"] = group["lr"]
                step_log_data.update(training_log_data)
                if accelerator.is_main_process:
                    print(training_log_data)
                running_loss = 0.0
                running_mae = 0.0
                running_xyz_loss = 0.0
                running_rot_loss = 0.0
                running_gripper_loss = 0.0
                running_step_time = 0.0
                running_steps = 0

            if (
                cfg.training.validate_every > 0
                and global_step % cfg.training.validate_every == 0
            ):
                validation = validate(
                    model,
                    val_loader,
                    accelerator,
                    cfg,
                    online_siglip=online_siglip,
                )
                step_log_data.update(validation)
                if accelerator.is_main_process:
                    print(validation)

            # Commit at most once for a given optimizer step. In particular,
            # training and validation commonly coincide (for example every 100
            # steps); separate committed W&B calls at the same explicit step can
            # cause the second record to be discarded.
            if step_log_data:
                log_metrics(
                    accelerator,
                    step_log_data,
                    step=global_step,
                    report_to=report_to,
                )

            if (
                cfg.training.save_every > 0
                and global_step % cfg.training.save_every == 0
            ):
                accelerator.wait_for_everyone()
                state_dict = model_state_dict_for_save(accelerator, model)
                if accelerator.is_main_process:
                    unwrapped = unwrap_model_without_optional_deepspeed(
                        accelerator, model
                    )
                    save_trainable_artifact(
                        unwrapped,
                        output_dir / f"checkpoint-{global_step}",
                        {
                            "global_step": global_step,
                            "effective_global_batch": effective_global_batch,
                            "pretrained_model": cfg.pretrained_model,
                            "model_report": model_report,
                            "config": asdict(cfg),
                        },
                        model_state_dict=state_dict,
                    )
                del state_dict
                accelerator.wait_for_everyone()

            # Start the next update timer after logging, validation, and saves,
            # so those maintenance costs do not inflate training step latency.
            update_started_at = time.perf_counter()
            if global_step >= cfg.training.max_steps:
                break
        if not saw_batch:
            raise RuntimeError(
                "Training dataloader yielded no batches. Check dataset filtering "
                "and whether drop_last exceeds the available sample count."
            )
        epoch += 1

    accelerator.wait_for_everyone()
    state_dict = model_state_dict_for_save(accelerator, model)
    if accelerator.is_main_process:
        unwrapped = unwrap_model_without_optional_deepspeed(accelerator, model)
        save_trainable_artifact(
            unwrapped,
            output_dir / "final",
            {
                "global_step": global_step,
                "effective_global_batch": effective_global_batch,
                "pretrained_model": cfg.pretrained_model,
                "model_report": model_report,
                "config": asdict(cfg),
            },
            model_state_dict=state_dict,
        )
    accelerator.wait_for_everyone()
    del state_dict
    accelerator.end_training()
