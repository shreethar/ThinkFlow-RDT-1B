from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_scheduler

from .checkpoint import save_trainable_artifact
from .config import ExperimentConfig
from .data import CachedFeatureDataset, RDTBatchCollator
from .model import SFTConditionedRDT


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloader(
    manifest: str,
    cfg: ExperimentConfig,
    shuffle: bool,
) -> DataLoader:
    dataset = CachedFeatureDataset(manifest)
    collator = RDTBatchCollator(
        max_lang_tokens=cfg.model.max_lang_tokens,
        image_tokens=cfg.model.image_tokens,
        pred_horizon=cfg.model.pred_horizon,
        feature_dim=cfg.model.qwen_hidden_size,
        state_dim=cfg.model.state_dim,
        action_dim=cfg.model.action_dim,
        lang_token_dim=cfg.model.lang_token_dim,
        img_token_dim=cfg.model.img_token_dim,
    )
    persistent = cfg.data.persistent_workers and cfg.data.num_workers > 0
    return DataLoader(
        dataset,
        batch_size=cfg.training.micro_batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=persistent,
        collate_fn=collator,
        drop_last=shuffle,
    )


def create_optimizer(model: SFTConditionedRDT, cfg: ExperimentConfig):
    lora_parameters = []
    interface_parameters = []
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


@torch.no_grad()
def validate(
    model: SFTConditionedRDT,
    dataloader: DataLoader,
    accelerator: Accelerator,
    cfg: ExperimentConfig,
) -> dict[str, float]:
    model.eval()
    losses: list[torch.Tensor] = []
    sample_mses: list[torch.Tensor] = []
    for index, batch in enumerate(dataloader):
        if index >= cfg.training.validation_batches:
            break
        metrics = model(batch)
        gathered_loss = accelerator.gather_for_metrics(metrics["loss"].detach())
        losses.append(gathered_loss.float().mean().cpu())
        if index < cfg.training.sample_validation_batches:
            unwrapped = accelerator.unwrap_model(model)
            prediction = unwrapped.sample_actions(batch)
            target = unwrapped.cast_batch(batch)["actions"]
            time_mask = batch["action_time_mask"].unsqueeze(-1).to(prediction.dtype)
            dim_mask = batch["action_dim_mask"].unsqueeze(1).to(prediction.dtype)
            valid = time_mask * dim_mask
            mse = ((prediction - target).pow(2) * valid).sum() / valid.sum().clamp_min(1)
            gathered_mse = accelerator.gather_for_metrics(mse.detach())
            sample_mses.append(gathered_mse.float().mean().cpu())
    model.train()
    return {
        "val/loss": float(torch.stack(losses).mean()) if losses else math.nan,
        "val/sample_mse": (
            float(torch.stack(sample_mses).mean()) if sample_mses else math.nan
        ),
    }


def train(cfg: ExperimentConfig, load_pretrained: bool = True) -> None:
    seed_everything(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=cfg.training.report_to,
        project_dir=str(output_dir / "logs"),
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("thinkflow-rdt")

    train_loader = create_dataloader(cfg.data.train_manifest, cfg, shuffle=True)
    val_loader = create_dataloader(cfg.data.val_manifest, cfg, shuffle=False)
    model = SFTConditionedRDT(cfg, load_pretrained=load_pretrained)
    if accelerator.is_main_process:
        print(json.dumps(model.trainable_parameter_report(), indent=2))
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

    global_step = 0
    running_loss = 0.0
    model.train()
    while global_step < cfg.training.max_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                metrics = model(batch)
                loss = metrics["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), cfg.training.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach())
            if not accelerator.sync_gradients:
                continue
            global_step += 1

            if global_step % cfg.training.log_every == 0:
                log_data = {
                    "train/loss": running_loss / cfg.training.log_every,
                    "train/lr_lora": optimizer.param_groups[0]["lr"],
                    "train/lr_interfaces": optimizer.param_groups[1]["lr"],
                    "train/step": global_step,
                }
                accelerator.log(log_data, step=global_step)
                if accelerator.is_main_process:
                    print(log_data)
                running_loss = 0.0

            if global_step % cfg.training.validate_every == 0:
                validation = validate(model, val_loader, accelerator, cfg)
                accelerator.log(validation, step=global_step)
                if accelerator.is_main_process:
                    print(validation)

            if global_step % cfg.training.save_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    save_trainable_artifact(
                        unwrapped,
                        output_dir / f"checkpoint-{global_step}",
                        {
                            "global_step": global_step,
                            "pretrained_model": cfg.pretrained_model,
                            "model_report": unwrapped.trainable_parameter_report(),
                        },
                    )

            if global_step >= cfg.training.max_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_trainable_artifact(
            unwrapped,
            output_dir / "final",
            {
                "global_step": global_step,
                "pretrained_model": cfg.pretrained_model,
                "model_report": unwrapped.trainable_parameter_report(),
            },
        )
    accelerator.end_training()
