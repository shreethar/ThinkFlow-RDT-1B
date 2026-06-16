from __future__ import annotations

import gc
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ExperimentConfig
from .lora import apply_lora, count_parameters
from .rdt_imports import import_rdt_runner


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def install_checkpointed_forward(rdt_core: nn.Module) -> None:
    """Install an RDT forward equivalent that checkpoints each Transformer block."""

    def checkpointed_forward(
        self,
        x,
        freq,
        t,
        lang_c,
        img_c,
        lang_mask=None,
        img_mask=None,
    ):
        t_embed = self.t_embedder(t).unsqueeze(1)
        freq_embed = self.freq_embedder(freq).unsqueeze(1)
        if t_embed.shape[0] == 1:
            t_embed = t_embed.expand(x.shape[0], -1, -1)
        x = torch.cat([t_embed, freq_embed, x], dim=1)
        x = x + self.x_pos_embed
        lang_c = lang_c + self.lang_cond_pos_embed[:, : lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed[:, : img_c.shape[1]]
        conditions = (lang_c, img_c)
        masks = (lang_mask, img_mask)
        for index, block in enumerate(self.blocks):
            condition = conditions[index % 2]
            mask = masks[index % 2]
            if self.training and torch.is_grad_enabled():
                def custom_forward(x_value, c_value, block=block, mask=mask):
                    return block(x_value, c_value, mask)
                x = checkpoint(
                    custom_forward,
                    x,
                    condition,
                    use_reentrant=False,
                )
            else:
                x = block(x, condition, mask)
        x = self.final_layer(x)
        return x[:, -self.horizon :]

    rdt_core.forward = types.MethodType(checkpointed_forward, rdt_core)


def _copy_selected_pretrained_weights(target_runner, source_runner, cfg: ExperimentConfig) -> dict[str, int]:
    """
    Copy the pretrained RDT core while deliberately leaving new Qwen interfaces,
    custom condition positional embeddings, state adaptor, and 7D output layer fresh.
    """
    source = source_runner.state_dict()
    target = target_runner.state_dict()
    allowed_prefixes = (
        "model.blocks.",
        "model.t_embedder.",
        "model.freq_embedder.",
        "model.x_pos_embed",
        "model.final_layer.norm_final.",
    )
    if cfg.model.copy_pretrained_final_fc1:
        allowed_prefixes = allowed_prefixes + ("model.final_layer.ffn_final.fc1.",)

    copied = 0
    skipped_shape = 0
    selected: dict[str, torch.Tensor] = {}
    for name, tensor in source.items():
        if not name.startswith(allowed_prefixes):
            continue
        if name not in target or target[name].shape != tensor.shape:
            skipped_shape += 1
            continue
        selected[name] = tensor
        copied += 1
    missing, unexpected = target_runner.load_state_dict(selected, strict=False)
    del source, target, selected
    return {
        "copied_tensors": copied,
        "skipped_shape": skipped_shape,
        "missing_after_partial_load": len(missing),
        "unexpected_after_partial_load": len(unexpected),
    }


class SFTConditionedRDT(nn.Module):
    def __init__(self, cfg: ExperimentConfig, load_pretrained: bool = True):
        super().__init__()
        self.cfg = cfg
        RDTRunner = import_rdt_runner(cfg.rdt_repo)
        dtype = resolve_dtype(cfg.model.dtype)
        self.compute_dtype = dtype
        runner_config = {
            "lang_adaptor": cfg.model.lang_adaptor,
            "img_adaptor": cfg.model.img_adaptor,
            "state_adaptor": cfg.model.state_adaptor,
            "rdt": {
                "hidden_size": cfg.model.hidden_size,
                "depth": cfg.model.depth,
                "num_heads": cfg.model.num_heads,
            },
            "noise_scheduler": {
                "num_train_timesteps": cfg.noise_scheduler.num_train_timesteps,
                "num_inference_timesteps": cfg.noise_scheduler.num_inference_timesteps,
                "beta_schedule": cfg.noise_scheduler.beta_schedule,
                "prediction_type": cfg.noise_scheduler.prediction_type,
                "clip_sample": cfg.noise_scheduler.clip_sample,
            },
        }
        self.runner = RDTRunner(
            action_dim=cfg.model.action_dim,
            pred_horizon=cfg.model.pred_horizon,
            config=runner_config,
            lang_token_dim=cfg.model.qwen_hidden_size,
            img_token_dim=cfg.model.qwen_hidden_size,
            state_token_dim=cfg.model.state_dim,
            max_lang_cond_len=cfg.model.max_lang_tokens,
            img_cond_len=cfg.model.image_tokens,
            lang_pos_embed_config=None,
            img_pos_embed_config=None,
            dtype=dtype,
        )
        self.pretrained_report: dict[str, int] | None = None

        if load_pretrained and cfg.pretrained_model:
            source = RDTRunner.from_pretrained(cfg.pretrained_model)
            self.pretrained_report = _copy_selected_pretrained_weights(
                self.runner, source, cfg
            )
            del source
            gc.collect()

        if cfg.model.gradient_checkpointing:
            install_checkpointed_forward(self.runner.model)

        self.runner.model, self.lora_targets = apply_lora(
            self.runner.model, cfg.lora
        )
        # Keep the newly initialized interfaces in the configured compute dtype.
        self.runner.lang_adaptor.to(dtype=dtype).requires_grad_(True)
        self.runner.img_adaptor.to(dtype=dtype).requires_grad_(True)
        self.runner.state_adaptor.to(dtype=dtype).requires_grad_(True)

    @property
    def model_dtype(self) -> torch.dtype:
        return self.compute_dtype

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Using forward, rather than calling compute_loss on a prepared DDP model,
        # lets Accelerate apply DDP hooks and mixed-precision autocast correctly.
        return self.compute_loss(batch)

    def trainable_parameter_report(self) -> dict[str, Any]:
        trainable, total = count_parameters(self)
        return {
            "trainable": trainable,
            "total": total,
            "percentage": 100.0 * trainable / max(total, 1),
            "lora_target_count": len(self.lora_targets),
            "pretrained": self.pretrained_report,
        }

    def cast_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        dtype = self.model_dtype
        float_keys = {
            "lang_tokens",
            "img_tokens",
            "state",
            "actions",
            "action_dim_mask",
            "ctrl_freq",
        }
        return {
            key: value.to(dtype=dtype) if key in float_keys else value
            for key, value in batch.items()
        }

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = self.cast_batch(batch)
        lang_tokens = batch["lang_tokens"]
        img_tokens = batch["img_tokens"]
        states = batch["state"].unsqueeze(1)
        actions = batch["actions"]
        lang_mask = batch["lang_mask"].bool()
        img_mask = batch["img_mask"].bool()
        time_mask = batch["action_time_mask"].bool()
        dim_mask = batch["action_dim_mask"].to(actions.dtype).unsqueeze(1)
        ctrl_freq = batch["ctrl_freq"]

        batch_size = actions.shape[0]
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0,
            self.runner.num_train_timesteps,
            (batch_size,),
            device=actions.device,
            dtype=torch.long,
        )
        noisy_actions = self.runner.noise_scheduler.add_noise(
            actions, noise, timesteps
        )

        state_action = torch.cat([states, noisy_actions], dim=1)
        state_action_dim_mask = dim_mask.expand(
            -1, state_action.shape[1], -1
        )
        state_action = torch.cat(
            [state_action, state_action_dim_mask], dim=-1
        )

        lang_cond, img_cond, state_action_cond = self.runner.adapt_conditions(
            lang_tokens, img_tokens, state_action
        )
        prediction = self.runner.model(
            state_action_cond,
            ctrl_freq,
            timesteps,
            lang_cond,
            img_cond,
            lang_mask=lang_mask,
            img_mask=img_mask,
        )

        if self.runner.prediction_type == "sample":
            target = actions
        elif self.runner.prediction_type == "epsilon":
            target = noise
        else:
            raise ValueError(
                f"Unsupported prediction type: {self.runner.prediction_type}"
            )

        valid = time_mask.unsqueeze(-1).to(prediction.dtype) * dim_mask
        print(f"DEBUG: time_mask sum={time_mask.sum().item()}, dim_mask sum={dim_mask.sum().item()}, valid sum={valid.sum().item()}")
        print(f"DEBUG: prediction max={prediction.max().item():.6f}, min={prediction.min().item():.6f}")
        print(f"DEBUG: target max={target.max().item():.6f}, min={target.min().item():.6f}")
        squared_error = (prediction - target).pow(2) * valid
        denominator = valid.sum().clamp_min(1.0)
        loss = squared_error.sum() / denominator
        mae = ((prediction - target).abs() * valid).sum() / denominator
        return {"loss": loss, "train_target_mae": mae.detach()}

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self.cast_batch(batch)
        lang_mask = batch["lang_mask"].bool()
        img_mask = batch["img_mask"].bool()
        dim_mask = batch["action_dim_mask"].to(batch["state"].dtype).unsqueeze(1)
        states = batch["state"].unsqueeze(1)

        state_input = torch.cat([states, dim_mask], dim=-1)
        lang_cond, img_cond, state_cond = self.runner.adapt_conditions(
            batch["lang_tokens"], batch["img_tokens"], state_input
        )
        noisy = torch.randn(
            states.shape[0],
            self.cfg.model.pred_horizon,
            self.cfg.model.action_dim,
            device=states.device,
            dtype=states.dtype,
        )
        scheduler = self.runner.noise_scheduler_sample
        try:
            scheduler.set_timesteps(
                self.runner.num_inference_timesteps, device=states.device
            )
        except TypeError:
            scheduler.set_timesteps(self.runner.num_inference_timesteps)

        expanded_dim_mask = dim_mask.expand(-1, noisy.shape[1], -1)
        for timestep in scheduler.timesteps:
            timestep = timestep.to(states.device)
            action_input = torch.cat([noisy, expanded_dim_mask], dim=-1)
            action_cond = self.runner.state_adaptor(action_input)
            state_action_cond = torch.cat([state_cond, action_cond], dim=1)
            model_timestep = timestep.reshape(1).expand(states.shape[0])
            output = self.runner.model(
                state_action_cond,
                batch["ctrl_freq"],
                model_timestep,
                lang_cond,
                img_cond,
                lang_mask=lang_mask,
                img_mask=img_mask,
            )
            noisy = scheduler.step(output, timestep, noisy).prev_sample
            noisy = noisy.to(states.dtype)
        return noisy * expanded_dim_mask
