from __future__ import annotations

import gc
import types
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


def _self_attention_with_external_kv(
    attention: nn.Module,
    x: torch.Tensor,
    external_kv: torch.Tensor | None,
) -> torch.Tensor:
    """Run timm Attention while appending one or more already projected K/V pairs."""
    if external_kv is None:
        return attention(x)

    batch_size, token_count, width = x.shape
    expected_width = width * 2
    if external_kv.ndim != 3 or external_kv.shape[:2] != (batch_size, 1):
        raise ValueError(
            "Projected Qwen KV must be [B, 1, 2 * hidden_size], got "
            f"{tuple(external_kv.shape)}"
        )
    if external_kv.shape[-1] != expected_width:
        raise ValueError(
            f"Projected Qwen KV width must be {expected_width}, got "
            f"{external_kv.shape[-1]}"
        )

    qkv = attention.qkv(x).reshape(
        batch_size,
        token_count,
        3,
        attention.num_heads,
        attention.head_dim,
    ).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)
    external_key, external_value = external_kv.chunk(2, dim=-1)
    external_key = external_key.reshape(
        batch_size, 1, attention.num_heads, attention.head_dim
    ).permute(0, 2, 1, 3)
    external_value = external_value.reshape(
        batch_size, 1, attention.num_heads, attention.head_dim
    ).permute(0, 2, 1, 3)

    query = attention.q_norm(query)
    key = attention.k_norm(key)
    external_key = attention.k_norm(external_key)
    key = torch.cat([key, external_key], dim=2)
    value = torch.cat([value, external_value], dim=2)

    if attention.fused_attn:
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=attention.attn_drop.p if attention.training else 0.0,
        )
    else:
        scores = (query * attention.scale) @ key.transpose(-2, -1)
        probabilities = attention.attn_drop(scores.softmax(dim=-1))
        output = probabilities @ value

    attention_width = getattr(attention, "attn_dim", width)
    output = output.transpose(1, 2).reshape(
        batch_size, token_count, attention_width
    )
    output = getattr(attention, "norm", nn.Identity())(output)
    output = attention.proj(output)
    return attention.proj_drop(output)


def _rdt_block_with_external_kv(
    block: nn.Module,
    x: torch.Tensor,
    condition: torch.Tensor,
    condition_mask: torch.Tensor | None,
    external_kv: torch.Tensor | None,
) -> torch.Tensor:
    residual = x
    x = _self_attention_with_external_kv(
        block.attn,
        block.norm1(x),
        external_kv,
    )
    x = x + residual

    residual = x
    x = block.cross_attn(block.norm2(x), condition, condition_mask)
    x = x + residual

    residual = x
    x = block.ffn(block.norm3(x))
    return x + residual


def install_kv_aware_forward(
    rdt_core: nn.Module,
    *,
    gradient_checkpointing: bool,
) -> None:
    """Install an upstream-equivalent RDT forward with optional Qwen KV injection."""

    def kv_aware_forward(
        self,
        x,
        freq,
        t,
        lang_c,
        img_c,
        lang_mask=None,
        img_mask=None,
        external_kv=None,
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
            if (
                gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                if external_kv is None:
                    def custom_forward(x_value, c_value, block=block, mask=mask):
                        return _rdt_block_with_external_kv(
                            block, x_value, c_value, mask, None
                        )

                    x = checkpoint(
                        custom_forward, x, condition, use_reentrant=False
                    )
                else:
                    def custom_forward(
                        x_value,
                        c_value,
                        kv_value,
                        block=block,
                        mask=mask,
                    ):
                        return _rdt_block_with_external_kv(
                            block, x_value, c_value, mask, kv_value
                        )

                    x = checkpoint(
                        custom_forward,
                        x,
                        condition,
                        external_kv,
                        use_reentrant=False,
                    )
            else:
                x = _rdt_block_with_external_kv(
                    block, x, condition, mask, external_kv
                )
        x = self.final_layer(x)
        return x[:, -self.horizon :]

    rdt_core.forward = types.MethodType(kv_aware_forward, rdt_core)


def _copy_selected_pretrained_weights(target_runner, source_runner, cfg: ExperimentConfig) -> dict[str, int]:
    """Copy either the legacy selected core or every compatible runner tensor."""
    source = source_runner.state_dict()
    target = target_runner.state_dict()
    allowed_prefixes: tuple[str, ...] | None
    if cfg.model.pretrained_copy_mode == "compatible":
        allowed_prefixes = None
    else:
        allowed_prefixes = (
            "model.blocks.",
            "model.t_embedder.",
            "model.freq_embedder.",
            "model.x_pos_embed",
            "model.img_cond_pos_embed",
            "model.final_layer.norm_final.",
        )
        if cfg.model.copy_pretrained_final_fc1:
            allowed_prefixes = allowed_prefixes + (
                "model.final_layer.ffn_final.fc1.",
            )

    copied = 0
    partially_copied = 0
    skipped_shape = 0
    state_adaptor_copied = 0
    selected: dict[str, torch.Tensor] = {}
    for name, tensor in source.items():
        if allowed_prefixes is not None and not name.startswith(allowed_prefixes):
            continue
        if name not in target:
            skipped_shape += 1
            continue
        if target[name].shape != tensor.shape:
            if (
                cfg.model.pretrained_copy_mode == "compatible"
                and name in {
                    "model.lang_cond_pos_embed",
                    "model.img_cond_pos_embed",
                }
                and tensor.ndim == 3
                and target[name].ndim == 3
                and tensor.shape[0] == target[name].shape[0]
                and tensor.shape[2] == target[name].shape[2]
                and tensor.shape[1] >= target[name].shape[1]
            ):
                selected[name] = tensor[:, : target[name].shape[1]].clone()
                partially_copied += 1
                continue
            skipped_shape += 1
            continue
        selected[name] = tensor
        copied += 1
        if name.startswith("state_adaptor."):
            state_adaptor_copied += 1
    missing, unexpected = target_runner.load_state_dict(selected, strict=False)
    del source, target, selected
    return {
        "copied_tensors": copied,
        "partially_copied_tensors": partially_copied,
        "skipped_shape": skipped_shape,
        "state_adaptor_copied_tensors": state_adaptor_copied,
        "state_adaptor_target_tensors": len(
            target_runner.state_adaptor.state_dict()
        ),
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
        self.horizon = cfg.model.pred_horizon
        self.rdt_state_dim = cfg.model.resolved_rdt_state_dim
        self.use_native_state_encoder = (
            cfg.model.state_encoder_layout == "rdt_eef"
        )
        self.use_native_action_encoder = (
            cfg.model.action_encoder_layout == "rdt_eef"
        )

        projector_width = cfg.model.hidden_size
        if cfg.model.qwen_fusion == "self_attention_kv":
            projector_width *= 2
        self.qwen_adaptor = nn.Linear(
            cfg.model.qwen_kv_dim,
            projector_width,
            dtype=dtype,
        )

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
        qwen_language_tokens = int(cfg.model.qwen_fusion == "language")
        self.runner = RDTRunner(
            action_dim=cfg.model.action_dim,
            pred_horizon=cfg.model.pred_horizon,
            config=runner_config,
            lang_token_dim=cfg.model.lang_token_dim,
            img_token_dim=cfg.model.img_token_dim,
            state_token_dim=self.rdt_state_dim,
            max_lang_cond_len=cfg.model.max_lang_tokens + qwen_language_tokens,
            img_cond_len=cfg.model.image_tokens,
            lang_pos_embed_config=None,
            img_pos_embed_config=None,
            dtype=dtype,
        )
        self.action_adaptor: nn.Module | None = None
        if self.use_native_state_encoder and not self.use_native_action_encoder:
            self.action_adaptor = self.runner.build_condition_adapter(
                cfg.model.state_adaptor,
                in_features=cfg.model.action_dim * 2,
                out_features=cfg.model.hidden_size,
            ).to(dtype=dtype)
        self.pretrained_report: dict[str, int] | None = None

        if load_pretrained and cfg.pretrained_model:
            source = RDTRunner.from_pretrained(cfg.pretrained_model)
            self.pretrained_report = _copy_selected_pretrained_weights(
                self.runner, source, cfg
            )
            del source
            gc.collect()

        if cfg.model.freeze_state_adaptor and self.use_native_state_encoder:
            if load_pretrained and cfg.pretrained_model:
                assert self.pretrained_report is not None
                if (
                    self.pretrained_report["state_adaptor_copied_tensors"]
                    != self.pretrained_report["state_adaptor_target_tensors"]
                ):
                    raise RuntimeError(
                        "Refusing to freeze an incompletely copied RDT state adaptor"
                    )
            elif not cfg.model.allow_random_frozen_state_adaptor:
                raise RuntimeError(
                    "Refusing to freeze a random RDT state adaptor; load pretrained "
                    "weights or explicitly enable allow_random_frozen_state_adaptor"
                )

        install_kv_aware_forward(
            self.runner.model,
            gradient_checkpointing=cfg.model.gradient_checkpointing,
        )
        if cfg.model.finetune_mode == "lora":
            self.runner.model, self.lora_targets = apply_lora(
                self.runner.model, cfg.lora
            )
        else:
            self.runner.model.requires_grad_(True)
            self.lora_targets = []

        # These adaptors are outside ``runner.model`` and can be frozen
        # independently of the RDT transformer.
        self.runner.lang_adaptor.to(dtype=dtype).requires_grad_(
            not cfg.model.freeze_condition_adaptors
        )
        self.runner.img_adaptor.to(dtype=dtype).requires_grad_(
            not cfg.model.freeze_condition_adaptors
        )
        self.runner.state_adaptor.to(dtype=dtype).requires_grad_(
            not cfg.model.freeze_state_adaptor
        )
        if self.action_adaptor is not None:
            self.action_adaptor.requires_grad_(True)
        self.qwen_adaptor.requires_grad_(True)

    @property
    def model_dtype(self) -> torch.dtype:
        return self.compute_dtype

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        sample: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        # Calling forward lets Accelerate apply DDP/FSDP hooks and autocast.
        if sample:
            return self.sample_actions(batch)
        return self.compute_loss(batch)

    def trainable_parameter_report(self) -> dict[str, Any]:
        trainable, total = count_parameters(self)
        return {
            "trainable": trainable,
            "total": total,
            "percentage": 100.0 * trainable / max(total, 1),
            "finetune_mode": self.cfg.model.finetune_mode,
            "qwen_fusion": self.cfg.model.qwen_fusion,
            "rdt_state_dim": self.rdt_state_dim,
            "state_adaptor_frozen": not any(
                parameter.requires_grad
                for parameter in self.runner.state_adaptor.parameters()
            ),
            "condition_adaptors_frozen": not any(
                parameter.requires_grad
                for module in (self.runner.lang_adaptor, self.runner.img_adaptor)
                for parameter in module.parameters()
            ),
            "native_state_encoder": self.use_native_state_encoder,
            "native_action_encoder": self.use_native_action_encoder,
            "action_adaptor_trainable": (
                self.action_adaptor is not None
                and any(
                    parameter.requires_grad
                    for parameter in self.action_adaptor.parameters()
                )
            ),
            "lora_target_count": len(self.lora_targets),
            "pretrained": self.pretrained_report,
        }

    def cast_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        dtype = self.model_dtype
        float_keys = {
            "lang_tokens",
            "img_tokens",
            "qwen_kv",
            "state",
            "actions",
            "action_dim_mask",
            "ctrl_freq",
        }
        return {
            key: value.to(dtype=dtype) if key in float_keys else value
            for key, value in batch.items()
        }

    @staticmethod
    def _euler_xyz_to_ortho6d(euler: torch.Tensor) -> torch.Tensor:
        """Convert XYZ Euler angles to RDT's first-two-columns rotation form."""
        roll, pitch, yaw = euler.unbind(dim=-1)
        cr, sr = roll.cos(), roll.sin()
        cp, sp = pitch.cos(), pitch.sin()
        cy, sy = yaw.cos(), yaw.sin()

        # R = Rz(yaw) @ Ry(pitch) @ Rx(roll), matching the XYZ convention
        # used by the RDT preprocessing utilities. RDT flattens columns 0 and 1.
        first_column = torch.stack(
            [cy * cp, sy * cp, -sp],
            dim=-1,
        )
        second_column = torch.stack(
            [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr],
            dim=-1,
        )
        return torch.cat([first_column, second_column], dim=-1)

    def _state_encoder_input(
        self,
        values: torch.Tensor,
        raw_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build the input consumed by the frozen current-state adaptor."""
        if values.shape != raw_mask.shape:
            raise ValueError(
                f"State values/mask shapes differ: {values.shape} vs {raw_mask.shape}"
            )
        if values.shape[-1] != self.cfg.model.action_dim:
            raise ValueError(
                f"Expected raw state width {self.cfg.model.action_dim}, got "
                f"{values.shape[-1]}"
            )
        if not self.use_native_state_encoder:
            return torch.cat([values * raw_mask, raw_mask], dim=-1)

        unified_values = values.new_zeros(*values.shape[:-1], self.rdt_state_dim)
        unified_masks = raw_mask.new_zeros(*raw_mask.shape[:-1], self.rdt_state_dim)

        # Cache state convention: absolute xyz, three orientation coordinates,
        # and binary gripper_closed. Native RDT state convention uses absolute
        # xyz, ortho6d rotation, and gripper_open.
        unified_values[..., 30:33] = values[..., :3] * raw_mask[..., :3]
        unified_masks[..., 30:33] = raw_mask[..., :3]

        rotation_valid = raw_mask[..., 3:6].amin(dim=-1, keepdim=True)
        rotation = self._euler_xyz_to_ortho6d(
            values[..., 3:6].float()
        ).to(values.dtype)
        unified_values[..., 33:39] = rotation * rotation_valid
        unified_masks[..., 33:39] = rotation_valid.expand_as(rotation)

        gripper_valid = raw_mask[..., 6]
        unified_values[..., 10] = (1.0 - values[..., 6]) * gripper_valid
        unified_masks[..., 10] = gripper_valid
        return torch.cat([unified_values, unified_masks], dim=-1)

    def _action_encoder_input(
        self,
        values: torch.Tensor,
        raw_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build action-token inputs for delta actions or absolute targets."""
        if self.use_native_action_encoder:
            return self._state_encoder_input(values, raw_mask)
        if values.shape != raw_mask.shape:
            raise ValueError(
                f"Action values/mask shapes differ: {values.shape} vs {raw_mask.shape}"
            )
        return torch.cat([values * raw_mask, raw_mask], dim=-1)

    def _adapt_actions(
        self,
        values: torch.Tensor,
        raw_mask: torch.Tensor,
    ) -> torch.Tensor:
        action_input = self._action_encoder_input(values, raw_mask)
        adaptor = (
            self.runner.state_adaptor
            if self.action_adaptor is None or self.use_native_action_encoder
            else self.action_adaptor
        )
        return adaptor(action_input)

    def _adapt_static_conditions(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        lang_cond = self.runner.lang_adaptor(batch["lang_tokens"])
        img_cond = self.runner.img_adaptor(batch["img_tokens"])
        lang_mask = batch["lang_mask"].bool()
        projected_qwen = self.qwen_adaptor(batch["qwen_kv"])
        external_kv: torch.Tensor | None = None
        if self.cfg.model.qwen_fusion == "language":
            lang_cond = torch.cat([projected_qwen, lang_cond], dim=1)
            qwen_mask = torch.ones(
                projected_qwen.shape[:2],
                dtype=torch.bool,
                device=projected_qwen.device,
            )
            lang_mask = torch.cat([qwen_mask, lang_mask], dim=1)
        else:
            external_kv = projected_qwen
        return lang_cond, img_cond, lang_mask, external_kv

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = self.cast_batch(batch)
        states = batch["state"].unsqueeze(1)
        actions = batch["actions"]
        time_mask = batch["action_time_mask"].bool()
        dim_mask = batch["action_dim_mask"].to(actions.dtype).unsqueeze(1)

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

        action_token_mask = (
            dim_mask.expand(-1, actions.shape[1], -1)
            * time_mask.unsqueeze(-1).to(actions.dtype)
        )
        # Padded future positions must not inject random noise into valid tokens.
        noisy_actions = noisy_actions * action_token_mask
        state_input = self._state_encoder_input(states, dim_mask)
        state_cond = self.runner.state_adaptor(state_input)
        action_cond = self._adapt_actions(noisy_actions, action_token_mask)
        state_action_cond = torch.cat([state_cond, action_cond], dim=1)

        lang_cond, img_cond, lang_mask, external_kv = (
            self._adapt_static_conditions(batch)
        )
        prediction = self.runner.model(
            state_action_cond,
            batch["ctrl_freq"],
            timesteps,
            lang_cond,
            img_cond,
            lang_mask=lang_mask,
            img_mask=batch["img_mask"].bool(),
            external_kv=external_kv,
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
        sample_valid_count = valid.sum(dim=(1, 2))
        sample_is_valid = (sample_valid_count > 0).to(prediction.dtype)
        sample_denominator = sample_valid_count.clamp_min(1.0)
        sample_loss = (
            ((prediction - target).pow(2) * valid).sum(dim=(1, 2))
            / sample_denominator
        )
        sample_mae = (
            ((prediction - target).abs() * valid).sum(dim=(1, 2))
            / sample_denominator
        )
        # The objective is the mean per-example denoising loss. This makes
        # microbatch accumulation exactly equivalent to a batch of the same
        # number of examples even when valid horizon lengths differ.
        loss_sum = (sample_loss * sample_is_valid).sum()
        mae_sum = (sample_mae * sample_is_valid).sum()
        valid_count = sample_is_valid.sum()
        return {
            "loss": loss_sum / valid_count.clamp_min(1.0),
            "train_target_mae": (
                mae_sum / valid_count.clamp_min(1.0)
            ).detach(),
            "loss_sum": loss_sum.detach(),
            "mae_sum": mae_sum.detach(),
            "valid_count": valid_count.detach(),
        }

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self.cast_batch(batch)
        states = batch["state"].unsqueeze(1)
        dim_mask = batch["action_dim_mask"].to(states.dtype).unsqueeze(1)
        state_input = self._state_encoder_input(states, dim_mask)
        state_cond = self.runner.state_adaptor(state_input)
        lang_cond, img_cond, lang_mask, external_kv = (
            self._adapt_static_conditions(batch)
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
                self.runner.num_inference_timesteps,
                device=states.device,
            )
        except TypeError:
            scheduler.set_timesteps(self.runner.num_inference_timesteps)

        expanded_dim_mask = dim_mask.expand(-1, noisy.shape[1], -1)
        for timestep in scheduler.timesteps:
            timestep = timestep.to(states.device)
            action_cond = self._adapt_actions(noisy, expanded_dim_mask)
            state_action_cond = torch.cat([state_cond, action_cond], dim=1)
            model_timestep = timestep.reshape(1).expand(states.shape[0])
            output = self.runner.model(
                state_action_cond,
                batch["ctrl_freq"],
                model_timestep,
                lang_cond,
                img_cond,
                lang_mask=lang_mask,
                img_mask=batch["img_mask"].bool(),
                external_kv=external_kv,
            )
            noisy = scheduler.step(output, timestep, noisy).prev_sample
            noisy = noisy.to(states.dtype) * expanded_dim_mask
        return noisy
