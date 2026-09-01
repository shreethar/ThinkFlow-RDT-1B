from __future__ import annotations

import gc
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .checkpoint import load_full_rdt_base
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
    if external_kv.ndim != 3 or external_kv.shape[0] != batch_size:
        raise ValueError(
            "Projected Qwen KV must be [B, tokens, 2 * hidden_size], got "
            f"{tuple(external_kv.shape)}"
        )
    external_token_count = int(external_kv.shape[1])
    if external_token_count <= 0:
        raise ValueError("Projected Qwen KV must contain at least one token")
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
        batch_size,
        external_token_count,
        attention.num_heads,
        attention.head_dim,
    ).permute(0, 2, 1, 3)
    external_value = external_value.reshape(
        batch_size,
        external_token_count,
        attention.num_heads,
        attention.head_dim,
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


def _cross_attention_with_external_kv(
    attention: nn.Module,
    x: torch.Tensor,
    condition: torch.Tensor,
    condition_mask: torch.Tensor | None,
    external_kv: torch.Tensor | None,
) -> torch.Tensor:
    """Run RDT cross-attention with already projected K/V appended directly."""
    if external_kv is None:
        return attention(x, condition, condition_mask)

    batch_size, query_count, width = x.shape
    condition_batch, condition_count, condition_width = condition.shape
    if condition_batch != batch_size or condition_width != width:
        raise ValueError(
            "Cross-attention condition must be [B, L, hidden_size], got "
            f"{tuple(condition.shape)} for queries {tuple(x.shape)}"
        )
    if external_kv.ndim != 3 or external_kv.shape[0] != batch_size:
        raise ValueError(
            "Projected external cross-attention KV must be "
            f"[B, tokens, 2 * hidden_size], got {tuple(external_kv.shape)}"
        )
    if external_kv.shape[-1] != 2 * width:
        raise ValueError(
            "Projected external cross-attention KV width must be "
            f"{2 * width}, got {external_kv.shape[-1]}"
        )

    query = attention.q(x).reshape(
        batch_size, query_count, attention.num_heads, attention.head_dim
    ).permute(0, 2, 1, 3)
    native_kv = attention.kv(condition).reshape(
        batch_size,
        condition_count,
        2,
        attention.num_heads,
        attention.head_dim,
    ).permute(2, 0, 3, 1, 4)
    key, value = native_kv.unbind(0)

    external_key, external_value = external_kv.chunk(2, dim=-1)
    external_token_count = external_key.shape[1]
    external_key = external_key.reshape(
        batch_size,
        external_token_count,
        attention.num_heads,
        attention.head_dim,
    ).permute(0, 2, 1, 3)
    external_value = external_value.reshape(
        batch_size,
        external_token_count,
        attention.num_heads,
        attention.head_dim,
    ).permute(0, 2, 1, 3)

    query = attention.q_norm(query)
    key = attention.k_norm(key)
    external_key = attention.k_norm(external_key)
    key = torch.cat([key, external_key], dim=2)
    value = torch.cat([value, external_value], dim=2)

    combined_mask = None
    if condition_mask is not None:
        condition_mask = condition_mask.to(device=x.device, dtype=torch.bool)
        if condition_mask.shape != (batch_size, condition_count):
            raise ValueError(
                "Cross-attention condition mask must be [B, L], got "
                f"{tuple(condition_mask.shape)}"
            )
        external_mask = torch.ones(
            batch_size,
            external_token_count,
            dtype=torch.bool,
            device=x.device,
        )
        combined_mask = torch.cat(
            [condition_mask, external_mask], dim=1
        ).reshape(batch_size, 1, 1, condition_count + external_token_count)
        combined_mask = combined_mask.expand(-1, -1, query_count, -1)

    if attention.fused_attn:
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=attention.attn_drop.p if attention.training else 0.0,
            attn_mask=combined_mask,
        )
    else:
        scores = (query * attention.scale) @ key.transpose(-2, -1)
        if combined_mask is not None:
            scores = scores.masked_fill(combined_mask.logical_not(), float("-inf"))
        probabilities = scores.softmax(dim=-1)
        if attention.attn_drop.p > 0:
            probabilities = attention.attn_drop(probabilities)
        output = probabilities @ value

    output = output.permute(0, 2, 1, 3).reshape(
        batch_size, query_count, width
    )
    output = attention.proj(output)
    if attention.proj_drop.p > 0:
        output = attention.proj_drop(output)
    return output


def _rdt_block_with_external_kv(
    block: nn.Module,
    x: torch.Tensor,
    condition: torch.Tensor,
    condition_mask: torch.Tensor | None,
    external_kv: torch.Tensor | None,
    external_cross_kv: torch.Tensor | None,
) -> torch.Tensor:
    residual = x
    x = _self_attention_with_external_kv(
        block.attn,
        block.norm1(x),
        external_kv,
    )
    x = x + residual

    residual = x
    x = _cross_attention_with_external_kv(
        block.cross_attn,
        block.norm2(x),
        condition,
        condition_mask,
        external_cross_kv,
    )
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
        external_cross_kv=None,
        extra_cross_cond=None,
        extra_cross_mask=None,
        unified_cross_attention=False,
    ):
        t_embed = self.t_embedder(t).unsqueeze(1)
        freq_embed = self.freq_embedder(freq).unsqueeze(1)
        if t_embed.shape[0] == 1:
            t_embed = t_embed.expand(x.shape[0], -1, -1)
        x = torch.cat([t_embed, freq_embed, x], dim=1)
        x = x + self.x_pos_embed
        lang_c = lang_c + self.lang_cond_pos_embed[:, : lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed[:, : img_c.shape[1]]
        if unified_cross_attention:
            condition_parts = [lang_c, img_c]
            if lang_mask is None:
                lang_mask = torch.ones(
                    lang_c.shape[:2], dtype=torch.bool, device=lang_c.device
                )
            if img_mask is None:
                img_mask = torch.ones(
                    img_c.shape[:2], dtype=torch.bool, device=img_c.device
                )
            mask_parts = [lang_mask, img_mask]
            if extra_cross_cond is not None:
                condition_parts.append(extra_cross_cond)
                if extra_cross_mask is None:
                    extra_cross_mask = torch.ones(
                        extra_cross_cond.shape[:2],
                        dtype=torch.bool,
                        device=extra_cross_cond.device,
                    )
                mask_parts.append(extra_cross_mask)
            unified_condition = torch.cat(condition_parts, dim=1)
            unified_mask = torch.cat(mask_parts, dim=1)
            conditions = (unified_condition, unified_condition)
            masks = (unified_mask, unified_mask)
        else:
            if extra_cross_cond is None:
                conditions = (lang_c, img_c)
                masks = (lang_mask, img_mask)
            else:
                if extra_cross_mask is None:
                    extra_cross_mask = torch.ones(
                        extra_cross_cond.shape[:2],
                        dtype=torch.bool,
                        device=extra_cross_cond.device,
                    )
                if lang_mask is None:
                    lang_mask = torch.ones(
                        lang_c.shape[:2],
                        dtype=torch.bool,
                        device=lang_c.device,
                    )
                if img_mask is None:
                    img_mask = torch.ones(
                        img_c.shape[:2],
                        dtype=torch.bool,
                        device=img_c.device,
                    )
                # Preserve RDT's alternating language/image blocks, but append
                # the same native proprioceptive state token to both contexts.
                # Qwen K/V is appended separately in
                # _cross_attention_with_external_kv after these native tokens
                # have passed through each block's condition projection.
                conditions = (
                    torch.cat([lang_c, extra_cross_cond], dim=1),
                    torch.cat([img_c, extra_cross_cond], dim=1),
                )
                masks = (
                    torch.cat([lang_mask, extra_cross_mask], dim=1),
                    torch.cat([img_mask, extra_cross_mask], dim=1),
                )
        for index, block in enumerate(self.blocks):
            condition = conditions[index % 2]
            mask = masks[index % 2]
            if (
                gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                def custom_forward(
                    x_value,
                    c_value,
                    self_kv_value,
                    cross_kv_value,
                    block=block,
                    mask=mask,
                ):
                    return _rdt_block_with_external_kv(
                        block,
                        x_value,
                        c_value,
                        mask,
                        self_kv_value,
                        cross_kv_value,
                    )

                x = checkpoint(
                    custom_forward,
                    x,
                    condition,
                    external_kv,
                    external_cross_kv,
                    use_reentrant=False,
                )
            else:
                x = _rdt_block_with_external_kv(
                    block,
                    x,
                    condition,
                    mask,
                    external_kv,
                    external_cross_kv,
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
    def __init__(
        self,
        cfg: ExperimentConfig,
        load_pretrained: bool = True,
        *,
        base_artifact: str | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        RDTRunner = import_rdt_runner(cfg.rdt_repo)

        dtype = resolve_dtype(cfg.model.dtype)
        self.compute_dtype = dtype
        self.horizon = cfg.model.pred_horizon
        self.rdt_state_dim = cfg.model.resolved_rdt_state_dim
        # Discrete gripper commands can otherwise leak their clean sign through
        # low-noise diffusion inputs.  Experiments may mask that input channel
        # without changing the model architecture or its 10D output contract.
        self.mask_noisy_gripper_input = False
        # Inference-only option: retain the scheduler result for continuous
        # motion, but take the gripper scalar from the final clean-x0 estimate.
        # This is separate from input masking so older checkpoints can opt into
        # clean-x0 decoding without changing their denoising input distribution.
        self.decode_clean_x0_gripper = False
        self.use_native_state_encoder = (
            cfg.model.state_encoder_layout
            in {"rdt_eef", "libero_ortho6d", "rdt_native_128"}
        )
        self.use_native_action_encoder = (
            cfg.model.action_encoder_layout
            in {"rdt_eef", "libero_ortho6d", "rdt_native_128"}
        )

        projector_width = cfg.model.hidden_size
        if cfg.model.qwen_fusion in {
            "self_attention_kv",
            "fastthinkact_state_kv",
            "cross_attention_kv",
            "fastthinkact_cross_attention_kv",
        }:
            projector_width *= 2
        self.qwen_adaptor = nn.Sequential(
            nn.Linear(cfg.model.qwen_kv_dim, projector_width, dtype=dtype),
            nn.GELU(),
            nn.Linear(projector_width, projector_width, dtype=dtype),
        )
        self.plan_hidden_norm: nn.Module | None = None
        self.waypoint_adaptor: nn.Module | None = None
        self.plan_adaptor: nn.Module | None = None
        self.plan_type_embedding: nn.Parameter | None = None
        self.plan_position_embedding: nn.Parameter | None = None
        if cfg.model.qwen_fusion in {
            "hidden_cross_attention",
            "hidden_waypoint_cross_attention",
        }:
            self.plan_hidden_norm = nn.LayerNorm(
                cfg.model.qwen_hidden_size,
                dtype=dtype,
            )
            plan_input_dim = cfg.model.qwen_hidden_size
            if cfg.model.qwen_fusion == "hidden_waypoint_cross_attention":
                self.waypoint_adaptor = nn.Sequential(
                    nn.Linear(
                        cfg.model.waypoint_dim,
                        cfg.model.waypoint_embed_dim,
                        dtype=dtype,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        cfg.model.waypoint_embed_dim,
                        cfg.model.waypoint_embed_dim,
                        dtype=dtype,
                    ),
                )
                plan_input_dim += cfg.model.waypoint_embed_dim
            self.plan_adaptor = nn.Sequential(
                nn.Linear(
                    plan_input_dim,
                    cfg.model.hidden_size,
                    dtype=dtype,
                ),
                nn.GELU(),
                nn.Linear(
                    cfg.model.hidden_size,
                    cfg.model.hidden_size,
                    dtype=dtype,
                ),
            )
            self.plan_type_embedding = nn.Parameter(
                torch.zeros(1, 1, cfg.model.hidden_size, dtype=dtype)
            )
            # These are trained by RDT, not extracted from Qwen. They preserve
            # which of the ordered five trajectory/spatial slots a token came
            # from while the hidden state carries the sample-specific content.
            self.plan_position_embedding = nn.Parameter(
                torch.zeros(
                    1,
                    cfg.model.spatial_token_count,
                    cfg.model.hidden_size,
                    dtype=dtype,
                )
            )
        self.unified_cross_extra_pos_embed = nn.Parameter(
            torch.zeros(1, 2, cfg.model.hidden_size, dtype=dtype)
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
        self.base_artifact_report: dict[str, Any] | None = None
        upstream_libero_base = bool(
            base_artifact
            and (
                (Path(base_artifact) / "ema" / "model.safetensors").is_file()
                or (Path(base_artifact) / "model.safetensors").is_file()
            )
        )

        # The upstream Libero_RDT EMA checkpoint is a complete runner. Avoid
        # loading the generic RDT-1B runner immediately before overwriting all
        # of it, which otherwise doubles startup I/O and peak host memory.
        if load_pretrained and cfg.pretrained_model and not upstream_libero_base:
            source = RDTRunner.from_pretrained(cfg.pretrained_model)
            self.pretrained_report = _copy_selected_pretrained_weights(
                self.runner, source, cfg
            )
            del source
            gc.collect()

        if cfg.model.freeze_state_adaptor and self.use_native_state_encoder:
            if upstream_libero_base:
                pass
            elif load_pretrained and cfg.pretrained_model:
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
        if base_artifact is not None:
            self.base_artifact_report = load_full_rdt_base(
                self,
                base_artifact,
                allow_output_head_mismatch=True,
                allow_language_position_mismatch=True,
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
            not cfg.model.resolved_freeze_language_adaptor
        )
        self.runner.img_adaptor.to(dtype=dtype).requires_grad_(
            not cfg.model.resolved_freeze_image_adaptor
        )
        self.runner.state_adaptor.to(dtype=dtype).requires_grad_(
            not cfg.model.freeze_state_adaptor
        )
        if self.action_adaptor is not None:
            self.action_adaptor.requires_grad_(True)
        self.qwen_adaptor.requires_grad_(
            cfg.model.qwen_fusion
            not in {"hidden_cross_attention", "hidden_waypoint_cross_attention"}
        )
        for module in (
            self.plan_hidden_norm,
            self.waypoint_adaptor,
            self.plan_adaptor,
        ):
            if module is not None:
                module.requires_grad_(True)
        if self.plan_type_embedding is not None:
            self.plan_type_embedding.requires_grad_(True)
        if self.plan_position_embedding is not None:
            self.plan_position_embedding.requires_grad_(True)
        self.unified_cross_extra_pos_embed.requires_grad_(
            cfg.model.qwen_fusion
            in {
                "unified_cross_attention",
                "fastthinkact_cross_attention_kv",
                "hidden_cross_attention",
                "hidden_waypoint_cross_attention",
            }
        )

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
            "plan_conditioning_trainable": (
                any(
                    parameter.requires_grad
                    for module in (
                        self.plan_hidden_norm,
                        self.waypoint_adaptor,
                        self.plan_adaptor,
                    )
                    if module is not None
                    for parameter in module.parameters()
                )
                or bool(
                    self.plan_type_embedding is not None
                    and self.plan_type_embedding.requires_grad
                )
                or bool(
                    self.plan_position_embedding is not None
                    and self.plan_position_embedding.requires_grad
                )
            ),
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
            "language_adaptor_frozen": not any(
                parameter.requires_grad
                for parameter in self.runner.lang_adaptor.parameters()
            ),
            "image_adaptor_frozen": not any(
                parameter.requires_grad
                for parameter in self.runner.img_adaptor.parameters()
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
            "base_artifact": self.base_artifact_report,
        }

    def cast_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        dtype = self.model_dtype
        float_keys = {
            "lang_tokens",
            "img_tokens",
            "qwen_kv",
            "qwen_hidden_states",
            "latent_waypoints",
            "state",
            "state_dim_mask",
            "actions",
            "action_dim_mask",
            "horizon_loss_weights",
            "gripper_loss_weights",
            "gripper_bce_weight",
            "gripper_bce_logit_scale",
            "xyz_loss_weight",
            "rotation_geodesic_weight",
            "qwen_fusion_loss_weight",
            "qwen_fusion_loss_margin",
            "diffusion_noise",
            "ctrl_freq",
        }
        return {
            key: value.to(dtype=dtype) if key in float_keys else value
            for key, value in batch.items()
        }

    def _adapt_hidden_plan(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.plan_hidden_norm is None
            or self.plan_adaptor is None
            or self.plan_type_embedding is None
            or self.plan_position_embedding is None
        ):
            raise RuntimeError(
                "Hidden plan modules were not constructed for this config"
            )
        hidden = batch.get("qwen_hidden_states")
        if hidden is None:
            raise KeyError("Hidden fusion requires qwen_hidden_states")
        expected_hidden = (
            hidden.shape[0],
            self.cfg.model.spatial_token_count,
            self.cfg.model.qwen_hidden_size,
        )
        if tuple(hidden.shape) != expected_hidden:
            raise ValueError(
                f"Expected hidden plan shape {expected_hidden}, got "
                f"{tuple(hidden.shape)}"
            )
        if not bool(torch.isfinite(hidden.float()).all()):
            raise ValueError("qwen_hidden_states contains NaN or Inf")

        hidden_features = self.plan_hidden_norm(hidden)
        plan_input = hidden_features
        if self.cfg.model.qwen_fusion == "hidden_waypoint_cross_attention":
            waypoints = batch.get("latent_waypoints")
            if waypoints is None or self.waypoint_adaptor is None:
                raise KeyError(
                    "hidden_waypoint_cross_attention requires latent_waypoints"
                )
            expected_waypoints = (
                hidden.shape[0],
                self.cfg.model.spatial_token_count,
                self.cfg.model.waypoint_dim,
            )
            if tuple(waypoints.shape) != expected_waypoints:
                raise ValueError(
                    f"Expected waypoint shape {expected_waypoints}, got "
                    f"{tuple(waypoints.shape)}"
                )
            if not bool(torch.isfinite(waypoints.float()).all()):
                raise ValueError("latent_waypoints contains NaN or Inf")
            # The student's waypoint head predicts normalized image coordinates.
            waypoint_features = self.waypoint_adaptor(waypoints * 2.0 - 1.0)
            plan_input = torch.cat([hidden_features, waypoint_features], dim=-1)
        plan_tokens = self.plan_adaptor(plan_input)
        plan_tokens = (
            plan_tokens
            + self.plan_type_embedding
            + self.plan_position_embedding
        )
        plan_mask = batch.get("plan_mask")
        if plan_mask is None:
            plan_mask = torch.ones(
                plan_tokens.shape[:2],
                dtype=torch.bool,
                device=plan_tokens.device,
            )
        else:
            plan_mask = plan_mask.bool()
            if tuple(plan_mask.shape) != tuple(plan_tokens.shape[:2]):
                raise ValueError(
                    "plan_mask must match plan token axes, got "
                    f"{tuple(plan_mask.shape)} vs {tuple(plan_tokens.shape[:2])}"
                )
        return plan_tokens, plan_mask

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

    @staticmethod
    def _ortho6d_to_rotation_matrix(ortho6d: torch.Tensor) -> torch.Tensor:
        """Differentiably project first-two-column 6D values onto SO(3)."""
        if ortho6d.shape[-1] != 6:
            raise ValueError(f"Expected ortho6D [...,6], got {tuple(ortho6d.shape)}")
        values = ortho6d.float()
        first_raw = values[..., :3]
        default_first = torch.zeros_like(first_raw)
        default_first[..., 0] = 1.0
        first = torch.where(
            first_raw.norm(dim=-1, keepdim=True) > 1e-6,
            first_raw,
            default_first,
        )
        first = F.normalize(first, dim=-1, eps=1e-6)

        second_raw = values[..., 3:6]
        second = second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first
        fallback_index = first.abs().argmin(dim=-1)
        fallback = F.one_hot(fallback_index, num_classes=3).to(first.dtype)
        fallback = fallback - (first * fallback).sum(dim=-1, keepdim=True) * first
        fallback = F.normalize(fallback, dim=-1, eps=1e-6)
        second = torch.where(
            second.norm(dim=-1, keepdim=True) > 1e-6,
            second,
            fallback,
        )
        second = F.normalize(second, dim=-1, eps=1e-6)
        third = torch.cross(first, second, dim=-1)
        return torch.stack([first, second, third], dim=-1)

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
        if values.shape[-1] != self.cfg.model.state_dim:
            raise ValueError(
                f"Expected raw state width {self.cfg.model.state_dim}, got "
                f"{values.shape[-1]}"
            )
        if not self.use_native_state_encoder:
            return torch.cat([values * raw_mask, raw_mask], dim=-1)

        if self.cfg.model.state_encoder_layout == "rdt_native_128":
            return torch.cat([values * raw_mask, raw_mask], dim=-1)

        unified_values = values.new_zeros(*values.shape[:-1], self.rdt_state_dim)
        unified_masks = raw_mask.new_zeros(*raw_mask.shape[:-1], self.rdt_state_dim)

        if self.cfg.model.state_encoder_layout == "libero_ortho6d":
            unified_values[..., 30:33] = values[..., :3] * raw_mask[..., :3]
            unified_masks[..., 30:33] = raw_mask[..., :3]
            unified_values[..., 33:39] = values[..., 3:9] * raw_mask[..., 3:9]
            unified_masks[..., 33:39] = raw_mask[..., 3:9]
            unified_values[..., 10:12] = values[..., 9:11] * raw_mask[..., 9:11]
            unified_masks[..., 10:12] = raw_mask[..., 9:11]
            return torch.cat([unified_values, unified_masks], dim=-1)

        # Legacy rdt_eef caches arrive here after the optional collator flip
        # from gripper_closed to the pretrained binary gripper_open convention.
        unified_values[..., 30:33] = values[..., :3] * raw_mask[..., :3]
        unified_masks[..., 30:33] = raw_mask[..., :3]

        rotation_valid = raw_mask[..., 3:6].amin(dim=-1, keepdim=True)
        rotation = self._euler_xyz_to_ortho6d(
            values[..., 3:6].float()
        ).to(values.dtype)
        unified_values[..., 33:39] = rotation * rotation_valid
        unified_masks[..., 33:39] = rotation_valid.expand_as(rotation)

        gripper_valid = raw_mask[..., 6]
        unified_values[..., 10] = values[..., 6] * gripper_valid
        unified_masks[..., 10] = gripper_valid
        return torch.cat([unified_values, unified_masks], dim=-1)

    def _action_encoder_input(
        self,
        values: torch.Tensor,
        raw_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build action-token inputs for delta actions or absolute targets."""
        if self.cfg.model.action_encoder_layout == "rdt_eef":
            return self._state_encoder_input(values, raw_mask)
        if self.cfg.model.action_encoder_layout == "rdt_native_128":
            return self._state_encoder_input(values, raw_mask)
        if values.shape != raw_mask.shape:
            raise ValueError(
                f"Action values/mask shapes differ: {values.shape} vs {raw_mask.shape}"
            )
        if values.shape[-1] != self.cfg.model.action_dim:
            raise ValueError(
                f"Expected raw action width {self.cfg.model.action_dim}, got "
                f"{values.shape[-1]}"
            )
        if self.cfg.model.action_encoder_layout != "libero_ortho6d":
            return torch.cat([values * raw_mask, raw_mask], dim=-1)

        unified_values = values.new_zeros(*values.shape[:-1], self.rdt_state_dim)
        unified_masks = raw_mask.new_zeros(*raw_mask.shape[:-1], self.rdt_state_dim)
        unified_values[..., 30:33] = values[..., :3] * raw_mask[..., :3]
        unified_masks[..., 30:33] = raw_mask[..., :3]
        unified_values[..., 33:39] = values[..., 3:9] * raw_mask[..., 3:9]
        unified_masks[..., 33:39] = raw_mask[..., 3:9]
        unified_values[..., 10] = values[..., 9] * raw_mask[..., 9]
        unified_masks[..., 10] = raw_mask[..., 9]
        return torch.cat([unified_values, unified_masks], dim=-1)

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
        state_cond: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        lang_cond = self.runner.lang_adaptor(batch["lang_tokens"])
        img_cond = self.runner.img_adaptor(batch["img_tokens"])
        lang_mask = batch["lang_mask"].bool()
        external_kv: torch.Tensor | None = None
        external_cross_kv: torch.Tensor | None = None
        extra_cross_cond: torch.Tensor | None = None
        extra_cross_mask: torch.Tensor | None = None
        if self.cfg.model.qwen_fusion == "none":
            pass
        elif self.cfg.model.qwen_fusion == "language":
            projected_qwen = self.qwen_adaptor(batch["qwen_kv"])
            lang_cond = torch.cat([projected_qwen, lang_cond], dim=1)
            qwen_mask = torch.ones(
                projected_qwen.shape[:2],
                dtype=torch.bool,
                device=projected_qwen.device,
            )
            lang_mask = torch.cat([qwen_mask, lang_mask], dim=1)
        elif self.cfg.model.qwen_fusion in {
            "self_attention_kv",
            "fastthinkact_state_kv",
        }:
            projected_qwen = self.qwen_adaptor(batch["qwen_kv"])
            external_kv = projected_qwen
        elif self.cfg.model.qwen_fusion == "cross_attention_kv":
            # This output is already an RDT-native [K; V] pair. Cross-attention
            # appends it after projecting its ordinary language/image context,
            # so the cached Qwen KV is not passed through attention.kv again.
            external_cross_kv = self.qwen_adaptor(batch["qwen_kv"])
        elif self.cfg.model.qwen_fusion == "fastthinkact_cross_attention_kv":
            if state_cond is None:
                raise ValueError(
                    "fastthinkact_cross_attention_kv requires state_cond"
                )
            # The native state token remains an ordinary condition and is
            # projected by each RDT cross-attention block's `kv` layer. Qwen's
            # cached K/V is adapted directly to [K; V] and appended afterwards,
            # bypassing that layer. This forms [native modality KV, state KV,
            # Qwen KV] for every action-query cross-attention operation.
            external_cross_kv = self.qwen_adaptor(batch["qwen_kv"])
            extra_cross_cond = (
                state_cond + self.unified_cross_extra_pos_embed[:, 0:1]
            )
            extra_cross_mask = torch.ones(
                state_cond.shape[:2],
                dtype=torch.bool,
                device=state_cond.device,
            )
        elif self.cfg.model.qwen_fusion == "unified_cross_attention":
            projected_qwen = self.qwen_adaptor(batch["qwen_kv"])
            extra_parts = []
            extra_masks = []
            extra_pos = self.unified_cross_extra_pos_embed
            if state_cond is not None:
                extra_parts.append(state_cond + extra_pos[:, 0:1])
                extra_masks.append(
                    torch.ones(
                        state_cond.shape[:2],
                        dtype=torch.bool,
                        device=state_cond.device,
                    )
                )
            extra_parts.append(projected_qwen + extra_pos[:, 1:2])
            extra_masks.append(
                torch.ones(
                    projected_qwen.shape[:2],
                    dtype=torch.bool,
                    device=projected_qwen.device,
                )
            )
            extra_cross_cond = torch.cat(extra_parts, dim=1)
            extra_cross_mask = torch.cat(extra_masks, dim=1)
        elif self.cfg.model.qwen_fusion in {
            "hidden_cross_attention",
            "hidden_waypoint_cross_attention",
        }:
            if state_cond is None:
                raise ValueError(
                    "hidden cross-attention fusion requires state_cond"
                )
            plan_tokens, plan_mask = self._adapt_hidden_plan(batch)
            state_token = state_cond + self.unified_cross_extra_pos_embed[:, 0:1]
            state_mask = torch.ones(
                state_cond.shape[:2],
                dtype=torch.bool,
                device=state_cond.device,
            )
            extra_cross_cond = torch.cat([state_token, plan_tokens], dim=1)
            extra_cross_mask = torch.cat([state_mask, plan_mask], dim=1)
        else:
            raise ValueError(f"Unsupported qwen_fusion: {self.cfg.model.qwen_fusion}")
        return (
            lang_cond,
            img_cond,
            lang_mask,
            external_kv,
            external_cross_kv,
            extra_cross_cond,
            extra_cross_mask,
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = self.cast_batch(batch)
        states = batch["state"].unsqueeze(1)
        actions = batch["actions"]
        time_mask = batch["action_time_mask"].bool()
        action_dim_mask = batch["action_dim_mask"].to(actions.dtype).unsqueeze(1)
        state_dim_mask = batch.get(
            "state_dim_mask",
            torch.ones_like(batch["state"]),
        ).to(states.dtype).unsqueeze(1)

        batch_size = actions.shape[0]
        noise = batch.get("diffusion_noise")
        if noise is None:
            noise = torch.randn_like(actions)
        elif noise.shape != actions.shape:
            raise ValueError(
                "diffusion_noise must match actions, got "
                f"{tuple(noise.shape)} vs {tuple(actions.shape)}"
            )
        timesteps = batch.get("diffusion_timesteps")
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.runner.num_train_timesteps,
                (batch_size,),
                device=actions.device,
                dtype=torch.long,
            )
        else:
            timesteps = timesteps.to(device=actions.device, dtype=torch.long)
            if timesteps.shape != (batch_size,):
                raise ValueError(
                    "diffusion_timesteps must have shape [B], got "
                    f"{tuple(timesteps.shape)}"
                )
            if bool((timesteps < 0).any()) or bool(
                (timesteps >= self.runner.num_train_timesteps).any()
            ):
                raise ValueError("diffusion_timesteps are outside the scheduler range")
        noisy_actions = self.runner.noise_scheduler.add_noise(
            actions, noise, timesteps
        )

        if self.cfg.model.action_encoder_layout == "rdt_native_128":
            gripper_input_index = 10
        elif self.cfg.model.action_encoder_layout == "libero_ortho6d":
            gripper_input_index = 9
        else:
            gripper_input_index = 6

        # The action adaptor must see the same 64 valid temporal positions in
        # training that it sees during sampling.  Passing `time_mask` here
        # exposes the padded episode suffix and lets the model infer release
        # timing from the distance to the end of a demonstration.  Keep the
        # temporal mask exclusively for supervised losses below.
        action_conditioning_mask = action_dim_mask.expand(
            -1,
            actions.shape[1],
            -1,
        )
        noisy_actions = noisy_actions * action_conditioning_mask
        model_noisy_actions = noisy_actions
        if self.mask_noisy_gripper_input:
            model_noisy_actions = noisy_actions.clone()
            model_noisy_actions[..., gripper_input_index] = 0.0
        state_input = self._state_encoder_input(states, state_dim_mask)
        state_cond = self.runner.state_adaptor(state_input)
        action_cond = self._adapt_actions(
            model_noisy_actions,
            action_conditioning_mask,
        )
        state_action_cond = torch.cat([state_cond, action_cond], dim=1)

        (
            lang_cond,
            img_cond,
            lang_mask,
            external_kv,
            external_cross_kv,
            extra_cross_cond,
            extra_cross_mask,
        ) = (
            self._adapt_static_conditions(batch, state_cond=state_cond)
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
            external_cross_kv=external_cross_kv,
            extra_cross_cond=extra_cross_cond,
            extra_cross_mask=extra_cross_mask,
            unified_cross_attention=(
                self.cfg.model.qwen_fusion == "unified_cross_attention"
            ),
        )
        if self.runner.prediction_type != "sample":
            raise ValueError(
                "This trainer optimizes imitation loss against clean action "
                "targets, so noise_scheduler.prediction_type must be 'sample'. "
                f"Got {self.runner.prediction_type!r}."
            )
        target = actions
        if self.cfg.model.action_encoder_layout == "rdt_native_128":
            if (
                self.cfg.model.native_rdt_128_mapping
                == "libero_joint_eef_delta"
            ):
                xyz_start, xyz_stop = 39, 42
                rotation_start, rotation_stop = 42, 45
            else:
                xyz_start, xyz_stop = 30, 33
                rotation_start, rotation_stop = 33, 39
            gripper_start, gripper_stop = 10, 11
        elif self.cfg.model.action_encoder_layout == "libero_ortho6d":
            xyz_start, xyz_stop = 0, 3
            rotation_start = 3
            rotation_stop = 9
            gripper_start, gripper_stop = 9, 10
        else:
            xyz_start, xyz_stop = 0, 3
            rotation_start = 3
            rotation_stop = 6
            gripper_start, gripper_stop = 6, 7

        valid = time_mask.unsqueeze(-1).to(prediction.dtype) * action_dim_mask
        sample_valid_count = valid.sum(dim=(1, 2))
        sample_is_valid = (sample_valid_count > 0).to(prediction.dtype)
        sample_denominator = sample_valid_count.clamp_min(1.0)
        diff = prediction - target
        if self.cfg.model.action_encoder_layout == "rdt_eef":
            # Legacy absolute-state targets store Euler angles.
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
        sample_unweighted_imitation_loss = (
            (diff.pow(2) * valid).sum(dim=(1, 2))
            / sample_denominator
        )
        horizon_loss_weights = batch.get("horizon_loss_weights")
        if horizon_loss_weights is None:
            objective_valid = valid
        else:
            weights = horizon_loss_weights.to(
                device=prediction.device,
                dtype=prediction.dtype,
            )
            if weights.ndim == 1:
                weights = weights.unsqueeze(0)
            if weights.ndim != 2 or weights.shape[-1] != actions.shape[1]:
                raise ValueError(
                    "horizon_loss_weights must be [H] or [B,H], got "
                    f"{tuple(weights.shape)} for horizon {actions.shape[1]}"
                )
            if weights.shape[0] not in {1, actions.shape[0]}:
                raise ValueError(
                    "horizon_loss_weights batch width must be 1 or match the "
                    f"action batch, got {weights.shape[0]} vs {actions.shape[0]}"
                )
            if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
                raise ValueError("horizon_loss_weights must be finite and non-negative")
            objective_valid = valid * weights.unsqueeze(-1)
        # Horizon weighting keeps its historical weighted-mean normalization.
        # Additional phase weights below intentionally do not enter this
        # denominator; otherwise a release-only sample multiplied by 5 would
        # also be divided by 5 and receive no extra gradient at all.
        objective_count = objective_valid.sum(dim=(1, 2))
        base_gripper_bce_valid = objective_valid[..., gripper_start].clone()
        gripper_loss_weights = batch.get("gripper_loss_weights")
        if gripper_loss_weights is not None:
            gripper_weights = gripper_loss_weights.to(
                device=prediction.device,
                dtype=prediction.dtype,
            )
            if gripper_weights.ndim == 1:
                gripper_weights = gripper_weights.unsqueeze(0)
            if gripper_weights.ndim != 2 or gripper_weights.shape[-1] != actions.shape[1]:
                raise ValueError(
                    "gripper_loss_weights must be [H] or [B,H], got "
                    f"{tuple(gripper_weights.shape)} for horizon {actions.shape[1]}"
                )
            if gripper_weights.shape[0] not in {1, actions.shape[0]}:
                raise ValueError(
                    "gripper_loss_weights batch width must be 1 or match the "
                    f"action batch, got {gripper_weights.shape[0]} vs {actions.shape[0]}"
                )
            if not bool(torch.isfinite(gripper_weights).all()) or bool(
                (gripper_weights < 0).any()
            ):
                raise ValueError("gripper_loss_weights must be finite and non-negative")
            objective_valid = objective_valid.clone()
            objective_valid[..., gripper_start:gripper_stop] *= (
                gripper_weights.unsqueeze(-1)
            )
        sample_imitation_loss = (
            (diff.pow(2) * objective_valid).sum(dim=(1, 2))
            / objective_count.clamp_min(1.0)
        )
        sample_mae = (
            (diff.abs() * valid).sum(dim=(1, 2))
            / sample_denominator
        )
        squared_error = diff.pow(2) * valid

        def component_losses(
            start: int,
            stop: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            component_valid = valid[..., start:stop]
            component_count = component_valid.sum(dim=(1, 2))
            component_is_valid = (component_count > 0).to(prediction.dtype)
            component_loss = (
                squared_error[..., start:stop].sum(dim=(1, 2))
                / component_count.clamp_min(1.0)
            )
            return component_loss, component_is_valid

        sample_xyz_loss, sample_xyz_valid = component_losses(
            xyz_start, xyz_stop
        )
        sample_rot_loss, sample_rot_valid = component_losses(
            rotation_start, rotation_stop
        )
        sample_gripper_loss, sample_gripper_valid = component_losses(
            gripper_start,
            gripper_stop,
        )
        xyz_loss_sum = (sample_xyz_loss * sample_xyz_valid).sum()
        xyz_valid_count = sample_xyz_valid.sum()
        rot_loss_sum = (sample_rot_loss * sample_rot_valid).sum()
        rot_valid_count = sample_rot_valid.sum()
        gripper_loss_sum = (
            sample_gripper_loss * sample_gripper_valid
        ).sum()
        gripper_valid_count = sample_gripper_valid.sum()

        # Unlike the unweighted component diagnostic above, this auxiliary XYZ
        # objective follows horizon_loss_weights. This lets deployment-relevant
        # near-term translation commands receive the same temporal emphasis as
        # the main imitation objective.
        xyz_objective_valid = objective_valid[..., xyz_start:xyz_stop]
        sample_xyz_objective_count = xyz_objective_valid.sum(dim=(1, 2))
        sample_xyz_auxiliary_loss = (
            (
                diff[..., xyz_start:xyz_stop].pow(2)
                * xyz_objective_valid
            ).sum(dim=(1, 2))
            / sample_xyz_objective_count.clamp_min(1.0)
        )
        xyz_auxiliary_valid = (
            sample_xyz_objective_count > 0
        ).to(prediction.dtype)
        xyz_auxiliary_loss = (
            sample_xyz_auxiliary_loss * xyz_auxiliary_valid
        ).sum() / xyz_auxiliary_valid.sum().clamp_min(1.0)
        xyz_loss_weight = batch.get("xyz_loss_weight")
        if xyz_loss_weight is None:
            xyz_loss_weight = prediction.new_tensor(0.0)
        if xyz_loss_weight.numel() != 1 or not bool(
            torch.isfinite(xyz_loss_weight).all()
        ) or bool((xyz_loss_weight < 0).any()):
            raise ValueError(
                "xyz_loss_weight must be one finite non-negative scalar"
            )
        horizon_loss_sum = squared_error.sum(dim=(0, 2))
        horizon_valid_count = valid.sum(dim=(0, 2))
        # The optimization objective is imitation loss: masked MSE between the
        # predicted clean action/target-state chunk and the ground-truth chunk.
        # The unweighted component values above remain diagnostics. Optional
        # auxiliary objectives below add explicitly configured component terms.
        # Ortho6D targets use ordinary residuals in imitation loss.
        loss_sum = (sample_imitation_loss * sample_is_valid).sum()
        mae_sum = (sample_mae * sample_is_valid).sum()
        valid_count = sample_is_valid.sum()
        imitation_loss = loss_sum / valid_count.clamp_min(1.0)

        # Optional counterfactual Qwen-conditioning objective. The ordinary
        # imitation objective can be minimized while treating Qwen KV as a
        # sample-independent bias. Re-run the exact same noisy action,
        # timestep, image, language, and state context with Qwen KV rolled
        # across the microbatch, then require the matched context to achieve a
        # lower per-sample denoising loss by a configured margin. Using the
        # same diffusion noise/timestep isolates Qwen identity as the changed
        # variable. The hinge stops rewarding arbitrary degradation after the
        # requested separation has been reached.
        qwen_fusion_loss_weight = batch.get("qwen_fusion_loss_weight")
        if qwen_fusion_loss_weight is None:
            qwen_fusion_loss_weight = prediction.new_tensor(0.0)
        qwen_fusion_loss_margin = batch.get("qwen_fusion_loss_margin")
        if qwen_fusion_loss_margin is None:
            qwen_fusion_loss_margin = prediction.new_tensor(0.0)
        for name, value in (
            ("qwen_fusion_loss_weight", qwen_fusion_loss_weight),
            ("qwen_fusion_loss_margin", qwen_fusion_loss_margin),
        ):
            if value.numel() != 1 or not bool(torch.isfinite(value).all()) or bool(
                (value < 0).any()
            ):
                raise ValueError(f"{name} must be one finite non-negative scalar")

        qwen_fusion_loss = prediction.new_tensor(0.0)
        shuffled_qwen_imitation_loss = imitation_loss.detach()
        qwen_fusion_margin_satisfied = prediction.new_tensor(1.0)
        if bool((qwen_fusion_loss_weight > 0).item()):
            if self.cfg.model.qwen_fusion == "none":
                raise ValueError("Qwen fusion loss requires Qwen conditioning")
            if batch_size < 2:
                raise ValueError(
                    "Qwen fusion loss requires micro_batch_size >= 2 for shuffling"
                )
            shuffled_batch = dict(batch)
            # Prefer negatives from the same source dataset so the hinge
            # cannot be satisfied merely by learning OXE dataset identity.
            permutation = torch.roll(
                torch.arange(batch_size, device=prediction.device),
                shifts=1,
                dims=0,
            )
            dataset_ids = batch.get("dataset_id")
            if isinstance(dataset_ids, (list, tuple)) and len(dataset_ids) == batch_size:
                buckets: dict[str, list[int]] = {}
                for sample_index, dataset_id in enumerate(dataset_ids):
                    buckets.setdefault(str(dataset_id), []).append(sample_index)
                for indices in buckets.values():
                    if len(indices) < 2:
                        continue
                    for offset, sample_index in enumerate(indices):
                        permutation[sample_index] = indices[(offset + 1) % len(indices)]
            if self.cfg.model.qwen_fusion in {
                "hidden_cross_attention",
                "hidden_waypoint_cross_attention",
            }:
                shuffled_batch["qwen_hidden_states"] = batch[
                    "qwen_hidden_states"
                ][permutation]
                if self.cfg.model.qwen_fusion == "hidden_waypoint_cross_attention":
                    shuffled_batch["latent_waypoints"] = batch[
                        "latent_waypoints"
                    ][permutation]
                if "plan_mask" in batch:
                    shuffled_batch["plan_mask"] = batch["plan_mask"][permutation]
            else:
                shuffled_batch["qwen_kv"] = batch["qwen_kv"][permutation]
            (
                shuffled_lang_cond,
                shuffled_img_cond,
                shuffled_lang_mask,
                shuffled_external_kv,
                shuffled_external_cross_kv,
                shuffled_extra_cross_cond,
                shuffled_extra_cross_mask,
            ) = self._adapt_static_conditions(
                shuffled_batch,
                state_cond=state_cond,
            )
            shuffled_prediction = self.runner.model(
                state_action_cond,
                batch["ctrl_freq"],
                timesteps,
                shuffled_lang_cond,
                shuffled_img_cond,
                lang_mask=shuffled_lang_mask,
                img_mask=batch["img_mask"].bool(),
                external_kv=shuffled_external_kv,
                external_cross_kv=shuffled_external_cross_kv,
                extra_cross_cond=shuffled_extra_cross_cond,
                extra_cross_mask=shuffled_extra_cross_mask,
                unified_cross_attention=(
                    self.cfg.model.qwen_fusion == "unified_cross_attention"
                ),
            )
            shuffled_diff = shuffled_prediction - target
            if self.cfg.model.action_encoder_layout == "rdt_eef":
                shuffled_diff = torch.cat(
                    [
                        shuffled_diff[..., :3],
                        torch.atan2(
                            torch.sin(shuffled_diff[..., 3:6]),
                            torch.cos(shuffled_diff[..., 3:6]),
                        ),
                        shuffled_diff[..., 6:],
                    ],
                    dim=-1,
                )
            sample_shuffled_imitation_loss = (
                (shuffled_diff.pow(2) * objective_valid).sum(dim=(1, 2))
                / objective_count.clamp_min(1.0)
            )
            shuffled_qwen_imitation_loss = (
                sample_shuffled_imitation_loss * sample_is_valid
            ).sum() / valid_count.clamp_min(1.0)
            sample_fusion_hinge = F.relu(
                qwen_fusion_loss_margin
                + sample_imitation_loss
                - sample_shuffled_imitation_loss
            )
            qwen_fusion_loss = (
                sample_fusion_hinge * sample_is_valid
            ).sum() / valid_count.clamp_min(1.0)
            qwen_fusion_margin_satisfied = (
                (
                    sample_shuffled_imitation_loss - sample_imitation_loss
                    >= qwen_fusion_loss_margin
                ).to(prediction.dtype)
                * sample_is_valid
            ).sum() / valid_count.clamp_min(1.0)

        # prediction_type="sample" means the RDT predicts the clean x0 action
        # chunk at each denoising call. Apply sign classification directly to
        # its existing gripper scalar; this adds no parameters or output head.
        # Rollout executes sign(prediction): <0 -> -1, >=0 -> +1. A literal
        # threshold in the training graph would have zero gradient almost
        # everywhere, so BCEWithLogits is its differentiable counterpart with
        # the same zero decision boundary and binarized {-1,+1} target.
        gripper_logits = prediction[..., gripper_start]
        gripper_labels = (target[..., gripper_start] >= 0).to(gripper_logits.dtype)
        logit_scale = batch.get("gripper_bce_logit_scale")
        if logit_scale is None:
            logit_scale = prediction.new_tensor(1.0)
        if logit_scale.numel() != 1 or not bool(torch.isfinite(logit_scale).all()) or bool(
            (logit_scale <= 0).any()
        ):
            raise ValueError("gripper_bce_logit_scale must be one finite positive scalar")
        gripper_bce_elements = F.binary_cross_entropy_with_logits(
            gripper_logits.float() * logit_scale.float(),
            gripper_labels.float(),
            reduction="none",
        ).to(prediction.dtype)
        gripper_bce_valid = objective_valid[..., gripper_start]
        sample_gripper_bce_count = base_gripper_bce_valid.sum(dim=1)
        sample_gripper_bce_loss = (
            (gripper_bce_elements * gripper_bce_valid).sum(dim=1)
            / sample_gripper_bce_count.clamp_min(1.0)
        )
        gripper_bce_sum = (sample_gripper_bce_loss * sample_is_valid).sum()
        gripper_bce_loss = gripper_bce_sum / valid_count.clamp_min(1.0)
        gripper_bce_weight = batch.get("gripper_bce_weight")
        if gripper_bce_weight is None:
            gripper_bce_weight = prediction.new_tensor(0.0)
        if gripper_bce_weight.numel() != 1 or not bool(
            torch.isfinite(gripper_bce_weight).all()
        ) or bool((gripper_bce_weight < 0).any()):
            raise ValueError("gripper_bce_weight must be one finite non-negative scalar")
        rotation_geodesic_weight = batch.get("rotation_geodesic_weight")
        if rotation_geodesic_weight is None:
            rotation_geodesic_weight = prediction.new_tensor(0.0)
        if rotation_geodesic_weight.numel() != 1 or not bool(
            torch.isfinite(rotation_geodesic_weight).all()
        ) or bool((rotation_geodesic_weight < 0).any()):
            raise ValueError(
                "rotation_geodesic_weight must be one finite non-negative scalar"
            )
        uses_ortho6d_rotation = (
            self.cfg.model.action_encoder_layout == "libero_ortho6d"
            or (
                self.cfg.model.action_encoder_layout == "rdt_native_128"
                and self.cfg.model.native_rdt_128_mapping
                != "libero_joint_eef_delta"
            )
        )
        if uses_ortho6d_rotation:
            predicted_rotation = self._ortho6d_to_rotation_matrix(
                prediction[..., rotation_start:rotation_stop]
            )
            target_rotation = self._ortho6d_to_rotation_matrix(
                target[..., rotation_start:rotation_stop]
            )
            relative_rotation = predicted_rotation.transpose(-1, -2) @ target_rotation
            cosine = (
                relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0
            ) * 0.5
            # 1-cos(theta) is the stable small-angle form of an SO(3)
            # geodesic penalty and avoids acos gradients near identity.
            rotation_geodesic_elements = 1.0 - cosine.clamp(-1.0, 1.0)
            rotation_objective_valid = objective_valid[
                ..., rotation_start:rotation_stop
            ].amin(dim=-1).float()
            sample_rotation_geodesic_count = rotation_objective_valid.sum(dim=1)
            sample_rotation_geodesic_loss = (
                (rotation_geodesic_elements * rotation_objective_valid).sum(dim=1)
                / sample_rotation_geodesic_count.clamp_min(1.0)
            )
            rotation_geodesic_loss = (
                (sample_rotation_geodesic_loss * sample_is_valid.float()).sum()
                / valid_count.float().clamp_min(1.0)
            )
        else:
            sample_rotation_geodesic_loss = prediction.new_zeros(batch_size)
            rotation_geodesic_loss = prediction.new_tensor(0.0)
            if bool((rotation_geodesic_weight > 0).any()):
                raise ValueError(
                    "rotation geodesic loss requires an orthogonal-6D action layout"
                )
        total_loss = (
            imitation_loss
            + xyz_loss_weight * xyz_auxiliary_loss
            + gripper_bce_weight * gripper_bce_loss
            + rotation_geodesic_weight.float() * rotation_geodesic_loss
            + qwen_fusion_loss_weight.float() * qwen_fusion_loss
        )
        result = {
            "loss": total_loss,
            "imitation_loss": imitation_loss.detach(),
            "xyz_auxiliary_loss": xyz_auxiliary_loss.detach(),
            "xyz_loss_weight": xyz_loss_weight.detach(),
            "gripper_bce_loss": gripper_bce_loss.detach(),
            "gripper_bce_weight": gripper_bce_weight.detach(),
            "rotation_geodesic_loss": rotation_geodesic_loss.detach(),
            "rotation_geodesic_weight": rotation_geodesic_weight.detach(),
            "qwen_fusion_loss": qwen_fusion_loss.detach(),
            "qwen_fusion_loss_weight": qwen_fusion_loss_weight.detach(),
            "qwen_fusion_loss_margin": qwen_fusion_loss_margin.detach(),
            "qwen_shuffled_imitation_loss": (
                shuffled_qwen_imitation_loss.detach()
            ),
            "qwen_fusion_margin_satisfied": (
                qwen_fusion_margin_satisfied.detach()
            ),
            "train_target_mae": (
                mae_sum / valid_count.clamp_min(1.0)
            ).detach(),
            "loss_sum": loss_sum.detach(),
            "mae_sum": mae_sum.detach(),
            "valid_count": valid_count.detach(),
            "xyz_loss_sum": xyz_loss_sum.detach(),
            "xyz_valid_count": xyz_valid_count.detach(),
            "rot_loss_sum": rot_loss_sum.detach(),
            "rot_valid_count": rot_valid_count.detach(),
            "gripper_loss_sum": gripper_loss_sum.detach(),
            "gripper_valid_count": gripper_valid_count.detach(),
            # Per-example values allow validation to aggregate by LIBERO suite.
            "sample_imitation_loss": sample_imitation_loss.detach(),
            "sample_unweighted_imitation_loss": (
                sample_unweighted_imitation_loss.detach()
            ),
            "sample_target_mae": sample_mae.detach(),
            "sample_is_valid": sample_is_valid.detach(),
            "sample_xyz_loss": sample_xyz_loss.detach(),
            "sample_xyz_auxiliary_loss": (
                sample_xyz_auxiliary_loss.detach()
            ),
            "sample_xyz_auxiliary_valid": xyz_auxiliary_valid.detach(),
            "sample_xyz_valid": sample_xyz_valid.detach(),
            "sample_rot_loss": sample_rot_loss.detach(),
            "sample_rot_valid": sample_rot_valid.detach(),
            "sample_gripper_loss": sample_gripper_loss.detach(),
            "sample_gripper_valid": sample_gripper_valid.detach(),
            "sample_gripper_bce_loss": sample_gripper_bce_loss.detach(),
            "sample_rotation_geodesic_loss": sample_rotation_geodesic_loss.detach(),
            # These are element-weighted MSE sums/counts for each future offset.
            "horizon_loss_sum": horizon_loss_sum.detach(),
            "horizon_valid_count": horizon_valid_count.detach(),
        }
        if bool(batch.get("return_denoising_prediction", False)):
            result["denoising_prediction"] = prediction.detach()
            result["diffusion_timesteps"] = timesteps.detach()
        return result

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self.cast_batch(batch)
        states = batch["state"].unsqueeze(1)
        state_dim_mask = batch.get(
            "state_dim_mask",
            torch.ones_like(batch["state"]),
        ).to(states.dtype).unsqueeze(1)
        action_dim_mask = batch["action_dim_mask"].to(states.dtype).unsqueeze(1)
        state_input = self._state_encoder_input(states, state_dim_mask)
        state_cond = self.runner.state_adaptor(state_input)
        (
            lang_cond,
            img_cond,
            lang_mask,
            external_kv,
            external_cross_kv,
            extra_cross_cond,
            extra_cross_mask,
        ) = (
            self._adapt_static_conditions(batch, state_cond=state_cond)
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

        expanded_dim_mask = action_dim_mask.expand(-1, noisy.shape[1], -1)
        if self.cfg.model.action_encoder_layout == "rdt_native_128":
            gripper_input_index = 10
        elif self.cfg.model.action_encoder_layout == "libero_ortho6d":
            gripper_input_index = 9
        else:
            gripper_input_index = 6
        final_clean_prediction = None
        for timestep in scheduler.timesteps:
            timestep = timestep.to(states.device)
            model_noisy = noisy
            if self.mask_noisy_gripper_input:
                model_noisy = noisy.clone()
                model_noisy[..., gripper_input_index] = 0.0
            action_cond = self._adapt_actions(model_noisy, expanded_dim_mask)
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
                external_cross_kv=external_cross_kv,
                extra_cross_cond=extra_cross_cond,
                extra_cross_mask=extra_cross_mask,
                unified_cross_attention=(
                    self.cfg.model.qwen_fusion == "unified_cross_attention"
                ),
            )
            final_clean_prediction = output
            noisy = scheduler.step(output, timestep, noisy).prev_sample
            noisy = noisy.to(states.dtype) * expanded_dim_mask
        if self.mask_noisy_gripper_input or self.decode_clean_x0_gripper:
            if final_clean_prediction is None:
                raise RuntimeError("Diffusion scheduler produced no inference timesteps")
            # With prediction_type="sample", `output` is the model's clean x0
            # estimate.  DPM-Solver is appropriate for continuous motion, but
            # integrating the binary gripper as a continuous trajectory can
            # reverse an otherwise correct release sign.  Decode that existing
            # output directly; no separate head or architecture change is used.
            noisy = noisy.clone()
            noisy[..., gripper_input_index] = (
                final_clean_prediction[..., gripper_input_index]
                * expanded_dim_mask[..., gripper_input_index]
            )
        return noisy
