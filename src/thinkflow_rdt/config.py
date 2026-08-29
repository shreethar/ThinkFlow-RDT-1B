from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    action_dim: int
    state_dim: int
    pred_horizon: int
    qwen_hidden_size: int
    qwen_kv_dim: int
    lang_token_dim: int
    img_token_dim: int
    max_lang_tokens: int
    image_tokens: int
    hidden_size: int
    depth: int
    num_heads: int
    dtype: str
    lang_adaptor: str
    img_adaptor: str
    state_adaptor: str
    copy_pretrained_final_fc1: bool
    gradient_checkpointing: bool
    # ``lora`` preserves the original lightweight fine-tuning path. ``full``
    # trains the complete RDT transformer while keeping selected encoders frozen.
    finetune_mode: str = "lora"
    # ``none`` runs the base RDT language/image/state conditioning without any
    # Qwen injection. This is useful for pretrained-only rollout baselines.
    # ``self_attention_kv`` projects the flattened Qwen K/V pair to one native
    # RDT K/V pair and appends it inside every RDT self-attention block.
    # ``fastthinkact_state_kv`` is the explicit experiment name for that same
    # native-RDT mapping: action queries attend state-derived and Qwen K/V in
    # the state/action stream, with no second projection of the Qwen pair.
    # ``cross_attention_kv`` projects cached Qwen K/V directly to native RDT K/V
    # and appends them after each cross-attention block's condition projection.
    # ``fastthinkact_cross_attention_kv`` additionally places the native state
    # encoder token in every cross-attention context. Each block projects that
    # state token to K/V, then directly appends the adapted Qwen K/V without a
    # second attention projection.
    # ``language`` is retained for backward compatibility with older artifacts.
    qwen_fusion: str = "language"
    # The official pretrained RDT state adaptor consumes a unified 128-D state.
    # Delta actions can remain 7-D by using the separate action adaptor.
    rdt_state_dim: int | None = None
    freeze_state_adaptor: bool = False
    freeze_condition_adaptors: bool = False
    # Optional per-modality overrides. ``None`` preserves the legacy combined
    # freeze_condition_adaptors behavior.
    freeze_language_adaptor: bool | None = None
    freeze_image_adaptor: bool | None = None
    # ``rdt_eef`` is the legacy compact 7D [xyz, Euler, binary gripper] layout.
    # ``libero_ortho6d`` uses 11D state [xyz, absolute ortho6D, finger0,
    # finger1] and 10D actions [dxyz, relative ortho6D, raw gripper command].
    state_encoder_layout: str = "raw"
    # ``raw`` keeps cached 7-D actions in the existing project convention.
    # ``rdt_eef`` treats action targets as absolute target states and encodes
    # them through the same native RDT EEF slots as current state observations.
    action_encoder_layout: str = "raw"
    # Compatibility switch for old caches that store binary gripper_closed in
    # dim 6. New LIBERO raw-command caches must leave this disabled.
    convert_cached_gripper_closed_to_open: bool = True
    # Only smoke tests should freeze a randomly initialized native state adaptor.
    allow_random_frozen_state_adaptor: bool = False
    # ``compatible`` copies every shape-compatible runner tensor, including the
    # original RDT condition/state adaptors; the new 7-D output row stays fresh.
    pretrained_copy_mode: str = "selected"
    # Widths stored in the cache can differ from the native RDT model widths.
    # In ``rdt_native_128`` mode the collator converts cached 7-D EEF vectors
    # to native 128-D value tensors before the model sees them.
    cache_state_dim: int | None = None
    cache_action_dim: int | None = None

    @property
    def resolved_rdt_state_dim(self) -> int:
        return self.state_dim if self.rdt_state_dim is None else self.rdt_state_dim

    @property
    def resolved_cache_state_dim(self) -> int:
        return self.state_dim if self.cache_state_dim is None else self.cache_state_dim

    @property
    def resolved_cache_action_dim(self) -> int:
        return self.action_dim if self.cache_action_dim is None else self.cache_action_dim

    @property
    def resolved_freeze_language_adaptor(self) -> bool:
        return (
            self.freeze_condition_adaptors
            if self.freeze_language_adaptor is None
            else self.freeze_language_adaptor
        )

    @property
    def resolved_freeze_image_adaptor(self) -> bool:
        return (
            self.freeze_condition_adaptors
            if self.freeze_image_adaptor is None
            else self.freeze_image_adaptor
        )

@dataclass(frozen=True)
class LoraConfigData:
    rank: int
    alpha: int
    dropout: float
    target_self_attention: bool
    target_cross_attention: bool
    target_ffn: bool
    train_final_layer: bool


@dataclass(frozen=True)
class NoiseSchedulerConfig:
    num_train_timesteps: int
    num_inference_timesteps: int
    beta_schedule: str
    prediction_type: str
    clip_sample: bool


@dataclass(frozen=True)
class DataConfig:
    train_manifest: str
    val_manifest: str
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    excluded_dataset_ids: tuple[str, ...] | list[str] = ()
    episode_aware_shuffle: bool = False
    shuffle_validation: bool = False
    stratified_validation: bool = False
    # Per-dataset q01/q99 files used to undo cached 7-D action normalization
    # before converting Euler rotation deltas to orthogonal 6-D.
    action_stats_paths: dict[str, str] | None = None


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate_lora: float
    learning_rate_interfaces: float
    weight_decay_interfaces: float
    warmup_steps: int
    max_grad_norm: float
    log_every: int
    validate_every: int
    save_every: int
    validation_batches: int
    sample_validation_batches: int
    mixed_precision: str
    report_to: str
    # If set, this is enforced as
    # micro_batch_size * accumulation * distributed_world_size.
    global_batch_size: int | None = None
    learning_rate: float | None = None
    wandb_project: str = "thinkflow-rdt"
    wandb_run_name: str | None = None
    validation_seed: int = 12345
    # Exact number of examples evaluated globally across all ranks. When null,
    # ``validation_batches`` retains its legacy per-rank meaning.
    validation_samples: int | None = None
    skip_nonfinite_updates: bool = True
    max_consecutive_nonfinite_updates: int = 10
    log_gradient_stats: bool = True
    qualitative_validation_examples: int = 2
    # Validation can use a larger batch than training because it has no
    # backward activations. ``None`` preserves the training micro batch size.
    validation_batch_size: int | None = None
    # Diffusion-sampled trajectory metrics are expensive. Keep this explicit so
    # experiments can evaluate only the deployed replanning horizon (LIBERO: 10).
    sampled_validation_horizons: tuple[int, ...] | list[int] = (1, 4, 8, 10, 64)
    # Optional counterfactual ranking objective for the action-model stage.
    # The Fast-ThinkAct paper itself uses only imitation loss here; this
    # experimental term requires the matched Qwen KV to outperform a shuffled
    # sample's KV by at least ``qwen_fusion_loss_margin``.
    qwen_fusion_loss_weight: float = 0.0
    qwen_fusion_loss_margin: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    rdt_repo: str
    pretrained_model: str | None
    output_dir: str
    model: ModelConfig
    lora: LoraConfigData
    noise_scheduler: NoiseSchedulerConfig
    data: DataConfig
    training: TrainingConfig

    def validate(self) -> None:
        if (
            self.model.action_dim != self.model.state_dim
            and self.model.state_encoder_layout != "libero_ortho6d"
        ):
            raise ValueError(
                "This RDT runner concatenates state and action tokens, so "
                "state_dim must equal action_dim. Pad/project proprioception "
                "to action_dim in the dataset adapter."
            )
        if self.model.hidden_size % self.model.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.model.pred_horizon <= 0:
            raise ValueError("pred_horizon must be positive")
        if self.model.finetune_mode not in {"lora", "full"}:
            raise ValueError("model.finetune_mode must be 'lora' or 'full'")
        if self.model.qwen_fusion not in {
            "none",
            "language",
            "self_attention_kv",
            "fastthinkact_state_kv",
            "cross_attention_kv",
            "fastthinkact_cross_attention_kv",
            "unified_cross_attention",
        }:
            raise ValueError(
                "model.qwen_fusion must be 'none', 'language', "
                "'self_attention_kv', 'fastthinkact_state_kv', "
                "'cross_attention_kv', 'fastthinkact_cross_attention_kv', or "
                "'unified_cross_attention'"
            )
        if self.model.pretrained_copy_mode not in {"selected", "compatible"}:
            raise ValueError(
                "model.pretrained_copy_mode must be 'selected' or 'compatible'"
            )
        if self.model.state_encoder_layout not in {
            "raw",
            "rdt_eef",
            "libero_ortho6d",
            "rdt_native_128",
        }:
            raise ValueError(
                "model.state_encoder_layout must be 'raw', 'rdt_eef', or "
                "'libero_ortho6d', or 'rdt_native_128'"
            )
        if self.model.action_encoder_layout not in {
            "raw",
            "rdt_eef",
            "libero_ortho6d",
            "rdt_native_128",
        }:
            raise ValueError(
                "model.action_encoder_layout must be 'raw', 'rdt_eef', or "
                "'libero_ortho6d', or 'rdt_native_128'"
            )
        if (
            self.model.action_encoder_layout == "rdt_eef"
            and self.model.state_encoder_layout != "rdt_eef"
        ):
            raise ValueError(
                "rdt_eef action layout requires model.state_encoder_layout=rdt_eef"
            )
        if self.model.finetune_mode == "lora" and self.lora.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.model.state_encoder_layout == "rdt_eef":
            if self.model.state_dim != 7 or self.model.action_dim != 7:
                raise ValueError("rdt_eef state layout requires 7-D state/actions")
            if self.model.resolved_rdt_state_dim < 39:
                raise ValueError("rdt_eef state layout requires rdt_state_dim >= 39")
        elif self.model.state_encoder_layout == "libero_ortho6d":
            if self.model.state_dim != 11 or self.model.action_dim != 10:
                raise ValueError(
                    "libero_ortho6d requires 11-D state and 10-D actions"
                )
            if self.model.action_encoder_layout != "libero_ortho6d":
                raise ValueError(
                    "libero_ortho6d state layout requires the matching action layout"
                )
            if self.model.resolved_rdt_state_dim < 39:
                raise ValueError(
                    "libero_ortho6d requires rdt_state_dim >= 39"
                )
            if self.model.convert_cached_gripper_closed_to_open:
                raise ValueError(
                    "libero_ortho6d preserves raw gripper values; set "
                    "convert_cached_gripper_closed_to_open=false"
                )
        elif self.model.state_encoder_layout == "rdt_native_128":
            if self.model.action_encoder_layout != "rdt_native_128":
                raise ValueError(
                    "rdt_native_128 state layout requires the matching action layout"
                )
            if self.model.state_dim != 128 or self.model.action_dim != 128:
                raise ValueError(
                    "rdt_native_128 requires model state_dim=action_dim=128"
                )
            if self.model.resolved_rdt_state_dim != 128:
                raise ValueError("rdt_native_128 requires rdt_state_dim=128")
            cache_layout = (
                self.model.resolved_cache_state_dim,
                self.model.resolved_cache_action_dim,
            )
            if cache_layout not in {(7, 7), (11, 10)}:
                raise ValueError(
                    "rdt_native_128 requires cached 7-D OXE state/actions or "
                    "11-D state + 10-D action LIBERO caches"
                )
        elif self.model.resolved_rdt_state_dim != self.model.state_dim:
            raise ValueError(
                "raw state layout requires rdt_state_dim to equal state_dim"
            )
        if self.noise_scheduler.prediction_type not in {"sample", "epsilon"}:
            raise ValueError("prediction_type must be 'sample' or 'epsilon'")
        if self.training.micro_batch_size <= 0:
            raise ValueError("training.micro_batch_size must be positive")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        if (
            self.training.global_batch_size is not None
            and self.training.global_batch_size <= 0
        ):
            raise ValueError("training.global_batch_size must be positive")
        if (
            self.training.validation_samples is not None
            and self.training.validation_samples <= 0
        ):
            raise ValueError("training.validation_samples must be positive")
        if self.data.stratified_validation and self.data.shuffle_validation:
            raise ValueError(
                "data.stratified_validation and shuffle_validation are mutually exclusive"
            )
        if self.training.max_consecutive_nonfinite_updates <= 0:
            raise ValueError(
                "training.max_consecutive_nonfinite_updates must be positive"
            )
        if self.training.qualitative_validation_examples < 0:
            raise ValueError(
                "training.qualitative_validation_examples must be non-negative"
            )
        if (
            self.training.validation_batch_size is not None
            and self.training.validation_batch_size <= 0
        ):
            raise ValueError("training.validation_batch_size must be positive")
        sampled_horizons = tuple(self.training.sampled_validation_horizons)
        if not sampled_horizons:
            raise ValueError("training.sampled_validation_horizons cannot be empty")
        if len(set(sampled_horizons)) != len(sampled_horizons):
            raise ValueError("training.sampled_validation_horizons must be unique")
        if any(horizon <= 0 for horizon in sampled_horizons):
            raise ValueError(
                "training.sampled_validation_horizons must contain positive values"
            )
        if self.training.qwen_fusion_loss_weight < 0:
            raise ValueError("training.qwen_fusion_loss_weight must be non-negative")
        if self.training.qwen_fusion_loss_margin < 0:
            raise ValueError("training.qwen_fusion_loss_margin must be non-negative")
        if (
            self.training.qwen_fusion_loss_weight > 0
            and self.model.qwen_fusion == "none"
        ):
            raise ValueError(
                "qwen_fusion_loss_weight requires a non-'none' Qwen fusion mode"
            )


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    cfg = ExperimentConfig(
        seed=int(_require(raw, "seed")),
        rdt_repo=str(_require(raw, "rdt_repo")),
        pretrained_model=raw.get("pretrained_model"),
        output_dir=str(_require(raw, "output_dir")),
        model=ModelConfig(**_require(raw, "model")),
        lora=LoraConfigData(**_require(raw, "lora")),
        noise_scheduler=NoiseSchedulerConfig(**_require(raw, "noise_scheduler")),
        data=DataConfig(**_require(raw, "data")),
        training=TrainingConfig(**_require(raw, "training")),
    )
    cfg.validate()
    return cfg
