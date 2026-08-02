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
    # ``language`` is retained for backward compatibility with older artifacts.
    qwen_fusion: str = "language"
    # The official pretrained RDT state adaptor consumes a unified 128-D state.
    # Delta actions can remain 7-D by using the separate action adaptor.
    rdt_state_dim: int | None = None
    freeze_state_adaptor: bool = False
    freeze_condition_adaptors: bool = False
    # Cached shards store [xyz, Euler/axis-angle xyz, gripper_closed], but the
    # collators flip dim 6 to RDT's native gripper_open convention at load time.
    # ``rdt_eef`` then maps loaded [xyz, Euler/axis-angle xyz, gripper_open]
    # observations to RDT's native [xyz, ortho6d, gripper_open] state slots.
    state_encoder_layout: str = "raw"
    # ``raw`` keeps cached 7-D actions in the existing project convention.
    # ``rdt_eef`` treats action targets as absolute target states and encodes
    # them through the same native RDT EEF slots as current state observations.
    action_encoder_layout: str = "raw"
    # Only smoke tests should freeze a randomly initialized native state adaptor.
    allow_random_frozen_state_adaptor: bool = False
    # ``compatible`` copies every shape-compatible runner tensor, including the
    # original RDT condition/state adaptors; the new 7-D output row stays fresh.
    pretrained_copy_mode: str = "selected"

    @property
    def resolved_rdt_state_dim(self) -> int:
        return self.state_dim if self.rdt_state_dim is None else self.rdt_state_dim

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
        if self.model.action_dim != self.model.state_dim:
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
            "unified_cross_attention",
        }:
            raise ValueError(
                "model.qwen_fusion must be 'none', 'language', "
                "'self_attention_kv', or 'unified_cross_attention'"
            )
        if self.model.pretrained_copy_mode not in {"selected", "compatible"}:
            raise ValueError(
                "model.pretrained_copy_mode must be 'selected' or 'compatible'"
            )
        if self.model.state_encoder_layout not in {"raw", "rdt_eef"}:
            raise ValueError(
                "model.state_encoder_layout must be 'raw' or 'rdt_eef'"
            )
        if self.model.action_encoder_layout not in {"raw", "rdt_eef"}:
            raise ValueError(
                "model.action_encoder_layout must be 'raw' or 'rdt_eef'"
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
