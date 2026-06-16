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
        if self.lora.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.noise_scheduler.prediction_type not in {"sample", "epsilon"}:
            raise ValueError("prediction_type must be 'sample' or 'epsilon'")


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
