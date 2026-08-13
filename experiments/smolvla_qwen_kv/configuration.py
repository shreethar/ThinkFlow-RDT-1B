"""Configuration helpers for the Qwen-KV-conditioned SmolVLA policy."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("smolvla_qwen_kv")
@dataclass
class KVSmolVLAConfig(SmolVLAConfig):
    """SmolVLA configuration extended with a `[K | V]` conditioning sequence.

    The input tensor is expected to have shape ``[B, T_kv, external_kv_width]``.
    Its last dimension is divided equally into the source key and source value.
    Separate learned projections are used for every Action Expert cross-attention
    layer because every layer has its own attention representation.
    """

    external_kv_key: str = "qwen_kv"
    external_kv_width: int = 2048
    external_kv_token_count: int = 1
    external_kv_required: bool = True
    external_kv_logit_bias_init: float = -4.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.external_kv_width <= 0 or self.external_kv_width % 2:
            raise ValueError(
                "external_kv_width must be a positive even number containing equal K and V halves"
            )
        if self.external_kv_token_count <= 0:
            raise ValueError("external_kv_token_count must be positive")
        if "cross" not in self.attention_mode:
            raise ValueError("Qwen KV injection requires a SmolVLA cross-attention mode")

    @classmethod
    def from_smolvla_config(cls, base: SmolVLAConfig, **overrides: Any) -> "KVSmolVLAConfig":
        """Copy a normal SmolVLA config without JSON-round-tripping nested features."""

        values = {
            item.name: getattr(base, item.name)
            for item in fields(SmolVLAConfig)
            if item.init and hasattr(base, item.name)
        }
        values.update(overrides)
        return cls(**values)


# LeRobot's third-party factory derives the modeling/processor module from the
# configuration class module name.  Point the class at the separately installed
# plugin facade while keeping this experiment package as the implementation
# source (and preserving the existing checkpoint's ``type`` discriminator).
KVSmolVLAConfig.__module__ = (
    "lerobot_policy_smolvla_qwen_kv.configuration_smolvla_qwen_kv"
)


def make_libero_kv_config(
    pretrained_name_or_path: str,
    *,
    device: str = "cuda",
    state_dim: int = 8,
    action_dim: int = 7,
    image_height: int = 128,
    image_width: int = 128,
    chunk_size: int = 50,
    n_action_steps: int = 4,
    train_expert_only: bool = True,
    freeze_vision_encoder: bool = True,
    external_kv_logit_bias_init: float = -4.0,
    external_kv_token_count: int = 1,
    local_files_only: bool = False,
) -> KVSmolVLAConfig:
    """Build a cache-compatible config while retaining pretrained architecture settings.

    SmolVLA pads state and action vectors to 32 internally, so changing the active
    feature sizes from the base checkpoint placeholders to LIBERO's 8D/7D schema does not resize pretrained
    projection weights.
    """

    # The serialized file contains the registry discriminator ``type``. Decode
    # through the registered base class rather than directly through the concrete
    # dataclass (draccus otherwise treats ``type`` as an unknown dataclass field).
    base = PreTrainedConfig.from_pretrained(
        pretrained_name_or_path,
        local_files_only=local_files_only,
    )
    if not isinstance(base, SmolVLAConfig):
        raise TypeError(
            f"Expected a SmolVLA checkpoint, got configuration {type(base).__name__}"
        )
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        f"{OBS_IMAGES}.image": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, image_height, image_width)
        ),
        f"{OBS_IMAGES}.image2": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, image_height, image_width)
        ),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    return KVSmolVLAConfig.from_smolvla_config(
        base,
        input_features=input_features,
        output_features=output_features,
        device=device,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        train_expert_only=train_expert_only,
        freeze_vision_encoder=freeze_vision_encoder,
        external_kv_logit_bias_init=external_kv_logit_bias_init,
        external_kv_token_count=external_kv_token_count,
        # The complete policy checkpoint is loaded immediately after construction.
        # Avoid separately downloading/loading another copy of the VLM weights first.
        load_vlm_weights=False,
        compile_model=False,
    )
