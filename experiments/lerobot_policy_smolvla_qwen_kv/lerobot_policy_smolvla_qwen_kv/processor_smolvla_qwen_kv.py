"""Use SmolVLA's official processor construction for the custom policy."""

from lerobot.policies.smolvla.processor_smolvla import (
    make_smolvla_pre_post_processors,
)


def make_smolvla_qwen_kv_pre_post_processors(config, dataset_stats=None):
    return make_smolvla_pre_post_processors(config, dataset_stats=dataset_stats)


__all__ = ["make_smolvla_qwen_kv_pre_post_processors"]
