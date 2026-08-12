"""LeRobot third-party registration for Qwen-KV SmolVLA.

The distribution name starts with ``lerobot_policy_`` so unmodified LeRobot
entrypoints discover it through ``register_third_party_plugins()``.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Editable installs keep this file inside the repository.  Add the repository
# root so the implementation can remain in the isolated experiment package.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# LeRobot's default batch converter keeps a conservative fixed list of metadata
# keys.  Register qwen_kv as complementary data so the official processor
# pipeline carries it through tokenization, normalization, and device transfer.
from lerobot.processor import converters as _converters

if "qwen_kv" not in _converters._COMPLEMENTARY_KEYS:
    _converters._COMPLEMENTARY_KEYS = (*_converters._COMPLEMENTARY_KEYS, "qwen_kv")


def _install_cached_dataset_factory() -> None:
    """Teach the literal ``lerobot-train`` entrypoint about cached shards.

    LeRobot has a third-party policy registry but no corresponding dataset
    registry.  Install an in-memory delegating factory: cached dataset IDs go
    to our IterableDataset and every normal LeRobot repo_id goes unchanged to
    the upstream factory.  No installed LeRobot source file is modified.
    """

    import lerobot.datasets.factory as dataset_factory

    from experiments.smolvla_qwen_kv.lerobot_integration import (
        make_cached_train_eval_datasets,
    )

    active_factory = dataset_factory.make_train_eval_datasets
    if getattr(active_factory, "_smolvla_qwen_cached_factory", False):
        integrated_factory = active_factory
    else:
        native_factory = active_factory

        def integrated_factory(cfg):
            cached = make_cached_train_eval_datasets(cfg)
            return native_factory(cfg) if cached is None else cached

        integrated_factory._smolvla_qwen_cached_factory = True
        dataset_factory.make_train_eval_datasets = integrated_factory

    # lerobot_train imports the factory into module scope before it discovers
    # plugins. Replace that already-bound reference as well.
    train_module = sys.modules.get("lerobot.scripts.lerobot_train")
    if train_module is not None:
        train_module.make_train_eval_datasets = integrated_factory


_install_cached_dataset_factory()

from .configuration_smolvla_qwen_kv import KVSmolVLAConfig

__all__ = ["KVSmolVLAConfig"]
