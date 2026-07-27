from types import SimpleNamespace

import pytest

from thinkflow_rdt.train import resolve_validation_batch_limit


def _config(*, validation_samples: int | None, validation_batches: int = 17):
    return SimpleNamespace(
        training=SimpleNamespace(
            validation_samples=validation_samples,
            validation_batches=validation_batches,
            micro_batch_size=1,
        )
    )


def test_validation_sample_budget_is_global_across_ranks() -> None:
    cfg = _config(validation_samples=256)
    for world_size, expected_local_rounds in ((1, 256), (2, 128), (8, 32)):
        accelerator = SimpleNamespace(num_processes=world_size)
        assert (
            resolve_validation_batch_limit(cfg, accelerator)
            == expected_local_rounds
        )


def test_validation_sample_budget_must_divide_distributed_round() -> None:
    cfg = _config(validation_samples=10)
    accelerator = SimpleNamespace(num_processes=4)
    with pytest.raises(ValueError, match="validation_samples must be divisible"):
        resolve_validation_batch_limit(cfg, accelerator)


def test_legacy_validation_batch_limit_is_preserved() -> None:
    cfg = _config(validation_samples=None, validation_batches=17)
    accelerator = SimpleNamespace(num_processes=8)
    assert resolve_validation_batch_limit(cfg, accelerator) == 17
