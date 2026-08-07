from types import SimpleNamespace

import pytest
import torch

from thinkflow_rdt.train import resolve_validation_batch_limit, validate


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


def test_validation_reports_per_suite_and_per_horizon_metrics() -> None:
    class FakeModel:
        def eval(self):
            return self

        def train(self):
            return self

        def __call__(self, _batch, *, sample=False):
            assert not sample
            return {
                "loss_sum": torch.tensor(4.0),
                "mae_sum": torch.tensor(1.5),
                "valid_count": torch.tensor(2.0),
                "xyz_loss_sum": torch.tensor(3.0),
                "xyz_valid_count": torch.tensor(2.0),
                "rot_loss_sum": torch.tensor(5.0),
                "rot_valid_count": torch.tensor(2.0),
                "gripper_loss_sum": torch.tensor(7.0),
                "gripper_valid_count": torch.tensor(2.0),
                "sample_imitation_loss": torch.tensor([1.0, 3.0]),
                "sample_target_mae": torch.tensor([0.5, 1.0]),
                "sample_is_valid": torch.ones(2),
                "sample_xyz_loss": torch.tensor([1.0, 2.0]),
                "sample_xyz_valid": torch.ones(2),
                "sample_rot_loss": torch.tensor([2.0, 3.0]),
                "sample_rot_valid": torch.ones(2),
                "sample_gripper_loss": torch.tensor([3.0, 4.0]),
                "sample_gripper_valid": torch.ones(2),
                "horizon_loss_sum": torch.tensor([7.0, 14.0]),
                "horizon_valid_count": torch.tensor([7.0, 7.0]),
            }

    class FakeAccelerator:
        device = torch.device("cpu")
        process_index = 0
        num_processes = 1

        @staticmethod
        def reduce(value, reduction="sum"):
            assert reduction == "sum"
            return value

    cfg = SimpleNamespace(
        model=SimpleNamespace(pred_horizon=2),
        training=SimpleNamespace(
            validation_samples=None,
            validation_batches=1,
            sample_validation_batches=0,
            validation_seed=123,
            micro_batch_size=2,
        ),
    )
    batch = {
        "dataset_id": ["libero_spatial", "libero_object"],
    }

    metrics = validate(FakeModel(), [batch], FakeAccelerator(), cfg)

    assert metrics["val/libero_spatial/loss"] == pytest.approx(1.0)
    assert metrics["val/libero_object/loss"] == pytest.approx(3.0)
    assert metrics["val/libero_spatial/examples"] == 1.0
    assert metrics["val/libero_object/examples"] == 1.0
    assert metrics["val/horizon_mse/step_00"] == pytest.approx(1.0)
    assert metrics["val/horizon_mse/step_01"] == pytest.approx(2.0)
