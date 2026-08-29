import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from thinkflow_rdt.train import (
    _binary_transition_events,
    _match_binary_transition_events,
    _validation_observation_grid,
    resolve_validation_batch_limit,
    validate,
)


def _jpeg(rgb: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), rgb).save(buffer, format="JPEG", quality=100)
    return buffer.getvalue()


def test_validation_observation_grid_uses_only_valid_image_slots() -> None:
    placeholders = [_jpeg((0, 0, 0)) for _ in range(3)]
    actual = _jpeg((200, 50, 25))
    grid = _validation_observation_grid(
        [*placeholders, actual, _jpeg((0, 0, 0)), _jpeg((0, 0, 0))],
        torch.tensor([False, False, False, True, False, False]),
    )
    assert grid is not None
    pixel = grid.getpixel((0, 0))
    assert pixel[0] > 150
    assert pixel[1] < 100


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


def test_native_gripper_transition_matching_includes_step_zero_and_timing() -> None:
    target = _binary_transition_events(
        torch.tensor([False, False, True, True, False]),
        initial_value=True,
    )
    predicted = _binary_transition_events(
        torch.tensor([False, False, False, True, True]),
        initial_value=True,
    )

    assert target == [(0, False), (2, True), (4, False)]
    assert predicted == [(0, False), (3, True)]
    metrics = _match_binary_transition_events(target, predicted)
    assert metrics["all"]["target"] == 3.0
    assert metrics["all"]["predicted"] == 2.0
    assert metrics["all"]["matched"] == 2.0
    assert metrics["all"]["exact"] == 1.0
    assert metrics["all"]["within_1"] == 2.0
    assert metrics["all"]["absolute_error_sum"] == 1.0
    assert metrics["open"]["signed_error_sum"] == 1.0
    assert metrics["close"]["matched"] == 1.0


def test_validation_reports_per_suite_without_denoising_horizon_metrics() -> None:
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
    assert not any(key.startswith("val/horizon_mse/") for key in metrics)


def test_validation_reports_controlled_qwen_ablation_metrics() -> None:
    class FakeModel:
        def eval(self):
            return self

        def train(self):
            return self

        def __call__(self, batch, *, sample=False):
            qwen_value = batch["qwen_kv"][:, 0, 0]
            prediction = torch.zeros_like(batch["actions"])
            prediction[..., 30] = qwen_value.unsqueeze(1)
            if sample:
                return prediction

            target = batch["actions"]
            sample_loss = (prediction[..., 30] - target[..., 30]).square().mean(1)
            sample_mae = (prediction[..., 30] - target[..., 30]).abs().mean(1)
            sample_valid = torch.ones_like(sample_loss)
            return {
                "loss_sum": sample_loss.sum(),
                "mae_sum": sample_mae.sum(),
                "valid_count": sample_valid.sum(),
                "xyz_loss_sum": sample_loss.sum(),
                "xyz_valid_count": sample_valid.sum(),
                "rot_loss_sum": torch.zeros(()),
                "rot_valid_count": sample_valid.sum(),
                "gripper_loss_sum": torch.zeros(()),
                "gripper_valid_count": sample_valid.sum(),
                "sample_imitation_loss": sample_loss,
                "sample_target_mae": sample_mae,
                "sample_is_valid": sample_valid,
                "sample_xyz_loss": sample_loss,
                "sample_xyz_valid": sample_valid,
                "sample_rot_loss": torch.zeros_like(sample_loss),
                "sample_rot_valid": sample_valid,
                "sample_gripper_loss": torch.zeros_like(sample_loss),
                "sample_gripper_valid": sample_valid,
            }

    class FakeAccelerator:
        device = torch.device("cpu")
        process_index = 0
        num_processes = 1
        is_main_process = True

        @staticmethod
        def reduce(value, reduction="sum"):
            assert reduction == "sum"
            return value

    cfg = SimpleNamespace(
        model=SimpleNamespace(
            pred_horizon=2,
            action_encoder_layout="rdt_native_128",
            qwen_fusion="fastthinkact_state_kv",
        ),
        noise_scheduler=SimpleNamespace(num_train_timesteps=1000),
        training=SimpleNamespace(
            validation_samples=None,
            validation_batches=1,
            sample_validation_batches=1,
            validation_seed=123,
            micro_batch_size=2,
            qualitative_validation_examples=0,
            report_to="none",
        ),
    )
    actions = torch.zeros(2, 2, 128)
    actions[0, :, 30] = 1.0
    actions[1, :, 30] = 3.0
    action_dim_mask = torch.zeros(2, 128)
    action_dim_mask[:, 30] = 1.0
    action_dim_mask[:, 10] = 1.0
    batch = {
        "dataset_id": ["bc_z", "bridge"],
        "state": torch.zeros(2, 128),
        "actions": actions,
        "action_time_mask": torch.ones(2, 2, dtype=torch.bool),
        "action_dim_mask": action_dim_mask,
        "qwen_kv": torch.tensor([[[1.0]], [[3.0]]]),
    }

    metrics = validate(FakeModel(), [batch], FakeAccelerator(), cfg)

    assert metrics["val/sample_mse"] == pytest.approx(0.0)
    assert metrics["val/qwen_ablation/reference/denoising_loss"] == pytest.approx(0.0)
    assert metrics["val/qwen_ablation/zero/denoising_loss_delta"] == pytest.approx(5.0)
    assert metrics["val/qwen_ablation/shuffled/denoising_loss_delta"] == pytest.approx(4.0)
    assert metrics["val/qwen_ablation/zero/sample_mse"] == pytest.approx(2.5)
    assert metrics["val/qwen_ablation/shuffled/sample_mse"] == pytest.approx(2.0)
    assert metrics[
        "val/qwen_ablation/zero/prediction_delta_rmse_native10"
    ] > 0.0
    assert metrics[
        "val/qwen_ablation/shuffled/sampled_native10/horizon_1/rmse"
    ] > 0.0
    assert metrics[
        "val/sampled_native10/horizon_1/gripper_command/accuracy"
    ] == 1.0
    assert metrics[
        "val/sampled_native10/horizon_1/gripper_command/f1"
    ] == 1.0
    assert metrics[
        "val/sampled_native10/horizon_1/gripper_transition/accuracy"
    ] == 1.0
