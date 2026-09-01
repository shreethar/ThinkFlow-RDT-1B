from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.generate_libero_recovery_cache import (
    first_gripper_transition,
    pad_cached_t5_features,
    position_feedback_action,
)
from thinkflow_rdt.config import load_config
from thinkflow_rdt.train import (
    apply_conditioning_warmup_lrs,
    create_optimizer,
    qwen_fusion_weight_for_step,
)


def test_first_gripper_transition_requires_persistence() -> None:
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[:, 6] = [-1, -1, 1, -1, -1, 1, 1, 1]
    assert first_gripper_transition(actions) == 5


def test_position_feedback_preserves_rotation_and_gripper() -> None:
    reference = np.asarray([0.1, 0.2, 0.3, 0.4, -0.5, 0.6, -1.0], dtype=np.float32)
    action = position_feedback_action(
        np.asarray([0.03, -0.02, 0.01]),
        np.zeros(3),
        reference,
        gain=20.0,
        command_limit=0.8,
    )
    np.testing.assert_allclose(action[:3], [0.6, -0.4, 0.2], atol=1e-6)
    np.testing.assert_allclose(action[3:], reference[3:])


def test_cached_t5_features_are_padded_without_changing_valid_tokens() -> None:
    tokens = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    output, mask = pad_cached_t5_features(
        ["task"],
        {"task": (tokens, torch.tensor([True, False, True]))},
        max_lang_tokens=5,
        expected_dim=4,
    )
    assert output.shape == (1, 5, 4)
    assert mask[0].tolist() == [True, True, False, False, False]
    torch.testing.assert_close(output[0, :2].float(), tokens[[0, 2]])


def test_recovery_config_unfreezes_only_image_condition_adaptor() -> None:
    cfg = load_config("configs/libero_b0_native128_recovery_continue.yaml")
    assert cfg.model.freeze_state_adaptor
    assert cfg.model.resolved_freeze_language_adaptor
    assert not cfg.model.resolved_freeze_image_adaptor


def test_full_optimizer_uses_interface_specific_learning_rate() -> None:
    class TinyRDT(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cross_attn = torch.nn.Linear(2, 2)
            self.ffn = torch.nn.Linear(2, 2)

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.runner = torch.nn.Module()
            self.runner.model = TinyRDT()
            self.runner.state_adaptor = torch.nn.Linear(2, 2)
            self.runner.state_adaptor.requires_grad_(False)
            self.runner.img_adaptor = torch.nn.Linear(2, 2)
            self.qwen_adaptor = torch.nn.Linear(2, 2)

    cfg = SimpleNamespace(
        model=SimpleNamespace(finetune_mode="full", freeze_state_adaptor=True),
        training=SimpleNamespace(
            learning_rate=1e-5,
            learning_rate_interfaces=5e-6,
            weight_decay_interfaces=0.01,
            conditioning_warmup_steps=2,
            learning_rate_rdt_backbone=1e-5,
            learning_rate_rdt_cross_attention=2e-5,
            learning_rate_plan_projector=1e-4,
            conditioning_warmup_cross_attention_learning_rate=1e-4,
        ),
    )
    optimizer = create_optimizer(TinyModel(), cfg)
    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert rates == {
        "rdt_cross_attention": pytest.approx(1e-4),
        "rdt_backbone": pytest.approx(1e-5),
        "plan_projector": pytest.approx(1e-4),
        "interfaces": pytest.approx(5e-6),
    }

    scheduled_lrs = [group["lr"] for group in optimizer.param_groups]
    assert apply_conditioning_warmup_lrs(
        optimizer,
        global_step=0,
        conditioning_warmup_steps=2,
        scheduled_lrs=scheduled_lrs,
    )
    warmup_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert warmup_rates["rdt_cross_attention"] == pytest.approx(1e-4)
    assert warmup_rates["plan_projector"] == pytest.approx(1e-4)
    assert warmup_rates["rdt_backbone"] == 0.0
    assert warmup_rates["interfaces"] == 0.0

    assert not apply_conditioning_warmup_lrs(
        optimizer,
        global_step=2,
        conditioning_warmup_steps=2,
        scheduled_lrs=scheduled_lrs,
    )
    restored_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    assert restored_rates == {
        "rdt_cross_attention": pytest.approx(2e-5),
        "rdt_backbone": pytest.approx(1e-5),
        "plan_projector": pytest.approx(1e-4),
        "interfaces": pytest.approx(5e-6),
    }


def test_qwen_fusion_weight_switches_after_conditioning_warmup() -> None:
    kwargs = {
        "conditioning_warmup_steps": 250,
        "post_warmup_weight": 3.0,
        "conditioning_warmup_weight": 10.0,
    }
    assert qwen_fusion_weight_for_step(global_step=0, **kwargs) == 10.0
    assert qwen_fusion_weight_for_step(global_step=249, **kwargs) == 10.0
    assert qwen_fusion_weight_for_step(global_step=250, **kwargs) == 3.0
