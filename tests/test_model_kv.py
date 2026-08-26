from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MethodType

import torch
from timm.models.vision_transformer import Attention

from thinkflow_rdt.config import load_config
from thinkflow_rdt.model import (
    SFTConditionedRDT,
    _cross_attention_with_external_kv,
    _self_attention_with_external_kv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyCrossAttention(torch.nn.Module):
    def __init__(self, width: int = 16, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = False
        self.q = torch.nn.Linear(width, width)
        self.kv = torch.nn.Linear(width, 2 * width)
        self.q_norm = torch.nn.LayerNorm(self.head_dim)
        self.k_norm = torch.nn.LayerNorm(self.head_dim)
        self.attn_drop = torch.nn.Dropout(0.0)
        self.proj = torch.nn.Linear(width, width)
        self.proj_drop = torch.nn.Dropout(0.0)

    def forward(self, x, condition, mask=None):
        return _cross_attention_with_external_kv(
            self, x, condition, mask, None
        )


def test_native_rdt_kv_injection_has_projector_gradient():
    attention = Attention(
        dim=16,
        num_heads=4,
        qkv_bias=True,
        qk_norm=True,
        norm_layer=torch.nn.LayerNorm,
    )
    values = torch.randn(2, 5, 16, requires_grad=True)
    external_kv = torch.randn(2, 5, 32, requires_grad=True)

    output = _self_attention_with_external_kv(attention, values, external_kv)
    output.square().mean().backward()

    assert output.shape == values.shape
    assert external_kv.grad is not None
    assert float(external_kv.grad.norm()) > 0.0


def test_direct_cross_attention_kv_bypasses_native_condition_projection():
    attention = TinyCrossAttention()
    queries = torch.randn(2, 3, 16, requires_grad=True)
    condition = torch.randn(2, 5, 16, requires_grad=True)
    condition_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    external_kv = torch.randn(2, 5, 32, requires_grad=True)
    projected_condition_lengths: list[int] = []

    hook = attention.kv.register_forward_pre_hook(
        lambda _module, inputs: projected_condition_lengths.append(inputs[0].shape[1])
    )
    output = _cross_attention_with_external_kv(
        attention,
        queries,
        condition,
        condition_mask,
        external_kv,
    )
    hook.remove()
    output.square().mean().backward()

    assert output.shape == queries.shape
    # Only the five native condition tokens pass through attention.kv. The five
    # external tokens are appended afterwards and therefore are not reprojected.
    assert projected_condition_lengths == [5]
    assert external_kv.grad is not None
    assert float(external_kv.grad.norm()) > 0.0


def test_direct_cross_attention_model_supports_five_spatial_kv_tokens():
    base = load_config(REPO_ROOT / "configs" / "tiny_smoke.yaml")
    model_config = replace(
        base.model,
        finetune_mode="full",
        qwen_fusion="cross_attention_kv",
    )
    model = SFTConditionedRDT(replace(base, model=model_config), load_pretrained=False)
    torch.nn.init.normal_(
        model.runner.model.final_layer.ffn_final.fc2.weight,
        std=0.02,
    )
    batch = {
        "lang_tokens": torch.randn(2, 12, 64),
        "lang_mask": torch.ones(2, 12, dtype=torch.bool),
        "img_tokens": torch.randn(2, 16, 64),
        "img_mask": torch.ones(2, 16, dtype=torch.bool),
        "qwen_kv": torch.randn(2, 5, 64),
        "state": torch.randn(2, 7),
        "actions": torch.randn(2, 16, 7),
        "action_time_mask": torch.ones(2, 16, dtype=torch.bool),
        "action_dim_mask": torch.ones(2, 7),
        "ctrl_freq": torch.full((2,), 10.0),
    }

    metrics = model(batch)
    metrics["loss"].backward()

    assert model.qwen_adaptor[-1].out_features == 2 * model_config.hidden_size
    assert model.qwen_adaptor[-1].weight.grad is not None
    assert float(model.qwen_adaptor[-1].weight.grad.norm()) > 0.0
    assert metrics["loss"].isfinite()


def test_full_kv_model_uses_128d_frozen_state_interface():
    base = load_config(REPO_ROOT / "configs" / "tiny_smoke.yaml")
    model_config = replace(
        base.model,
        finetune_mode="full",
        qwen_fusion="fastthinkact_state_kv",
        rdt_state_dim=128,
        state_encoder_layout="rdt_eef",
        freeze_state_adaptor=True,
        freeze_condition_adaptors=True,
        allow_random_frozen_state_adaptor=True,
    )
    cfg = replace(base, model=model_config)
    cfg.validate()
    model = SFTConditionedRDT(cfg, load_pretrained=False)

    # Upstream initializes the last output projection to zero. Make it nonzero
    # so a single backward pass can exercise the projector gradient path.
    torch.nn.init.normal_(
        model.runner.model.final_layer.ffn_final.fc2.weight,
        std=0.02,
    )
    batch_size = 2
    batch = {
        "lang_tokens": torch.randn(batch_size, 12, 64),
        "lang_mask": torch.ones(batch_size, 12, dtype=torch.bool),
        "img_tokens": torch.randn(batch_size, 16, 64),
        "img_mask": torch.ones(batch_size, 16, dtype=torch.bool),
        "qwen_kv": torch.randn(batch_size, 5, 64),
        "state": torch.randn(batch_size, 7),
        "actions": torch.randn(batch_size, 16, 7),
        "action_time_mask": torch.ones(batch_size, 16, dtype=torch.bool),
        "action_dim_mask": torch.ones(batch_size, 7),
        "ctrl_freq": torch.full((batch_size,), 10.0),
    }

    metrics = model(batch)
    metrics["loss"].backward()

    assert model.qwen_adaptor[-1].out_features == 2 * cfg.model.hidden_size
    assert model.runner.state_adaptor[0].in_features == 2 * 128
    assert model.action_adaptor is not None
    assert model.action_adaptor[0].in_features == 2 * 7
    assert model.runner.model.final_layer.ffn_final.fc2.out_features == 7
    assert model.qwen_adaptor[-1].weight.grad is not None
    assert float(model.qwen_adaptor[-1].weight.grad.norm()) > 0.0
    assert model.action_adaptor[0].weight.grad is not None
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.runner.state_adaptor.parameters()
    )
    assert metrics["loss"].isfinite()


def test_native_128d_output_is_supervised_only_on_ten_eef_slots():
    base = load_config(REPO_ROOT / "configs" / "tiny_smoke.yaml")
    model_config = replace(
        base.model,
        action_dim=128,
        state_dim=128,
        cache_action_dim=7,
        cache_state_dim=7,
        finetune_mode="full",
        qwen_fusion="self_attention_kv",
        rdt_state_dim=128,
        state_encoder_layout="rdt_native_128",
        action_encoder_layout="rdt_native_128",
        freeze_state_adaptor=True,
        freeze_condition_adaptors=True,
        convert_cached_gripper_closed_to_open=False,
        allow_random_frozen_state_adaptor=True,
    )
    cfg = replace(base, model=model_config)
    cfg.validate()
    model = SFTConditionedRDT(cfg, load_pretrained=False)
    torch.nn.init.normal_(
        model.runner.model.final_layer.ffn_final.fc2.weight,
        std=0.02,
    )
    batch_size = 2
    active = torch.zeros(128)
    active[[10, *range(30, 39)]] = 1.0
    batch = {
        "lang_tokens": torch.randn(batch_size, 12, 64),
        "lang_mask": torch.ones(batch_size, 12, dtype=torch.bool),
        "img_tokens": torch.randn(batch_size, 16, 64),
        "img_mask": torch.ones(batch_size, 16, dtype=torch.bool),
        "qwen_kv": torch.randn(batch_size, 1, 64),
        "state": torch.randn(batch_size, 128) * active,
        "state_dim_mask": active.expand(batch_size, -1),
        "actions": torch.randn(batch_size, 16, 128) * active,
        "action_time_mask": torch.ones(batch_size, 16, dtype=torch.bool),
        "action_dim_mask": active.expand(batch_size, -1),
        "ctrl_freq": torch.full((batch_size,), 10.0),
    }

    metrics = model(batch)
    metrics["loss"].backward()

    assert model.runner.model.final_layer.ffn_final.fc2.out_features == 128
    assert model.action_adaptor is None
    assert model.runner.state_adaptor[0].in_features == 256
    inactive = active == 0
    final_gradient = model.runner.model.final_layer.ffn_final.fc2.weight.grad
    assert final_gradient is not None
    assert torch.count_nonzero(final_gradient[inactive]).item() == 0
    assert torch.count_nonzero(final_gradient[active.bool()]).item() > 0
    assert metrics["loss"].isfinite()


def test_native_state_conversion_uses_ortho6d_and_open_gripper():
    base = load_config(REPO_ROOT / "configs" / "tiny_smoke.yaml")
    model_config = replace(
        base.model,
        finetune_mode="full",
        qwen_fusion="self_attention_kv",
        rdt_state_dim=128,
        state_encoder_layout="rdt_eef",
        freeze_state_adaptor=True,
        allow_random_frozen_state_adaptor=True,
    )
    model = SFTConditionedRDT(replace(base, model=model_config), load_pretrained=False)
    state = torch.tensor([[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]])
    mask = torch.ones_like(state)

    encoded = model._state_encoder_input(state, mask)
    values, indicator = encoded.chunk(2, dim=-1)
    torch.testing.assert_close(values[0, 0, 30:33], state[0, 0, :3])
    torch.testing.assert_close(
        values[0, 0, 33:39],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )
    # The collator has already converted cached gripper_closed to gripper_open.
    assert values[0, 0, 10].item() == 1.0
    assert indicator[0, 0, 10].item() == 1.0
    assert indicator[0, 0, 33:39].sum().item() == 6.0


def test_libero_ortho6d_layout_maps_raw_fingers_and_raw_command_slots():
    base = load_config(REPO_ROOT / "configs" / "tiny_smoke.yaml")
    model_config = replace(
        base.model,
        action_dim=10,
        state_dim=11,
        finetune_mode="full",
        rdt_state_dim=128,
        state_encoder_layout="libero_ortho6d",
        action_encoder_layout="libero_ortho6d",
        convert_cached_gripper_closed_to_open=False,
        freeze_state_adaptor=True,
        allow_random_frozen_state_adaptor=True,
    )
    cfg = replace(base, model=model_config)
    cfg.validate()
    model = SFTConditionedRDT(cfg, load_pretrained=False)

    state = torch.tensor(
        [[[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.04, -0.04]]]
    )
    state_encoded = model._state_encoder_input(state, torch.ones_like(state))
    state_values, state_mask = state_encoded.chunk(2, dim=-1)
    torch.testing.assert_close(state_values[0, 0, 30:39], state[0, 0, :9])
    torch.testing.assert_close(state_values[0, 0, 10:12], state[0, 0, 9:11])
    assert state_mask[0, 0, 10:12].sum().item() == 2.0

    action = torch.tensor(
        [[[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0]]]
    )
    action_encoded = model._action_encoder_input(action, torch.ones_like(action))
    action_values, action_mask = action_encoded.chunk(2, dim=-1)
    torch.testing.assert_close(action_values[0, 0, 30:39], action[0, 0, :9])
    assert action_values[0, 0, 10].item() == -1.0
    assert action_mask[0, 0, 10].item() == 1.0
    assert model.action_adaptor is None
    assert model.runner.model.final_layer.ffn_final.fc2.out_features == 10


def test_masked_loss_is_mean_of_per_example_losses():
    base = load_config(REPO_ROOT / "configs" / "tiny_smoke.yaml")
    model_config = replace(
        base.model,
        finetune_mode="full",
        qwen_fusion="self_attention_kv",
        rdt_state_dim=128,
        state_encoder_layout="rdt_eef",
        freeze_state_adaptor=True,
        allow_random_frozen_state_adaptor=True,
    )
    model = SFTConditionedRDT(replace(base, model=model_config), load_pretrained=False)

    def zero_prediction(self, x, *_args, **_kwargs):
        return x.new_zeros(x.shape[0], model.horizon, 7)

    model.runner.model.forward = MethodType(zero_prediction, model.runner.model)
    actions = torch.stack(
        [
            torch.ones(model.horizon, 7),
            torch.full((model.horizon, 7), 2.0),
        ]
    )
    time_mask = torch.zeros(2, model.horizon, dtype=torch.bool)
    time_mask[0] = True
    time_mask[1, 0] = True
    batch = {
        "lang_tokens": torch.randn(2, 12, 64),
        "lang_mask": torch.ones(2, 12, dtype=torch.bool),
        "img_tokens": torch.randn(2, 16, 64),
        "img_mask": torch.ones(2, 16, dtype=torch.bool),
        "qwen_kv": torch.randn(2, 1, 64),
        "state": torch.zeros(2, 7),
        "actions": actions,
        "action_time_mask": time_mask,
        "action_dim_mask": torch.ones(2, 7),
        "ctrl_freq": torch.full((2,), 10.0),
    }

    metrics = model(batch)
    torch.testing.assert_close(metrics["loss"], torch.tensor(2.5))
    torch.testing.assert_close(metrics["train_target_mae"], torch.tensor(1.5))
    assert metrics["valid_count"].item() == 2.0
    assert metrics["sample_imitation_loss"].shape == (2,)
    assert metrics["horizon_loss_sum"].shape == (model.horizon,)
    assert metrics["horizon_valid_count"].shape == (model.horizon,)


def test_horizon_loss_weights_change_only_the_optimization_objective():
    base = load_config(REPO_ROOT / "configs" / "tiny_smoke.yaml")
    model_config = replace(
        base.model,
        finetune_mode="full",
        qwen_fusion="self_attention_kv",
        rdt_state_dim=128,
        state_encoder_layout="rdt_eef",
        freeze_state_adaptor=True,
        allow_random_frozen_state_adaptor=True,
    )
    model = SFTConditionedRDT(replace(base, model=model_config), load_pretrained=False)

    def zero_prediction(self, x, *_args, **_kwargs):
        return x.new_zeros(x.shape[0], model.horizon, 7)

    model.runner.model.forward = MethodType(zero_prediction, model.runner.model)
    actions = torch.ones(1, model.horizon, 7)
    actions[:, 0] = 2.0
    batch = {
        "lang_tokens": torch.randn(1, 12, 64),
        "lang_mask": torch.ones(1, 12, dtype=torch.bool),
        "img_tokens": torch.randn(1, 16, 64),
        "img_mask": torch.ones(1, 16, dtype=torch.bool),
        "qwen_kv": torch.randn(1, 1, 64),
        "state": torch.zeros(1, 7),
        "actions": actions,
        "action_time_mask": torch.ones(1, model.horizon, dtype=torch.bool),
        "action_dim_mask": torch.ones(1, 7),
        "horizon_loss_weights": torch.tensor([5.0] + [1.0] * (model.horizon - 1)),
        "ctrl_freq": torch.full((1,), 10.0),
    }

    metrics = model(batch)
    expected_unweighted = (4.0 + (model.horizon - 1)) / model.horizon
    expected_weighted = (5.0 * 4.0 + (model.horizon - 1)) / (
        5.0 + model.horizon - 1
    )
    torch.testing.assert_close(
        metrics["sample_unweighted_imitation_loss"],
        torch.tensor([expected_unweighted]),
    )
    torch.testing.assert_close(metrics["loss"], torch.tensor(expected_weighted))


def test_auxiliary_gripper_bce_uses_existing_clean_output_without_a_head():
    base = load_config(REPO_ROOT / "configs" / "tiny_smoke.yaml")
    model_config = replace(
        base.model,
        finetune_mode="full",
        qwen_fusion="self_attention_kv",
        rdt_state_dim=128,
        state_encoder_layout="rdt_eef",
        freeze_state_adaptor=True,
        allow_random_frozen_state_adaptor=True,
    )
    model = SFTConditionedRDT(replace(base, model=model_config), load_pretrained=False)

    def zero_prediction(self, x, *_args, **_kwargs):
        return x.new_zeros(x.shape[0], model.horizon, 7)

    model.runner.model.forward = MethodType(zero_prediction, model.runner.model)
    actions = torch.zeros(1, model.horizon, 7)
    actions[..., 6] = 1.0
    batch = {
        "lang_tokens": torch.randn(1, 12, 64),
        "lang_mask": torch.ones(1, 12, dtype=torch.bool),
        "img_tokens": torch.randn(1, 16, 64),
        "img_mask": torch.ones(1, 16, dtype=torch.bool),
        "qwen_kv": torch.randn(1, 1, 64),
        "state": torch.zeros(1, 7),
        "actions": actions,
        "action_time_mask": torch.ones(1, model.horizon, dtype=torch.bool),
        "action_dim_mask": torch.ones(1, 7),
        "gripper_bce_weight": torch.tensor(2.0),
        "gripper_bce_logit_scale": torch.tensor(5.0),
        "ctrl_freq": torch.full((1,), 10.0),
    }

    metrics = model(batch)
    expected_imitation = torch.tensor(1.0 / 7.0)
    expected_bce = torch.log(torch.tensor(2.0))
    torch.testing.assert_close(metrics["imitation_loss"], expected_imitation)
    torch.testing.assert_close(metrics["gripper_bce_loss"], expected_bce)
    torch.testing.assert_close(
        metrics["loss"],
        expected_imitation + 2.0 * expected_bce,
    )

    batch["gripper_loss_weights"] = torch.full((1, model.horizon), 5.0)
    weighted_metrics = model(batch)
    torch.testing.assert_close(
        weighted_metrics["imitation_loss"],
        torch.tensor(5.0 / 7.0),
    )
    torch.testing.assert_close(
        weighted_metrics["gripper_bce_loss"],
        5.0 * expected_bce,
    )


def test_noisy_gripper_mask_is_applied_in_training_and_sampling() -> None:
    base = load_config(REPO_ROOT / "configs" / "tiny_smoke.yaml")
    model_config = replace(
        base.model,
        finetune_mode="full",
        qwen_fusion="self_attention_kv",
        rdt_state_dim=128,
        state_encoder_layout="rdt_eef",
        freeze_state_adaptor=True,
        allow_random_frozen_state_adaptor=True,
    )
    model = SFTConditionedRDT(replace(base, model=model_config), load_pretrained=False)
    model.mask_noisy_gripper_input = True

    def zero_prediction(self, x, *_args, **_kwargs):
        prediction = x.new_zeros(x.shape[0], model.horizon, 7)
        prediction[..., 6] = -0.75
        return prediction

    model.runner.model.forward = MethodType(zero_prediction, model.runner.model)
    captured_action_inputs: list[torch.Tensor] = []
    captured_action_masks: list[torch.Tensor] = []
    original_adapt_actions = model._adapt_actions

    def capture_action_input(self, actions, action_mask):
        captured_action_inputs.append(actions.detach().clone())
        captured_action_masks.append(action_mask.detach().clone())
        return original_adapt_actions(actions, action_mask)

    model._adapt_actions = MethodType(capture_action_input, model)
    batch = {
        "lang_tokens": torch.randn(1, 12, 64),
        "lang_mask": torch.ones(1, 12, dtype=torch.bool),
        "img_tokens": torch.randn(1, 16, 64),
        "img_mask": torch.ones(1, 16, dtype=torch.bool),
        "qwen_kv": torch.randn(1, 1, 64),
        "state": torch.zeros(1, 7),
        "actions": torch.ones(1, model.horizon, 7),
        "action_time_mask": torch.ones(1, model.horizon, dtype=torch.bool),
        "action_dim_mask": torch.ones(1, 7),
        "ctrl_freq": torch.full((1,), 10.0),
        "diffusion_noise": torch.randn(1, model.horizon, 7),
        "diffusion_timesteps": torch.tensor(
            [model.runner.num_train_timesteps - 1]
        ),
        "return_denoising_prediction": True,
    }

    metrics = model(batch)
    assert "denoising_prediction" in metrics
    assert torch.count_nonzero(captured_action_inputs[-1][..., 6]) == 0
    assert torch.count_nonzero(captured_action_inputs[-1][..., :6]) > 0

    # Episode padding is a loss-only concern. Changing only the temporal loss
    # mask must not change the action values or indicator mask presented to the
    # diffusion model; inference always conditions on a full action horizon.
    full_horizon_action_input = captured_action_inputs[-1].clone()
    full_horizon_conditioning_mask = captured_action_masks[-1].clone()
    batch["action_time_mask"][:, model.horizon // 2 :] = False
    model(batch)
    torch.testing.assert_close(
        captured_action_inputs[-1],
        full_horizon_action_input,
    )
    torch.testing.assert_close(
        captured_action_masks[-1],
        full_horizon_conditioning_mask,
    )
    torch.testing.assert_close(
        captured_action_masks[-1],
        batch["action_dim_mask"].unsqueeze(1).expand_as(batch["actions"]),
    )

    captured_action_inputs.clear()
    captured_action_masks.clear()
    sampling_batch = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "actions",
            "action_time_mask",
            "diffusion_noise",
            "diffusion_timesteps",
            "return_denoising_prediction",
        }
    }
    sampled = model.sample_actions(sampling_batch)
    assert captured_action_inputs
    assert all(
        torch.count_nonzero(action_input[..., 6]) == 0
        for action_input in captured_action_inputs
    )
    # The masked discrete channel is decoded from the model's final clean x0,
    # rather than the continuously integrated solver state.
    torch.testing.assert_close(
        sampled[..., 6],
        torch.full_like(sampled[..., 6], -0.75),
    )

    # Output-only clean-x0 decoding must not alter the noisy gripper input. This
    # supports older checkpoints that were trained without gripper masking.
    model.mask_noisy_gripper_input = False
    model.decode_clean_x0_gripper = True
    captured_action_inputs.clear()
    sampled = model.sample_actions(sampling_batch)
    assert any(
        torch.count_nonzero(action_input[..., 6]) > 0
        for action_input in captured_action_inputs
    )
    torch.testing.assert_close(
        sampled[..., 6],
        torch.full_like(sampled[..., 6], -0.75),
    )


def test_ortho6d_so3_penalty_measures_decoded_rotation() -> None:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    quarter_turn_z = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
    matrices = SFTConditionedRDT._ortho6d_to_rotation_matrix(
        torch.stack([identity, quarter_turn_z])
    )

    relative = matrices[0].transpose(-1, -2) @ matrices[1]
    cosine = (relative.diagonal().sum() - 1.0) * 0.5
    penalty = 1.0 - cosine

    torch.testing.assert_close(matrices[0], torch.eye(3))
    torch.testing.assert_close(penalty, torch.tensor(1.0))
