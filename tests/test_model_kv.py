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
    external_kv = torch.randn(2, 1, 32, requires_grad=True)

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
        qwen_fusion="self_attention_kv",
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
        "qwen_kv": torch.randn(batch_size, 1, 64),
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
