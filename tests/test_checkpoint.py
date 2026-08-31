from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import thinkflow_rdt.checkpoint as checkpoint
from thinkflow_rdt.checkpoint import (
    ADAPTER_DIR,
    FULL_RDT_FILE,
    INTERFACE_FILE,
    METADATA_FILE,
    TRAINER_STATE_FILE,
    load_trainer_state,
    load_full_rdt_base,
    load_trainable_artifact,
    save_trainer_state,
    save_trainable_artifact,
)


class FakeFullCore(nn.Module):
    """Full RDT stand-in that also has the hub model save capability."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Linear(3, 3)

    def save_pretrained(self, _path: str | Path) -> None:
        raise AssertionError("A non-PEFT model must use the full-state artifact")


class FakeTransferCore(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 4,
        lang_tokens: int = 128,
    ) -> None:
        super().__init__()
        self.lang_cond_pos_embed = nn.Parameter(
            torch.zeros(1, lang_tokens, hidden_dim)
        )
        self.block = nn.Linear(hidden_dim, hidden_dim)
        self.final_layer = nn.Module()
        self.final_layer.ffn_final = nn.Module()
        self.final_layer.ffn_final.fc2 = nn.Linear(hidden_dim, action_dim)


class FakeLoraCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_layer = nn.Linear(3, 3)
        self.lora_A = nn.Linear(3, 2, bias=False)
        self.modules_to_save = nn.Linear(3, 3)
        self.peft_config = {"default": object()}

    def save_pretrained(
        self,
        path: str | Path,
        *,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.state_dict() if state_dict is None else state_dict,
            output / "adapter_model.pt",
        )


class FakeRunner(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.model = core
        self.lang_adaptor = nn.Linear(4, 3)
        self.img_adaptor = nn.Linear(5, 3)
        self.state_adaptor = nn.Linear(6, 3)


class FakeConditionedRDT(nn.Module):
    def __init__(self, core: nn.Module, finetune_mode: str | None) -> None:
        super().__init__()
        model_config = SimpleNamespace()
        if finetune_mode is not None:
            model_config.finetune_mode = finetune_mode
        self.cfg = SimpleNamespace(model=model_config)
        self.qwen_adaptor = nn.Linear(7, 3)
        self.runner = FakeRunner(core)


def add_hidden_waypoint_interface(model: FakeConditionedRDT) -> None:
    model.cfg.model.qwen_fusion = "hidden_waypoint_cross_attention"
    model.cfg.model.conditioning_variant = "b2"
    model.cfg.model.spatial_token_count = 5
    model.cfg.model.qwen_hidden_size = 8
    model.cfg.model.waypoint_dim = 2
    model.cfg.model.waypoint_embed_dim = 4
    model.cfg.model.hidden_size = 6
    model.plan_hidden_norm = nn.LayerNorm(8)
    model.waypoint_adaptor = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 4))
    model.plan_adaptor = nn.Sequential(nn.Linear(12, 6), nn.Linear(6, 6))
    model.plan_type_embedding = nn.Parameter(torch.zeros(1, 1, 6))
    model.plan_position_embedding = nn.Parameter(torch.zeros(1, 5, 6))


def add_b0_hidden_interface(model: FakeConditionedRDT) -> None:
    model.cfg.model.qwen_fusion = "hidden_cross_attention"
    model.cfg.model.conditioning_variant = "b0"
    model.cfg.model.spatial_token_count = 1
    model.cfg.model.qwen_hidden_size = 8
    model.cfg.model.hidden_size = 6
    model.plan_hidden_norm = nn.LayerNorm(8)
    model.plan_adaptor = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 6))
    model.plan_type_embedding = nn.Parameter(torch.zeros(1, 1, 6))
    model.plan_position_embedding = nn.Parameter(torch.zeros(1, 1, 6))


def fill_parameters(model: nn.Module, start: float) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_(start + index)


def clone_states(model: FakeConditionedRDT) -> dict[str, dict[str, torch.Tensor]]:
    modules = {
        "core": model.runner.model,
        "qwen": model.qwen_adaptor,
        "lang": model.runner.lang_adaptor,
        "img": model.runner.img_adaptor,
        "state": model.runner.state_adaptor,
    }
    return {
        name: {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
        }
        for name, module in modules.items()
    }


def assert_states_equal(
    model: FakeConditionedRDT,
    expected: dict[str, dict[str, torch.Tensor]],
) -> None:
    modules = {
        "core": model.runner.model,
        "qwen": model.qwen_adaptor,
        "lang": model.runner.lang_adaptor,
        "img": model.runner.img_adaptor,
        "state": model.runner.state_adaptor,
    }
    for module_name, module in modules.items():
        actual_state = module.state_dict()
        assert actual_state.keys() == expected[module_name].keys()
        for key, expected_tensor in expected[module_name].items():
            torch.testing.assert_close(actual_state[key], expected_tensor)


def stochastic_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.randn(8, 3)
    inputs += random.random() + float(np.random.random())
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def test_trainer_state_resume_matches_uninterrupted_next_update(tmp_path):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    uninterrupted = nn.Sequential(nn.Linear(3, 5), nn.Dropout(0.25), nn.Linear(5, 2))
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-3)
    uninterrupted_scheduler = torch.optim.lr_scheduler.LambdaLR(
        uninterrupted_optimizer,
        lambda step: 0.9**step,
    )

    stochastic_optimizer_step(
        uninterrupted,
        uninterrupted_optimizer,
        uninterrupted_scheduler,
    )
    boundary_model_state = {
        key: value.detach().clone()
        for key, value in uninterrupted.state_dict().items()
    }
    progress = {
        "global_step": 1,
        "epoch": 4,
        "batches_consumed_in_epoch": 17,
    }
    contract = {"world_size": 1, "effective_global_batch": 32}
    save_trainer_state(
        tmp_path,
        optimizer=uninterrupted_optimizer,
        scheduler=uninterrupted_scheduler,
        scaler=None,
        progress=progress,
        resume_contract=contract,
        process_index=0,
        num_processes=1,
        is_main_process=True,
    )
    assert (tmp_path / TRAINER_STATE_FILE).is_file()

    expected_loss = stochastic_optimizer_step(
        uninterrupted,
        uninterrupted_optimizer,
        uninterrupted_scheduler,
    )
    expected_model_state = {
        key: value.detach().clone()
        for key, value in uninterrupted.state_dict().items()
    }

    resumed = nn.Sequential(nn.Linear(3, 5), nn.Dropout(0.25), nn.Linear(5, 2))
    resumed.load_state_dict(boundary_model_state)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
        resumed_optimizer,
        lambda step: 0.9**step,
    )
    restored_progress = load_trainer_state(
        tmp_path,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=None,
        process_index=0,
        num_processes=1,
        expected_contract=contract,
    )
    resumed_loss = stochastic_optimizer_step(
        resumed,
        resumed_optimizer,
        resumed_scheduler,
    )

    assert restored_progress == progress
    assert resumed_loss == expected_loss
    assert resumed_scheduler.state_dict() == uninterrupted_scheduler.state_dict()
    for name, expected in expected_model_state.items():
        torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)


def test_exact_resume_rejects_changed_contract(tmp_path):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    save_trainer_state(
        tmp_path,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        progress={"global_step": 1},
        resume_contract={"micro_batch_size": 8},
        process_index=0,
        num_processes=1,
        is_main_process=True,
    )

    with pytest.raises(ValueError, match="micro_batch_size"):
        load_trainer_state(
            tmp_path,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            process_index=0,
            num_processes=1,
            expected_contract={"micro_batch_size": 16},
        )


def test_full_rdt_artifact_round_trip_includes_qwen_and_interfaces(tmp_path):
    source = FakeConditionedRDT(FakeFullCore(), finetune_mode="full")
    fill_parameters(source, start=1.0)
    expected = clone_states(source)

    save_trainable_artifact(source, tmp_path, {"global_step": 17})

    assert (tmp_path / FULL_RDT_FILE).is_file()
    assert not (tmp_path / ADAPTER_DIR).exists()
    assert json.loads((tmp_path / METADATA_FILE).read_text()) == {"global_step": 17}
    interfaces = torch.load(tmp_path / INTERFACE_FILE, weights_only=True)
    assert "qwen_adaptor" in interfaces

    restored = FakeConditionedRDT(FakeFullCore(), finetune_mode="full")
    fill_parameters(restored, start=-10.0)
    load_trainable_artifact(restored, tmp_path, trainable=False)

    assert_states_equal(restored, expected)
    assert all(not parameter.requires_grad for parameter in restored.parameters())


def test_artifact_restores_noisy_gripper_mask_behavior(tmp_path):
    source = FakeConditionedRDT(FakeFullCore(), finetune_mode="full")
    save_trainable_artifact(
        source,
        tmp_path,
        {"mask_noisy_gripper_input": True},
    )

    restored = FakeConditionedRDT(FakeFullCore(), finetune_mode="full")
    restored.mask_noisy_gripper_input = False
    load_trainable_artifact(restored, tmp_path, trainable=False)

    assert restored.mask_noisy_gripper_input is True


def test_non_peft_core_uses_full_state_even_if_it_has_save_pretrained(tmp_path):
    model = FakeConditionedRDT(FakeFullCore(), finetune_mode=None)

    save_trainable_artifact(model, tmp_path, {})

    assert (tmp_path / FULL_RDT_FILE).is_file()
    assert not (tmp_path / ADAPTER_DIR).exists()


def test_full_artifact_accepts_accelerator_gathered_state(tmp_path):
    source = FakeConditionedRDT(FakeFullCore(), finetune_mode="full")
    fill_parameters(source, start=3.0)
    expected = clone_states(source)

    save_trainable_artifact(
        source,
        tmp_path,
        {"global_step": 9},
        model_state_dict=source.state_dict(),
    )

    restored = FakeConditionedRDT(FakeFullCore(), finetune_mode="full")
    fill_parameters(restored, start=-30.0)
    load_trainable_artifact(restored, tmp_path, trainable=False)
    assert_states_equal(restored, expected)


def test_hidden_waypoint_interfaces_and_metadata_round_trip(tmp_path):
    source = FakeConditionedRDT(FakeFullCore(), finetune_mode="full")
    add_hidden_waypoint_interface(source)
    fill_parameters(source, start=5.0)
    expected = {
        name: tensor.detach().clone()
        for name, tensor in source.state_dict().items()
    }

    save_trainable_artifact(
        source,
        tmp_path,
        {"global_step": 3},
        model_state_dict=source.state_dict(),
    )

    interfaces = torch.load(tmp_path / INTERFACE_FILE, weights_only=True)
    for name in (
        "plan_hidden_norm",
        "waypoint_adaptor",
        "plan_adaptor",
        "plan_type_embedding",
        "plan_position_embedding",
    ):
        assert name in interfaces
    metadata = json.loads((tmp_path / METADATA_FILE).read_text())
    assert metadata["conditioning_interface"] == {
        "type": "hidden_waypoint_cross_attention",
        "conditioning_variant": "b2",
        "cached_kv_retained_but_unused": True,
        "spatial_token_count": 5,
        "qwen_hidden_size": 8,
        "waypoint_dim": 2,
        "waypoint_embed_dim": 4,
        "rdt_condition_dim": 6,
    }

    restored = FakeConditionedRDT(FakeFullCore(), finetune_mode="full")
    add_hidden_waypoint_interface(restored)
    fill_parameters(restored, start=-5.0)
    load_trainable_artifact(restored, tmp_path, trainable=False)

    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, expected[name])
    assert all(not parameter.requires_grad for parameter in restored.parameters())


def test_b0_hidden_interfaces_and_metadata_round_trip(tmp_path):
    source = FakeConditionedRDT(FakeFullCore(), finetune_mode="full")
    add_b0_hidden_interface(source)
    fill_parameters(source, start=7.0)
    expected = {
        name: tensor.detach().clone()
        for name, tensor in source.state_dict().items()
    }

    save_trainable_artifact(
        source,
        tmp_path,
        {"global_step": 4},
        model_state_dict=source.state_dict(),
    )

    metadata = json.loads((tmp_path / METADATA_FILE).read_text())
    assert metadata["conditioning_interface"] == {
        "type": "hidden_cross_attention",
        "cached_kv_retained_but_unused": True,
        "spatial_token_count": 1,
        "qwen_hidden_size": 8,
        "rdt_condition_dim": 6,
        "qwen_token_selector": "think_end",
        "conditioning_variant": "b0",
    }

    restored = FakeConditionedRDT(FakeFullCore(), finetune_mode="full")
    add_b0_hidden_interface(restored)
    fill_parameters(restored, start=-7.0)
    load_trainable_artifact(restored, tmp_path, trainable=False)

    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, expected[name])


def test_b0_artifact_can_initialize_new_plan_modules_but_resume_stays_strict(
    tmp_path,
):
    source = FakeConditionedRDT(FakeFullCore(), finetune_mode="full")
    fill_parameters(source, start=2.0)
    save_trainable_artifact(source, tmp_path, {"global_step": 20})

    target = FakeConditionedRDT(FakeFullCore(), finetune_mode="full")
    add_hidden_waypoint_interface(target)
    plan_before = {
        name: tensor.detach().clone()
        for name, tensor in target.state_dict().items()
        if name.startswith("plan_") or name.startswith("waypoint_adaptor.")
    }

    with pytest.raises(KeyError, match="plan"):
        load_trainable_artifact(target, tmp_path, trainable=True)

    load_trainable_artifact(
        target,
        tmp_path,
        trainable=True,
        allow_missing_plan_interfaces=True,
    )
    for name, expected in plan_before.items():
        torch.testing.assert_close(target.state_dict()[name], expected)
    assert all(parameter.requires_grad for parameter in target.parameters())


def test_full_base_can_reinitialize_only_a_changed_action_output_head(tmp_path):
    source_core = FakeTransferCore(action_dim=7)
    fill_parameters(source_core, start=3.0)
    torch.save(source_core.state_dict(), tmp_path / FULL_RDT_FILE)

    target_core = FakeTransferCore(action_dim=10)
    original_head = {
        name: tensor.detach().clone()
        for name, tensor in target_core.final_layer.ffn_final.fc2.state_dict().items()
    }
    model = SimpleNamespace(runner=SimpleNamespace(model=target_core))
    report = load_full_rdt_base(
        model,
        tmp_path,
        allow_output_head_mismatch=True,
    )

    torch.testing.assert_close(target_core.block.weight, source_core.block.weight)
    torch.testing.assert_close(target_core.block.bias, source_core.block.bias)
    for name, tensor in target_core.final_layer.ffn_final.fc2.state_dict().items():
        torch.testing.assert_close(tensor, original_head[name])
    assert report["reinitialized_tensors"] == [
        "final_layer.ffn_final.fc2.bias",
        "final_layer.ffn_final.fc2.weight",
    ]


def test_full_base_still_rejects_other_shape_mismatches(tmp_path):
    source_core = FakeTransferCore(action_dim=7, hidden_dim=4)
    torch.save(source_core.state_dict(), tmp_path / FULL_RDT_FILE)
    target_core = FakeTransferCore(action_dim=10, hidden_dim=5)
    model = SimpleNamespace(runner=SimpleNamespace(model=target_core))

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_full_rdt_base(
            model,
            tmp_path,
            allow_output_head_mismatch=True,
        )


def test_full_base_truncates_legacy_extra_language_position(tmp_path):
    source_core = FakeTransferCore(action_dim=7, lang_tokens=129)
    fill_parameters(source_core, start=3.0)
    torch.save(source_core.state_dict(), tmp_path / FULL_RDT_FILE)

    target_core = FakeTransferCore(action_dim=10, lang_tokens=128)
    original_head = {
        name: tensor.detach().clone()
        for name, tensor in target_core.final_layer.ffn_final.fc2.state_dict().items()
    }
    model = SimpleNamespace(runner=SimpleNamespace(model=target_core))
    report = load_full_rdt_base(
        model,
        tmp_path,
        allow_output_head_mismatch=True,
        allow_language_position_mismatch=True,
    )

    torch.testing.assert_close(
        target_core.lang_cond_pos_embed,
        source_core.lang_cond_pos_embed[:, :128],
    )
    for name, tensor in target_core.final_layer.ffn_final.fc2.state_dict().items():
        torch.testing.assert_close(tensor, original_head[name])
    assert report["adapted_tensors"] == ["lang_cond_pos_embed"]


def test_lora_artifact_round_trip_preserves_adapter_and_qwen(
    tmp_path,
    monkeypatch,
):
    def fake_load_peft_weights(path: str, device: str):
        assert device == "cpu"
        return torch.load(Path(path) / "adapter_model.pt", weights_only=True)

    def fake_set_peft_model_state_dict(model, state, adapter_name: str):
        assert adapter_name == "default"
        model.load_state_dict(state)

    monkeypatch.setattr(checkpoint, "load_peft_weights", fake_load_peft_weights)
    monkeypatch.setattr(
        checkpoint,
        "set_peft_model_state_dict",
        fake_set_peft_model_state_dict,
    )

    source = FakeConditionedRDT(FakeLoraCore(), finetune_mode=None)
    fill_parameters(source, start=2.0)
    expected = clone_states(source)

    save_trainable_artifact(source, tmp_path, {"global_step": 4})

    assert (tmp_path / ADAPTER_DIR / "adapter_model.pt").is_file()
    assert not (tmp_path / FULL_RDT_FILE).exists()

    restored = FakeConditionedRDT(FakeLoraCore(), finetune_mode=None)
    fill_parameters(restored, start=-20.0)
    load_trainable_artifact(restored, tmp_path, trainable=False)

    assert_states_equal(restored, expected)
    assert all(not parameter.requires_grad for parameter in restored.parameters())


def test_load_rejects_artifact_without_qwen_projector(tmp_path, monkeypatch):
    adapter_dir = tmp_path / ADAPTER_DIR
    adapter_dir.mkdir()
    torch.save({}, adapter_dir / "adapter_model.pt")
    torch.save(
        {
            "lang_adaptor": nn.Linear(4, 3).state_dict(),
            "img_adaptor": nn.Linear(5, 3).state_dict(),
            "state_adaptor": nn.Linear(6, 3).state_dict(),
        },
        tmp_path / INTERFACE_FILE,
    )
    monkeypatch.setattr(checkpoint, "load_peft_weights", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        checkpoint,
        "set_peft_model_state_dict",
        lambda *_args, **_kwargs: None,
    )

    model = FakeConditionedRDT(FakeLoraCore(), finetune_mode=None)
    with pytest.raises(KeyError, match="qwen_adaptor"):
        load_trainable_artifact(model, tmp_path, trainable=False)
