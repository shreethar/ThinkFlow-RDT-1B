from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import thinkflow_rdt.checkpoint as checkpoint
from thinkflow_rdt.checkpoint import (
    ADAPTER_DIR,
    FULL_RDT_FILE,
    INTERFACE_FILE,
    METADATA_FILE,
    load_full_rdt_base,
    load_trainable_artifact,
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
