from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict


INTERFACE_FILE = "interfaces.pt"
METADATA_FILE = "metadata.json"
ADAPTER_DIR = "rdt_lora"
FULL_RDT_FILE = "rdt_full.pt"

_FORMAT_KEY = "_rdt_artifact_format"
_LORA_FORMAT = "lora"
_FULL_FORMAT = "full"
_LORA_MODES = {"lora", "peft", "adapter"}
_FULL_MODES = {"full", "full_rdt", "full-rdt", "full_finetune", "full_finetuning"}


def _configured_artifact_format(model) -> str | None:
    """Return an explicitly configured fine-tuning format, when available."""
    model_config = getattr(getattr(model, "cfg", None), "model", None)
    mode = getattr(model_config, "finetune_mode", None)
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    if normalized in _LORA_MODES:
        return _LORA_FORMAT
    if normalized in _FULL_MODES:
        return _FULL_FORMAT
    raise ValueError(f"Unsupported model.finetune_mode for checkpointing: {mode!r}")


def _is_peft_model(module) -> bool:
    """Detect PEFT without mistaking hub models' save_pretrained for LoRA."""
    return (
        isinstance(module, PeftModel)
        or getattr(module, "peft_config", None) is not None
    )


def _artifact_format_for_save(model) -> str:
    configured = _configured_artifact_format(model)
    if configured is not None:
        return configured
    return _LORA_FORMAT if _is_peft_model(model.runner.model) else _FULL_FORMAT


def _component_state(
    model_state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    component = {
        name[len(prefix) :]: tensor
        for name, tensor in model_state_dict.items()
        if name.startswith(prefix)
    }
    if not component:
        raise KeyError(f"Full model state has no tensors under prefix {prefix!r}")
    return component


def _interface_state(
    model,
    artifact_format: str,
    model_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if model_state_dict is not None:
        state = {
            _FORMAT_KEY: artifact_format,
            "qwen_adaptor": _component_state(
                model_state_dict, "qwen_adaptor."
            ),
            "lang_adaptor": _component_state(
                model_state_dict, "runner.lang_adaptor."
            ),
            "img_adaptor": _component_state(
                model_state_dict, "runner.img_adaptor."
            ),
            "state_adaptor": _component_state(
                model_state_dict, "runner.state_adaptor."
            ),
        }
        if getattr(model, "action_adaptor", None) is not None:
            state["action_adaptor"] = _component_state(
                model_state_dict, "action_adaptor."
            )
        if "unified_cross_extra_pos_embed" in model_state_dict:
            state["unified_cross_extra_pos_embed"] = model_state_dict[
                "unified_cross_extra_pos_embed"
            ]
        return state
    state = {
        _FORMAT_KEY: artifact_format,
        "qwen_adaptor": model.qwen_adaptor.state_dict(),
        "lang_adaptor": model.runner.lang_adaptor.state_dict(),
        "img_adaptor": model.runner.img_adaptor.state_dict(),
        "state_adaptor": model.runner.state_adaptor.state_dict(),
    }
    if getattr(model, "action_adaptor", None) is not None:
        state["action_adaptor"] = model.action_adaptor.state_dict()
    if getattr(model, "unified_cross_extra_pos_embed", None) is not None:
        state["unified_cross_extra_pos_embed"] = (
            model.unified_cross_extra_pos_embed.detach().cpu()
        )
    return state


def _load_interfaces(model, interfaces: dict[str, Any], *, trainable: bool) -> None:
    required = {
        "qwen_adaptor",
        "lang_adaptor",
        "img_adaptor",
        "state_adaptor",
    }
    action_adaptor = getattr(model, "action_adaptor", None)
    checkpoint_has_action_adaptor = "action_adaptor" in interfaces
    if checkpoint_has_action_adaptor != (action_adaptor is not None):
        raise ValueError(
            "Checkpoint/model action-adaptor layouts differ; use the same "
            "model.action_encoder_layout that created the artifact"
        )
    if action_adaptor is not None:
        required.add("action_adaptor")
    missing = required.difference(interfaces)
    if missing:
        raise KeyError(
            f"{INTERFACE_FILE} is missing checkpoint components: {sorted(missing)}"
        )

    modules = {
        "qwen_adaptor": model.qwen_adaptor,
        "lang_adaptor": model.runner.lang_adaptor,
        "img_adaptor": model.runner.img_adaptor,
        "state_adaptor": model.runner.state_adaptor,
    }
    if action_adaptor is not None:
        modules["action_adaptor"] = action_adaptor
    for name, module in modules.items():
        module.load_state_dict(interfaces[name])
        if not trainable:
            module.requires_grad_(False)
    if (
        getattr(model, "unified_cross_extra_pos_embed", None) is not None
        and "unified_cross_extra_pos_embed" in interfaces
    ):
        model.unified_cross_extra_pos_embed.data.copy_(
            interfaces["unified_cross_extra_pos_embed"].to(
                device=model.unified_cross_extra_pos_embed.device,
                dtype=model.unified_cross_extra_pos_embed.dtype,
            )
        )
        model.unified_cross_extra_pos_embed.requires_grad = trainable


def save_trainable_artifact(
    model,
    output_dir: str | Path,
    metadata: dict[str, Any],
    *,
    model_state_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact_format = _artifact_format_for_save(model)
    if artifact_format == _LORA_FORMAT:
        if not _is_peft_model(model.runner.model):
            raise TypeError(
                "model.finetune_mode='lora' requires a PEFT-wrapped RDT model"
            )
        save_pretrained = getattr(model.runner.model, "save_pretrained", None)
        if not callable(save_pretrained):
            raise TypeError("LoRA RDT model does not implement save_pretrained")
        if model_state_dict is None:
            save_pretrained(output / ADAPTER_DIR)
        else:
            save_pretrained(
                output / ADAPTER_DIR,
                state_dict=_component_state(
                    model_state_dict, "runner.model."
                ),
            )
    else:
        if _is_peft_model(model.runner.model):
            raise TypeError(
                "model.finetune_mode='full' requires a non-PEFT RDT model"
            )
        rdt_state = (
            model.runner.model.state_dict()
            if model_state_dict is None
            else _component_state(model_state_dict, "runner.model.")
        )
        torch.save(rdt_state, output / FULL_RDT_FILE)

    torch.save(
        _interface_state(model, artifact_format, model_state_dict),
        output / INTERFACE_FILE,
    )
    with (output / METADATA_FILE).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_trainable_artifact(model, artifact_dir: str | Path, trainable: bool) -> None:
    artifact = Path(artifact_dir)
    interfaces_path = artifact / INTERFACE_FILE
    if not interfaces_path.exists():
        raise FileNotFoundError(interfaces_path)
    interfaces = torch.load(
        interfaces_path,
        map_location="cpu",
        weights_only=True,
    )

    adapter_dir = artifact / ADAPTER_DIR
    full_rdt_path = artifact / FULL_RDT_FILE
    artifact_format = interfaces.get(_FORMAT_KEY)
    if artifact_format is None:
        # Compatibility with earlier LoRA artifacts, whose interfaces file had
        # no format marker. Such artifacts still need a Qwen projector to be
        # usable with the current model contract.
        if adapter_dir.exists() and not full_rdt_path.exists():
            artifact_format = _LORA_FORMAT
        elif full_rdt_path.exists() and not adapter_dir.exists():
            artifact_format = _FULL_FORMAT
        else:
            artifact_format = _configured_artifact_format(model)

    if artifact_format == _LORA_FORMAT:
        load_lora_rdt_core(model, artifact, trainable=trainable)
    elif artifact_format == _FULL_FORMAT:
        if not full_rdt_path.exists():
            raise FileNotFoundError(full_rdt_path)
        if _is_peft_model(model.runner.model):
            raise TypeError("A full-RDT artifact requires a non-PEFT RDT model")
        state_dict = torch.load(
            full_rdt_path,
            map_location="cpu",
            weights_only=True,
        )
        model.runner.model.load_state_dict(state_dict)
        model.runner.model.requires_grad_(trainable)
    else:
        raise ValueError(f"Unsupported RDT artifact format: {artifact_format!r}")

    _load_interfaces(model, interfaces, trainable=trainable)
    metadata_path = artifact / METADATA_FILE
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if "mask_noisy_gripper_input" in metadata:
            model.mask_noisy_gripper_input = bool(
                metadata["mask_noisy_gripper_input"]
            )


def load_lora_rdt_core(model, artifact_dir: str | Path, *, trainable: bool) -> None:
    """Load only an RDT LoRA adapter, without touching interface modules.

    This is useful when an old checkpoint's Qwen/interface shapes no longer
    match the current fusion design, but its RDT LoRA weights should still be
    merged into the base transformer.
    """
    artifact = Path(artifact_dir)
    adapter_dir = artifact / ADAPTER_DIR
    if not adapter_dir.exists():
        raise FileNotFoundError(adapter_dir)
    if not _is_peft_model(model.runner.model):
        raise TypeError("A LoRA artifact requires a PEFT-wrapped RDT model")
    # The model constructor has already created the same default LoRA layout.
    # Load into it instead of wrapping the model a second time.
    adapter_state = load_peft_weights(str(adapter_dir), device="cpu")
    set_peft_model_state_dict(
        model.runner.model,
        adapter_state,
        adapter_name="default",
    )
    if not trainable:
        model.runner.model.requires_grad_(False)
    for name, parameter in model.runner.model.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            parameter.requires_grad = trainable


def load_full_rdt_base(
    model,
    artifact_dir: str | Path,
    *,
    strict: bool = True,
    allow_output_head_mismatch: bool = False,
    allow_language_position_mismatch: bool = False,
) -> dict[str, Any]:
    """Load a merged/full RDT core before wrapping the core with fresh LoRA.

    Use this for two-stage fine-tuning: first merge an earlier LoRA run into the
    base RDT transformer, then initialize a new LoRA adapter for the next
    dataset. This intentionally loads only ``rdt_full.pt``; interface modules
    such as the Qwen projector may have changed shape between fusion designs.
    """
    artifact = Path(artifact_dir)
    full_rdt_path = artifact / FULL_RDT_FILE
    if not full_rdt_path.exists():
        raise FileNotFoundError(full_rdt_path)
    if _is_peft_model(model.runner.model):
        raise TypeError("load_full_rdt_base must run before applying PEFT/LoRA")
    state_dict = torch.load(
        full_rdt_path,
        map_location="cpu",
        weights_only=True,
    )
    target_state = model.runner.model.state_dict()
    shape_mismatches = {
        name: (tuple(tensor.shape), tuple(target_state[name].shape))
        for name, tensor in state_dict.items()
        if name in target_state and tensor.shape != target_state[name].shape
    }
    if not shape_mismatches:
        model.runner.model.load_state_dict(state_dict, strict=strict)
        return {
            "loaded_tensors": len(state_dict),
            "reinitialized_tensors": [],
            "adapted_tensors": [],
            "shape_mismatches": {},
        }

    output_head_keys = {
        "final_layer.ffn_final.fc2.weight",
        "final_layer.ffn_final.fc2.bias",
    }
    language_position_key = "lang_cond_pos_embed"
    permitted_mismatches: set[str] = set()
    if allow_output_head_mismatch:
        permitted_mismatches.update(output_head_keys)
    if allow_language_position_mismatch:
        permitted_mismatches.add(language_position_key)
    if not set(shape_mismatches).issubset(permitted_mismatches):
        # Preserve PyTorch's detailed size-mismatch error for any unsupported
        # architecture difference.
        model.runner.model.load_state_dict(state_dict, strict=strict)
        raise AssertionError("unreachable")

    mismatched_output_keys = set(shape_mismatches).intersection(output_head_keys)
    if mismatched_output_keys:
        source_weight = state_dict["final_layer.ffn_final.fc2.weight"]
        target_weight = target_state["final_layer.ffn_final.fc2.weight"]
        source_bias = state_dict["final_layer.ffn_final.fc2.bias"]
        target_bias = target_state["final_layer.ffn_final.fc2.bias"]
        valid_action_width_change = (
            mismatched_output_keys == output_head_keys
            and source_weight.ndim == target_weight.ndim == 2
            and source_weight.shape[1] == target_weight.shape[1]
            and source_bias.ndim == target_bias.ndim == 1
            and source_weight.shape[0] == source_bias.shape[0]
            and target_weight.shape[0] == target_bias.shape[0]
        )
        if not valid_action_width_change:
            model.runner.model.load_state_dict(state_dict, strict=strict)
            raise AssertionError("unreachable")

    adapted_tensors: list[str] = []
    if language_position_key in shape_mismatches:
        source_position = state_dict[language_position_key]
        target_position = target_state[language_position_key]
        # The legacy ``qwen_fusion=language`` model reserved one additional
        # language position for its projected Qwen token.  Cross-attention KV
        # fusion no longer places that token in the language sequence.  Keep
        # all 128 shared learned positions and discard only the obsolete 129th.
        valid_language_width_change = (
            source_position.ndim == target_position.ndim == 3
            and source_position.shape[0] == target_position.shape[0]
            and source_position.shape[2] == target_position.shape[2]
            and source_position.shape[1] == target_position.shape[1] + 1
        )
        if not valid_language_width_change:
            model.runner.model.load_state_dict(state_dict, strict=strict)
            raise AssertionError("unreachable")
        state_dict[language_position_key] = source_position[
            :, : target_position.shape[1], :
        ]
        adapted_tensors.append(language_position_key)

    compatible_state = {
        name: tensor
        for name, tensor in state_dict.items()
        if name not in mismatched_output_keys
    }
    missing, unexpected = model.runner.model.load_state_dict(
        compatible_state,
        strict=False,
    )
    if set(missing) != mismatched_output_keys or unexpected:
        raise RuntimeError(
            "Full RDT base differs outside the permitted transfer tensors: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "loaded_tensors": len(compatible_state),
        "reinitialized_tensors": sorted(mismatched_output_keys),
        "adapted_tensors": adapted_tensors,
        "shape_mismatches": shape_mismatches,
    }
