#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


VISUAL_PREFIX = "model.language_model.visual."
FIXED_VISUAL_PREFIX = "model.visual."

MODEL_FILES = ("config.json", "generation_config.json", "README.md")
PROCESSOR_FILES = (
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "special_tokens_map.json",
)


def copy_if_present(src_dir: Path, dst_dir: Path, names: tuple[str, ...]) -> list[str]:
    copied: list[str] = []
    for name in names:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied.append(name)
    return copied


def fixed_key(key: str) -> str:
    if key.startswith(VISUAL_PREFIX):
        return FIXED_VISUAL_PREFIX + key[len(VISUAL_PREFIX) :]
    return key


def convert_safetensors(src_model: Path, dst_model: Path) -> dict[str, int]:
    state = load_file(str(src_model))
    fixed = {}
    renamed = 0
    for key, value in state.items():
        new_key = fixed_key(key)
        if new_key != key:
            renamed += 1
        if new_key in fixed:
            raise ValueError(f"Renaming creates duplicate tensor key: {new_key}")
        fixed[new_key] = value
    save_file(fixed, str(dst_model))
    return {"tensors": len(state), "renamed_visual_tensors": renamed}


def convert_model_weights(src: Path, dst: Path) -> dict[str, object]:
    index_path = src / "model.safetensors.index.json"
    single_path = src / "model.safetensors"

    if index_path.exists():
        with index_path.open(encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid or missing weight_map in {index_path}")
        shard_names = sorted(set(weight_map.values()))
    elif single_path.exists():
        shard_names = [single_path.name]
        index = None
    else:
        shard_names = sorted(path.name for path in src.glob("model-*-of-*.safetensors"))
        index = None
        if not shard_names:
            raise FileNotFoundError(
                f"No model.safetensors, model.safetensors.index.json, or numbered shards in {src}"
            )

    if len(shard_names) > 1 and index is None:
        weight_map = {}
        for shard_name in shard_names:
            with safe_open(src / shard_name, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in weight_map:
                        raise ValueError(f"Tensor key occurs in multiple shards: {key}")
                    weight_map[key] = shard_name
        index = {"metadata": {}, "weight_map": weight_map}

    totals = {"tensors": 0, "renamed_visual_tensors": 0}
    for shard_name in shard_names:
        shard = src / shard_name
        if not shard.exists():
            raise FileNotFoundError(f"Shard listed by checkpoint is missing: {shard}")
        counts = convert_safetensors(shard, dst / shard_name)
        totals["tensors"] += counts["tensors"]
        totals["renamed_visual_tensors"] += counts["renamed_visual_tensors"]

    if index is not None:
        new_weight_map = {}
        for key, shard_name in weight_map.items():
            new_key = fixed_key(key)
            if new_key in new_weight_map:
                raise ValueError(f"Renaming creates duplicate index key: {new_key}")
            new_weight_map[new_key] = shard_name
        index["weight_map"] = new_weight_map
        with (dst / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)
            handle.write("\n")

    return {**totals, "weight_files": shard_names, "sharded": len(shard_names) > 1}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LatentStudent checkpoint keys from "
            "model.language_model.visual.* to model.visual.* and copy processor files."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/workspace/model/LatentStudent-ckpt-400"),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/workspace/model/LatentStudent-ckpt-400-fixed"),
    )
    parser.add_argument(
        "--processor-src",
        type=Path,
        default=Path("/workspace/model/stage1_unsloth"),
    )
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    processor_src = args.processor_src.expanduser().resolve()
    if not processor_src.exists():
        raise FileNotFoundError(processor_src)

    dst.mkdir(parents=True, exist_ok=True)
    copied_model_files = copy_if_present(src, dst, MODEL_FILES)
    copied_processor_files = copy_if_present(processor_src, dst, PROCESSOR_FILES)
    counts = convert_model_weights(src, dst)

    copied_extra_files: list[str] = []
    spatial_params = src / "spatial_parameters.pt"
    if spatial_params.exists():
        shutil.copy2(spatial_params, dst / "spatial_parameters.pt")
        copied_extra_files.append("spatial_parameters.pt")

    report = {
        "source": str(src),
        "destination": str(dst),
        "processor_source": str(processor_src),
        "copied_model_files": copied_model_files,
        "copied_processor_files": copied_processor_files,
        "copied_extra_files": copied_extra_files,
        **counts,
    }
    with (dst / "conversion_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(f"wrote {dst}")
    print(f"renamed {counts['renamed_visual_tensors']} / {counts['tensors']} tensors")


if __name__ == "__main__":
    main()
