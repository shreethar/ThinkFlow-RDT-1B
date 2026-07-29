#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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


def convert_safetensors(src_model: Path, dst_model: Path) -> dict[str, int]:
    state = load_file(str(src_model))
    fixed = {}
    renamed = 0
    for key, value in state.items():
        if key.startswith(VISUAL_PREFIX):
            key = FIXED_VISUAL_PREFIX + key[len(VISUAL_PREFIX) :]
            renamed += 1
        fixed[key] = value
    save_file(fixed, str(dst_model))
    return {"tensors": len(state), "renamed_visual_tensors": renamed}


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
    src_model = src / "model.safetensors"
    dst_model = dst / "model.safetensors"

    if not src_model.exists():
        raise FileNotFoundError(src_model)
    if not processor_src.exists():
        raise FileNotFoundError(processor_src)

    dst.mkdir(parents=True, exist_ok=True)
    copied_model_files = copy_if_present(src, dst, MODEL_FILES)
    copied_processor_files = copy_if_present(processor_src, dst, PROCESSOR_FILES)
    counts = convert_safetensors(src_model, dst_model)

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
