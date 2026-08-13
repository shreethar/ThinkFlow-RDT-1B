#!/usr/bin/env python
"""Plot the current agent and wrist images from one sample per LIBERO suite."""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


DEFAULT_SUITES = ("libero_10", "libero_spatial", "libero_goal", "libero_object")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("cache_features_libero_b2_native"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument(
        "--camera-slots",
        nargs=2,
        type=int,
        default=(3, 4),
        metavar=("AGENT_SLOT", "WRIST_SLOT"),
        help="Current agent/wrist slots for two-frame, three-camera cache layout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/libero_shard_first_samples.png"),
    )
    return parser.parse_args()


def image_to_rgb(value: Any) -> np.ndarray:
    if isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(value))) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)

    array = np.asarray(torch.as_tensor(value).cpu())
    if array.ndim != 3:
        raise ValueError(f"Expected a three-dimensional image, got {array.shape}")
    if array.shape[-1] == 3:
        rgb = array
    elif array.shape[0] == 3:
        rgb = np.moveaxis(array, 0, -1)
    else:
        raise ValueError(f"Expected HWC or CHW RGB image, got {array.shape}")
    if np.issubdtype(rgb.dtype, np.floating):
        if float(np.nanmax(rgb)) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8, copy=False)


def sample_images(
    pack: dict[str, Any],
    sample_index: int,
    camera_slots: tuple[int, int],
) -> list[np.ndarray]:
    count = int(pack.get("num_samples", len(pack["sample_image_indices"])))
    if not 0 <= sample_index < count:
        raise IndexError(f"sample-index {sample_index} is outside shard sample count {count}")

    pool = pack.get("image_arrays")
    if pool is None:
        pool = pack.get("image_jpegs")
    if pool is None:
        raise KeyError("Shard contains neither image_arrays nor image_jpegs")

    indices = torch.as_tensor(pack["sample_image_indices"])[sample_index]
    mask = torch.as_tensor(pack["sample_image_mask"], dtype=torch.bool)[sample_index]
    result = []
    for slot in camera_slots:
        if slot >= len(indices) or not bool(mask[slot]):
            raise ValueError(f"Camera slot {slot} is invalid for sample {sample_index}")
        result.append(image_to_rgb(pool[int(indices[slot])]))
    return result


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    camera_slots = tuple(args.camera_slots)

    figure, axes = plt.subplots(
        len(args.suites),
        2,
        figsize=(10, 4 * len(args.suites)),
        squeeze=False,
    )
    for row, suite in enumerate(args.suites):
        shard = (
            cache_root
            / suite
            / args.split
            / f"shard_{args.shard_index:09d}.pt"
        )
        if not shard.exists():
            raise FileNotFoundError(shard)
        pack = torch.load(shard, map_location="cpu", weights_only=False)
        images = sample_images(pack, args.sample_index, camera_slots)
        metadata = pack.get("metadata", [{}])[args.sample_index]
        step = metadata.get("step_idx", "?")
        for column, (name, image) in enumerate(zip(("agent", "wrist"), images)):
            axes[row, column].imshow(image)
            axes[row, column].set_title(f"{suite} | {name} | sample {args.sample_index} | step {step}")
            axes[row, column].axis("off")
        print(
            f"{suite}: {shard.name}, sample={args.sample_index}, "
            f"agent={images[0].shape}, wrist={images[1].shape}"
        )

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
