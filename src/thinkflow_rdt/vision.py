from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from PIL import Image


def pad_siglip_image_to_square(
    image: Image.Image,
    image_mean: Iterable[float],
) -> Image.Image:
    """Pad without cropping, using the SigLIP mean as neutral background."""
    image = image.convert("RGB")
    width, height = image.size
    if width == height:
        return image
    background = tuple(int(float(value) * 255) for value in image_mean)
    side = max(width, height)
    output = Image.new("RGB", (side, side), background)
    output.paste(image, ((side - width) // 2, (side - height) // 2))
    return output


def prepare_siglip_images(
    images: Iterable[Image.Image],
    processor: Any,
) -> list[Image.Image]:
    """Apply Libero_RDT's aspect-ratio padding before HF preprocessing."""
    image_mean = getattr(processor, "image_mean", (0.5, 0.5, 0.5))
    return [pad_siglip_image_to_square(image, image_mean) for image in images]


def validate_siglip_384_processor(processor: Any) -> dict[str, Any]:
    """Fail fast when preprocessing differs from the RDT-1B SigLIP contract."""
    size = getattr(processor, "size", {})
    height = int(size.get("height", 0))
    width = int(size.get("width", 0))
    mean = tuple(float(value) for value in processor.image_mean)
    std = tuple(float(value) for value in processor.image_std)
    if (height, width) != (384, 384):
        raise ValueError(
            "RDT-1B expects SigLIP-384 preprocessing, got "
            f"{height}x{width}. A 336 processor would produce a different "
            "patch-token count and cannot use the 4,374-token image table."
        )
    expected = (0.5, 0.5, 0.5)
    if mean != expected or std != expected:
        raise ValueError(
            "Unexpected SigLIP normalization: "
            f"mean={mean}, std={std}, expected={expected}"
        )
    return {"height": height, "width": width, "mean": mean, "std": std}
