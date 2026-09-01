from types import SimpleNamespace

from PIL import Image
import pytest

from thinkflow_rdt.vision import (
    pad_siglip_image_to_square,
    validate_siglip_384_processor,
)


def test_siglip_square_padding_uses_neutral_mean_without_resizing():
    image = Image.new("RGB", (128, 64), (255, 0, 0))
    padded = pad_siglip_image_to_square(image, (0.5, 0.5, 0.5))
    assert padded.size == (128, 128)
    assert padded.getpixel((0, 0)) == (127, 127, 127)
    assert padded.getpixel((64, 64)) == (255, 0, 0)


def test_siglip_square_image_is_not_resized_by_padding_step():
    image = Image.new("RGB", (128, 128), (1, 2, 3))
    result = pad_siglip_image_to_square(image, (0.5, 0.5, 0.5))
    assert result.size == (128, 128)
    assert result.getpixel((0, 0)) == (1, 2, 3)


def test_siglip_processor_contract_is_384_not_336():
    processor = SimpleNamespace(
        size={"height": 384, "width": 384},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    assert validate_siglip_384_processor(processor)["width"] == 384
    processor.size = {"height": 336, "width": 336}
    with pytest.raises(ValueError, match="SigLIP-384"):
        validate_siglip_384_processor(processor)
