from __future__ import annotations

import io
import json
import numpy as np
from PIL import Image
import torch
from types import SimpleNamespace

from scripts.precompute_all_features import (
    anchor_kind,
    episode_sample_count_is_allowed,
    episode_pack_relative_path,
    extract_qwen_kv,
    extract_siglip_features,
    image_bytes_to_image,
    image_to_jpeg_bytes,
    image_to_lossless_png_bytes,
    iter_episode_sample_groups,
    save_episode_anchor_pack_job,
    save_sample_shard,
    select_episode_qwen_anchors,
)


class _FakeQwenTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return 2 if token == "<|im_end|>" else -1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [99] if text == "</think>" else []


class _FakeQwenProcessor:
    tokenizer = _FakeQwenTokenizer()

    def apply_chat_template(self, messages, **kwargs) -> str:
        del messages, kwargs
        return "prompt with literal closing think token"

    def __call__(self, **kwargs):
        batch_size = len(kwargs["text"])
        return {
            "input_ids": torch.tensor([[11, 99, 12]] * batch_size),
            "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        }


class _FakeQwenModel:
    def __call__(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        batch_size, sequence_length = input_ids.shape
        positions = torch.arange(sequence_length, dtype=torch.float32)
        keys = positions.view(1, 1, sequence_length, 1).repeat(
            batch_size, 1, 1, 2
        )
        values = (positions + 10).view(1, 1, sequence_length, 1).repeat(
            batch_size, 1, 1, 2
        )
        hidden = positions.view(1, sequence_length, 1).repeat(
            batch_size, 1, 3
        )
        return SimpleNamespace(
            past_key_values=((keys, values),),
            hidden_states=(torch.zeros_like(hidden), hidden),
        )


class _FakeSiglipProcessor:
    image_mean = (0.5, 0.5, 0.5)

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros(len(images), 3, 2, 2)}


class _FakeSiglipEncoder:
    def __call__(self, *, pixel_values):
        count = pixel_values.shape[0]
        values = torch.arange(count * 3, dtype=torch.float32).reshape(count, 1, 3)
        return SimpleNamespace(last_hidden_state=values)


def test_libero_siglip_can_encode_neutral_placeholder_slots():
    batch = {
        "siglip_image_slots": [[
            Image.new("RGB", (8, 8), (255, 0, 0)),
            Image.new("RGB", (8, 8), (127, 127, 127)),
        ]],
        "siglip_slot_mask": [[True, False]],
    }
    tokens, mask = extract_siglip_features(
        batch,
        _FakeSiglipProcessor(),
        _FakeSiglipEncoder(),
        max_img_tokens=2,
        expected_dim=3,
        device=torch.device("cpu"),
        encode_invalid_slots=True,
    )
    assert mask.tolist() == [[True, True]]
    torch.testing.assert_close(tokens[0, 1].float(), torch.tensor([3.0, 4.0, 5.0]))


def test_b0_extraction_selects_literal_think_end_for_kv_and_final_hidden():
    batch = {
        "instructions": ["pick up the block"],
        "qwen_images": [[Image.new("RGB", (8, 8))]],
    }

    qwen_kv, qwen_hidden = extract_qwen_kv(
        batch,
        _FakeQwenProcessor(),
        _FakeQwenModel(),
        device=torch.device("cpu"),
        layer_index=0,
        max_new_tokens=8,
        expected_dim=4,
        return_hidden_state=True,
        think_token_selector="think_end",
    )

    # The literal </think> is at padded sequence index 1. The old legacy
    # selector would have gathered index 0 instead.
    torch.testing.assert_close(
        qwen_kv.float(),
        torch.tensor([[[1.0, 1.0, 11.0, 11.0]]]),
    )
    torch.testing.assert_close(
        qwen_hidden,
        torch.tensor([[[1.0, 1.0, 1.0]]]),
    )


def test_b0_sample_shard_retains_kv_hidden_and_native_libero_schema(tmp_path):
    batch_size = 2
    batch = {
        "metadata": [
            {
                "dataset_id": "libero_spatial",
                "episode_id": "demo-0",
                "step_idx": str(index),
            }
            for index in range(batch_size)
        ],
        "instructions": ["pick up the object"] * batch_size,
        "joint_state": torch.arange(14, dtype=torch.float32).reshape(2, 7),
        "libero_native_state": torch.tensor(
            [[0.1, 0.2, 0.3, 0, 0, 0, -0.04245, 0.05185]] * batch_size,
            dtype=torch.float32,
        ),
        "libero_native_actions": torch.randn(batch_size, 64, 7),
        "action_time_mask": torch.ones(batch_size, 64, dtype=torch.bool),
        "ctrl_freq": torch.full((batch_size,), 20.0),
        "siglip_image_slots": [
            [Image.new("RGB", (8, 8), color=(index, 0, 0))]
            for index in range(batch_size)
        ],
        "siglip_slot_mask": torch.ones(batch_size, 1, dtype=torch.bool),
    }
    manifest = io.StringIO()
    qwen_kv = torch.randn(batch_size, 1, 2048, dtype=torch.bfloat16)
    qwen_hidden = torch.randn(batch_size, 1, 2560, dtype=torch.bfloat16)

    count, manifest_line = save_sample_shard(
        split_dir=tmp_path,
        manifest_handle=manifest,
        shard_index=0,
        sample_start_index=0,
        batch=batch,
        qwen_kv=qwen_kv,
        qwen_hidden_states=qwen_hidden,
        lang_tokens=torch.randn(1, 3, 16),
        lang_mask=torch.ones(1, 3, dtype=torch.bool),
        sample_lang_index=torch.zeros(batch_size, dtype=torch.long),
        image_history_size=1,
        image_jpeg_quality=100,
        save_padded_features=False,
        image_codec="png",
        cache_proprioception_schema="libero_native",
        qwen_token_selector="think_end",
    )

    entry = json.loads(manifest_line)
    shard = torch.load(
        tmp_path / entry["path"], map_location="cpu", weights_only=True
    )
    assert count == 2
    assert shard["conditioning_variant"] == "b0"
    assert shard["qwen_token_selector"] == "think_end"
    assert shard["proprioception_schema"] == (
        "libero_joint7_gripper2_norm01_action7_v1"
    )
    torch.testing.assert_close(shard["qwen_kv"], qwen_kv)
    torch.testing.assert_close(shard["qwen_hidden_states"], qwen_hidden)
    assert shard["state"].shape == (2, 9)
    torch.testing.assert_close(
        shard["state"][:, 7:9], torch.tensor([[0.0, 1.0]] * 2)
    )
    assert shard["actions"].shape == (2, 64, 7)
    torch.testing.assert_close(
        shard["eef_position"], torch.tensor([[0.1, 0.2, 0.3]] * 2)
    )
    assert entry["has_eef_position"] is True
    assert entry["eef_position_dim"] == 3


def make_sample(
    step_idx: int,
    gripper: float = 0.0,
    *,
    dataset_id: str = "bridge",
    episode_id: str = "reused-id",
) -> dict:
    actions = np.zeros((2, 7), dtype=np.float32)
    actions[:, 6] = gripper
    return {
        "dataset_id": dataset_id,
        "episode_id": episode_id,
        "step_idx": str(step_idx),
        "instruction": "move the object",
        "actions": actions,
    }


def test_cached_png_round_trip_is_pixel_exact() -> None:
    pixels = np.arange(17 * 19 * 3, dtype=np.uint16).reshape(17, 19, 3)
    pixels = (pixels % 256).astype(np.uint8)

    payload = image_to_lossless_png_bytes(Image.fromarray(pixels, mode="RGB"))
    decoded = np.asarray(image_bytes_to_image(payload))

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    np.testing.assert_array_equal(decoded, pixels)


def test_cached_jpeg_uses_requested_lossy_codec() -> None:
    pixels = np.full((32, 48, 3), 127, dtype=np.uint8)
    payload = image_to_jpeg_bytes(Image.fromarray(pixels, mode="RGB"), quality=90)
    decoded = image_bytes_to_image(payload)

    assert payload.startswith(b"\xff\xd8\xff")
    assert decoded.size == (48, 32)
    assert np.abs(np.asarray(decoded, dtype=np.int16) - pixels).mean() < 2.0


def test_episode_pack_directory_buckets_are_one_based() -> None:
    assert episode_pack_relative_path(0, shards_per_directory=500).as_posix() == (
        "episodes_000000001_000000500/episode_000000001.pt"
    )
    assert episode_pack_relative_path(499, shards_per_directory=500).as_posix() == (
        "episodes_000000001_000000500/episode_000000500.pt"
    )
    assert episode_pack_relative_path(500, shards_per_directory=500).as_posix() == (
        "episodes_000000501_000001000/episode_000000501.pt"
    )


def test_exact_episode_policy_allows_only_configured_short_dataset() -> None:
    kuka = [make_sample(step, dataset_id="kuka") for step in range(24)]
    bridge = [make_sample(step, dataset_id="bridge") for step in range(24)]
    full_bridge = [make_sample(step, dataset_id="bridge") for step in range(32)]

    assert episode_sample_count_is_allowed(
        kuka,
        required_samples=32,
        allow_short_dataset_ids={"kuka"},
    )
    assert not episode_sample_count_is_allowed(
        bridge,
        required_samples=32,
        allow_short_dataset_ids={"kuka"},
    )
    assert episode_sample_count_is_allowed(
        full_bridge,
        required_samples=32,
        allow_short_dataset_ids={"kuka"},
    )


def test_per_sample_qwen_episode_pack_keeps_one_feature_and_instruction_per_sample(
    tmp_path,
) -> None:
    sample_count = 2
    slots = [
        [Image.new("RGB", (16, 16), color=(index * 20, 0, 0))]
        for index in range(sample_count)
    ]
    batch = {
        "metadata": [
            {
                "dataset_id": "bridge",
                "episode_id": "episode-a",
                "step_idx": str(index * 4),
                "image_count": 1,
            }
            for index in range(sample_count)
        ],
        "instructions": ["pick up the cup"] * sample_count,
        "siglip_image_slots": slots,
        "siglip_slot_mask": torch.ones(sample_count, 1, dtype=torch.bool),
        "state": torch.zeros(sample_count, 7),
        "state_dim_mask": torch.ones(sample_count, 7),
        "actions": torch.zeros(sample_count, 64, 7),
        "action_time_mask": torch.ones(sample_count, 64, dtype=torch.bool),
        "action_dim_mask": torch.ones(sample_count, 7),
        "ctrl_freq": torch.full((sample_count,), 10.0),
    }
    anchors = [
        {"step_idx": str(index * 4), "instruction": "pick up the cup"}
        for index in range(sample_count)
    ]
    qwen = torch.arange(sample_count * 8, dtype=torch.bfloat16).reshape(
        sample_count, 1, 8
    )

    count, manifest_line, _ = save_episode_anchor_pack_job(
        split_dir=tmp_path,
        episode_index=500,
        start_index=123,
        batch=batch,
        anchors=anchors,
        qwen_kv_by_anchor=qwen,
        lang_tokens=torch.zeros(1, 3, 4, dtype=torch.bfloat16),
        lang_mask=torch.ones(1, 3, dtype=torch.bool),
        save_padded_features=False,
        image_history_size=1,
        image_jpeg_quality=90,
        image_codec="jpeg",
        qwen_cache_scope="per_sample",
        episode_shards_per_directory=500,
        actions_normalized=True,
    )

    manifest = json.loads(manifest_line)
    path = tmp_path / manifest["path"]
    pack = torch.load(path, map_location="cpu", weights_only=True)
    assert count == sample_count
    assert manifest["path"] == (
        "episodes_000000501_000001000/episode_000000501.pt"
    )
    assert pack["qwen_cache_scope"] == "per_sample"
    assert pack["actions_normalized"] is True
    assert pack["sample_anchor_index"].tolist() == [0, 1]
    assert torch.equal(pack["qwen_anchor_kv"], qwen)
    assert pack["instruction"] == "pick up the cup"
    assert pack["instructions"] == ["pick up the cup", "pick up the cup"]
    assert pack["qwen_anchor_kind"] == ["per_sample", "per_sample"]
    assert all(payload.startswith(b"\xff\xd8\xff") for payload in pack["image_jpegs"])


def test_anchor_policy_keeps_only_first_step_without_gripper_change() -> None:
    samples = [make_sample(step, gripper=0.0) for step in (0, 2, 5, 9)]

    anchors = select_episode_qwen_anchors(
        samples,
        normalized_actions=False,
        max_anchors=2,
    )

    assert [anchor["step_idx"] for anchor in anchors] == ["0"]
    assert [anchor_kind(index, anchor) for index, anchor in enumerate(anchors)] == [
        "first_step"
    ]
    assert all("_qwen_anchor_kind" not in sample for sample in samples)


def test_anchor_policy_uses_first_gripper_transition_only() -> None:
    samples = [
        make_sample(0, gripper=0.0),
        make_sample(1, gripper=0.0),
        make_sample(3, gripper=1.0),
        make_sample(4, gripper=0.0),
    ]

    anchors = select_episode_qwen_anchors(
        samples,
        normalized_actions=False,
        max_anchors=2,
    )

    assert [anchor["step_idx"] for anchor in anchors] == ["0", "3"]
    assert [anchor_kind(index, anchor) for index, anchor in enumerate(anchors)] == [
        "first_step",
        "first_gripper_change",
    ]


def test_anchor_policy_never_fills_extra_uniform_anchors() -> None:
    samples = [make_sample(step, gripper=0.0) for step in range(5)]

    anchors = select_episode_qwen_anchors(
        samples,
        normalized_actions=False,
        max_anchors=8,
    )

    assert len(anchors) == 1
    assert anchors[0]["_qwen_anchor_kind"] == "first_step"


def test_episode_grouping_splits_reused_id_at_step_reset() -> None:
    stream = [
        make_sample(0),
        make_sample(2),
        make_sample(5),
        make_sample(0),
        make_sample(1),
        make_sample(4),
    ]

    groups = list(iter_episode_sample_groups(stream))

    assert [[sample["step_idx"] for sample in group] for group in groups] == [
        ["0", "2", "5"],
        ["0", "1", "4"],
    ]


def test_episode_grouping_tracks_occurrences_when_public_key_reappears() -> None:
    stream = [
        make_sample(0, episode_id="a"),
        make_sample(2, episode_id="a"),
        make_sample(0, episode_id="b"),
        make_sample(1, episode_id="b"),
        make_sample(0, episode_id="a"),
        make_sample(3, episode_id="a"),
    ]

    groups = list(iter_episode_sample_groups(stream))

    assert [group[0]["episode_id"] for group in groups] == ["a", "b", "a"]
    assert [[sample["step_idx"] for sample in group] for group in groups] == [
        ["0", "2"],
        ["0", "1"],
        ["0", "3"],
    ]


def test_episode_grouping_treats_duplicate_step_as_new_occurrence() -> None:
    stream = [make_sample(step) for step in (0, 2, 2, 6)]

    groups = list(iter_episode_sample_groups(stream))

    assert [[sample["step_idx"] for sample in group] for group in groups] == [
        ["0", "2"],
        ["2", "6"],
    ]
