from __future__ import annotations

import json
import numpy as np
from PIL import Image
import torch

from scripts.precompute_all_features import (
    anchor_kind,
    episode_pack_relative_path,
    image_bytes_to_image,
    image_to_jpeg_bytes,
    image_to_lossless_png_bytes,
    iter_episode_sample_groups,
    save_episode_anchor_pack_job,
    select_episode_qwen_anchors,
)


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
