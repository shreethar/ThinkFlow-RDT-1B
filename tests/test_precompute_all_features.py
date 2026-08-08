from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.precompute_all_features import (
    anchor_kind,
    image_bytes_to_image,
    image_to_lossless_png_bytes,
    iter_episode_sample_groups,
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
