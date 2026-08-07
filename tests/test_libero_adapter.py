from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from thinkflow_rdt.adapters.libero import (
    convert_libero_demo,
    libero_gripper_closed,
    libero_image_to_rgb,
    libero_observation_to_rdt,
)


class FakeGroup(dict):
    def __init__(self, *args, name: str, parent=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.attrs = {}
        self.parent = parent or SimpleNamespace(attrs={})


def test_project_gripper_command_convention() -> None:
    commands = np.array([-1.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(libero_gripper_closed(commands), [1.0, 0.0])


def test_libero_image_flip_uses_height_axis_for_frames_and_sequences() -> None:
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    sequence = np.stack([frame, frame + 20], axis=0)

    np.testing.assert_array_equal(libero_image_to_rgb(frame), frame[::-1])
    np.testing.assert_array_equal(libero_image_to_rgb(sequence), sequence[:, ::-1])
    assert libero_image_to_rgb(frame).flags.c_contiguous


def test_live_libero_observation_flips_agent_and_wrist_images_once() -> None:
    agent = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    wrist = agent + 40
    converted = libero_observation_to_rdt(
        {
            "ee_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "ee_ori": np.zeros(3, dtype=np.float32),
            "gripper_states": np.array([0.04, -0.04], dtype=np.float32),
            "agentview_image": agent,
            "robot0_eye_in_hand_image": wrist,
        }
    )

    np.testing.assert_array_equal(converted["primary"], agent[::-1])
    np.testing.assert_array_equal(converted["wrist"], wrist[::-1])


def test_demo_conversion_flips_images_and_uses_project_gripper_convention() -> None:
    agent = np.arange(2 * 2 * 3 * 3, dtype=np.uint8).reshape(2, 2, 3, 3)
    wrist = agent + 50
    obs = FakeGroup(
        {
            "ee_pos": np.zeros((2, 3), dtype=np.float32),
            "ee_ori": np.zeros((2, 3), dtype=np.float32),
            "gripper_states": np.array(
                [[0.04, -0.04], [0.01, -0.01]], dtype=np.float32
            ),
            "agentview_rgb": agent,
            "eye_in_hand_rgb": wrist,
        },
        name="/data/demo_0/obs",
    )
    group = FakeGroup(
        {
            "obs": obs,
            "actions": np.array(
                [
                    [0, 0, 0, 0, 0, 0, -1],
                    [0, 0, 0, 0, 0, 0, +1],
                ],
                dtype=np.float32,
            ),
        },
        name="/data/demo_0",
    )

    episode = convert_libero_demo(group, episode_id="demo_0")

    np.testing.assert_array_equal(episode.primary_images, agent[:, ::-1])
    np.testing.assert_array_equal(episode.wrist_images, wrist[:, ::-1])
    np.testing.assert_array_equal(episode.actions[:, 6], [1.0, 0.0])
    np.testing.assert_array_equal(episode.states[:, 6], [0.0, 1.0])
