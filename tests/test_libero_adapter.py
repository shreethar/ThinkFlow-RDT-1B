from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from thinkflow_rdt.adapters.libero import (
    LIBERO_ACTION_DIM,
    LIBERO_STATE_DIM,
    convert_libero_demo,
    libero_action_to_rdt,
    libero_image_to_rgb,
    libero_observation_to_rdt,
    libero_orientation_to_ortho6d,
    libero_rot_command_to_ortho6d,
    ortho6d_to_libero_orientation,
    ortho6d_to_libero_rot_command,
    rdt_action_to_libero,
)
from scripts.replay_libero_demo_actions import codec_roundtrip


class FakeGroup(dict):
    def __init__(self, *args, name: str, parent=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.attrs = {}
        self.parent = parent or SimpleNamespace(attrs={})


def test_rotation_6d_round_trips_absolute_pose_and_raw_command() -> None:
    rotvec = np.array(
        [[0.2, -0.4, 0.1], [3.16408952, -0.21493082, -0.41219167]],
        dtype=np.float32,
    )
    pose_6d = libero_orientation_to_ortho6d(rotvec)
    reconstructed = ortho6d_to_libero_orientation(pose_6d)
    original_matrix = Rotation.from_rotvec(rotvec).as_matrix()
    reconstructed_matrix = Rotation.from_rotvec(reconstructed).as_matrix()
    np.testing.assert_allclose(reconstructed_matrix, original_matrix, atol=1e-6)

    command = np.array([[0.4, -0.2, 0.8], [-1.0, 1.0, 0.0]], dtype=np.float32)
    encoded = libero_rot_command_to_ortho6d(command)
    np.testing.assert_allclose(
        ortho6d_to_libero_rot_command(encoded),
        command,
        atol=1e-6,
    )


def test_raw_7d_action_round_trips_through_10d_without_gripper_remapping() -> None:
    raw = np.array(
        [[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -1.0],
         [-0.3, 0.2, -0.1, -0.8, 0.7, -0.6, 1.0]],
        dtype=np.float32,
    )
    encoded = libero_action_to_rdt(raw)
    assert encoded.shape == (2, LIBERO_ACTION_DIM)
    np.testing.assert_array_equal(encoded[:, 9], raw[:, 6])
    np.testing.assert_allclose(rdt_action_to_libero(encoded), raw, atol=1e-6)


def test_replay_codec_report_detects_bitwise_near_roundtrip() -> None:
    raw = np.array(
        [[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -1.0]],
        dtype=np.float32,
    )
    decoded, report = codec_roundtrip(raw)

    np.testing.assert_allclose(decoded, raw, atol=1e-6)
    assert report["allclose_atol_1e-6"] is True
    assert report["encoded_shape"] == [1, 10]
    assert report["translation_max_l2"] == 0.0
    assert report["gripper_max_abs_error"] == 0.0


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
    assert converted["state"].shape == (LIBERO_STATE_DIM,)
    np.testing.assert_allclose(converted["state"][9:11], [0.04, -0.04])


def test_demo_conversion_flips_images_and_preserves_raw_fingers_and_commands() -> None:
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
    assert episode.states.shape == (2, LIBERO_STATE_DIM)
    assert episode.actions.shape == (2, LIBERO_ACTION_DIM)
    assert episode.native_states.shape == (2, 8)
    assert episode.native_actions.shape == (2, 7)
    np.testing.assert_array_equal(episode.native_states[:, :3], 0.0)
    np.testing.assert_array_equal(episode.native_states[:, 3:6], 0.0)
    np.testing.assert_allclose(
        episode.native_states[:, 6:8],
        [[0.04, -0.04], [0.01, -0.01]],
    )
    np.testing.assert_array_equal(episode.native_actions, group["actions"])
    np.testing.assert_allclose(
        episode.states[:, 9:11],
        [[0.04, -0.04], [0.01, -0.01]],
    )
    np.testing.assert_array_equal(episode.actions[:, 9], [-1.0, 1.0])
