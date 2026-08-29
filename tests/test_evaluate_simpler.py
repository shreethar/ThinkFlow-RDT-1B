from __future__ import annotations

import os

import numpy as np

from scripts.evaluate_simpler_b0_oxe import (
    AF_UNIX_SAFE_PATH_BYTES,
    ActionStats,
    GoogleGripperTargetAdapter,
    decode_native_actions,
    resolve_qwen_extraction_mode,
    worker_socket_path,
)


def test_qwen_extraction_mode_auto_detects_b2_artifacts() -> None:
    assert resolve_qwen_extraction_mode(
        "auto",
        checkpoint="output_2/fractal_b2/checkpoint-5000",
        config="configs/part3_rdt1b.yaml",
    ) == "b2"
    assert resolve_qwen_extraction_mode(
        "auto",
        checkpoint="output_2/b0/checkpoint-20000",
        config="configs/part3_rdt1b.yaml",
    ) == "b0"
    assert resolve_qwen_extraction_mode(
        "b2",
        checkpoint="arbitrary",
        config="arbitrary",
    ) == "b2"


def test_worker_socket_path_stays_below_linux_limit(monkeypatch, tmp_path):
    deliberately_long = tmp_path / ("long-output-component-" * 8)
    monkeypatch.setenv("THINKFLOW_SIMPLER_SOCKET_DIR", str(deliberately_long))

    socket_path = worker_socket_path()

    assert len(os.fsencode(socket_path)) <= AF_UNIX_SAFE_PATH_BYTES
    assert socket_path.suffix == ".sock"
    assert socket_path.name.startswith("tfse_")
    assert socket_path.parent != deliberately_long


def test_google_gripper_sticky_transition_ignores_one_step_chatter():
    adapter = GoogleGripperTargetAdapter(sticky_steps=3, assumed_open=True)

    commands = [
        adapter.command(target)
        for target in (True, False, True, True, True, True, True)
    ]

    assert commands == [0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    assert adapter.assumed_open is True
    assert adapter.remaining_steps == 0


def test_google_gripper_zero_sticky_steps_preserves_stateless_commands():
    adapter = GoogleGripperTargetAdapter(sticky_steps=0, assumed_open=True)

    assert adapter.command(True) == -1.0
    assert adapter.command(False) == 1.0


def test_rotation_scale_multiplies_decoded_controller_rotation():
    native = np.zeros((1, 128), dtype=np.float32)
    native[:, 30:33] = 0.0
    native[:, 33:39] = np.array(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.2], dtype=np.float32
    )
    native[:, 10] = 1.0
    stats = ActionStats(
        q01=np.array([-1.0] * 7, dtype=np.float32),
        q99=np.array([1.0] * 7, dtype=np.float32),
    )

    full = decode_native_actions(
        native,
        dataset="fractal",
        task="google_robot_pick_coke_can",
        stats=stats,
    )
    quarter = decode_native_actions(
        native,
        dataset="fractal",
        task="google_robot_pick_coke_can",
        stats=stats,
        rotation_scale=0.25,
    )

    np.testing.assert_allclose(
        quarter["environment_rotation"], full["environment_rotation"] * 0.25
    )
