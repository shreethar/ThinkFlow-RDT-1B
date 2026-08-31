from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from rollout_libero_rdt import (  # noqa: E402
    native_rdt_action_to_libero_7d,
    native_rdt_policy_inputs,
)


def test_libero_joint_eef_rollout_uses_author_native_slots() -> None:
    state = torch.zeros(1, 11)
    state[0, 9:11] = torch.tensor([-0.04245, 0.05185])
    joints = torch.arange(7, dtype=torch.float32).unsqueeze(0)
    packed_state, state_mask, action_mask = native_rdt_policy_inputs(
        state,
        torch.ones_like(state),
        torch.ones(1, 7),
        mapping="libero_joint_eef_delta",
        joint_state=joints,
    )

    torch.testing.assert_close(packed_state[0, :7], joints[0])
    torch.testing.assert_close(
        packed_state[0, 10:12], torch.tensor([0.0, 1.0])
    )
    assert torch.nonzero(state_mask[0]).flatten().tolist() == [
        *range(7),
        10,
        11,
    ]
    assert torch.nonzero(action_mask[0]).flatten().tolist() == [
        10,
        *range(39, 45),
    ]


def test_libero_joint_eef_rollout_decodes_clipped_binary_command() -> None:
    native = np.zeros((2, 128), dtype=np.float32)
    native[0, 39:45] = np.array([-2.0, -0.5, 0.25, 0.5, 1.5, 2.0])
    native[0, 10] = -0.01
    native[1, 10] = 0.0

    decoded = native_rdt_action_to_libero_7d(native)

    np.testing.assert_allclose(
        decoded[0, :6], [-1.0, -0.5, 0.25, 0.5, 1.0, 1.0]
    )
    np.testing.assert_array_equal(decoded[:, 6], [-1.0, 1.0])
