from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from precompute_latent_student_kv import save_sample_shard  # noqa: E402


def test_libero_sample_shard_saves_joint7_and_normalized_gripper2(tmp_path) -> None:
    batch = {
        "libero_native_state": torch.tensor(
            [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -0.04245, 0.05185]]
        ),
        "libero_native_actions": torch.zeros(1, 64, 7),
        "joint_state": torch.arange(7, dtype=torch.float32).unsqueeze(0),
        "joint_states": torch.zeros(1, 64, 7),
        "joint_states_mask": torch.ones(1, 64, dtype=torch.bool),
        "action_time_mask": torch.ones(1, 64, dtype=torch.bool),
        "ctrl_freq": torch.tensor([20.0]),
        "metadata": [
            {
                "dataset_id": "libero_spatial",
                "episode_id": "demo_0",
                "step_idx": "0",
            }
        ],
        "instructions": ["pick up the object"],
    }
    saved, manifest_line = save_sample_shard(
        split_dir=tmp_path,
        shard_index=0,
        sample_start_index=0,
        batch=batch,
        latent_kv=torch.zeros(1, 5, 2048),
        spatial_hidden_states=torch.zeros(1, 5, 2560),
        waypoints=torch.zeros(1, 5, 2),
        lang_tokens=None,
        lang_mask=None,
        sample_lang_index=None,
        image_history_size=2,
        image_jpeg_quality=100,
        cache_image_slots=False,
        save_padded_features=False,
        cache_proprioception_schema="libero_native",
        image_storage="raw_uint8",
        feature_variant="b2",
    )

    assert saved == 1
    shard = torch.load(tmp_path / "shard_000000000.pt", weights_only=True)
    assert shard["state"].shape == (1, 9)
    torch.testing.assert_close(
        shard["state"][0],
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 1.0]),
    )
    assert shard["actions"].shape == (1, 64, 7)
    torch.testing.assert_close(
        shard["eef_position"], torch.tensor([[0.1, 0.2, 0.3]])
    )
    assert shard["proprioception_schema"] == (
        "libero_joint7_gripper2_norm01_action7_v1"
    )
    manifest = json.loads(manifest_line)
    assert manifest["state_dim"] == 9
    assert manifest["action_dim"] == 7
    assert manifest["conditioning_variant"] == "b2"
    assert manifest["has_eef_position"] is True
    assert manifest["eef_position_dim"] == 3
