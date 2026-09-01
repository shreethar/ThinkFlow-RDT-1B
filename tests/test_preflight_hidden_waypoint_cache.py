from __future__ import annotations

import json
import sys

import torch

from scripts.preflight_hidden_waypoint_cache import main


def test_b0_preflight_accepts_cache_without_optional_eef_position(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    shard_path = tmp_path / "shard_000000000.pt"
    torch.save(
        {
            "cache_layout": "sample_shard",
            "conditioning_variant": "b0",
            "qwen_token_selector": "think_end",
            "dataset_id": "libero_spatial",
            "num_samples": 2,
            "qwen_kv": torch.zeros(2, 1, 2048),
            "qwen_hidden_states": torch.zeros(2, 1, 2560),
            "state": torch.tensor(
                [[0.0] * 7 + [0.0, 1.0], [0.0] * 7 + [1.0, 0.0]]
            ),
            "actions": torch.zeros(2, 64, 7),
        },
        shard_path,
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "path": shard_path.name,
                "dataset_id": "libero_spatial",
                "conditioning_variant": "b0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight_hidden_waypoint_cache.py",
            "--manifest",
            str(manifest),
            "--expected-variant",
            "b0",
            "--expected-dataset",
            "libero_spatial",
        ],
    )

    main()

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["diagnostic_sidecar"] == "not cached (optional)"

