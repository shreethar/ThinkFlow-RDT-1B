from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.run_precompute_32frame_episode_packs_latent_student_kv as extractor


def sample(dataset_id: str, episode_id: str, step: int) -> dict[str, str]:
    return {
        "dataset_id": dataset_id,
        "episode_id": episode_id,
        "step_idx": str(step),
    }


def test_round_robin_schedule_does_not_exhaust_first_dataset() -> None:
    first = [
        sample("bc_z", "a1", 0),
        sample("bc_z", "a1", 1),
        sample("bc_z", "a2", 0),
    ]
    second = [sample("bridge", "b1", 0)]
    combined = SimpleNamespace(
        members=[
            SimpleNamespace(dataset_id="bc_z", dataset=first),
            SimpleNamespace(dataset_id="bridge", dataset=second),
        ]
    )

    identities = [
        extractor.episode_identity(group)
        for group in extractor.iter_scheduled_episode_groups(
            combined,
            schedule="round_robin",
        )
    ]

    assert identities == [("bc_z", "a1"), ("bridge", "b1"), ("bc_z", "a2")]


def test_qwen_batch_size_is_honored(monkeypatch) -> None:
    calls: list[int] = []

    def fake_extract(batch, **_kwargs):
        size = len(batch["instructions"])
        calls.append(size)
        return (
            torch.zeros(size, 5, 2048),
            torch.zeros(size, 5, 2560, dtype=torch.bfloat16),
            torch.zeros(size, 5, 2),
        )

    monkeypatch.setattr(extractor, "extract_latent_student_spatial_kv", fake_extract)
    batch = {
        "metadata": [{} for _ in range(5)],
        "instructions": ["instruction"] * 5,
        "qwen_images": [[object()]] * 5,
    }

    spatial_kv, hidden_states, waypoints = extractor.extract_latent_student_spatial_kv_chunked(
        batch,
        student=object(),
        processor=object(),
        device=torch.device("cpu"),
        layer_index=7,
        expected_dim=2048,
        spatial_token_count=5,
        prompt_template="{task}",
        batch_size=2,
    )

    assert calls == [2, 2, 1]
    assert spatial_kv.shape == (5, 5, 2048)
    assert hidden_states.shape == (5, 5, 2560)
    assert hidden_states.dtype == torch.bfloat16
    assert waypoints.shape == (5, 5, 2)


def write_pack_and_row(split_dir: Path, episode_number: int = 1) -> Path:
    relative = Path("episodes_000000001_000000500") / (
        f"episode_{episode_number:09d}.pt"
    )
    pack = split_dir / relative
    pack.parent.mkdir(parents=True, exist_ok=True)
    pack.write_bytes(b"pack")
    row = {
        "path": relative.as_posix(),
        "dataset_id": "bc_z",
        "episode_id": f"episode-{episode_number}",
        "num_samples": 32,
    }
    (split_dir / "manifest.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    return pack


def test_resume_validates_manifest_and_continues_file_number(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    write_pack_and_row(split_dir)

    manifest, rows, next_index = extractor.prepare_episode_split_output(
        split_dir,
        resume=True,
        overwrite=False,
    )

    assert manifest == split_dir / "manifest.jsonl"
    assert len(rows) == 1
    assert next_index == 1


def test_resume_rejects_pack_missing_from_manifest(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    write_pack_and_row(split_dir)
    orphan = split_dir / "episodes_000000001_000000500" / "episode_000000002.pt"
    orphan.write_bytes(b"orphan")

    with pytest.raises(ValueError, match="not recorded"):
        extractor.prepare_episode_split_output(
            split_dir,
            resume=True,
            overwrite=False,
        )


class FakeTokenizer:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "</think>"
        return 42


class FakeStudent:
    def __init__(self) -> None:
        self.M = 6
        self.K = 5
        self.end_think_token_id = 42
        self.spatial_tokens = torch.ones(5, 2560)
        self.spatial_mlp = torch.nn.Linear(2560, 2)
        self._language_model = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=2560,
                num_hidden_layers=32,
                num_key_value_heads=4,
                head_dim=256,
                layer_types=[
                    "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
                    for index in range(32)
                ],
            )
        )


def test_student_contract_accepts_layer_7_full_attention() -> None:
    args = SimpleNamespace(layer_index=7, latent_count=6, spatial_token_count=5)
    cfg = SimpleNamespace(
        model=SimpleNamespace(qwen_hidden_size=2560, qwen_kv_dim=2048)
    )

    contract = extractor.validate_student_runtime_contract(
        FakeStudent(),
        SimpleNamespace(tokenizer=FakeTokenizer()),
        args=args,
        cfg=cfg,
    )

    assert contract["layer_type"] == "full_attention"
    assert contract["flattened_kv_dim"] == 2048


def test_student_contract_rejects_linear_attention_layer() -> None:
    args = SimpleNamespace(layer_index=6, latent_count=6, spatial_token_count=5)
    cfg = SimpleNamespace(
        model=SimpleNamespace(qwen_hidden_size=2560, qwen_kv_dim=2048)
    )

    with pytest.raises(ValueError, match="full-attention"):
        extractor.validate_student_runtime_contract(
            FakeStudent(),
            SimpleNamespace(tokenizer=FakeTokenizer()),
            args=args,
            cfg=cfg,
        )
