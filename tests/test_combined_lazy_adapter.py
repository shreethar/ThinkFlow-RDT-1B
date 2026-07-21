from __future__ import annotations

from thinkflow_rdt.adapters.combined_lazy import (
    LazyCombinedStandardizedDataset,
    default_lazy_standardized_dataset_configs,
    episode_belongs_to_stage,
    episode_stage_index,
    episode_split_name,
    is_missing_local_shard_error,
    sample_belongs_to_stage,
    sample_stage_index,
)


def test_episode_split_name_is_deterministic_and_partitioned():
    episode_id = "episode_123"

    first = episode_split_name("bridge", episode_id, seed=7)
    second = episode_split_name("bridge", episode_id, seed=7)

    assert first == second
    assert first in {"train", "validation", "test"}


def test_lazy_combined_dataset_streams_members_in_sequence():
    left = [{"dataset_id": "left", "value": 0}, {"dataset_id": "left", "value": 1}]
    right = [{"dataset_id": "right", "value": 2}]
    dataset = LazyCombinedStandardizedDataset({"left": left, "right": right})

    samples = list(dataset)

    assert samples == [
        {"dataset_id": "left", "value": 0},
        {"dataset_id": "left", "value": 1},
        {"dataset_id": "right", "value": 2},
    ]
    assert dataset.dataset_ids == ["left", "right"]


def test_missing_shard_error_is_detected():
    error = RuntimeError(
        "open() failed: No such file or directory; opening "
        "/tmp/bc_z-train.array_record-00390-of-01024"
    )

    assert is_missing_local_shard_error(error)


def test_lazy_combined_dataset_continues_after_missing_shard_error():
    def broken_member():
        yield {"dataset_id": "broken", "value": 0}
        raise RuntimeError(
            "open() failed: No such file or directory; opening "
            "/tmp/bc_z-train.array_record-00390-of-01024"
        )

    right = [{"dataset_id": "right", "value": 1}]
    dataset = LazyCombinedStandardizedDataset({"broken": broken_member(), "right": right})

    samples = list(dataset)

    assert samples == [
        {"dataset_id": "broken", "value": 0},
        {"dataset_id": "right", "value": 1},
    ]


def test_default_configs_accept_mock_dataset_root_directly(tmp_path):
    mock_root = tmp_path / "mock_dataset"
    data_dir = mock_root / "bc_z_dataset" / "data"
    data_dir.mkdir(parents=True)

    configs = default_lazy_standardized_dataset_configs(
        dataset_ids=["bc_z"],
        root=mock_root,
    )

    assert configs[0].data_dir == data_dir


def test_stage_assignment_is_deterministic_and_droid_skips_stage_three():
    episode_id = "episode_123"

    first = sample_stage_index("bridge", episode_id, 12, seed=7)
    second = sample_stage_index("bridge", episode_id, 12, seed=7)

    assert first == second
    assert first in {1, 2, 3}
    assert sample_belongs_to_stage("droid", episode_id, 12, 3, seed=7) is False


def test_legacy_episode_stage_helpers_remain_available():
    episode_id = "episode_123"

    first = episode_stage_index("bridge", episode_id, seed=7)
    second = episode_stage_index("bridge", episode_id, seed=7)

    assert first == second
    assert first in {1, 2, 3}
    assert episode_belongs_to_stage("droid", episode_id, 3, seed=7) is False
