"""Bridge cached Qwen/image shards into LeRobot's official training loop."""

from __future__ import annotations

from pathlib import Path

from .cached_libero import LeRobotCachedLiberoDataset, list_shards
from .stats import load_or_compute_cache_stats


LIBERO_SUITES = ("libero_10", "libero_spatial", "libero_goal", "libero_object")
DATASET_PREFIX = "cached_libero_qwen"


def suites_from_repo_id(repo_id: str) -> list[str] | None:
    """Decode ``cached_libero_qwen:all`` or a comma-separated suite list."""

    if repo_id == DATASET_PREFIX:
        return list(LIBERO_SUITES)
    prefix = f"{DATASET_PREFIX}:"
    if not repo_id.startswith(prefix):
        return None
    value = repo_id[len(prefix) :]
    suites = list(LIBERO_SUITES) if value == "all" else value.split(",")
    invalid = [suite for suite in suites if suite not in LIBERO_SUITES]
    if invalid:
        raise ValueError(f"Unknown cached LIBERO suites {invalid}; choices={LIBERO_SUITES}")
    return list(dict.fromkeys(suites))


def _stats_path_for_config(cfg, cache_root: Path, suites: list[str]) -> Path:
    pretrained = getattr(cfg.trainable_config, "pretrained_path", None)
    # checkpoint-014000 contains statistics pooled across all four suites. Do
    # not accidentally reuse those numbers for a suite-specific fine-tune.
    if pretrained and suites == list(LIBERO_SUITES):
        candidate = Path(pretrained).expanduser().resolve().parent / "cache_stats.pt"
        if candidate.exists():
            return candidate
    label = "all" if suites == list(LIBERO_SUITES) else "_".join(suites)
    return cache_root / f"smolvla_native_stats_{label}.pt"


def make_cached_train_eval_datasets(cfg):
    """Factory compatible with ``lerobot.datasets.factory``.

    Return ``None`` for unrelated dataset IDs so the wrapper can delegate to
    LeRobot's native factory.
    """

    suites = suites_from_repo_id(cfg.dataset.repo_id)
    if suites is None:
        return None
    if cfg.dataset.root is None:
        raise ValueError("Cached Qwen LIBERO integration requires --dataset.root")
    if cfg.dataset.eval_split:
        raise ValueError(
            "The iterable cached integration currently requires --dataset.eval_split=0; "
            "use the standalone checkpoint evaluator for closed-loop validation."
        )
    cache_root = Path(cfg.dataset.root).expanduser().resolve()
    chunk_size = int(getattr(cfg.trainable_config, "chunk_size", 50))
    shard_paths = [
        path
        for suite in suites
        for path in list_shards(cache_root, suite, split="train")
    ]
    stats, num_samples = load_or_compute_cache_stats(
        _stats_path_for_config(cfg, cache_root, suites),
        shard_paths,
        chunk_size=chunk_size,
    )
    # LIBERO's standard suites contain 50 demonstrations per task.  This count
    # is informational for LeRobot logs/model cards; streaming does not sample
    # from it.
    approximate_episodes = len(suites) * 10 * 50
    dataset = LeRobotCachedLiberoDataset(
        shard_paths,
        cache_root=cache_root,
        repo_id=cfg.dataset.repo_id,
        stats=stats,
        num_samples=num_samples,
        chunk_size=chunk_size,
        seed=cfg.seed if cfg.seed is not None else 0,
        repeat=True,
        approximate_episodes=approximate_episodes,
        expected_qwen_tokens=int(cfg.trainable_config.external_kv_token_count),
    )
    return dataset, None
