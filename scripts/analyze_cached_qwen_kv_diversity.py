#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thinkflow_rdt.data import CachedFeatureDataset  # noqa: E402


def json_default(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def resolve_manifest_line_path(
    item: object,
    *,
    manifest_dir: Path,
) -> tuple[object, Path]:
    if isinstance(item, str):
        path = Path(item)
        resolved = path if path.is_absolute() else (manifest_dir / path).resolve()
        return str(resolved), resolved
    if not isinstance(item, dict):
        raise TypeError(f"Manifest line must be a JSON string/object, got {type(item)}")
    path_value = item.get("path")
    if not path_value:
        raise ValueError(f"Manifest object has no path: {item}")
    path = Path(str(path_value))
    resolved = path if path.is_absolute() else (manifest_dir / path).resolve()
    rewritten = dict(item)
    rewritten["path"] = str(resolved)
    return rewritten, resolved


def merge_manifests(input_manifests: list[Path], output_manifest: Path) -> int:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_manifest.open("w", encoding="utf-8") as out:
        for manifest in input_manifests:
            manifest = manifest.expanduser().resolve()
            if not manifest.exists():
                raise FileNotFoundError(manifest)
            with manifest.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    item = json.loads(stripped)
                    rewritten, resolved = resolve_manifest_line_path(
                        item,
                        manifest_dir=manifest.parent,
                    )
                    if not resolved.exists():
                        raise FileNotFoundError(
                            f"{manifest}:{line_number} points to missing cache file {resolved}"
                        )
                    out.write(json.dumps(rewritten) + "\n")
                    count += 1
    return count


def manifests_from_cache_roots(cache_roots: list[Path], *, split: str) -> list[Path]:
    manifests = []
    for root in cache_roots:
        manifest = root.expanduser().resolve() / split / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        manifests.append(manifest)
    return manifests


def manifests_from_cache_parts_root(parts_root: Path, *, parts: list[int], split: str) -> list[Path]:
    root = parts_root.expanduser().resolve()
    manifests = []
    for part in parts:
        manifest = root / f"part_{int(part)}" / split / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        manifests.append(manifest)
    return manifests


def selected_indices(dataset_size: int, *, count: int, seed: int, mode: str) -> list[int]:
    count = min(max(0, count), dataset_size)
    if mode == "first":
        return list(range(count))
    if mode == "even":
        if count <= 1:
            return [0] if count == 1 else []
        return [round(index * (dataset_size - 1) / (count - 1)) for index in range(count)]
    rng = random.Random(seed)
    return sorted(rng.sample(range(dataset_size), count))


def prepare_kv(value: Any, *, pool: str) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.as_tensor(value).detach().float().cpu()
    if tensor.ndim == 1:
        tokens = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tokens = tensor
    else:
        raise ValueError(f"Expected qwen_kv [D] or [T,D], got {tuple(tensor.shape)}")
    if pool == "mean":
        pooled = tokens.mean(dim=0)
    elif pool == "first":
        pooled = tokens[0]
    elif pool == "flatten":
        pooled = tokens.flatten()
    else:
        raise ValueError(f"Unsupported pool mode: {pool}")
    return tokens, pooled


def scalar_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {"count": 0.0}
    return {
        "count": float(values.numel()),
        "min": float(values.min()),
        "p01": float(values.quantile(0.01)),
        "p05": float(values.quantile(0.05)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
        "std": float(values.std(unbiased=False)),
    }


def sample_pair_indices(n: int, count: int, seed: int) -> list[tuple[int, int]]:
    if n < 2 or count <= 0:
        return []
    rng = random.Random(seed)
    max_pairs = n * (n - 1) // 2
    count = min(count, max_pairs)
    if max_pairs <= count * 4:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(pairs)
        return pairs[:count]
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < count:
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def pairwise_cosine_sample(
    normalized: torch.Tensor,
    *,
    pair_count: int,
    seed: int,
) -> torch.Tensor:
    pairs = sample_pair_indices(normalized.shape[0], pair_count, seed)
    if not pairs:
        return torch.empty(0)
    left = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    right = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    return (normalized[left] * normalized[right]).sum(dim=-1)


def full_offdiag_cosine_stats(normalized: torch.Tensor) -> dict[str, float] | None:
    n = normalized.shape[0]
    if n < 2 or n > 4096:
        return None
    cosine = normalized @ normalized.T
    mask = ~torch.eye(n, dtype=torch.bool)
    return scalar_stats(cosine[mask])


def top_neighbors(normalized: torch.Tensor, metadata: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    n = normalized.shape[0]
    if n < 2 or top_k <= 0:
        return []
    result = []
    block = 1024
    for start in range(0, n, block):
        stop = min(start + block, n)
        cosine = normalized[start:stop] @ normalized.T
        row_indices = torch.arange(start, stop)
        cosine[torch.arange(stop - start), row_indices] = -float("inf")
        values, indices = cosine.max(dim=1)
        for local_index, (value, neighbor) in enumerate(zip(values.tolist(), indices.tolist())):
            sample_index = start + local_index
            result.append(
                {
                    "cosine": float(value),
                    "sample_index": sample_index,
                    "neighbor_index": int(neighbor),
                    "sample": metadata[sample_index],
                    "neighbor": metadata[int(neighbor)],
                }
            )
    return sorted(result, key=lambda item: item["cosine"], reverse=True)[:top_k]


def effective_rank(vectors: torch.Tensor) -> dict[str, Any]:
    n, d = vectors.shape
    if n < 2:
        return {"rank": 0, "participation_ratio": 0.0}
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    # SVD on the smaller covariance side is much cheaper when D is large.
    if n <= d:
        covariance = centered @ centered.T / max(n - 1, 1)
    else:
        covariance = centered.T @ centered / max(n - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).float()
    total = eigenvalues.sum()
    if float(total) <= 0:
        return {"rank": 0, "participation_ratio": 0.0}
    probs = eigenvalues / total
    entropy_rank = torch.exp(-(probs * probs.clamp_min(1e-30).log()).sum())
    participation = total.pow(2) / eigenvalues.pow(2).sum().clamp_min(1e-30)
    top = torch.flip(eigenvalues, dims=[0])[:20]
    explained = top / total
    return {
        "entropy_effective_rank": float(entropy_rank),
        "participation_ratio": float(participation),
        "top20_explained_variance": explained.tolist(),
        "nonzero_eigenvalues_gt_1e-8": int((eigenvalues > 1e-8).sum()),
    }


def quantized_duplicate_rate(vectors: torch.Tensor, *, decimals: int) -> dict[str, Any]:
    rounded = torch.round(vectors.float() * (10**decimals)).to(torch.int32)
    seen: set[bytes] = set()
    duplicates = 0
    for row in rounded:
        payload = row.numpy().tobytes()
        if payload in seen:
            duplicates += 1
        else:
            seen.add(payload)
    return {
        "decimals": decimals,
        "unique": len(seen),
        "duplicates": duplicates,
        "duplicate_rate": duplicates / max(vectors.shape[0], 1),
    }


def grouped_pair_stats(
    normalized: torch.Tensor,
    metadata: list[dict[str, Any]],
    *,
    key: str,
    max_groups: int,
    pair_count_per_group: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        value = row.get(key)
        if value is not None:
            groups[str(value)].append(index)
    eligible = [(group_key, indices) for group_key, indices in groups.items() if len(indices) >= 2]
    eligible.sort(key=lambda item: len(item[1]), reverse=True)
    summary = {}
    for offset, (group_key, indices) in enumerate(eligible[:max_groups]):
        subset = normalized[torch.tensor(indices, dtype=torch.long)]
        values = pairwise_cosine_sample(
            subset,
            pair_count=pair_count_per_group,
            seed=seed + offset,
        )
        summary[group_key] = {
            "samples": len(indices),
            "pair_cosine": scalar_stats(values),
        }
    return summary


def metadata_from_sample(sample: dict[str, Any], index: int) -> dict[str, Any]:
    row = {
        "selected_index": index,
        "dataset_id": sample.get("dataset_id"),
        "episode_id": sample.get("episode_id"),
        "step_idx": sample.get("step_idx"),
        "path": sample.get("_path"),
    }
    for key in (
        "qwen_cache_scope",
        "qwen_anchor_kind",
        "qwen_anchor_original_kind",
        "qwen_anchor_step_idx",
        "qwen_anchor_count",
    ):
        if key in sample:
            row[key] = sample[key]
    return row


def analyze_vectors(
    *,
    pooled: torch.Tensor,
    token_tensor: torch.Tensor | None,
    metadata: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    norms = pooled.norm(dim=-1)
    normalized = torch.nn.functional.normalize(pooled, dim=-1)
    pair_values = pairwise_cosine_sample(
        normalized,
        pair_count=args.pair_count,
        seed=args.seed,
    )
    report: dict[str, Any] = {
        "shape": list(pooled.shape),
        "norms": scalar_stats(norms),
        "pair_cosine_sample": scalar_stats(pair_values),
        "full_pair_cosine": full_offdiag_cosine_stats(normalized),
        "effective_rank": effective_rank(pooled),
        "quantized_duplicates": [
            quantized_duplicate_rate(pooled, decimals=decimals)
            for decimals in args.duplicate_decimals
        ],
        "top_nearest_neighbors": top_neighbors(
            normalized,
            metadata,
            top_k=args.top_k,
        ),
        "grouped_similarity": {
            "dataset_id": grouped_pair_stats(
                normalized,
                metadata,
                key="dataset_id",
                max_groups=args.max_groups,
                pair_count_per_group=args.group_pair_count,
                seed=args.seed + 111,
            ),
            "episode_id": grouped_pair_stats(
                normalized,
                metadata,
                key="episode_id",
                max_groups=args.max_groups,
                pair_count_per_group=args.group_pair_count,
                seed=args.seed + 222,
            ),
        },
    }
    if token_tensor is not None and token_tensor.ndim == 3:
        token_count = token_tensor.shape[1]
        per_token = {}
        for token_index in range(token_count):
            token_vectors = token_tensor[:, token_index, :]
            token_norm = torch.nn.functional.normalize(token_vectors, dim=-1)
            per_token[str(token_index)] = {
                "shape": list(token_vectors.shape),
                "norms": scalar_stats(token_vectors.norm(dim=-1)),
                "pair_cosine_sample": scalar_stats(
                    pairwise_cosine_sample(
                        token_norm,
                        pair_count=args.pair_count,
                        seed=args.seed + 1000 + token_index,
                    )
                ),
                "effective_rank": effective_rank(token_vectors),
            }
        report["per_token"] = per_token
        if token_count >= 2:
            intra = []
            for sample_tokens in token_tensor:
                token_norm = torch.nn.functional.normalize(sample_tokens, dim=-1)
                cosine = token_norm @ token_norm.T
                mask = ~torch.eye(token_count, dtype=torch.bool)
                intra.append(cosine[mask])
            report["within_sample_token_pair_cosine"] = scalar_stats(torch.cat(intra))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze raw cached Qwen/LatentStudent KV diversity before the RDT "
            "Qwen projector. Low diversity means the cache itself may not carry "
            "sample-specific information; high diversity plus weak ablation points "
            "toward fusion/projector/training usage."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--cache-root", type=Path, action="append", default=[])
    parser.add_argument("--cache-parts-root", type=Path)
    parser.add_argument("--cache-parts", type=int, nargs="+", choices=[1, 2, 3], default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--sample-mode", choices=["random", "first", "even"], default="random")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--pool",
        choices=["mean", "first", "flatten"],
        default="mean",
        help="How to collapse multi-token KV to one vector for sample-level similarity.",
    )
    parser.add_argument("--pair-count", type=int, default=200000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-groups", type=int, default=10)
    parser.add_argument("--group-pair-count", type=int, default=20000)
    parser.add_argument("--duplicate-decimals", type=int, nargs="+", default=[3, 4, 5])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifests = [path.expanduser().resolve() for path in args.manifest]
    manifests.extend(manifests_from_cache_roots(args.cache_root, split=args.split))
    if args.cache_parts_root is not None:
        manifests.extend(
            manifests_from_cache_parts_root(
                args.cache_parts_root,
                parts=args.cache_parts or [1, 2, 3],
                split=args.split,
            )
        )
    if not manifests:
        raise ValueError("Provide --manifest, --cache-root, or --cache-parts-root")

    merged_manifest = args.output.parent / f"merged_{args.split}_kv_manifest.jsonl"
    merged_rows = merge_manifests(manifests, merged_manifest)
    dataset = CachedFeatureDataset(merged_manifest, required_keys={"qwen_kv"})
    indices = selected_indices(
        len(dataset),
        count=args.num_samples,
        seed=args.seed,
        mode=args.sample_mode,
    )
    token_tensors = []
    pooled_tensors = []
    metadata = []
    token_shapes: dict[str, int] = defaultdict(int)
    for offset, index in enumerate(indices):
        sample = dataset[index]
        tokens, pooled = prepare_kv(sample["qwen_kv"], pool=args.pool)
        token_shapes[str(tuple(tokens.shape))] += 1
        token_tensors.append(tokens)
        pooled_tensors.append(pooled)
        metadata.append(metadata_from_sample(sample, index))
        if (offset + 1) % 500 == 0:
            print(f"loaded {offset + 1}/{len(indices)} samples", flush=True)

    pooled = torch.stack(pooled_tensors, dim=0).float()
    token_tensor = None
    if len({tuple(tokens.shape) for tokens in token_tensors}) == 1:
        token_tensor = torch.stack(token_tensors, dim=0).float()

    report = {
        "manifests": [str(path) for path in manifests],
        "merged_manifest": str(merged_manifest.resolve()),
        "merged_manifest_rows": merged_rows,
        "dataset_samples": len(dataset),
        "selected_samples": len(indices),
        "selected_indices_preview": indices[:20],
        "pool": args.pool,
        "token_shapes": dict(token_shapes),
        "analysis": analyze_vectors(
            pooled=pooled,
            token_tensor=token_tensor,
            metadata=metadata,
            args=args,
        ),
        "interpretation_hints": [
            "Pair cosine near 1.0 and low effective rank indicate low KV diversity.",
            "High diversity but RDT ablation shuffle_qwen ~= baseline suggests projector/fusion/training is not using sample-specific KV.",
            "B2 spatial tokens should ideally show lower pair cosine and higher effective rank than B0 </think> KV.",
            "If nearest neighbors mostly come from the same episode/instruction, KV may encode episode/task context rather than action-relevant sample details.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, default=json_default) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    pair_stats = report["analysis"]["pair_cosine_sample"]
    rank = report["analysis"]["effective_rank"]
    print(
        "pair cosine sample: "
        f"mean={pair_stats.get('mean', 0):.6f} "
        f"median={pair_stats.get('median', 0):.6f} "
        f"p95={pair_stats.get('p95', 0):.6f} "
        f"p99={pair_stats.get('p99', 0):.6f}"
    )
    print(
        "effective rank: "
        f"entropy={rank.get('entropy_effective_rank', 0):.2f} "
        f"participation={rank.get('participation_ratio', 0):.2f}"
    )
    print(f"token shapes: {dict(token_shapes)}")


if __name__ == "__main__":
    main()
