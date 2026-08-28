#!/usr/bin/env bash
set -euo pipefail

# Full Fast-ThinkAct B2 post-training from cached five-spatial-token Qwen KV.
# B2_CACHE_ROOT must contain train/ and validation/ episode-pack directories.

CONFIG=${CONFIG:-configs/part3_rdt1b.yaml}
B2_CACHE_ROOT=${B2_CACHE_ROOT:-cache_features}
OUTPUT_DIR=${OUTPUT_DIR:-output_2/b2_oxe_native128_full}
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-/home/ubuntu/models/siglip-so400m-patch14-384}
SIGLIP_FALLBACK_MODEL_ID=${SIGLIP_FALLBACK_MODEL_ID:-google/siglip-so400m-patch14-384}
WANDB_PROJECT=${WANDB_PROJECT:-"ThinkLite B2 OXE"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-oxe-b2-rdt1b-native128-fastthinkact-full}
WANDB_ENTITY=${WANDB_ENTITY:-}
RESUME_FROM=${RESUME_FROM:-}
SKIP_CACHE_PREFLIGHT=${SKIP_CACHE_PREFLIGHT:-0}

resolve_split_manifest() {
  local split_name=$1
  local preferred=$2
  local candidate
  for candidate in \
    "$B2_CACHE_ROOT/$split_name/manifest.jsonl" \
    "$B2_CACHE_ROOT/$split_name/$preferred" \
    "$B2_CACHE_ROOT/$preferred"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf 'No %s manifest found under B2_CACHE_ROOT=%s\n' \
    "$split_name" "$B2_CACHE_ROOT" >&2
  return 1
}

TRAIN_MANIFEST=${TRAIN_MANIFEST:-$(resolve_split_manifest train train_manifest.jsonl)}
VAL_MANIFEST=${VAL_MANIFEST:-$(resolve_split_manifest validation validation_manifest.jsonl)}

if [[ -z "$RESUME_FROM" ]] && compgen -G "$OUTPUT_DIR/checkpoint-*" > /dev/null; then
  echo "Refusing to start a fresh run in an existing checkpoint directory: $OUTPUT_DIR" >&2
  echo "Set RESUME_FROM to a checkpoint or choose a new OUTPUT_DIR." >&2
  exit 1
fi

if [[ "$SKIP_CACHE_PREFLIGHT" != "1" ]]; then
  uv run --no-sync python - "$TRAIN_MANIFEST" "$VAL_MANIFEST" <<'PY'
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import torch

EXPECTED_DATASETS = {"bc_z", "bridge", "droid", "fractal", "kuka"}
EXPECTED_FEATURE = "latent_student_spatial_kv"


def resolve_record_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def inspect_manifest(raw_path: str) -> tuple[int, int, Counter[str]]:
    manifest = Path(raw_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    rows = 0
    samples = 0
    datasets: Counter[str] = Counter()
    representative: dict[str, Path] = {}
    for line_number, line in enumerate(manifest.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("feature_type") != EXPECTED_FEATURE:
            raise ValueError(
                f"{manifest}:{line_number} is not B2: "
                f"feature_type={item.get('feature_type')!r}"
            )
        if int(item.get("qwen_token_count", -1)) != 5:
            raise ValueError(f"{manifest}:{line_number} does not contain five Qwen tokens")
        if int(item.get("qwen_kv_dim", -1)) != 2048:
            raise ValueError(f"{manifest}:{line_number} has the wrong Qwen KV width")
        if item.get("qwen_cache_scope") != "per_sample":
            raise ValueError(f"{manifest}:{line_number} is not per-sample Qwen KV")

        dataset_id = str(item.get("dataset_id"))
        path = resolve_record_path(manifest, str(item["path"]))
        if not path.is_file():
            raise FileNotFoundError(
                f"{manifest}:{line_number} points to missing B2 shard {path}"
            )
        representative.setdefault(dataset_id, path)
        rows += 1
        samples += int(item["num_samples"])
        datasets[dataset_id] += int(item["num_samples"])

    missing_datasets = EXPECTED_DATASETS - set(datasets)
    if missing_datasets:
        raise ValueError(f"{manifest} is missing datasets: {sorted(missing_datasets)}")

    for dataset_id, path in representative.items():
        pack = torch.load(path, map_location="cpu", weights_only=True)
        num_samples = int(pack["num_samples"])
        qwen_kv = torch.as_tensor(pack["qwen_anchor_kv"])
        if tuple(qwen_kv.shape) != (num_samples, 5, 2048):
            raise ValueError(
                f"{path} has Qwen KV {tuple(qwen_kv.shape)}, expected "
                f"({num_samples}, 5, 2048)"
            )
        if qwen_kv.dtype != torch.bfloat16:
            raise ValueError(f"{path} Qwen KV is {qwen_kv.dtype}, expected bfloat16")
        if not torch.isfinite(qwen_kv.float()).all():
            raise ValueError(f"{path} contains non-finite Qwen KV")
        anchors = torch.as_tensor(pack["sample_anchor_index"], dtype=torch.long)
        if tuple(anchors.shape) != (num_samples,):
            raise ValueError(f"{path} has invalid sample_anchor_index shape")
        if anchors.min().item() < 0 or anchors.max().item() >= num_samples:
            raise ValueError(f"{path} contains invalid per-sample anchor indices")

    print(
        f"B2 cache verified: {manifest} | rows={rows} samples={samples} "
        f"datasets={dict(sorted(datasets.items()))}"
    )
    return rows, samples, datasets


inspect_manifest(sys.argv[1])
inspect_manifest(sys.argv[2])
PY
fi

if [[ -n "$WANDB_ENTITY" ]]; then
  export WANDB_ENTITY
fi

RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
  RESUME_ARGS+=(--resume-from "$RESUME_FROM")
fi

uv run --no-sync python scripts/train_b0_cached_features.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  "${RESUME_ARGS[@]}" \
  --online-siglip \
  --siglip-model-id "$SIGLIP_MODEL_ID" \
  --siglip-fallback-model-id "$SIGLIP_FALLBACK_MODEL_ID" \
  --learning-rate 1e-4 \
  --max-steps 20000 \
  --micro-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --global-batch-size 32 \
  --validate-every 500 \
  --validation-batch-size 32 \
  --validation-samples 256 \
  --save-every 1000 \
  --sample-validation-batches 1 \
  --qualitative-validation-examples 32 \
  --skip-nonfinite-updates \
  --log-gradient-stats \
  --report-to wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME"
