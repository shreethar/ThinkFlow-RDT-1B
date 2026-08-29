#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/fractal_b2_from_oxe5k_native128_full.yaml}
CACHE_ROOT=${CACHE_ROOT:-cache_features}
INIT_ARTIFACT=${INIT_ARTIFACT:-output_2/b2_oxe_native128_full/checkpoint-5000}
OUTPUT_DIR=${OUTPUT_DIR:-output_2/fractal_b2_from_oxe5k_native128_full_10k}
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-/home/ubuntu/models/siglip-so400m-patch14-384}
SIGLIP_FALLBACK_MODEL_ID=${SIGLIP_FALLBACK_MODEL_ID:-google/siglip-so400m-patch14-384}
WANDB_PROJECT=${WANDB_PROJECT:-"ThinkLite B2 Fractal"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-fractal-b2-from-oxe5k-native128-full-10k}
MAX_STEPS=${MAX_STEPS:-10000}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

TRAIN_MANIFEST=${TRAIN_MANIFEST:-$CACHE_ROOT/train/manifest.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-$CACHE_ROOT/validation/manifest.jsonl}

for path in "$TRAIN_MANIFEST" "$VAL_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing manifest: $path" >&2; exit 1; }
done
for name in rdt_full.pt interfaces.pt metadata.json; do
  [[ -f "$INIT_ARTIFACT/$name" ]] || {
    echo "Incomplete initialization artifact: $INIT_ARTIFACT/$name" >&2
    exit 1
  }
done
if [[ -z "${RESUME_FROM:-}" ]] && compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
  echo "Refusing to overwrite existing checkpoints in $OUTPUT_DIR" >&2
  echo "Set RESUME_FROM to resume, or choose another OUTPUT_DIR." >&2
  exit 1
fi

uv run --no-sync python - "$CONFIG" "$INIT_ARTIFACT" "$TRAIN_MANIFEST" "$VAL_MANIFEST" <<'PY'
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import sys

from thinkflow_rdt.config import load_config

cfg = load_config(sys.argv[1])
artifact = Path(sys.argv[2])
metadata = json.loads((artifact / "metadata.json").read_text())
old = metadata["config"]["model"]
for key in (
    "action_dim", "state_dim", "cache_action_dim", "cache_state_dim",
    "pred_horizon", "qwen_kv_dim", "qwen_fusion", "state_encoder_layout",
    "action_encoder_layout",
):
    expected = getattr(cfg.model, key)
    if old.get(key) != expected:
        raise ValueError(f"checkpoint/config mismatch for model.{key}: {old.get(key)!r} vs {expected!r}")

for raw in sys.argv[3:]:
    manifest = Path(raw)
    counts = Counter()
    rows = Counter()
    for line in manifest.open():
        if not line.strip():
            continue
        item = json.loads(line)
        counts[str(item.get("dataset_id"))] += int(item["num_samples"])
        rows[str(item.get("dataset_id"))] += 1
        if item.get("feature_type") != "latent_student_spatial_kv":
            raise ValueError(f"{manifest} contains a non-B2 row")
        if int(item.get("qwen_token_count", -1)) != 5:
            raise ValueError(f"{manifest} contains a row without five Qwen tokens")
    if counts.get("fractal", 0) <= 0:
        raise ValueError(f"{manifest} contains no Fractal examples")
    print(
        f"Fractal B2 preflight: {manifest} | rows={rows['fractal']} "
        f"samples={counts['fractal']} (other datasets excluded by config)"
    )
print(f"Initialization artifact verified at source step {metadata.get('global_step')}")
PY

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "Fractal B2 preflight completed; training will not start."
  exit 0
fi

ARTIFACT_ARGS=(--init-artifact "$INIT_ARTIFACT")
if [[ -n "${RESUME_FROM:-}" ]]; then
  ARTIFACT_ARGS=(--resume-from "$RESUME_FROM")
fi

uv run --no-sync python scripts/train_b0_cached_features.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  "${ARTIFACT_ARGS[@]}" \
  --online-siglip \
  --siglip-model-id "$SIGLIP_MODEL_ID" \
  --siglip-fallback-model-id "$SIGLIP_FALLBACK_MODEL_ID" \
  --max-steps "$MAX_STEPS" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --skip-nonfinite-updates \
  --log-gradient-stats \
  --report-to wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME"
