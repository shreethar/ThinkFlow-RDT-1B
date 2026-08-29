#!/usr/bin/env bash
set -euo pipefail

# Continue the original 20K LIBERO B0 model with simulator-labelled spatial
# recovery examples. Five thousand updates is the default decision point. Set
# MAX_STEPS=10000 only after checkpoint-5000 improves fixed online rollouts.

CACHE_ROOT=${CACHE_ROOT:-cache_features_libero_b0_raw_ortho6d}
RECOVERY_CACHE_ROOT=${RECOVERY_CACHE_ROOT:-cache_features_libero_b0_recovery}
CONFIG=${CONFIG:-configs/libero_b0_native128_recovery_continue.yaml}
INIT_ARTIFACT=${INIT_ARTIFACT:-output_2/libero_b0_from_oxe20k_v2/checkpoint-20000}
OUTPUT_DIR=${OUTPUT_DIR:-output_2/libero_b0_recovery_from_libero20k_5k}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}

MAX_STEPS=${MAX_STEPS:-5000}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
LEARNING_RATE_INTERFACES=${LEARNING_RATE_INTERFACES:-5e-6}
XYZ_LOSS_WEIGHT=${XYZ_LOSS_WEIGHT:-1.0}
HORIZON_LOSS_SCHEDULE=${HORIZON_LOSS_SCHEDULE:-"1-10:4,11-20:2,21-64:0.5"}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
VALIDATION_BATCH_SIZE=${VALIDATION_BATCH_SIZE:-32}
VALIDATION_SAMPLES=${VALIDATION_SAMPLES:-256}
QUALITATIVE_VALIDATION_EXAMPLES=${QUALITATIVE_VALIDATION_EXAMPLES:-32}
VALIDATE_EVERY=${VALIDATE_EVERY:-500}
SAVE_EVERY=${SAVE_EVERY:-1000}
WARMUP_STEPS=${WARMUP_STEPS:-250}
NUM_WORKERS=${NUM_WORKERS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-ThinkLite B0 LIBERO Recovery}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-libero-b0-20k-recovery-plus5k}
SIGLIP_MODEL_ID=${SIGLIP_MODEL_ID:-/home/ubuntu/models/siglip-so400m-patch14-384}
SIGLIP_FALLBACK_MODEL_ID=${SIGLIP_FALLBACK_MODEL_ID:-google/siglip-so400m-patch14-384}
RESUME_FROM=${RESUME_FROM:-}
SKIP_CACHE_PREFLIGHT=${SKIP_CACHE_PREFLIGHT:-0}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

if [[ "$MAX_STEPS" != "5000" && "$MAX_STEPS" != "10000" ]]; then
  echo "Warning: expected MAX_STEPS=5000 or 10000; got $MAX_STEPS" >&2
fi

CACHE_ARGS=()
for suite in $SUITES; do
  suite_root="$CACHE_ROOT/$suite"
  for split in train validation; do
    if [[ ! -f "$suite_root/$split/manifest.jsonl" ]]; then
      echo "Missing original cache manifest: $suite_root/$split/manifest.jsonl" >&2
      exit 1
    fi
  done
  CACHE_ARGS+=(--cache-root "$suite_root")
done
for split in train validation; do
  if [[ ! -f "$RECOVERY_CACHE_ROOT/$split/manifest.jsonl" ]]; then
    echo "Missing recovery cache manifest: $RECOVERY_CACHE_ROOT/$split/manifest.jsonl" >&2
    echo "Run scripts/run_generate_libero_recovery_cache.sh first." >&2
    exit 1
  fi
done
if [[ ! -f "$RECOVERY_CACHE_ROOT/precompute_metadata.json" ]]; then
  echo "Recovery cache is incomplete (missing precompute_metadata.json): $RECOVERY_CACHE_ROOT" >&2
  exit 1
fi
CACHE_ARGS+=(--cache-root "$RECOVERY_CACHE_ROOT")

for required in rdt_full.pt interfaces.pt metadata.json; do
  if [[ ! -f "$INIT_ARTIFACT/$required" ]]; then
    echo "Incomplete initialization artifact: $INIT_ARTIFACT/$required" >&2
    exit 1
  fi
done

if [[ -z "$RESUME_FROM" ]] && compgen -G "$OUTPUT_DIR/checkpoint-*" > /dev/null; then
  echo "Refusing to start a fresh continuation in an existing checkpoint directory: $OUTPUT_DIR" >&2
  echo "Set RESUME_FROM to one of its checkpoints or choose a new OUTPUT_DIR." >&2
  exit 1
fi

if [[ "$SKIP_CACHE_PREFLIGHT" != "1" ]]; then
  uv run --no-sync python - "$RECOVERY_CACHE_ROOT" "$INIT_ARTIFACT" "$CONFIG" <<'PY'
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import torch

from thinkflow_rdt.config import load_config

root = Path(sys.argv[1]).expanduser().resolve()
artifact = Path(sys.argv[2]).expanduser().resolve()
cfg = load_config(sys.argv[3])
metadata = json.loads((root / "precompute_metadata.json").read_text())
if metadata.get("generator") != "offline_translation_perturbation_feedback_oracle_v1":
    raise ValueError("Recovery cache was produced by an unexpected generator")
if metadata.get("strict_dagger") is not False:
    raise ValueError("Expected offline recovery cache metadata with strict_dagger=false")

split_episodes: dict[str, set[str]] = {}
for split, expected_unique, expected_repeat in (
    ("train", 1024, int(metadata["config"]["train_repeat"])),
    ("validation", 128, 1),
):
    split_dir = root / split
    unique_manifest = split_dir / "manifest_unique.jsonl"
    repeated_manifest = split_dir / "manifest.jsonl"
    unique_lines = [
        line for line in unique_manifest.read_text().splitlines() if line.strip()
    ]
    repeated_lines = [
        line for line in repeated_manifest.read_text().splitlines() if line.strip()
    ]
    if len(repeated_lines) != len(unique_lines) * expected_repeat:
        raise ValueError(
            f"{split} manifest repetition mismatch: {len(repeated_lines)} vs "
            f"{len(unique_lines)} x {expected_repeat}"
        )
    if Counter(repeated_lines) != Counter(
        {line: expected_repeat for line in unique_lines}
    ):
        raise ValueError(f"{split} manifest does not repeat every unique shard equally")

    sample_count = 0
    dataset_ids: set[str] = set()
    episodes: set[str] = set()
    for line in unique_lines:
        item = json.loads(line)
        path = Path(item["path"])
        path = path if path.is_absolute() else split_dir / path
        pack = torch.load(path, map_location="cpu", weights_only=True)
        n = int(pack["num_samples"])
        sample_count += n
        qwen = torch.as_tensor(pack["qwen_kv"])
        state = torch.as_tensor(pack["state"])
        actions = torch.as_tensor(pack["actions"])
        time_mask = torch.as_tensor(pack["action_time_mask"], dtype=torch.bool)
        image_mask = torch.as_tensor(pack["sample_image_mask"], dtype=torch.bool)
        expected_shapes = {
            "qwen_kv": ((n, 1, cfg.model.qwen_kv_dim), tuple(qwen.shape)),
            "state": ((n, cfg.model.resolved_cache_state_dim), tuple(state.shape)),
            "actions": (
                (n, cfg.model.pred_horizon, cfg.model.resolved_cache_action_dim),
                tuple(actions.shape),
            ),
            "action_time_mask": ((n, cfg.model.pred_horizon), tuple(time_mask.shape)),
            "sample_image_mask": ((n, 6), tuple(image_mask.shape)),
        }
        for name, (expected, actual) in expected_shapes.items():
            if actual != expected:
                raise ValueError(f"{path}: {name} {actual}, expected {expected}")
        if qwen.dtype != torch.bfloat16 or not torch.isfinite(qwen.float()).all():
            raise ValueError(f"{path}: Qwen KV is not finite bfloat16")
        if not torch.isfinite(state).all() or not torch.isfinite(actions).all():
            raise ValueError(f"{path}: state/actions contain non-finite values")
        if not bool(time_mask.all()):
            raise ValueError(f"{path}: recovery horizon is unexpectedly padded")
        if not bool((image_mask.sum(dim=1) > 0).all()):
            raise ValueError(f"{path}: a recovery sample has no valid observation image")
        for language in pack["lang_tokens"]:
            language = torch.as_tensor(language)
            if language.ndim != 2 or language.shape[-1] != cfg.model.lang_token_dim:
                raise ValueError(f"{path}: invalid cached T5 shape {tuple(language.shape)}")
            if not torch.isfinite(language.float()).all():
                raise ValueError(f"{path}: cached T5 contains non-finite values")
        for sample_metadata in pack["metadata"]:
            recovery = sample_metadata.get("recovery", {})
            if recovery.get("accepted") is not True:
                raise ValueError(f"{path}: contains a recovery sample that failed its gate")
            dataset_ids.add(str(sample_metadata["dataset_id"]))
            episodes.add(str(sample_metadata["episode_id"]))
    if sample_count != expected_unique:
        raise ValueError(
            f"{split} unique sample count is {sample_count}, expected {expected_unique}"
        )
    expected_datasets = {
        "libero_spatial_recovery",
        "libero_object_recovery",
        "libero_goal_recovery",
        "libero_10_recovery",
    }
    if dataset_ids != expected_datasets:
        raise ValueError(f"{split} recovery datasets are {sorted(dataset_ids)}")
    split_episodes[split] = episodes
    print(
        f"Recovery cache verified: split={split} unique_samples={sample_count} "
        f"effective_presentations={sample_count * expected_repeat} "
        f"datasets={sorted(dataset_ids)}"
    )

overlap = split_episodes["train"] & split_episodes["validation"]
if overlap:
    raise ValueError(f"Recovery train/validation episode leakage: {sorted(overlap)[:3]}")

checkpoint_metadata = json.loads((artifact / "metadata.json").read_text())
checkpoint_model = checkpoint_metadata.get("config", {}).get("model", {})
for key, expected in {
    "action_dim": cfg.model.action_dim,
    "state_dim": cfg.model.state_dim,
    "pred_horizon": cfg.model.pred_horizon,
    "qwen_kv_dim": cfg.model.qwen_kv_dim,
    "qwen_fusion": cfg.model.qwen_fusion,
    "state_encoder_layout": cfg.model.state_encoder_layout,
    "action_encoder_layout": cfg.model.action_encoder_layout,
    "cache_state_dim": cfg.model.resolved_cache_state_dim,
    "cache_action_dim": cfg.model.resolved_cache_action_dim,
}.items():
    if checkpoint_model.get(key) != expected:
        raise ValueError(
            f"Initialization checkpoint mismatch for model.{key}: "
            f"{checkpoint_model.get(key)!r} vs {expected!r}"
        )
print(
    "Initialization checkpoint verified: "
    f"{artifact} at step {checkpoint_metadata.get('global_step')}"
)
PY
fi

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "Recovery continuation preflight completed; training will not start."
  exit 0
fi

ARTIFACT_ARGS=(--init-artifact "$INIT_ARTIFACT")
if [[ -n "$RESUME_FROM" ]]; then
  ARTIFACT_ARGS=(--resume-from "$RESUME_FROM")
fi

uv run --no-sync python scripts/train_b0_cached_features.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  "${CACHE_ARGS[@]}" \
  --online-siglip \
  --siglip-model-id "$SIGLIP_MODEL_ID" \
  --siglip-fallback-model-id "$SIGLIP_FALLBACK_MODEL_ID" \
  --max-steps "$MAX_STEPS" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --learning-rate-interfaces "$LEARNING_RATE_INTERFACES" \
  --warmup-steps "$WARMUP_STEPS" \
  --horizon-loss-schedule "$HORIZON_LOSS_SCHEDULE" \
  --xyz-loss-weight "$XYZ_LOSS_WEIGHT" \
  --mask-noisy-gripper-input \
  --gripper-bce-weight 1.0 \
  --gripper-bce-logit-scale 5.0 \
  --rotation-geodesic-weight 1.0 \
  --validation-batch-size "$VALIDATION_BATCH_SIZE" \
  --validation-samples "$VALIDATION_SAMPLES" \
  --sample-validation-batches 1 \
  --qualitative-validation-examples "$QUALITATIVE_VALIDATION_EXAMPLES" \
  --validate-every "$VALIDATE_EVERY" \
  --save-every "$SAVE_EVERY" \
  --log-every 10 \
  --num-workers "$NUM_WORKERS" \
  --pin-memory \
  --persistent-workers \
  --skip-nonfinite-updates \
  --log-gradient-stats \
  --report-to wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  "${ARTIFACT_ARGS[@]}"
