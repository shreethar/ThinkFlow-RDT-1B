#!/usr/bin/env bash
set -euo pipefail

SOURCE_CACHE_ROOT="${SOURCE_CACHE_ROOT:-cache_features/part_3}"
TARGET_CACHE_ROOT="${TARGET_CACHE_ROOT:-cache_features/part_3_target_state}"
CONFIG="${CONFIG:-configs/part3_rdt1b_lora32_target_state.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/part3_rdt1b_lora32_target_state}"
TARGET_OFFSET="${TARGET_OFFSET:-1}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-validation}"

mkdir -p "${TARGET_CACHE_ROOT}/${TRAIN_SPLIT}" "${TARGET_CACHE_ROOT}/${VAL_SPLIT}" "${OUTPUT_DIR}/audits"

echo "[1/5] Auditing source train cache"
python scripts/audit_cached_action_targets.py \
  --manifest "${SOURCE_CACHE_ROOT}/${TRAIN_SPLIT}/manifest.jsonl" \
  --max-packs "${AUDIT_MAX_PACKS:-32}" \
  --max-samples-per-pack "${AUDIT_MAX_SAMPLES_PER_PACK:-128}" \
  | tee "${OUTPUT_DIR}/audits/source_train_action_audit.txt"

echo "[2/5] Materializing target-state train cache"
python scripts/materialize_target_state_cache.py \
  --input-manifest "${SOURCE_CACHE_ROOT}/${TRAIN_SPLIT}/manifest.jsonl" \
  --output-manifest "${TARGET_CACHE_ROOT}/${TRAIN_SPLIT}/manifest.jsonl" \
  --target-offset "${TARGET_OFFSET}"

echo "[3/5] Materializing target-state validation cache"
python scripts/materialize_target_state_cache.py \
  --input-manifest "${SOURCE_CACHE_ROOT}/${VAL_SPLIT}/manifest.jsonl" \
  --output-manifest "${TARGET_CACHE_ROOT}/${VAL_SPLIT}/manifest.jsonl" \
  --target-offset "${TARGET_OFFSET}"

echo "[4/5] Auditing converted train cache"
python scripts/audit_cached_action_targets.py \
  --manifest "${TARGET_CACHE_ROOT}/${TRAIN_SPLIT}/manifest.jsonl" \
  --max-packs "${AUDIT_MAX_PACKS:-32}" \
  --max-samples-per-pack "${AUDIT_MAX_SAMPLES_PER_PACK:-128}" \
  | tee "${OUTPUT_DIR}/audits/target_state_train_action_audit.txt"

echo "[5/5] Launching target-state training"
python scripts/train_b0_cached_features.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --train-manifest "${TARGET_CACHE_ROOT}/${TRAIN_SPLIT}/manifest.jsonl" \
  --val-manifest "${TARGET_CACHE_ROOT}/${VAL_SPLIT}/manifest.jsonl" \
  "$@"
