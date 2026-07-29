#!/usr/bin/env bash
set -euo pipefail

SRC="${SRC:-/workspace/model/LatentStudent-ckpt-400}"
DST="${DST:-/workspace/model/LatentStudent-ckpt-400-fixed}"
PROCESSOR_SRC="${PROCESSOR_SRC:-/workspace/model/stage1_unsloth}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      SRC="$2"
      shift 2
      ;;
    --dst)
      DST="$2"
      shift 2
      ;;
    --processor-src)
      PROCESSOR_SRC="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash scripts/convert_model.sh [--src PATH] [--dst PATH] [--processor-src PATH]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

uv run python scripts/convert_latent_student_checkpoint.py \
  --src "${SRC}" \
  --dst "${DST}" \
  --processor-src "${PROCESSOR_SRC}"
