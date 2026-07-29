#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-dataset/hf_parts}"
PARTS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_dataset_part.sh [1|2|3|all]
  bash scripts/download_dataset_part.sh --part 3 --dataset-root dataset/hf_parts

Defaults to part 3 when no part is provided.
EOF
}

add_part() {
  case "$1" in
    1|2|3)
      PARTS+=("$1")
      ;;
    all)
      PARTS+=(1 2 3)
      ;;
    *)
      echo "Dataset part must be 1, 2, 3, or all; got: $1" >&2
      exit 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --part)
      add_part "$2"
      shift 2
      ;;
    --all)
      add_part all
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      add_part "$1"
      shift
      ;;
  esac
done

if [[ ${#PARTS[@]} -eq 0 ]]; then
  PARTS=(3)
fi

mkdir -p "${DATASET_ROOT}"

for part in "${PARTS[@]}"; do
  repo="shreethar/FYP-Stage-3-part-${part}"
  local_dir="${DATASET_ROOT}/part_${part}"
  echo "Downloading dataset part ${part}:"
  echo "  repo: ${repo}"
  echo "  dir:  ${local_dir}"
  uv run hf download "${repo}" --repo-type dataset --local-dir "${local_dir}"
done

echo "Dataset downloads complete."
