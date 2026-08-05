#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(command -v python || command -v python3)"
fi

DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data}"
FEATURE_ROOT="${FEATURE_ROOT:-${DATA_ROOT}/feature_extraction}"
DEFAULT_CACHE_ROOT="/share/home/tm902089733300000/a903202310/lys/research"
DEFAULT_CACHE_ROOT+="/Unsupervised-hypergraph/outputs/attention_cache"
DEFAULT_CACHE_ROOT+="/fresh_attention_c8847872bedf_20260731T074520Z_p876"
ATTENTION_CACHE_ROOT="${ATTENTION_CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"
CACHE_TAG="$(basename -- "${ATTENTION_CACHE_ROOT}")"
TAU="${TAU:-0.05}"
LIMIT="${LIMIT:-}"
GRAPH_TAG="${CACHE_TAG}_tau${TAU/./p}"
if [[ -n "${LIMIT}" ]]; then
  GRAPH_TAG+="_limit${LIMIT}"
fi
GRAPH_DIR="${GRAPH_DIR:-${FEATURE_ROOT}/ragtruth_original_attribute_graphs/${GRAPH_TAG}}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAINING_DIR="${TRAINING_DIR:-${GRAPH_DIR}/training/${RUN_TAG}}"

DEVICE="${DEVICE:-cuda}"
BUILD_DEVICE="${BUILD_DEVICE:-${DEVICE}}"
SPLITS="${SPLITS:-train test}"
BUILD_ONLY="${BUILD_ONLY:-0}"
AUTO_PREPARE_CACHE="${AUTO_PREPARE_CACHE:-1}"
ALLOW_PARTIAL_CACHE="${ALLOW_PARTIAL_CACHE:-0}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.20}"

read -r -a SPLIT_ARGS <<< "${SPLITS}"
read -r -a SEED_ARGS <<< "${SEEDS}"

if [[ "${AUTO_PREPARE_CACHE}" != "0" && "${AUTO_PREPARE_CACHE}" != "1" ]]; then
  printf 'AUTO_PREPARE_CACHE must be 0 or 1, got %s\n' \
    "${AUTO_PREPARE_CACHE}" >&2
  exit 1
fi
if [[ "${ALLOW_PARTIAL_CACHE}" != "0" && "${ALLOW_PARTIAL_CACHE}" != "1" ]]; then
  printf 'ALLOW_PARTIAL_CACHE must be 0 or 1, got %s\n' \
    "${ALLOW_PARTIAL_CACHE}" >&2
  exit 1
fi

cache_split_complete() {
  "${PYTHON_BIN}" -m original.cli audit \
    --cache-root "${ATTENTION_CACHE_ROOT}" \
    --splits "$1" >/dev/null 2>&1
}

for split in "${SPLIT_ARGS[@]}"; do
  if cache_split_complete "${split}"; then
    continue
  fi
  if [[ "${ALLOW_PARTIAL_CACHE}" == "1" ]] && \
     compgen -G "${ATTENTION_CACHE_ROOT}/${split}/attention_*.pt" >/dev/null; then
    printf 'WARNING: explicitly using partial %s attention cache.\n' "${split}" >&2
    continue
  fi
  if [[ "${AUTO_PREPARE_CACHE}" != "1" ]]; then
    printf 'RAGTruth %s attention cache is incomplete: %s\n' \
      "${split}" "${ATTENTION_CACHE_ROOT}/${split}" >&2
    exit 1
  fi
  SPLIT="${split}" \
  ATTENTION_CACHE_ROOT="${ATTENTION_CACHE_ROOT}" \
  DATA_ROOT="${DATA_ROOT}" \
  DEVICE="${DEVICE}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/prepare_attention_split.sh"
  if [[ "${ALLOW_PARTIAL_CACHE}" != "1" ]] && \
     ! cache_split_complete "${split}"; then
    printf 'RAGTruth %s cache remains incomplete after resume.\n' "${split}" >&2
    exit 1
  fi
done

BUILD_ARGS=(
  -m original.cli build
  --cache-root "${ATTENTION_CACHE_ROOT}"
  --output-dir "${GRAPH_DIR}"
  --tau "${TAU}"
  --splits "${SPLIT_ARGS[@]}"
  --device "${BUILD_DEVICE}"
  --resume
)
if [[ -n "${LIMIT}" ]]; then
  BUILD_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${ALLOW_PARTIAL_CACHE}" != "1" ]]; then
  BUILD_ARGS+=(--require-complete-cache)
fi

printf 'attention_cache=%s\n' "${ATTENTION_CACHE_ROOT}"
printf 'persistent_graphs=%s\n' "${GRAPH_DIR}"
printf 'tau=%s build_device=%s splits=%s\n' "${TAU}" "${BUILD_DEVICE}" "${SPLITS}"
"${PYTHON_BIN}" "${BUILD_ARGS[@]}"

if [[ "${BUILD_ONLY}" == "1" ]]; then
  printf 'Original attributed graphs ready: %s\n' "${GRAPH_DIR}"
  exit 0
elif [[ "${BUILD_ONLY}" != "0" ]]; then
  printf 'BUILD_ONLY must be 0 or 1, got %s\n' "${BUILD_ONLY}" >&2
  exit 1
fi
if [[ " ${SPLITS} " != *" train "* ]] || [[ " ${SPLITS} " != *" test "* ]]; then
  printf 'Training requires SPLITS="train test". Use BUILD_ONLY=1 otherwise.\n' >&2
  exit 1
fi

printf 'training_output=%s\n' "${TRAINING_DIR}"
TRAIN_ARGS=(
  -m original.cli train
  --graph-root "${GRAPH_DIR}"
  --output-dir "${TRAINING_DIR}"
  --seeds "${SEED_ARGS[@]}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --batch-size "${BATCH_SIZE}"
  --validation-fraction "${VALIDATION_FRACTION}"
  --device "${DEVICE}"
)
if [[ "${ALLOW_PARTIAL_CACHE}" == "1" ]]; then
  TRAIN_ARGS+=(--allow-partial-cache)
fi
"${PYTHON_BIN}" "${TRAIN_ARGS[@]}"

printf 'Original CHARM experiment complete: %s\n' "${TRAINING_DIR}"
