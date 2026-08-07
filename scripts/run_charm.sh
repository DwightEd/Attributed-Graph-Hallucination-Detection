#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data}"
FEATURE_ROOT="${FEATURE_ROOT:-${DATA_ROOT}/feature_extraction}"
DEFAULT_CACHE_ROOT="/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876"
ATTENTION_CACHE_ROOT="${ATTENTION_CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"
CACHE_TAG="$(basename -- "${ATTENTION_CACHE_ROOT}")"
TAU="${TAU:-0.05}"
LIMIT="${LIMIT:-}"
GRAPH_TAG="${CACHE_TAG}_tau${TAU/./p}"
if [[ -n "${LIMIT}" ]]; then GRAPH_TAG+="_limit${LIMIT}"; fi
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

cache_split_complete() {
  "${PYTHON_BIN}" -m original.cli audit \
    --cache-root "${ATTENTION_CACHE_ROOT}" --splits "$1" >/dev/null 2>&1
}

for split in "${SPLIT_ARGS[@]}"; do
  if cache_split_complete "${split}"; then continue; fi
  if [[ "${ALLOW_PARTIAL_CACHE}" == "1" ]] && compgen -G "${ATTENTION_CACHE_ROOT}/${split}/attention_*.pt" >/dev/null; then
    printf 'WARNING: explicitly using partial %s attention cache.\n' "${split}" >&2
    continue
  fi
  if [[ "${AUTO_PREPARE_CACHE}" != "1" ]]; then
    printf 'RAGTruth %s attention cache is incomplete: %s\n' "${split}" "${ATTENTION_CACHE_ROOT}/${split}" >&2
    exit 1
  fi
  SPLIT="${split}" ATTENTION_CACHE_ROOT="${ATTENTION_CACHE_ROOT}" DATA_ROOT="${DATA_ROOT}" DEVICE="${DEVICE}" PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/prepare_attention_split.sh"
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
if [[ -n "${LIMIT}" ]]; then BUILD_ARGS+=(--limit "${LIMIT}"); fi
if [[ "${ALLOW_PARTIAL_CACHE}" != "1" ]]; then BUILD_ARGS+=(--require-complete-cache); fi

printf 'attention_cache=%s\npersistent_graphs=%s\ntau=%s build_device=%s splits=%s\n' \
  "${ATTENTION_CACHE_ROOT}" "${GRAPH_DIR}" "${TAU}" "${BUILD_DEVICE}" "${SPLITS}"
"${PYTHON_BIN}" "${BUILD_ARGS[@]}"

if [[ "${BUILD_ONLY}" == "1" ]]; then
  printf 'Original attributed graphs ready: %s\n' "${GRAPH_DIR}"
  exit 0
fi

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
if [[ "${ALLOW_PARTIAL_CACHE}" == "1" ]]; then TRAIN_ARGS+=(--allow-partial-cache); fi
"${PYTHON_BIN}" "${TRAIN_ARGS[@]}"
printf 'Original CHARM experiment complete: %s\n' "${TRAINING_DIR}"
