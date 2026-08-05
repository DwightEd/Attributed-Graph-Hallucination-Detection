#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(command -v python || command -v python3)"
fi

SPLIT="${SPLIT:?Set SPLIT=train or SPLIT=test}"
if [[ "${SPLIT}" != "train" && "${SPLIT}" != "test" ]]; then
  printf 'SPLIT must be train or test, got %s\n' "${SPLIT}" >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/RAGTruth/dataset}"
MODEL_PATH="${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
HYPERGRAPH_PROJECT="${HYPERGRAPH_PROJECT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph}"
ATTENTION_CACHE_ROOT="${ATTENTION_CACHE_ROOT:?Set ATTENTION_CACHE_ROOT}"
CACHE_TAG="$(basename -- "${ATTENTION_CACHE_ROOT}")"
# The legacy fresh_hypergraphs_* files predate the mandatory cache SHA-256
# binding.  Reusing them makes the current validator fail before it can resume
# the attention cache, so cache-bound replay artifacts have their own stable
# schema-specific directory.  Existing attention_*.pt files remain untouched.
REPLAY_GRAPH_ROOT="${REPLAY_GRAPH_ROOT:-${DATA_ROOT}/RAGTruth/hypergraphs/cache_bound_sha256/${CACHE_TAG}}"
REPLAY_GRAPH_DIR="${REPLAY_GRAPH_DIR:-${REPLAY_GRAPH_ROOT}/${SPLIT}}"
LOG_DIR="${LOG_DIR:-${DATA_ROOT}/feature_extraction/ragtruth_original_attribute_graphs/logs/${CACHE_TAG}}"

ATTENTION_CACHE_DTYPE="${ATTENTION_CACHE_DTYPE:-float16}"
ATTENTION_CACHE_FLOOR="${ATTENTION_CACHE_FLOOR:-0.01}"
ATTENTION_TAU="${ATTENTION_TAU:-0.01}"
DTYPE="${DTYPE:-auto}"
GENERATOR_MODEL="${GENERATOR_MODEL:-llama-2-7b-chat}"
TASK_TYPE="${TASK_TYPE:-all}"
DEVICE="${DEVICE:-cuda}"

for required_file in \
  "${HYPERGRAPH_PROJECT}/get_response_attention.py" \
  "${DATASET_DIR}/source_info.jsonl" \
  "${DATASET_DIR}/response.jsonl" \
  "${MODEL_PATH}/config.json"; do
  if [[ ! -f "${required_file}" ]]; then
    printf 'Missing attention extraction input: %s\n' "${required_file}" >&2
    exit 1
  fi
done

mkdir -p \
  "${ATTENTION_CACHE_ROOT}/${SPLIT}" \
  "${REPLAY_GRAPH_DIR}" \
  "${LOG_DIR}"

printf 'Resuming RAGTruth %s attention extraction.\n' "${SPLIT}"
printf 'attention_cache=%s\n' "${ATTENTION_CACHE_ROOT}/${SPLIT}"
printf 'replay_hypergraphs=%s\n' "${REPLAY_GRAPH_DIR}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" -u "${HYPERGRAPH_PROJECT}/get_response_attention.py" \
  --dataset-dir "${DATASET_DIR}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${REPLAY_GRAPH_DIR}" \
  --attention-cache-dir "${ATTENTION_CACHE_ROOT}/${SPLIT}" \
  --attention-cache-dtype "${ATTENTION_CACHE_DTYPE}" \
  --attention-cache-floor "${ATTENTION_CACHE_FLOOR}" \
  --split "${SPLIT}" \
  --generator-model "${GENERATOR_MODEL}" \
  --task-type "${TASK_TYPE}" \
  --tau "${ATTENTION_TAU}" \
  --device "${DEVICE}" \
  --attn-implementation eager \
  --dtype "${DTYPE}" \
  --resume-existing \
  2>&1 | tee "${LOG_DIR}/extract_${SPLIT}_attention.log"

printf 'RAGTruth %s attention extraction returned: %s\n' \
  "${SPLIT}" "${ATTENTION_CACHE_ROOT}/${SPLIT}"
