#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(command -v python || command -v python3)"
fi

DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data}"
FEATURE_ROOT="${FEATURE_ROOT:-${DATA_ROOT}/feature_extraction}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/RAGTruth/dataset}"
MODEL_PATH="${MODEL_PATH:-${TOKENIZER:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}}"
HYPERGRAPH_PROJECT="${HYPERGRAPH_PROJECT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph}"
ATTENTION_DIR="${ATTENTION_DIR:-${HYPERGRAPH_PROJECT}/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}"
CACHE_TAG="$(basename -- "${ATTENTION_DIR}")"
TEST_GRAPH_DIR="${TEST_GRAPH_DIR:-${FEATURE_ROOT}/attention_preparation/${CACHE_TAG}/test_hypergraphs}"
LOG_DIR="${LOG_DIR:-${FEATURE_ROOT}/attention_preparation/${CACHE_TAG}/logs}"

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
    printf 'Missing test-attention extraction input: %s\n' "${required_file}" >&2
    exit 1
  fi
done

mkdir -p "${ATTENTION_DIR}/test" "${TEST_GRAPH_DIR}" "${LOG_DIR}"

printf 'Official test attention is missing; extracting only RAGTruth test.\n'
printf 'Raw dataset:       %s\n' "${DATASET_DIR}"
printf 'Observer model:    %s\n' "${MODEL_PATH}"
printf 'Test cache:        %s\n' "${ATTENTION_DIR}/test"
printf 'Replay hypergraph: %s\n' "${TEST_GRAPH_DIR}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" -u "${HYPERGRAPH_PROJECT}/get_response_attention.py" \
  --dataset-dir "${DATASET_DIR}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${TEST_GRAPH_DIR}" \
  --attention-cache-dir "${ATTENTION_DIR}/test" \
  --attention-cache-dtype "${ATTENTION_CACHE_DTYPE}" \
  --attention-cache-floor "${ATTENTION_CACHE_FLOOR}" \
  --split test \
  --generator-model "${GENERATOR_MODEL}" \
  --task-type "${TASK_TYPE}" \
  --tau "${ATTENTION_TAU}" \
  --device "${DEVICE}" \
  --attn-implementation eager \
  --dtype "${DTYPE}" \
  --resume-existing \
  2>&1 | tee "${LOG_DIR}/extract_test_attention.log"

if ! compgen -G "${ATTENTION_DIR}/test/attention_*.pt" >/dev/null; then
  printf 'Test extraction returned without producing attention files.\n' >&2
  exit 1
fi

printf 'Official RAGTruth test attention ready: %s\n' "${ATTENTION_DIR}/test"
