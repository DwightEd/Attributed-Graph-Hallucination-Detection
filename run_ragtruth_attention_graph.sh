#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(command -v python || command -v python3)"
fi

DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data}"
FEATURE_ROOT="${FEATURE_ROOT:-${DATA_ROOT}/feature_extraction}"
DEFAULT_ATTENTION_CACHE_ROOT="/share/home/tm902089733300000/a903202310/lys"
DEFAULT_ATTENTION_CACHE_ROOT+="/research/Unsupervised-hypergraph/outputs/attention_cache"
DEFAULT_ATTENTION_CACHE_ROOT+="/fresh_attention_c8847872bedf_20260731T074520Z_p876"
ATTENTION_CACHE_ROOT="${ATTENTION_CACHE_ROOT:-${DEFAULT_ATTENTION_CACHE_ROOT}}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${FEATURE_ROOT}/ragtruth_attention_graph_${RUN_TAG}}"
GRAPH_DIR="${GRAPH_DIR:-${OUTPUT_DIR}/graphs}"
DEVICE="${DEVICE:-cuda}"
SELECTION="${SELECTION:-threshold}"
GRAPH_TRANSFORM="${GRAPH_TRANSFORM:-none}"
THRESHOLD="${THRESHOLD:-}"
TOP_K="${TOP_K:-8}"
MAX_EDGES_PER_TARGET="${MAX_EDGES_PER_TARGET:-64}"
QUERY_BLOCK="${QUERY_BLOCK:-32}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
MESSAGE_PASSING_STEPS="${MESSAGE_PASSING_STEPS:-2}"
DROPOUT="${DROPOUT:-0.10}"
SUPPORT_WEIGHT="${SUPPORT_WEIGHT:-1.0}"
ATTENTION_WEIGHT="${ATTENTION_WEIGHT:-1.0}"
DISTRIBUTION_WEIGHT="${DISTRIBUTION_WEIGHT:-1.0}"
NODE_WEIGHT="${NODE_WEIGHT:-0.25}"
EMBEDDING_ONLY_SCORING="${EMBEDDING_ONLY_SCORING:-0}"
NUM_SCORE_VIEWS="${NUM_SCORE_VIEWS:-4}"
MAX_SUPPORT_EDGES="${MAX_SUPPORT_EDGES:-8192}"
MAX_WEIGHT_TRACES="${MAX_WEIGHT_TRACES:-65536}"
MAX_DISTRIBUTION_GROUPS="${MAX_DISTRIBUTION_GROUPS:-512}"
DECODER_CHUNK_SIZE="${DECODER_CHUNK_SIZE:-16384}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.20}"
LIMIT="${LIMIT:-}"
SKIP_EVALUATION="${SKIP_EVALUATION:-0}"
REQUIRE_COMPLETE_CACHE="${REQUIRE_COMPLETE_CACHE:-0}"
RESPONSES="${RESPONSES:-${DATA_ROOT}/RAGTruth/dataset/response.jsonl}"
SOURCES="${SOURCES:-${DATA_ROOT}/RAGTruth/dataset/source_info.jsonl}"
DEFAULT_TOKENIZER="/share/home/tm902089733300000/a903202310/lys/models"
DEFAULT_TOKENIZER+="/Meta-Llama-3.1-8B-Instruct"
TOKENIZER="${TOKENIZER:-${DEFAULT_TOKENIZER}}"
SENTENCE_OUTPUT="${SENTENCE_OUTPUT:-}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -d "${ATTENTION_CACHE_ROOT}/train" ]] || \
   ! compgen -G "${ATTENTION_CACHE_ROOT}/train/attention_*.pt" >/dev/null; then
  printf 'Official RAGTruth train attention cache is absent: %s\n' \
    "${ATTENTION_CACHE_ROOT}/train" >&2
  exit 1
fi

if [[ ! -d "${ATTENTION_CACHE_ROOT}/test" ]] || \
   ! compgen -G "${ATTENTION_CACHE_ROOT}/test/attention_*.pt" >/dev/null; then
  printf 'Official RAGTruth test attention cache is absent; extracting only test.\n'
  ATTENTION_DIR="${ATTENTION_CACHE_ROOT}" \
  DATA_ROOT="${DATA_ROOT}" \
  FEATURE_ROOT="${FEATURE_ROOT}" \
  DEVICE="${DEVICE}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/scripts/data/prepare_ragtruth_test_attention.sh"
fi

ARGS=(
  --cache-root "${ATTENTION_CACHE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --graph-dir "${GRAPH_DIR}"
  --device "${DEVICE}"
  --selection "${SELECTION}"
  --graph-transform "${GRAPH_TRANSFORM}"
  --top-k "${TOP_K}"
  --max-edges-per-target "${MAX_EDGES_PER_TARGET}"
  --query-block "${QUERY_BLOCK}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --embedding-dim "${EMBEDDING_DIM}"
  --message-passing-steps "${MESSAGE_PASSING_STEPS}"
  --dropout "${DROPOUT}"
  --support-weight "${SUPPORT_WEIGHT}"
  --attention-weight "${ATTENTION_WEIGHT}"
  --distribution-weight "${DISTRIBUTION_WEIGHT}"
  --node-weight "${NODE_WEIGHT}"
  --num-score-views "${NUM_SCORE_VIEWS}"
  --max-support-edges "${MAX_SUPPORT_EDGES}"
  --max-weight-traces "${MAX_WEIGHT_TRACES}"
  --max-distribution-groups "${MAX_DISTRIBUTION_GROUPS}"
  --decoder-chunk-size "${DECODER_CHUNK_SIZE}"
  --validation-fraction "${VALIDATION_FRACTION}"
  --seed "${SEED}"
  --responses "${RESPONSES}"
  --sources "${SOURCES}"
  --tokenizer "${TOKENIZER}"
)
if [[ -n "${THRESHOLD}" ]]; then
  ARGS+=(--threshold "${THRESHOLD}")
fi
if [[ "${EMBEDDING_ONLY_SCORING}" == "1" ]]; then
  ARGS+=(--embedding-only-scoring)
fi
if [[ -n "${LIMIT}" ]]; then
  printf 'WARNING: LIMIT=%s is smoke-only and applies independently to train/test.\n' \
    "${LIMIT}" >&2
  ARGS+=(--limit "${LIMIT}")
fi
if [[ "${SKIP_EVALUATION}" == "1" ]]; then
  if [[ -z "${LIMIT}" ]]; then
    printf 'SKIP_EVALUATION=1 is only valid together with a smoke LIMIT.\n' >&2
    exit 1
  fi
  ARGS+=(--skip-evaluation)
fi
if [[ "${REQUIRE_COMPLETE_CACHE}" == "1" ]]; then
  ARGS+=(--require-complete-cache)
elif [[ "${REQUIRE_COMPLETE_CACHE}" != "0" ]]; then
  printf 'REQUIRE_COMPLETE_CACHE must be 0 or 1, got %s\n' \
    "${REQUIRE_COMPLETE_CACHE}" >&2
  exit 1
fi
if [[ -n "${SENTENCE_OUTPUT}" ]]; then
  ARGS+=(--sentence-output "${SENTENCE_OUTPUT}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" run "${ARGS[@]}"

printf 'Attention-graph run complete: %s\n' "${OUTPUT_DIR}"
