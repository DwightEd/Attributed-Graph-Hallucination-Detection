#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(command -v python || command -v python3)"
fi
DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data}"
FEATURE_ROOT="${FEATURE_ROOT:-${DATA_ROOT}/feature_extraction}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/RAGTruth/dataset}"
RESPONSES="${RESPONSES:-${DATASET_DIR}/response.jsonl}"
SOURCES="${SOURCES:-${DATASET_DIR}/source_info.jsonl}"
TOKENIZER="${TOKENIZER:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
ATTENTION_DIR="${ATTENTION_DIR:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}"
RUN_DIR="${RUN_DIR:-${FEATURE_ROOT}/ragtruth_attribute_graph_typed_mae_fresh_attention_c8847872bedf_official}"
DEVICE="${DEVICE:-cuda:0}"
RESIDENCY="${RESIDENCY:-cuda}"
MAX_RESIDENT_GIB="${MAX_RESIDENT_GIB:-0}"
PREFIX_TOP_K="${PREFIX_TOP_K:-8}"
HISTORY_TOP_K="${HISTORY_TOP_K:-8}"
QUERY_BLOCK="${QUERY_BLOCK:-64}"
LAYER_CHUNK="${LAYER_CHUNK:-2}"
MAX_NODES="${MAX_NODES:-12000}"
MAX_EDGES="${MAX_EDGES:-192000}"
MASK_STRIDE="${MASK_STRIDE:-8}"
AMP="${AMP:-bfloat16}"
LABEL_SHIFT="${LABEL_SHIFT:-0}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
REGISTER_HYPERGRAPH_RESULT="${REGISTER_HYPERGRAPH_RESULT:-1}"
AUTO_PREPARE_TEST_ATTENTION="${AUTO_PREPARE_TEST_ATTENTION:-1}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.15}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-6}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"

GRAPH_DIR="${RUN_DIR}/compact_graphs"
TRAIN_DIR="${RUN_DIR}/training"
SCORE_FILE="${RUN_DIR}/test_token_scores.jsonl"
SENTENCE_SCORE_FILE="${RUN_DIR}/test_sentence_scores.jsonl"
METRIC_FILE="${RUN_DIR}/test_metrics.json"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export AMP

for required_file in "${RESPONSES}" "${SOURCES}" "${TOKENIZER}/config.json"; do
  if [[ ! -f "${required_file}" ]]; then
    printf 'Missing required input: %s\n' "${required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${ATTENTION_DIR}/train" ]] || ! compgen -G "${ATTENTION_DIR}/train/attention_*.pt" >/dev/null; then
  printf 'Official RAGTruth train attention cache is absent: %s\n' "${ATTENTION_DIR}/train" >&2
  exit 1
fi
if [[ ! -d "${ATTENTION_DIR}/test" ]] || ! compgen -G "${ATTENTION_DIR}/test/attention_*.pt" >/dev/null; then
  if [[ "${AUTO_PREPARE_TEST_ATTENTION}" != "1" ]]; then
    printf 'Official RAGTruth test attention cache is absent: %s\n' "${ATTENTION_DIR}/test" >&2
    printf 'Rerun with AUTO_PREPARE_TEST_ATTENTION=1 to extract only official test.\n' >&2
    exit 1
  fi
  DATA_ROOT="${DATA_ROOT}" \
  FEATURE_ROOT="${FEATURE_ROOT}" \
  DATASET_DIR="${DATASET_DIR}" \
  TOKENIZER="${TOKENIZER}" \
  ATTENTION_DIR="${ATTENTION_DIR}" \
  DEVICE="${DEVICE}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${SCRIPT_DIR}/scripts/data/prepare_ragtruth_test_attention.sh"
fi

mkdir -p "${FEATURE_ROOT}" "${RUN_DIR}"
if [[ "${REGISTER_HYPERGRAPH_RESULT}" == "1" ]]; then
  if ! FEATURE_ROOT="${FEATURE_ROOT}" bash "${SCRIPT_DIR}/register_ragtruth_hypergraph_result.sh"; then
    printf 'Warning: historical hypergraph result was not registered; attribute-graph run continues.\n' >&2
  fi
fi

"${PYTHON_BIN}" - <<'PY'
import sys
import os
import sklearn
import torch
import tqdm

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in this Python environment")
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("sklearn:", sklearn.__version__)
if os.environ["AMP"] == "bfloat16" and not torch.cuda.is_bf16_supported():
    raise RuntimeError("GPU does not support bfloat16; rerun with AMP=float16")
PY

COMPACT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  COMPACT_ARGS+=(--limit "${LIMIT}")
fi

"${PYTHON_BIN}" -m unsupervised_token_graph.ragtruth_cli compact \
  --attention-dir "${ATTENTION_DIR}" \
  --output-dir "${GRAPH_DIR}" \
  --device "${DEVICE}" \
  --prefix-top-k "${PREFIX_TOP_K}" \
  --history-top-k "${HISTORY_TOP_K}" \
  --query-block "${QUERY_BLOCK}" \
  --layer-chunk "${LAYER_CHUNK}" \
  --storage-dtype float16 \
  --resume \
  "${COMPACT_ARGS[@]}"

"${PYTHON_BIN}" -m unsupervised_token_graph.ragtruth_cli train \
  --graph-dir "${GRAPH_DIR}" \
  --output-dir "${TRAIN_DIR}" \
  --device "${DEVICE}" \
  --residency "${RESIDENCY}" \
  --max-resident-gib "${MAX_RESIDENT_GIB}" \
  --hidden-dim 192 \
  --num-layers 2 \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --max-nodes "${MAX_NODES}" \
  --max-edges "${MAX_EDGES}" \
  --mask-ratio 0.20 \
  --neighborhood-weight 0.25 \
  --route-weight 0.10 \
  --split-policy official \
  --validation-fraction "${VALIDATION_FRACTION}" \
  --seed "${SEED}" \
  --amp "${AMP}"

"${PYTHON_BIN}" -m unsupervised_token_graph.ragtruth_cli score \
  --checkpoint "${TRAIN_DIR}/best.pt" \
  --graph-dir "${GRAPH_DIR}" \
  --split-file "${TRAIN_DIR}/splits.json" \
  --partition test \
  --output "${SCORE_FILE}" \
  --device "${DEVICE}" \
  --residency "${RESIDENCY}" \
  --max-resident-gib "${MAX_RESIDENT_GIB}" \
  --mask-stride "${MASK_STRIDE}" \
  --amp "${AMP}"

"${PYTHON_BIN}" -m unsupervised_token_graph.ragtruth_cli sentences \
  --scores "${SCORE_FILE}" \
  --attention-dir "${ATTENTION_DIR}" \
  --graph-dir "${GRAPH_DIR}" \
  --responses "${RESPONSES}" \
  --sources "${SOURCES}" \
  --tokenizer "${TOKENIZER}" \
  --top-fraction 0.20 \
  --output "${SENTENCE_SCORE_FILE}"

if [[ "${RUN_EVALUATION}" == "1" ]]; then
  "${PYTHON_BIN}" -m unsupervised_token_graph.ragtruth_cli evaluate \
    --scores "${SCORE_FILE}" \
    --attention-dir "${ATTENTION_DIR}" \
    --graph-dir "${GRAPH_DIR}" \
    --sentence-scores "${SENTENCE_SCORE_FILE}" \
    --output "${METRIC_FILE}" \
    --label-shift "${LABEL_SHIFT}"
fi

echo "compact graphs: ${GRAPH_DIR}"
echo "checkpoint: ${TRAIN_DIR}/best.pt"
echo "token scores: ${SCORE_FILE}"
echo "sentence scores: ${SENTENCE_SCORE_FILE}"
if [[ "${RUN_EVALUATION}" == "1" ]]; then
  echo "metrics: ${METRIC_FILE}"
fi
