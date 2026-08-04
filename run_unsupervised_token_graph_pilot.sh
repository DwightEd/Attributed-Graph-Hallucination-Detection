#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DATASET="${DATASET:-halueval_qa}"
SOURCE_DATA="${SOURCE_DATA:?Set SOURCE_DATA to HaluEval qa_data.json or BoolQ JSONL}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local Hugging Face model}"
RUN_DIR="${RUN_DIR:-outputs/unsupervised_token_graph/${DATASET}}"
PILOT_LIMIT="${PILOT_LIMIT:-300}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_ATTENTION_GIB="${MAX_ATTENTION_GIB:-12}"
RUN_TRAINING="${RUN_TRAINING:-0}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
POSTPROCESS_DEVICE="${POSTPROCESS_DEVICE:-auto}"
RETAIN_DENSE_ATTENTION="${RETAIN_DENSE_ATTENTION:-0}"
INCLUDE_LOGIT_NODE_FEATURES="${INCLUDE_LOGIT_NODE_FEATURES:-1}"
PURE_ATTENTION="${PURE_ATTENTION:-0}"
CPU_THREADS="${CPU_THREADS:-4}"

PREPARED_DIR="${RUN_DIR}/prepared"
EXTRACTION_DIR="${RUN_DIR}/extraction"
AUDIT_DIR="${RUN_DIR}/pattern_audit"
TRAINING_DIR="${RUN_DIR}/training"

mkdir -p "${RUN_DIR}"

case "${POSTPROCESS_DEVICE}" in
  auto|cpu|model) ;;
  *)
    echo "POSTPROCESS_DEVICE must be auto, cpu, or model" >&2
    exit 2
    ;;
esac
case "${RETAIN_DENSE_ATTENTION}" in
  0|1) ;;
  *)
    echo "RETAIN_DENSE_ATTENTION must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${INCLUDE_LOGIT_NODE_FEATURES}" in
  0|1) ;;
  *)
    echo "INCLUDE_LOGIT_NODE_FEATURES must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${PURE_ATTENTION}" in
  0|1) ;;
  *)
    echo "PURE_ATTENTION must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ "${PURE_ATTENTION}" == "1" && "${INCLUDE_LOGIT_NODE_FEATURES}" != "0" ]]; then
  echo "PURE_ATTENTION=1 requires INCLUDE_LOGIT_NODE_FEATURES=0" >&2
  exit 2
fi
if [[ ! "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPU_THREADS must be a positive integer" >&2
  exit 2
fi

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export BLIS_NUM_THREADS="${CPU_THREADS}"
export TOKENIZERS_PARALLELISM=false

EXTRACTION_STORAGE_ARGS=()
if [[ "${RETAIN_DENSE_ATTENTION}" == "0" ]]; then
  EXTRACTION_STORAGE_ARGS+=(--discard-dense-attention)
fi

GRAPH_NODE_FEATURE_ARGS=()
if [[ "${INCLUDE_LOGIT_NODE_FEATURES}" == "0" ]]; then
  GRAPH_NODE_FEATURE_ARGS+=(--exclude-logit-node-features)
fi
if [[ "${PURE_ATTENTION}" == "1" ]]; then
  GRAPH_NODE_FEATURE_ARGS+=(--pure-attention)
fi

if [[ "${DATASET}" == "halueval_qa" ]]; then
  "${PYTHON_BIN}" -m unsupervised_token_graph.prepare \
    --dataset halueval_qa \
    --input "${SOURCE_DATA}" \
    --output-dir "${PREPARED_DIR}"
elif [[ "${DATASET}" == "boolq" ]]; then
  BOOLQ_PREDICTIONS="${BOOLQ_PREDICTIONS:-${RUN_DIR}/boolq_predictions.jsonl}"
  "${PYTHON_BIN}" -m unsupervised_token_graph.generate_boolq \
    --input "${SOURCE_DATA}" \
    --model "${MODEL_PATH}" \
    --output "${BOOLQ_PREDICTIONS}" \
    --limit "${PILOT_LIMIT}" \
    --max-input-tokens "${MAX_TOKENS}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
  "${PYTHON_BIN}" -m unsupervised_token_graph.prepare \
    --dataset boolq \
    --input "${SOURCE_DATA}" \
    --predictions "${BOOLQ_PREDICTIONS}" \
    --allow-missing-predictions \
    --output-dir "${PREPARED_DIR}"
else
  echo "DATASET must be halueval_qa or boolq" >&2
  exit 2
fi

"${PYTHON_BIN}" -m unsupervised_token_graph.extract \
  --examples "${PREPARED_DIR}/examples.jsonl" \
  --model "${MODEL_PATH}" \
  --output-dir "${EXTRACTION_DIR}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-attention-gib "${MAX_ATTENTION_GIB}" \
  --limit "${PILOT_LIMIT}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --postprocess-device "${POSTPROCESS_DEVICE}" \
  "${GRAPH_NODE_FEATURE_ARGS[@]}" \
  "${EXTRACTION_STORAGE_ARGS[@]}"

if [[ "${PURE_ATTENTION}" == "1" ]]; then
  echo "Pattern audit skipped: its scalar feature report includes non-attention diagnostics."
else
  "${PYTHON_BIN}" -m unsupervised_token_graph.audit \
    --features "${EXTRACTION_DIR}/features.jsonl" \
    --evaluation-labels "${PREPARED_DIR}/evaluation_labels.jsonl" \
    --examples "${PREPARED_DIR}/examples.jsonl" \
    --output-dir "${AUDIT_DIR}"

  echo "Pattern audit: ${AUDIT_DIR}/pattern_audit.md"
fi

if [[ "${RUN_TRAINING}" == "1" ]]; then
  "${PYTHON_BIN}" -m unsupervised_token_graph.train \
    --graph-dir "${EXTRACTION_DIR}/graphs" \
    --output-dir "${TRAINING_DIR}"
  "${PYTHON_BIN}" -m unsupervised_token_graph.evaluate_scores \
    --scores "${TRAINING_DIR}/unsupervised_scores.jsonl" \
    --evaluation-labels "${PREPARED_DIR}/evaluation_labels.jsonl" \
    --split test \
    --output "${TRAINING_DIR}/evaluation_only_metrics.json"
fi
