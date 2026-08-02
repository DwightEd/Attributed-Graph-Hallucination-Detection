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

PREPARED_DIR="${RUN_DIR}/prepared"
EXTRACTION_DIR="${RUN_DIR}/extraction"
AUDIT_DIR="${RUN_DIR}/pattern_audit"
TRAINING_DIR="${RUN_DIR}/training"

mkdir -p "${RUN_DIR}"

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
  --dtype "${DTYPE}"

"${PYTHON_BIN}" -m unsupervised_token_graph.audit \
  --features "${EXTRACTION_DIR}/features.jsonl" \
  --evaluation-labels "${PREPARED_DIR}/evaluation_labels.jsonl" \
  --examples "${PREPARED_DIR}/examples.jsonl" \
  --output-dir "${AUDIT_DIR}"

echo "Pattern audit: ${AUDIT_DIR}/pattern_audit.md"

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
