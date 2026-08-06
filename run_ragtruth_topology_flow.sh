#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_ATTENTION_DIR:?Set TRAIN_ATTENTION_DIR to the label-free train attention cache}"
: "${TEST_ATTENTION_DIR:?Set TEST_ATTENTION_DIR to the label-free test attention cache}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new empty output directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-80}"
MASS_COVER="${MASS_COVER:-0.80}"
RELAY_DISCOUNT="${RELAY_DISCOUNT:-0.85}"
VALIDATE_LIMIT="${VALIDATE_LIMIT:-32}"

mkdir -p "${OUTPUT_DIR}"
if find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 | grep -q .; then
  echo "OUTPUT_DIR must be empty: ${OUTPUT_DIR}" >&2
  exit 2
fi

"${PYTHON_BIN}" main.py validate \
  --input-dir "${TRAIN_ATTENTION_DIR}" \
  --limit "${VALIDATE_LIMIT}" \
  --output "${OUTPUT_DIR}/train.cache_validation.json"

"${PYTHON_BIN}" main.py validate \
  --input-dir "${TEST_ATTENTION_DIR}" \
  --limit "${VALIDATE_LIMIT}" \
  --output "${OUTPUT_DIR}/test.cache_validation.json"

# train_directory requires an empty directory, so validation artifacts live in
# the parent while model artifacts are isolated below.
MODEL_DIR="${OUTPUT_DIR}/model"
"${PYTHON_BIN}" main.py train \
  --input-dir "${TRAIN_ATTENTION_DIR}" \
  --output-dir "${MODEL_DIR}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --mass-cover "${MASS_COVER}" \
  --relay-discount "${RELAY_DISCOUNT}"

"${PYTHON_BIN}" main.py score \
  --input-dir "${TEST_ATTENTION_DIR}" \
  --checkpoint "${MODEL_DIR}/topology_flow_ranker.pt" \
  --output "${OUTPUT_DIR}/test.topology_scores.jsonl"

if [[ -n "${RAGTRUTH_RESPONSES:-}" && -n "${RAGTRUTH_SOURCES:-}" ]]; then
  "${PYTHON_BIN}" main.py evaluate \
    --scores "${OUTPUT_DIR}/test.topology_scores.jsonl" \
    --responses "${RAGTRUTH_RESPONSES}" \
    --sources "${RAGTRUTH_SOURCES}" \
    --output "${OUTPUT_DIR}/evaluation.json"
fi
