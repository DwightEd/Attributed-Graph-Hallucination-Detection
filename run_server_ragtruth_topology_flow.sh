#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BASE_HOME="/share/home/tm902089733300000/a903202310/lys"
PYTHON_BIN="${PYTHON_BIN:-${BASE_HOME}/conda_envs/research/bin/python}"
DATA_ROOT="${DATA_ROOT:-${BASE_HOME}/data}"
RAGTRUTH_DIR="${RAGTRUTH_DIR:-${DATA_ROOT}/RAGTruth/dataset}"
ORIGINAL_GRAPH_ROOT="${ORIGINAL_GRAPH_ROOT:-${DATA_ROOT}/feature_extraction/ragtruth_original_attribute_graphs/fresh_attention_c8847872bedf_20260731T074520Z_p876_tau0p05}"
DEFAULT_ATTENTION_ROOT="${BASE_HOME}/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876"

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-80}"
MASS_COVER="${MASS_COVER:-0.80}"
RELAY_DISCOUNT="${RELAY_DISCOUNT:-0.85}"
HEAD_REDUCER="${HEAD_REDUCER:-median}"
CORRUPTION_MODE="${CORRUPTION_MODE:-all}"
VALIDATE_LIMIT="${VALIDATE_LIMIT:-32}"
SKIP_EVALUATION="${SKIP_EVALUATION:-0}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${SCRIPT_DIR}/main.py" ]]; then
  echo "Run this script from the PATF repository checkout: ${SCRIPT_DIR}" >&2
  exit 2
fi

# Prefer the known raw sparse-CSR cache. If it moved, recover its root from a
# previously built v2 original graph's source_cache_path.
ATTENTION_ROOT="${ATTENTION_ROOT:-}"
if [[ -z "${ATTENTION_ROOT}" && -d "${DEFAULT_ATTENTION_ROOT}/train" && -d "${DEFAULT_ATTENTION_ROOT}/test" ]]; then
  ATTENTION_ROOT="${DEFAULT_ATTENTION_ROOT}"
fi
if [[ -z "${ATTENTION_ROOT}" ]]; then
  FIRST_OLD_GRAPH="$(find "${ORIGINAL_GRAPH_ROOT}/graphs/train" -maxdepth 1 -type f -name '*.graph.pt' 2>/dev/null | sort | head -n 1 || true)"
  if [[ -n "${FIRST_OLD_GRAPH}" ]]; then
    ATTENTION_ROOT="$(${PYTHON_BIN} - "${FIRST_OLD_GRAPH}" <<'PY'
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1])
graph = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
source = graph.get("source_cache_path")
if source:
    print(Path(source).resolve().parent.parent)
PY
)"
  fi
fi
if [[ -z "${ATTENTION_ROOT}" ]]; then
  echo "Could not locate raw attention cache." >&2
  echo "Set ATTENTION_ROOT=/path/to/fresh_attention_... and rerun." >&2
  exit 2
fi

TRAIN_ATTENTION_DIR="${TRAIN_ATTENTION_DIR:-${ATTENTION_ROOT}/train}"
TEST_ATTENTION_DIR="${TEST_ATTENTION_DIR:-${ATTENTION_ROOT}/test}"
CACHE_TAG="$(basename -- "${ATTENTION_ROOT}")"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/feature_extraction/ragtruth_topology_flow_runs/${CACHE_TAG}/${RUN_TAG}}"
MODEL_DIR="${OUTPUT_DIR}/model"
RESPONSE_FILE="${RAGTRUTH_RESPONSES:-${RAGTRUTH_DIR}/response.jsonl}"
SOURCE_FILE="${RAGTRUTH_SOURCES:-${RAGTRUTH_DIR}/source_info.jsonl}"

for split_dir in "${TRAIN_ATTENTION_DIR}" "${TEST_ATTENTION_DIR}"; do
  if [[ ! -d "${split_dir}" ]]; then
    echo "Attention split directory not found: ${split_dir}" >&2
    exit 2
  fi
  if ! compgen -G "${split_dir}/attention_*.pt" >/dev/null; then
    echo "No attention_*.pt files found in: ${split_dir}" >&2
    exit 2
  fi
done
if [[ "${SKIP_EVALUATION}" != "1" ]]; then
  for metadata in "${RESPONSE_FILE}" "${SOURCE_FILE}"; do
    if [[ ! -f "${metadata}" ]]; then
      echo "RAGTruth metadata not found: ${metadata}" >&2
      exit 2
    fi
  done
fi
if [[ -e "${OUTPUT_DIR}" ]] && find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 | grep -q .; then
  echo "OUTPUT_DIR must be new or empty: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/run.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "PATF repository:       ${SCRIPT_DIR}"
echo "Python:                ${PYTHON_BIN}"
echo "Raw attention root:    ${ATTENTION_ROOT}"
echo "Train attention:       ${TRAIN_ATTENTION_DIR}"
echo "Test attention:        ${TEST_ATTENTION_DIR}"
echo "Output:                ${OUTPUT_DIR}"
echo "Device:                ${DEVICE}"
echo "Epochs:                ${EPOCHS}"
echo "Mass cover:            ${MASS_COVER}"
echo "Relay discount:        ${RELAY_DISCOUNT}"
echo "Head reducer:          ${HEAD_REDUCER}"
echo "Corruption mode:       ${CORRUPTION_MODE}"

# Show the actual raw-cache contract before expensive work. y_token may be
# present physically; the PATF loader ignores it through a field whitelist.
FIRST_ATTENTION="$(find "${TRAIN_ATTENTION_DIR}" -maxdepth 1 -type f -name 'attention_*.pt' | sort | head -n 1)"
"${PYTHON_BIN}" - "${FIRST_ATTENTION}" <<'PY'
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1])
sample = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
print("preflight_file=", path)
print("attention_cache_schema=", sample.get("attention_cache_schema"))
print("response_id=", sample.get("response_id"))
print("source_id=", sample.get("source_id"))
print("response_idx=", sample.get("response_idx"))
print("num_attention_layers=", sample.get("num_attention_layers"))
print("num_attention_heads=", sample.get("num_attention_heads"))
print("attention_diagonal_shape=", tuple(sample["attention_diagonal"].shape))
print("response_row_ptr_shape=", tuple(sample["response_row_ptr"].shape))
print("response_values_shape=", tuple(sample["response_values"].shape))
print("contains_y_token_but_ignored=", "y_token" in sample)
PY

"${PYTHON_BIN}" main.py validate \
  --input-dir "${TRAIN_ATTENTION_DIR}" \
  --limit "${VALIDATE_LIMIT}" \
  --output "${OUTPUT_DIR}/train.cache_validation.json"

"${PYTHON_BIN}" main.py validate \
  --input-dir "${TEST_ATTENTION_DIR}" \
  --limit "${VALIDATE_LIMIT}" \
  --output "${OUTPUT_DIR}/test.cache_validation.json"

"${PYTHON_BIN}" main.py train \
  --input-dir "${TRAIN_ATTENTION_DIR}" \
  --output-dir "${MODEL_DIR}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --mass-cover "${MASS_COVER}" \
  --relay-discount "${RELAY_DISCOUNT}" \
  --head-reducer "${HEAD_REDUCER}" \
  --corruption-mode "${CORRUPTION_MODE}"

"${PYTHON_BIN}" main.py score \
  --input-dir "${TEST_ATTENTION_DIR}" \
  --checkpoint "${MODEL_DIR}/topology_flow_ranker.pt" \
  --output "${OUTPUT_DIR}/test.topology_scores.jsonl"

if [[ "${SKIP_EVALUATION}" != "1" ]]; then
  "${PYTHON_BIN}" main.py evaluate \
    --scores "${OUTPUT_DIR}/test.topology_scores.jsonl" \
    --responses "${RESPONSE_FILE}" \
    --sources "${SOURCE_FILE}" \
    --output "${OUTPUT_DIR}/evaluation.json"

  "${PYTHON_BIN}" - "${OUTPUT_DIR}/evaluation.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
overall = report["overall"]
print("\n===== PATF evaluation =====")
print(f"samples={overall['samples']}")
print(f"positive_samples={overall['positive_samples']}")
print(f"positive_fraction={overall['positive_fraction']:.6f}")
print(f"response_AUROC={overall['auroc']:.6f}")
print(f"response_AUPRC={overall['average_precision']:.6f}")
print("\nBy task:")
for task, row in report.get("strata", {}).get("task", {}).items():
    print(
        f"  {task}: n={row['samples']} AUROC={row['auroc']:.6f} "
        f"AUPRC={row['average_precision']:.6f}"
    )
PY
fi

echo "PATF run complete: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
