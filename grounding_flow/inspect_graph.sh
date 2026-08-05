#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python}"
DEFAULT_RUN_DIR="/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/halueval_grounding_flow_20260805T015520Z_seed42"
RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" && $# -gt 0 && "${1}" != -* ]]; then
  RUN_DIR="${1}"
  shift
fi
RUN_DIR="${RUN_DIR:-${DEFAULT_RUN_DIR}}"
TOP_EDGES="${TOP_EDGES:-20}"
MAX_TARGETS="${MAX_TARGETS:-64}"
RESPONSE_ID="${RESPONSE_ID:-}"
# Keep inspection output beside the immutable experiment directory so a later
# Grounding Flow --resume is not rejected by an unexpected file inside it.
REPORT_PATH="${REPORT_PATH:-${RUN_DIR%/}.graph_structure.json}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 2
fi
INDEX_PATH=""
for candidate in \
  "${RUN_DIR}/prepared/graphs/index.json" \
  "${RUN_DIR}/graphs/index.json" \
  "${RUN_DIR}/index.json"; do
  if [[ -s "${candidate}" ]]; then
    INDEX_PATH="${candidate}"
    break
  fi
done
if [[ -z "${INDEX_PATH}" ]]; then
  echo "Prepared graph index is missing under: ${RUN_DIR}" >&2
  echo "Expected prepared/graphs/index.json, graphs/index.json, or index.json" >&2
  exit 2
fi

arguments=(
  --run-dir "${RUN_DIR}"
  --top-edges "${TOP_EDGES}"
  --max-targets "${MAX_TARGETS}"
  --output "${REPORT_PATH}"
)
if [[ -n "${RESPONSE_ID}" ]]; then
  arguments+=(--response-id "${RESPONSE_ID}")
fi
arguments+=("$@")

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m attention_graph.inspect "${arguments[@]}"

echo "Attention-graph structure report: ${REPORT_PATH}"
