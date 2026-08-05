#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python}"
DEFAULT_RUN_DIR="/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/halueval_grounding_flow_20260805T015520Z_seed42"
RUN_DIR="${RUN_DIR:-${1:-${DEFAULT_RUN_DIR}}}"
HALUEVAL_QA="${HALUEVAL_QA:-/share/home/tm902089733300000/a903202310/lys/data/HaluEval/qa_data.json}"
EVALUATION_LABELS="${EVALUATION_LABELS:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -s "${RUN_DIR}/score_freeze.json" ]]; then
  echo "Completed score freeze is missing: ${RUN_DIR}/score_freeze.json" >&2
  exit 2
fi

arguments=(--run-dir "${RUN_DIR}")
if [[ -n "${EVALUATION_LABELS}" ]]; then
  arguments+=(--labels "${EVALUATION_LABELS}")
fi
if [[ -s "${HALUEVAL_QA}" ]]; then
  arguments+=(--halueval-qa "${HALUEVAL_QA}")
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m grounding_flow.graph_index "${arguments[@]}"

echo "PUBLIC LABELED DATASET INDEX: ${RUN_DIR}/prepared/graphs/index.json"
echo "The command output above includes first_record; it must contain split and label."
echo "INTERNAL TECHNICAL INVENTORY: ${RUN_DIR}/prepared/graphs/artifact_index.json"
