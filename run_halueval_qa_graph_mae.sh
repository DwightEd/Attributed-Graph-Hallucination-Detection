#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python}"
DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data}"
MODEL_PATH="${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
SOURCE_DATA="${SOURCE_DATA:-${DATA_ROOT}/HaluEval/qa_data.json}"

# HaluEval has two candidate responses per pair. The old default of 300
# candidates produced 150 pairs and therefore only 30 held-out test pairs.
# 2000 candidates produce 1000 pairs: about 700/100/200 train/val/test pairs.
PILOT_LIMIT="${PILOT_LIMIT:-2000}"
GPU_ID="${GPU_ID:-0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_ATTENTION_GIB="${MAX_ATTENTION_GIB:-12}"
CPU_THREADS="${CPU_THREADS:-4}"
INCLUDE_LOGIT_NODE_FEATURES="${INCLUDE_LOGIT_NODE_FEATURES:-0}"

case "${INCLUDE_LOGIT_NODE_FEATURES}" in
  0) NODE_FEATURE_TAG="no_logits" ;;
  1) NODE_FEATURE_TAG="with_logits" ;;
  *)
    echo "INCLUDE_LOGIT_NODE_FEATURES must be 0 or 1" >&2
    exit 2
    ;;
esac
RUN_DIR="${RUN_DIR:-${DATA_ROOT}/feature_extraction/halueval_qa_graph_mae_${NODE_FEATURE_TAG}_${PILOT_LIMIT}_$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -s "${SOURCE_DATA}" ]]; then
  echo "HaluEval QA data does not exist: ${SOURCE_DATA}" >&2
  echo "The earlier download already completed HaluEval before the BoolQ error." >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory does not exist: ${MODEL_PATH}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"
printf '%s\n' "${RUN_DIR}" \
  > "${DATA_ROOT}/feature_extraction/LATEST_HALUEVAL_GRAPH_MAE_RUN.txt"

exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

echo "HaluEval graph-MAE experiment"
echo "candidates=${PILOT_LIMIT}"
echo "expected_pairs=$((PILOT_LIMIT / 2))"
echo "expected_test_pairs=$((PILOT_LIMIT / 10))"
echo "include_logit_node_features=${INCLUDE_LOGIT_NODE_FEATURES}"
echo "run_dir=${RUN_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --id="${GPU_ID}" \
    --query-gpu=index,name,memory.used,memory.total \
    --format=csv,noheader
fi

env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  DATASET=halueval_qa \
  SOURCE_DATA="${SOURCE_DATA}" \
  MODEL_PATH="${MODEL_PATH}" \
  RUN_DIR="${RUN_DIR}" \
  PILOT_LIMIT="${PILOT_LIMIT}" \
  MAX_TOKENS="${MAX_TOKENS}" \
  MAX_ATTENTION_GIB="${MAX_ATTENTION_GIB}" \
  DEVICE=cuda:0 \
  DTYPE=float16 \
  POSTPROCESS_DEVICE=auto \
  RETAIN_DENSE_ATTENTION=0 \
  INCLUDE_LOGIT_NODE_FEATURES="${INCLUDE_LOGIT_NODE_FEATURES}" \
  CPU_THREADS="${CPU_THREADS}" \
  RUN_TRAINING=1 \
  bash "${SCRIPT_DIR}/run_unsupervised_token_graph_pilot.sh"

echo "Experiment complete: ${RUN_DIR}"
echo "Metrics: ${RUN_DIR}/training/evaluation_only_metrics.json"
