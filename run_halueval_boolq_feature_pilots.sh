#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data}"
MODEL_PATH="${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
GPU_ID="${GPU_ID:-0}"
DEVICE="cuda:0"
DTYPE="${DTYPE:-float16}"
PILOT_LIMIT="${PILOT_LIMIT:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_ATTENTION_GIB="${MAX_ATTENTION_GIB:-6}"
GPU_BUSY_LIMIT_MIB="${GPU_BUSY_LIMIT_MIB:-512}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
RUN_ROOT="${RUN_ROOT:-${DATA_ROOT}/feature_extraction/token_graph_pilot_v3_${PILOT_LIMIT}}"

case "${DTYPE}" in
  float16|bfloat16|float32) ;;
  *)
    echo "DTYPE must be float16, bfloat16, or float32" >&2
    exit 2
    ;;
esac

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory does not exist: ${MODEL_PATH}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the GPU preflight" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to prevent concurrent GPU/cache writers" >&2
  exit 2
fi

mkdir -p "${DATA_ROOT}/feature_extraction/.locks" "${RUN_ROOT}"
exec 9>"${DATA_ROOT}/feature_extraction/.locks/gpu_${GPU_ID}.lock"
if ! flock -n 9; then
  echo "Another managed pilot already holds the GPU ${GPU_ID} lock" >&2
  exit 2
fi
exec 8>"${RUN_ROOT}/.pilot.lock"
if ! flock -n 8; then
  echo "Another process is already writing ${RUN_ROOT}" >&2
  exit 2
fi

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  DATA_ROOT="${DATA_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/download_halueval_boolq.sh"
else
  DATA_ROOT="${DATA_ROOT}" PYTHON_BIN="${PYTHON_BIN}" OFFLINE_ONLY=1 \
    bash "${SCRIPT_DIR}/download_halueval_boolq.sh"
fi

HALUEVAL_DATA="${DATA_ROOT}/HaluEval/qa_data.json"
BOOLQ_DATA="${DATA_ROOT}/BoolQ/dev.jsonl"
for dataset_file in "${HALUEVAL_DATA}" "${BOOLQ_DATA}"; do
  if [[ ! -s "${dataset_file}" ]]; then
    echo "Missing validated dataset: ${dataset_file}" >&2
    exit 2
  fi
done

GPU_MEMORY_USED_MIB="$(
  nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used \
    --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]'
)"
if [[ ! "${GPU_MEMORY_USED_MIB}" =~ ^[0-9]+$ ]]; then
  echo "Could not read GPU ${GPU_ID} memory usage" >&2
  exit 2
fi
if (( GPU_MEMORY_USED_MIB > GPU_BUSY_LIMIT_MIB )) && [[ "${ALLOW_BUSY_GPU}" != "1" ]]; then
  echo "GPU ${GPU_ID} is using ${GPU_MEMORY_USED_MIB} MiB; refusing to share it." >&2
  echo "Set ALLOW_BUSY_GPU=1 only after checking the other process." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export DTYPE

"${PYTHON_BIN}" - <<'PY'
import json
import os
import torch
import transformers

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible to PyTorch")
requested_dtype = os.environ["DTYPE"]
if requested_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
    raise SystemExit("The visible GPU does not support bfloat16")
x = torch.tensor([2.0, 3.0], device="cuda:0")
witness = {
    "event": "GPU_WITNESS",
    "cuda_device_count": torch.cuda.device_count(),
    "device_name": torch.cuda.get_device_name(0),
    "tensor_sum": float(x.sum().item()),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "requested_dtype": requested_dtype,
}
print(json.dumps(witness, ensure_ascii=False), flush=True)
PY

run_dataset() {
  local dataset_name="$1"
  local source_data="$2"
  local dataset_run_dir="${RUN_ROOT}/${dataset_name}"
  echo "Starting ${dataset_name}; resumable output: ${dataset_run_dir}"
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    DATASET="${dataset_name}" \
    SOURCE_DATA="${source_data}" \
    MODEL_PATH="${MODEL_PATH}" \
    RUN_DIR="${dataset_run_dir}" \
    PILOT_LIMIT="${PILOT_LIMIT}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    MAX_ATTENTION_GIB="${MAX_ATTENTION_GIB}" \
    DEVICE="${DEVICE}" \
    DTYPE="${DTYPE}" \
    RUN_TRAINING=0 \
    bash "${SCRIPT_DIR}/run_unsupervised_token_graph_pilot.sh"
  echo "Completed ${dataset_name}: ${dataset_run_dir}/pattern_audit/pattern_audit.md"
}

# These calls are deliberately sequential: each loads the same 8B model and the
# dense all-layer attention tensors are too expensive to run concurrently.
run_dataset halueval_qa "${HALUEVAL_DATA}"
run_dataset boolq "${BOOLQ_DATA}"

du -sh "${RUN_ROOT}" || true
echo "Both feature pilots completed under ${RUN_ROOT}"
