#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/research/data}"
MODEL_PATH="${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
GPU_ID="${GPU_ID:-0}"
DEVICE="cuda:0"
DTYPE="${DTYPE:-float16}"
POSTPROCESS_DEVICE="${POSTPROCESS_DEVICE:-auto}"
RETAIN_DENSE_ATTENTION="${RETAIN_DENSE_ATTENTION:-0}"
CPU_THREADS="${CPU_THREADS:-4}"
PILOT_LIMIT="${PILOT_LIMIT:-64}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_ATTENTION_GIB="${MAX_ATTENTION_GIB:-6}"
GPU_BUSY_LIMIT_MIB="${GPU_BUSY_LIMIT_MIB:-512}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
RUN_ROOT="${RUN_ROOT:-${DATA_ROOT}/feature_extraction/token_graph_pilot_v4_gpu_${PILOT_LIMIT}}"
HALUEVAL_DATA="${DATA_ROOT}/HaluEval/qa_data.json"
BOOLQ_DATA="${DATA_ROOT}/BoolQ/dev.jsonl"
HALUEVAL_URL="https://raw.githubusercontent.com/RUCAIBox/HaluEval/b7253db3cdaa0ab2c382f92b26b390109174f77e/data/qa_data.json"
BOOLQ_URL="https://storage.googleapis.com/boolq/dev.jsonl"

case "${DTYPE}" in
  float16|bfloat16|float32) ;;
  *)
    echo "DTYPE must be float16, bfloat16, or float32" >&2
    exit 2
    ;;
esac
case "${POSTPROCESS_DEVICE}" in
  auto|cpu|model) ;;
  *)
    echo "POSTPROCESS_DEVICE must be auto, cpu, or model" >&2
    exit 2
    ;;
esac
if [[ "${RETAIN_DENSE_ATTENTION}" != "0" && "${RETAIN_DENSE_ATTENTION}" != "1" ]]; then
  echo "RETAIN_DENSE_ATTENTION must be 0 or 1" >&2
  exit 2
fi
if [[ ! "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPU_THREADS must be a positive integer" >&2
  exit 2
fi
case "${DOWNLOAD_DATA}" in
  0|1) ;;
  *)
    echo "DOWNLOAD_DATA must be 0 or 1" >&2
    exit 2
    ;;
esac

download_if_missing() {
  local dataset_name="$1"
  local source_url="$2"
  local destination="$3"
  local temporary="${destination}.$$.part"

  if [[ -s "${destination}" ]]; then
    echo "Reusing ${dataset_name}: ${destination}"
    return
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download ${dataset_name}" >&2
    exit 2
  fi

  mkdir -p "$(dirname -- "${destination}")"
  echo "Downloading ${dataset_name}: ${source_url}"
  if ! curl \
      --fail \
      --location \
      --retry 5 \
      --retry-delay 2 \
      --connect-timeout 30 \
      --output "${temporary}" \
      "${source_url}"; then
    rm -f -- "${temporary}"
    echo "Failed to download ${dataset_name}" >&2
    exit 2
  fi
  if [[ ! -s "${temporary}" ]]; then
    echo "Downloaded an empty ${dataset_name} file: ${temporary}" >&2
    exit 2
  fi
  mv -- "${temporary}" "${destination}"
  echo "Downloaded ${dataset_name}: ${destination}"
}

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  download_if_missing halueval_qa "${HALUEVAL_URL}" "${HALUEVAL_DATA}"
  download_if_missing boolq_dev "${BOOLQ_URL}" "${BOOLQ_DATA}"
fi

for dataset_file in "${HALUEVAL_DATA}" "${BOOLQ_DATA}"; do
  if [[ ! -s "${dataset_file}" ]]; then
    echo "Missing dataset file: ${dataset_file}" >&2
    exit 2
  fi
done

echo "Datasets are ready; no dataset manifest is required."

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
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export BLIS_NUM_THREADS="${CPU_THREADS}"

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
    POSTPROCESS_DEVICE="${POSTPROCESS_DEVICE}" \
    RETAIN_DENSE_ATTENTION="${RETAIN_DENSE_ATTENTION}" \
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
