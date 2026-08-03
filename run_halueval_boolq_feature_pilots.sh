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
BOOLQ_HF_ROWS_URL="https://datasets-server.huggingface.co/rows"

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
    rm -f -- "${temporary}"
    echo "Downloaded an empty ${dataset_name} file: ${temporary}" >&2
    exit 2
  fi
  mv -- "${temporary}" "${destination}"
  echo "Downloaded ${dataset_name}: ${destination}"
}

validate_boolq_file() {
  local source_path="$1"
  "${PYTHON_BIN}" - "${source_path}" <<'PY'
import json
import sys
from pathlib import Path


source_path = Path(sys.argv[1])
expected_rows = 3270
rows = 0
with source_path.open(encoding="utf-8") as source:
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            raise ValueError(f"BoolQ line {line_number} is empty")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"BoolQ line {line_number} is not an object")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"BoolQ line {line_number} has an invalid question")
        if not isinstance(row.get("passage"), str) or not row["passage"].strip():
            raise ValueError(f"BoolQ line {line_number} has an invalid passage")
        if not isinstance(row.get("answer"), bool):
            raise ValueError(f"BoolQ line {line_number} has a non-boolean answer")
        rows += 1
if rows != expected_rows:
    raise ValueError(f"Expected {expected_rows} BoolQ rows, found {rows}")
print(f"Validated BoolQ: {rows} rows", flush=True)
PY
}

download_boolq_from_huggingface() {
  local destination="$1"
  local temporary="${destination}.$$.part"

  mkdir -p "$(dirname -- "${destination}")"
  echo "Downloading the Google BoolQ validation split from Hugging Face"
  if "${PYTHON_BIN}" - "${temporary}" "${BOOLQ_HF_ROWS_URL}" <<'PY'
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


output_path = Path(sys.argv[1])
endpoint = sys.argv[2]
expected_rows = 3270
page_size = 100
retry_delay_seconds = float(
    os.environ.get("BOOLQ_DOWNLOAD_RETRY_DELAY_SECONDS", "2")
)
if retry_delay_seconds < 0:
    raise ValueError("BOOLQ_DOWNLOAD_RETRY_DELAY_SECONDS cannot be negative")


def fetch_page(offset: int, length: int) -> dict[str, object]:
    query = urllib.parse.urlencode(
        {
            "dataset": "google/boolq",
            "config": "default",
            "split": "validation",
            "offset": offset,
            "length": length,
        }
    )
    request = urllib.request.Request(
        f"{endpoint}?{query}",
        headers={"User-Agent": "boolq-attention-pilot/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError("Hugging Face rows response is not an object")
            return payload
        except Exception as error:
            last_error = error
            if attempt < 4:
                time.sleep(retry_delay_seconds * 2 ** (attempt - 1))
    raise RuntimeError(
        f"Could not fetch BoolQ rows at offset {offset}: {last_error}"
    )


written = 0
with output_path.open("w", encoding="utf-8") as output:
    while written < expected_rows:
        payload = fetch_page(written, min(page_size, expected_rows - written))
        if payload.get("num_rows_total") != expected_rows:
            raise ValueError(
                "Google BoolQ validation split no longer contains "
                f"{expected_rows} rows: {payload.get('num_rows_total')!r}"
            )
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"No BoolQ rows returned at offset {written}")
        for item in rows:
            if not isinstance(item, dict) or item.get("row_idx") != written:
                raise ValueError(f"Unexpected BoolQ row index at offset {written}")
            if item.get("truncated_cells"):
                raise ValueError(f"BoolQ row {written} contains truncated cells")
            row = item.get("row")
            if not isinstance(row, dict):
                raise ValueError(f"BoolQ row {written} is not an object")
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError(f"BoolQ row {written} has an invalid question")
            if not isinstance(row.get("passage"), str) or not row["passage"].strip():
                raise ValueError(f"BoolQ row {written} has an invalid passage")
            if not isinstance(row.get("answer"), bool):
                raise ValueError(f"BoolQ row {written} has a non-boolean answer")
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
        print(f"Downloaded BoolQ validation: {written}/{expected_rows}", flush=True)
    output.flush()
    os.fsync(output.fileno())

if written != expected_rows:
    raise RuntimeError(f"Expected {expected_rows} BoolQ rows, wrote {written}")
PY
  then
    if ! mv -- "${temporary}" "${destination}"; then
      rm -f -- "${temporary}"
      echo "Failed to publish BoolQ dataset: ${destination}" >&2
      return 1
    fi
    echo "Downloaded boolq_dev: ${destination}"
  else
    rm -f -- "${temporary}"
    echo "Failed to download BoolQ from Hugging Face" >&2
    return 1
  fi
}

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  download_if_missing halueval_qa "${HALUEVAL_URL}" "${HALUEVAL_DATA}"
  if [[ -s "${BOOLQ_DATA}" ]] && validate_boolq_file "${BOOLQ_DATA}"; then
    echo "Reusing boolq_dev: ${BOOLQ_DATA}"
  else
    if [[ -e "${BOOLQ_DATA}" ]]; then
      echo "Existing BoolQ file is invalid; downloading a verified replacement"
    fi
    download_boolq_from_huggingface "${BOOLQ_DATA}"
    validate_boolq_file "${BOOLQ_DATA}"
  fi
else
  if [[ ! -s "${BOOLQ_DATA}" ]]; then
    echo "Missing dataset file: ${BOOLQ_DATA}" >&2
    exit 2
  fi
  validate_boolq_file "${BOOLQ_DATA}"
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
