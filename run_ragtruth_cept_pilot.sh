#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DEFAULT_PYTHON="/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${DEFAULT_PYTHON}" ]]; then
    PYTHON_BIN="${DEFAULT_PYTHON}"
  else
    PYTHON_BIN="$(command -v python || command -v python3)"
  fi
fi

DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/RAGTruth/dataset}"
MODEL_PATH="${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
FEATURE_ROOT="${FEATURE_ROOT:-${DATA_ROOT}/feature_extraction}"
PYTHON_TAG="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
CEPT_VENV="${CEPT_VENV:-${DATA_ROOT}/.venvs/cept-py${PYTHON_TAG}-transformers-4.52.3}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
MAX_SEQUENCE_TOKENS="${MAX_SEQUENCE_TOKENS:-2048}"
MAX_COUNTERFACTUALS="${MAX_COUNTERFACTUALS:-2}"
HISTORY_BLOCK_SIZE="${HISTORY_BLOCK_SIZE:-4}"
MAX_HISTORY_BLOCKS="${MAX_HISTORY_BLOCKS:-8}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${FEATURE_ROOT}/ragtruth_cept_gate01_${RUN_TAG}_seed${SEED}}"
ELIGIBLE_RESPONSE_IDS="${ELIGIBLE_RESPONSE_IDS:-}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
RESUME="${RESUME:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${ELIGIBLE_RESPONSE_IDS}" ]]; then
  printf 'ELIGIBLE_RESPONSE_IDS is required and must come from a verified train cache inventory.\n' >&2
  exit 1
fi

for required in \
  "${DATASET_DIR}/response.jsonl" \
  "${DATASET_DIR}/source_info.jsonl" \
  "${MODEL_PATH}/config.json" \
  "${ELIGIBLE_RESPONSE_IDS}"; do
  if [[ ! -f "${required}" ]]; then
    printf 'Missing CEPT input: %s\n' "${required}" >&2
    exit 1
  fi
done

if ! command -v flock >/dev/null 2>&1; then
  printf 'The CEPT runner requires flock for safe concurrent setup.\n' >&2
  exit 1
fi

mkdir -p "$(dirname -- "${CEPT_VENV}")"
exec 9>"${CEPT_VENV}.bootstrap.lock"
flock -x 9

BASE_IDENTITY="$("${PYTHON_BIN}" - <<'PY'
import json
import sys
from pathlib import Path

import torch

print(json.dumps({
    "python": str(Path(sys.executable).resolve()),
    "python_version": sys.version,
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
}, sort_keys=True))
PY
)"
CREATED_VENV=0
if [[ ! -x "${CEPT_VENV}/bin/python" ]]; then
  printf 'Creating isolated CEPT environment: %s\n' "${CEPT_VENV}"
  "${PYTHON_BIN}" -m venv --system-site-packages "${CEPT_VENV}"
  CREATED_VENV=1
fi
CEPT_PYTHON="${CEPT_VENV}/bin/python"
BASE_IDENTITY_FILE="${CEPT_VENV}/.cept-base-identity.json"

if [[ "${CREATED_VENV}" == "1" ]]; then
  printf '%s\n' "${BASE_IDENTITY}" >"${BASE_IDENTITY_FILE}"
elif [[ ! -f "${BASE_IDENTITY_FILE}" ]]; then
  printf 'Existing CEPT_VENV lacks its base identity: %s\n' "${CEPT_VENV}" >&2
  printf 'Choose a fresh CEPT_VENV path; the existing directory was not modified.\n' >&2
  exit 1
elif [[ "$(<"${BASE_IDENTITY_FILE}")" != "${BASE_IDENTITY}" ]]; then
  printf 'CEPT_VENV was created from a different Python/Torch/CUDA base: %s\n' \
    "${CEPT_VENV}" >&2
  printf 'Choose a fresh CEPT_VENV path for the current PYTHON_BIN.\n' >&2
  exit 1
fi

if [[ ! -f "${CEPT_VENV}/pyvenv.cfg" ]] || \
   ! grep -Eiq '^include-system-site-packages[[:space:]]*=[[:space:]]*true[[:space:]]*$' \
      "${CEPT_VENV}/pyvenv.cfg"; then
  printf 'CEPT_VENV must be a venv created with --system-site-packages: %s\n' \
    "${CEPT_VENV}" >&2
  printf 'Choose a fresh CEPT_VENV path; the existing directory was not modified.\n' >&2
  exit 1
fi

"${CEPT_PYTHON}" - <<'PY'
import sys
import torch

if sys.prefix == sys.base_prefix:
    raise SystemExit("CEPT runtime is not an isolated virtual environment")
if not torch.cuda.is_available():
    raise SystemExit(
        "the CEPT venv cannot access CUDA torch from its base environment; "
        "set PYTHON_BIN to the CUDA-enabled research Python and use a fresh CEPT_VENV"
    )
PY

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  if ! "${CEPT_PYTHON}" - <<'PY'
import sys
from pathlib import Path

import accelerate
import packaging
import safetensors
import transformers

venv_prefix = Path(sys.prefix).resolve()
transformers_path = Path(transformers.__file__).resolve()
try:
    transformers_path.relative_to(venv_prefix)
except ValueError:
    raise SystemExit(1)
raise SystemExit(transformers.__version__ != "4.52.3")
PY
  then
    printf 'Installing pinned CEPT dependencies inside %s\n' "${CEPT_VENV}"
    "${CEPT_PYTHON}" -m pip install --disable-pip-version-check \
      --ignore-installed --no-deps transformers==4.52.3
    "${CEPT_PYTHON}" -m pip install \
      --disable-pip-version-check \
      -r requirements-counterfactual-grounding.txt
  fi
elif [[ "${INSTALL_DEPS}" != "0" ]]; then
  printf 'INSTALL_DEPS must be 0 or 1, got %s\n' "${INSTALL_DEPS}" >&2
  exit 1
fi

"${CEPT_PYTHON}" - <<'PY'
import sys
from pathlib import Path

import torch
import transformers

if transformers.__version__ != "4.52.3":
    raise SystemExit(
        f"CEPT requires transformers==4.52.3, found {transformers.__version__}; "
        "run with INSTALL_DEPS=1"
    )
venv_prefix = Path(sys.prefix).resolve()
transformers_path = Path(transformers.__file__).resolve()
try:
    transformers_path.relative_to(venv_prefix)
except ValueError:
    raise SystemExit(
        "transformers resolves outside CEPT_VENV; run with INSTALL_DEPS=1"
    )
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; Gate 1 requires a CUDA observer replay")
print(f"cept_python={sys.executable}")
print(f"cept_prefix={sys.prefix} base_prefix={sys.base_prefix}")
print(f"torch={torch.__version__} torch_path={torch.__file__}")
print(f"transformers={transformers.__version__} transformers_path={transformers.__file__}")
print(f"gpu={torch.cuda.get_device_name(0)} visible_device={torch.cuda.current_device()}")
PY

flock -u 9
exec 9>&-

if [[ "${RESUME}" == "1" ]]; then
  if [[ ! -f "${OUTPUT_DIR}/manifest.json" ]]; then
    printf 'Cannot resume without the original manifest: %s\n' \
      "${OUTPUT_DIR}/manifest.json" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume)
elif [[ "${RESUME}" == "0" ]]; then
  mkdir -p "$(dirname -- "${OUTPUT_DIR}")"
  if ! mkdir "${OUTPUT_DIR}"; then
    printf 'Output already exists; set RESUME=1 only for the identical interrupted run: %s\n' \
      "${OUTPUT_DIR}" >&2
    exit 1
  fi
  RESUME_ARGS=()
else
  printf 'RESUME must be 0 or 1, got %s\n' "${RESUME}" >&2
  exit 1
fi
exec 8>"${OUTPUT_DIR}/.cept-shell.lock"
if ! flock -n 8; then
  printf 'Another CEPT shell owns this output directory: %s\n' "${OUTPUT_DIR}" >&2
  exit 1
fi
printf 'method=CEPT Gate0/1 (no training, no AUROC)\n'
printf 'dataset=%s model=%s\n' "${DATASET_DIR}" "${MODEL_PATH}"
printf 'samples=%s seed=%s device=%s dtype=%s\n' \
  "${NUM_SAMPLES}" "${SEED}" "${DEVICE}" "${DTYPE}"
printf 'history_block_size=%s max_history_blocks=%s\n' \
  "${HISTORY_BLOCK_SIZE}" "${MAX_HISTORY_BLOCKS}"
printf 'output=%s\n' "${OUTPUT_DIR}"

"${CEPT_PYTHON}" -u -m counterfactual_grounding.cli pilot \
  --responses "${DATASET_DIR}/response.jsonl" \
  --sources "${DATASET_DIR}/source_info.jsonl" \
  --model "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --eligible-response-ids "${ELIGIBLE_RESPONSE_IDS}" \
  --split train \
  --num-samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max-sequence-tokens "${MAX_SEQUENCE_TOKENS}" \
  --max-counterfactuals "${MAX_COUNTERFACTUALS}" \
  --history-block-size "${HISTORY_BLOCK_SIZE}" \
  --max-history-blocks "${MAX_HISTORY_BLOCKS}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "${OUTPUT_DIR}/run.log"

printf 'CEPT Gate 0/1 artifacts: %s\n' "${OUTPUT_DIR}"
