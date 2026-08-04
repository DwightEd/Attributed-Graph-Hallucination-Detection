#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# This launcher is intentionally self-contained.  Edit the defaults below when
# needed; no positional command-line arguments are accepted.
if [[ "$#" -ne 0 ]]; then
  printf 'Usage: bash ./run_halueval_attention_graph.sh\n' >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python}"
SOURCE_RUN_FILE="${SOURCE_RUN_FILE:-/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/LATEST_HALUEVAL_GRAPH_MAE_RUN.txt}"
SOURCE_RUN="${SOURCE_RUN:-}"
if [[ -z "${SOURCE_RUN}" ]]; then
  if [[ ! -s "${SOURCE_RUN_FILE}" ]]; then
    printf 'SOURCE_RUN_FILE does not exist or is empty: %s\n' "${SOURCE_RUN_FILE}" >&2
    exit 2
  fi
  SOURCE_RUN="$(<"${SOURCE_RUN_FILE}")"
fi
EXTRACTION_DIR="${EXTRACTION_DIR:-${SOURCE_RUN}/extraction}"
EXAMPLES="${EXAMPLES:-${SOURCE_RUN}/prepared/examples.jsonl}"
EVALUATION_LABELS="${EVALUATION_LABELS:-${SOURCE_RUN}/prepared/evaluation_labels.jsonl}"

SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOURCE_RUN}/attention_graph_${RUN_TAG}}"
DEVICE="${DEVICE:-cuda}"
SELECTION="${SELECTION:-threshold}"
THRESHOLD="${THRESHOLD:-}"
TOP_K="${TOP_K:-8}"
MAX_EDGES_PER_TARGET="${MAX_EDGES_PER_TARGET:-64}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.10}"
TEST_FRACTION="${TEST_FRACTION:-0.20}"
LIMIT_PAIRS="${LIMIT_PAIRS:-}"
REQUIRE_COMPLETE_CACHE="${REQUIRE_COMPLETE_CACHE:-0}"
SKIP_EVALUATION="${SKIP_EVALUATION:-0}"
GROUP_BY_PROMPT="${GROUP_BY_PROMPT:-0}"
CONVERSION_CHUNK_EDGES="${CONVERSION_CHUNK_EDGES:-8192}"

# Preserve an explicitly supplied CUDA mask while providing a reproducible default.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python executable does not exist: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi

# Fail before allocating GPU memory if Python resolves an installed/stale
# package instead of this repository's dedicated HaluEval entrypoint.
RESOLVED_HALUEVAL_CLI="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import attention_graph.halueval_cli as module

root = Path.cwd().resolve()
expected = (root / "attention_graph" / "halueval_cli.py").resolve()
actual = Path(module.__file__).resolve()
if actual != expected:
    raise SystemExit(
        f"Unexpected HaluEval CLI module: expected {expected}, resolved {actual}"
    )
print(actual)
PY
)"

if [[ -z "${SOURCE_RUN}" || ! -d "${EXTRACTION_DIR}" ]]; then
  printf 'Legacy HaluEval extraction directory does not exist: %s\n' "${EXTRACTION_DIR}" >&2
  exit 2
fi
SOURCE_COMPLETION="${SOURCE_RUN}/training/evaluation_only_metrics.json"
if [[ ! -s "${SOURCE_COMPLETION}" ]]; then
  printf 'The source Graph-MAE run has not completed: %s\n' "${SOURCE_RUN}" >&2
  printf 'Wait for its evaluation file before starting another GPU job: %s\n' "${SOURCE_COMPLETION}" >&2
  exit 2
fi
if [[ ! -s "${EXAMPLES}" ]]; then
  printf 'Label-free examples file does not exist: %s\n' "${EXAMPLES}" >&2
  exit 2
fi
if [[ "${SKIP_EVALUATION}" == "0" && ! -s "${EVALUATION_LABELS}" ]]; then
  printf 'Evaluation-label sidecar does not exist: %s\n' "${EVALUATION_LABELS}" >&2
  exit 2
fi
if [[ "${REQUIRE_COMPLETE_CACHE}" != "0" && "${REQUIRE_COMPLETE_CACHE}" != "1" ]]; then
  printf 'REQUIRE_COMPLETE_CACHE must be 0 or 1, got %s\n' "${REQUIRE_COMPLETE_CACHE}" >&2
  exit 2
fi
if [[ "${SKIP_EVALUATION}" != "0" && "${SKIP_EVALUATION}" != "1" ]]; then
  printf 'SKIP_EVALUATION must be 0 or 1, got %s\n' "${SKIP_EVALUATION}" >&2
  exit 2
fi
if [[ "${GROUP_BY_PROMPT}" != "0" && "${GROUP_BY_PROMPT}" != "1" ]]; then
  printf 'GROUP_BY_PROMPT must be 0 or 1, got %s\n' "${GROUP_BY_PROMPT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

printf 'HaluEval attention-only graph experiment\n'
printf 'entrypoint=attention_graph.halueval_cli\n'
printf 'resolved_cli=%s\n' "${RESOLVED_HALUEVAL_CLI}"
printf 'source_run=%s\n' "${SOURCE_RUN}"
printf 'output_dir=%s\n' "${OUTPUT_DIR}"
printf 'gpu=%s device=%s seed=%s epochs=%s selection=%s\n' \
  "${CUDA_VISIBLE_DEVICES}" "${DEVICE}" "${SEED}" "${EPOCHS}" "${SELECTION}"
printf 'Warning: legacy tau-censored compatibility; this is not the dense/floor-0.01 protocol.\n' >&2

ARGS=(
  --source-run "${SOURCE_RUN}"
  --extraction-dir "${EXTRACTION_DIR}"
  --examples "${EXAMPLES}"
  --evaluation-labels "${EVALUATION_LABELS}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --selection "${SELECTION}"
  --top-k "${TOP_K}"
  --max-edges-per-target "${MAX_EDGES_PER_TARGET}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
  --validation-fraction "${VALIDATION_FRACTION}"
  --test-fraction "${TEST_FRACTION}"
  --conversion-chunk-edges "${CONVERSION_CHUNK_EDGES}"
  --seed "${SEED}"
)
if [[ -n "${THRESHOLD}" ]]; then
  ARGS+=(--threshold "${THRESHOLD}")
fi
if [[ -n "${LIMIT_PAIRS}" ]]; then
  ARGS+=(--limit-pairs "${LIMIT_PAIRS}")
fi
if [[ "${REQUIRE_COMPLETE_CACHE}" == "1" ]]; then
  ARGS+=(--require-complete-cache)
fi
if [[ "${SKIP_EVALUATION}" == "1" ]]; then
  ARGS+=(--skip-evaluation)
fi
if [[ "${GROUP_BY_PROMPT}" == "1" ]]; then
  ARGS+=(--group-by-prompt)
fi

"${PYTHON_BIN}" -m attention_graph.halueval_cli run "${ARGS[@]}"
