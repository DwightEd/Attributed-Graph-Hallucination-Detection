#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python}"
DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/feature_extraction}"
LATEST_RUN_FILE="${LATEST_RUN_FILE:-${DATA_ROOT}/feature_extraction/LATEST_HALUEVAL_GRAPH_MAE_RUN.txt}"
SOURCE_RUN="${SOURCE_RUN:-}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"

# Dataset/split settings. LIMIT_PAIRS=none consumes every complete extracted pair.
LIMIT_PAIRS="${LIMIT_PAIRS:-none}"
EXPECTED_CANDIDATES="${EXPECTED_CANDIDATES:-2000}"
MIN_TEST_PAIR_COVERAGE="${MIN_TEST_PAIR_COVERAGE:-0.90}"
FAIL_ON_LOW_COVERAGE="${FAIL_ON_LOW_COVERAGE:-0}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.10}"
TEST_FRACTION="${TEST_FRACTION:-0.20}"
SPLIT_SEED="${SPLIT_SEED:-42}"
GROUP_BY_PROMPT="${GROUP_BY_PROMPT:-1}"
REQUIRE_COMPLETE_CACHE="${REQUIRE_COMPLETE_CACHE:-1}"

# Grounding-flow conditional null. These are method parameters, not MAE loss weights.
EVIDENCE_SEGMENTS="${EVIDENCE_SEGMENTS:-1}"
NULL_SAMPLES="${NULL_SAMPLES:-32}"
NULL_SWAPS_PER_EDGE="${NULL_SWAPS_PER_EDGE:-4}"
LAG_BOUNDARIES="${LAG_BOUNDARIES:-4,8,16,32,64,128}"
NULL_MAX_ATTEMPT_FACTOR="${NULL_MAX_ATTEMPT_FACTOR:-12}"
NULL_STD_FLOOR="${NULL_STD_FLOOR:-1e-6}"

# Label-free PCA + two-state HMM training.
PCA_COMPONENTS="${PCA_COMPONENTS:-32}"
PCA_FIT_TOKENS="${PCA_FIT_TOKENS:-20000}"
HMM_ITERATIONS="${HMM_ITERATIONS:-50}"
HMM_TOLERANCE="${HMM_TOLERANCE:-1e-4}"
HMM_VARIANCE_FLOOR="${HMM_VARIANCE_FLOOR:-1e-4}"
SEED="${SEED:-42}"

CONVERSION_CHUNK_EDGES="${CONVERSION_CHUNK_EDGES:-8192}"
QUERY_BLOCK="${QUERY_BLOCK:-32}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
RESUME="${RESUME:-1}"
SKIP_EVALUATION="${SKIP_EVALUATION:-0}"

for setting in GROUP_BY_PROMPT REQUIRE_COMPLETE_CACHE FAIL_ON_LOW_COVERAGE RESUME SKIP_EVALUATION; do
  value="${!setting}"
  case "${value}" in
    0|1) ;;
    *)
      echo "${setting} must be 0 or 1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${SOURCE_RUN}" ]]; then
  if [[ ! -s "${LATEST_RUN_FILE}" ]]; then
    echo "Latest HaluEval source-run pointer is missing: ${LATEST_RUN_FILE}" >&2
    echo "Set SOURCE_RUN to the completed pure-attention extraction run." >&2
    exit 2
  fi
  IFS= read -r SOURCE_RUN < "${LATEST_RUN_FILE}"
fi
SOURCE_RUN="${SOURCE_RUN%/}"
EXTRACTION_DIR="${EXTRACTION_DIR:-${SOURCE_RUN}/extraction}"
EXAMPLES="${EXAMPLES:-${SOURCE_RUN}/prepared/examples.jsonl}"
EVALUATION_LABELS="${EVALUATION_LABELS:-${SOURCE_RUN}/prepared/evaluation_labels.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/halueval_grounding_flow_$(date -u +%Y%m%dT%H%M%SZ)_seed${SEED}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -s "${EXTRACTION_DIR}/extraction_manifest.json" ]]; then
  echo "HaluEval extraction manifest is missing: ${EXTRACTION_DIR}/extraction_manifest.json" >&2
  exit 2
fi
if [[ ! -s "${EXAMPLES}" ]]; then
  echo "Label-free HaluEval examples are missing: ${EXAMPLES}" >&2
  exit 2
fi
if [[ "${SKIP_EVALUATION}" == "0" && ! -s "${EVALUATION_LABELS}" ]]; then
  echo "Evaluation label sidecar is missing: ${EVALUATION_LABELS}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/run.log") 2>&1

echo "Grounding-flow attention-only experiment"
echo "source_run=${SOURCE_RUN}"
echo "extraction=${EXTRACTION_DIR}"
echo "output=${OUTPUT_DIR}"
echo "limit_pairs=${LIMIT_PAIRS} expected_candidates=${EXPECTED_CANDIDATES} require_complete_cache=${REQUIRE_COMPLETE_CACHE} min_test_pair_coverage=${MIN_TEST_PAIR_COVERAGE} fail_on_low_coverage=${FAIL_ON_LOW_COVERAGE}"
echo "gpu=${GPU_ID} device=${DEVICE} seed=${SEED}"
echo "null_samples=${NULL_SAMPLES} swaps_per_edge=${NULL_SWAPS_PER_EDGE} lag_boundaries=${LAG_BOUNDARIES}"
echo "pca_components=${PCA_COMPONENTS} hmm_iterations=${HMM_ITERATIONS}"
echo "This is the independent grounding_flow pipeline; no MAE or pattern-audit stage is invoked."

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --id="${GPU_ID}" \
    --query-gpu=index,name,memory.used,memory.total \
    --format=csv,noheader
fi

arguments=(
  run
  --extraction-dir "${EXTRACTION_DIR}"
  --examples "${EXAMPLES}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --limit-pairs "${LIMIT_PAIRS}"
  --expected-candidates "${EXPECTED_CANDIDATES}"
  --min-test-pair-coverage "${MIN_TEST_PAIR_COVERAGE}"
  --validation-fraction "${VALIDATION_FRACTION}"
  --test-fraction "${TEST_FRACTION}"
  --split-seed "${SPLIT_SEED}"
  --evidence-segments "${EVIDENCE_SEGMENTS}"
  --num-nulls "${NULL_SAMPLES}"
  --null-swaps-per-edge "${NULL_SWAPS_PER_EDGE}"
  --lag-boundaries "${LAG_BOUNDARIES}"
  --null-max-attempt-factor "${NULL_MAX_ATTEMPT_FACTOR}"
  --null-std-floor "${NULL_STD_FLOOR}"
  --pca-components "${PCA_COMPONENTS}"
  --pca-fit-tokens "${PCA_FIT_TOKENS}"
  --hmm-iterations "${HMM_ITERATIONS}"
  --hmm-tolerance "${HMM_TOLERANCE}"
  --hmm-variance-floor "${HMM_VARIANCE_FLOOR}"
  --conversion-chunk-edges "${CONVERSION_CHUNK_EDGES}"
  --query-block "${QUERY_BLOCK}"
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}"
  --seed "${SEED}"
)

if [[ "${GROUP_BY_PROMPT}" == "1" ]]; then
  arguments+=(--group-by-prompt)
else
  arguments+=(--no-group-by-prompt)
fi
if [[ "${REQUIRE_COMPLETE_CACHE}" == "1" ]]; then
  arguments+=(--require-complete-cache)
else
  arguments+=(--allow-partial-cache)
fi
if [[ "${FAIL_ON_LOW_COVERAGE}" == "1" ]]; then
  arguments+=(--fail-on-low-coverage)
fi
if [[ "${RESUME}" == "0" ]]; then
  arguments+=(--no-resume)
fi
if [[ "${SKIP_EVALUATION}" == "1" ]]; then
  arguments+=(--skip-evaluation)
else
  arguments+=(--evaluation-labels "${EVALUATION_LABELS}")
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
"${PYTHON_BIN}" -m grounding_flow.cli run "${arguments[@]:1}"

echo "Grounding-flow experiment complete: ${OUTPUT_DIR}"
echo "Metrics: ${OUTPUT_DIR}/evaluation.json"
