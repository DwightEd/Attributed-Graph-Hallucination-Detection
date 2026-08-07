#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE="${BASE:-/share/home/tm902089733300000/a903202310/lys}"
PYTHON_BIN="${PYTHON_BIN:-$BASE/conda_envs/research/bin/python}"
PATF_OUTPUT="${PATF_OUTPUT:-$BASE/data/feature_extraction/patf/20260806T172826Z}"
RAGTRUTH_ROOT="${RAGTRUTH_ROOT:-$BASE/data/RAGTruth/dataset}"
FEATURE_SPLIT="${FEATURE_SPLIT:-train}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$PATF_OUTPUT/diagnostics/$FEATURE_SPLIT}"
FIT_RANKER="${FIT_RANKER:-1}"
COUNTERFACTUAL="${COUNTERFACTUAL:-auto}"
HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0.20}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-48}"
DEVICE="${DEVICE:-cpu}"
LIMIT="${LIMIT:-}"

ARGS=(
  --output-dir "$PATF_OUTPUT"
  --ragtruth-root "$RAGTRUTH_ROOT"
  --feature-split "$FEATURE_SPLIT"
  --analysis-dir "$ANALYSIS_DIR"
  --counterfactual "$COUNTERFACTUAL"
  --holdout-fraction "$HOLDOUT_FRACTION"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --hidden-dim "$HIDDEN_DIM"
  --device "$DEVICE"
)

if [[ "$FIT_RANKER" == "1" ]]; then
  ARGS+=(--fit-ranker)
elif [[ "$FIT_RANKER" != "0" ]]; then
  echo "FIT_RANKER must be 0 or 1" >&2
  exit 2
fi

if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

printf 'saved_patf_output=%s\n' "$PATF_OUTPUT"
printf 'ragtruth_root=%s\n' "$RAGTRUTH_ROOT"
printf 'analysis_dir=%s\n' "$ANALYSIS_DIR"
printf 'feature_split=%s fit_ranker=%s counterfactual=%s device=%s\n' \
  "$FEATURE_SPLIT" "$FIT_RANKER" "$COUNTERFACTUAL" "$DEVICE"

"$PYTHON_BIN" scripts/analyze_saved_patf.py "${ARGS[@]}" \
  2>&1 | tee "$ANALYSIS_DIR/analysis.log"
