#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE="${BASE:-/share/home/tm902089733300000/a903202310/lys}"
PYTHON="${PYTHON:-$BASE/conda_envs/research/bin/python}"
ATTENTION_ROOT="${ATTENTION_ROOT:-$BASE/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}"
RAGTRUTH_ROOT="${RAGTRUTH_ROOT:-$BASE/data/RAGTruth/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/data/feature_extraction/patf/$(date -u +%Y%m%dT%H%M%SZ)}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-80}"
WORKERS="${WORKERS:-8}"
TORCH_THREADS="${TORCH_THREADS:-1}"

export OMP_NUM_THREADS="$TORCH_THREADS"
export MKL_NUM_THREADS="$TORCH_THREADS"
export OPENBLAS_NUM_THREADS="$TORCH_THREADS"
export NUMEXPR_NUM_THREADS="$TORCH_THREADS"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR"
printf 'attention_root=%s\nragtruth_root=%s\noutput_dir=%s\n' \
  "$ATTENTION_ROOT" "$RAGTRUTH_ROOT" "$OUTPUT_DIR"
printf 'device=%s epochs=%s workers=%s torch_threads=%s\n' \
  "$DEVICE" "$EPOCHS" "$WORKERS" "$TORCH_THREADS"

"$PYTHON" main.py \
  --config configs/patf.json \
  --attention-root "$ATTENTION_ROOT" \
  --ragtruth-root "$RAGTRUTH_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --workers "$WORKERS" \
  --torch-threads "$TORCH_THREADS" \
  2>&1 | tee -a "$OUTPUT_DIR/run.log"
