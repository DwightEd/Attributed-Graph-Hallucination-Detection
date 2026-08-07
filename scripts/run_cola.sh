#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE="${BASE:-/share/home/tm902089733300000/a903202310/lys}"
PYTHON="${PYTHON:-$BASE/conda_envs/research/bin/python}"
GRAPH_ROOT="${GRAPH_ROOT:-$BASE/data/feature_extraction/ragtruth_original_attribute_graphs/fresh_attention_c8847872bedf_20260731T074520Z_p876_tau0p05}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/data/feature_extraction/cola/$(date -u +%Y%m%dT%H%M%SZ)}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-300}"
SUBGRAPH_SIZE="${SUBGRAPH_SIZE:-4}"
TEST_ROUNDS="${TEST_ROUNDS:-256}"
SEED="${SEED:-1}"

mkdir -p "$OUTPUT_DIR"

"$PYTHON" -m baselines.cola.run \
  --graph-root "$GRAPH_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --subgraph-size "$SUBGRAPH_SIZE" \
  --test-rounds "$TEST_ROUNDS" \
  --seed "$SEED" \
  2>&1 | tee -a "$OUTPUT_DIR/run.log"
