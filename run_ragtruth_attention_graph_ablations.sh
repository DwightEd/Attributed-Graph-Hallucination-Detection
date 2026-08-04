#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data}"
FEATURE_ROOT="${FEATURE_ROOT:-${DATA_ROOT}/feature_extraction}"
SEED="${SEED:-42}"
ABLATION_TAG="${ABLATION_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_seed${SEED}}"
ABLATION_ROOT="${ABLATION_ROOT:-${FEATURE_ROOT}/ragtruth_attention_graph_ablations_${ABLATION_TAG}}"
GRAPH_STORE_ROOT="${GRAPH_STORE_ROOT:-${FEATURE_ROOT}/ragtruth_attention_graph_prepared}"
RUN_SELECTION_ABLATIONS="${RUN_SELECTION_ABLATIONS:-1}"
REQUIRE_COMPLETE_CACHE="${REQUIRE_COMPLETE_CACHE:-0}"

run_variant() {
  local name="$1"
  local transform="$2"
  local message_steps="$3"
  local selection="$4"
  local graph_key="$5"
  local support_weight="1.0"
  local attention_weight="1.0"
  local distribution_weight="1.0"
  local node_weight="0.25"
  local embedding_only="0"
  if [[ "${name}" == "feature_only" ]]; then
    support_weight="0.0"
    attention_weight="0.0"
    distribution_weight="0.0"
    node_weight="1.0"
    embedding_only="1"
  fi

  printf '\n=== %s ===\n' "${name}"
  OUTPUT_DIR="${ABLATION_ROOT}/${name}" \
  GRAPH_DIR="${GRAPH_STORE_ROOT}/${graph_key}" \
  GRAPH_TRANSFORM="${transform}" \
  MESSAGE_PASSING_STEPS="${message_steps}" \
  SELECTION="${selection}" \
  SUPPORT_WEIGHT="${support_weight}" \
  ATTENTION_WEIGHT="${attention_weight}" \
  DISTRIBUTION_WEIGHT="${distribution_weight}" \
  NODE_WEIGHT="${node_weight}" \
  EMBEDDING_ONLY_SCORING="${embedding_only}" \
  REQUIRE_COMPLETE_CACHE="${REQUIRE_COMPLETE_CACHE}" \
  SEED="${SEED}" \
    bash "${SCRIPT_DIR}/run_ragtruth_attention_graph.sh"
}

# These six runs share the exact same threshold+cap64 prepared graphs.
run_variant full none 2 threshold threshold_cap64
run_variant no_message none 0 threshold threshold_cap64
run_variant feature_only none 0 threshold threshold_cap64
run_variant source_shuffle source_shuffle 2 threshold threshold_cap64
run_variant collapse_relations collapse_relations 2 threshold threshold_cap64
run_variant mean_heads mean_heads 2 threshold threshold_cap64

if [[ "${RUN_SELECTION_ABLATIONS}" == "1" ]]; then
  run_variant global_topk none 2 global_topk global_topk_k8
  run_variant typed_topk none 2 typed_topk typed_topk_k8
fi

printf '\nAblations complete: %s\n' "${ABLATION_ROOT}"
