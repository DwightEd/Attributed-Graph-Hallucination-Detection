#!/usr/bin/env bash
set -euo pipefail

PILOT_ROOT="${PILOT_ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/feature_audit/partial_cache_pilot_ragtruth_partial_cache_20260802T165711Z}"
HYPERGRAPH_RESULT="${HYPERGRAPH_RESULT:-${PILOT_ROOT}/hypergraph_ssl_full_hierarchical_ssl_v2_full}"
FEATURE_ROOT="${FEATURE_ROOT:-/share/home/tm902089733300000/a903202310/lys/data/feature_extraction}"
HYPERGRAPH_LINK="${HYPERGRAPH_LINK:-${FEATURE_ROOT}/ragtruth_hypergraph_ssl_v2_full}"

if [[ ! -d "${HYPERGRAPH_RESULT}" ]]; then
  printf 'Historical full-v2 hypergraph result not found: %s\n' "${HYPERGRAPH_RESULT}" >&2
  printf 'Known hypergraph result directories under the pilot root:\n' >&2
  find "${PILOT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'hypergraph_ssl*' -print >&2 || true
  exit 1
fi

mkdir -p "${FEATURE_ROOT}"
if [[ -L "${HYPERGRAPH_LINK}" ]]; then
  current="$(readlink -f "${HYPERGRAPH_LINK}")"
  expected="$(readlink -f "${HYPERGRAPH_RESULT}")"
  if [[ "${current}" != "${expected}" ]]; then
    printf 'Existing link points elsewhere: %s -> %s\n' "${HYPERGRAPH_LINK}" "${current}" >&2
    exit 1
  fi
elif [[ -e "${HYPERGRAPH_LINK}" ]]; then
  printf 'Refusing to overwrite an existing non-link path: %s\n' "${HYPERGRAPH_LINK}" >&2
  exit 1
else
  ln -s "${HYPERGRAPH_RESULT}" "${HYPERGRAPH_LINK}"
fi

printf 'Historical hypergraph result: %s -> %s\n' "${HYPERGRAPH_LINK}" "${HYPERGRAPH_RESULT}"
