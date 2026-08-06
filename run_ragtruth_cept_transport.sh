#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/RAGTruth/dataset}"
MODEL_PATH="${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}"
ATTENTION_CACHE_ROOT="${ATTENTION_CACHE_ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}"
FEATURE_ROOT="${FEATURE_ROOT:-${DATA_ROOT}/feature_extraction}"
PYTHON_BIN="${PYTHON_BIN:-/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python}"
PYTHON_TAG="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
CEPT_VENV="${CEPT_VENV:-${DATA_ROOT}/.venvs/cept-py${PYTHON_TAG}-transformers-4.52.3}"
GRAPH_STORE="${GRAPH_STORE:-${FEATURE_ROOT}/ragtruth_cept_prediction_graphs_fresh_attention_c8847872bedf_layerstate_v5_runtimebound}"
SEED="${SEED:-42}"
TEACHER_SAMPLES="${TEACHER_SAMPLES:-300}"
HISTORY_BLOCK_SIZE="${HISTORY_BLOCK_SIZE:-4}"
MAX_HISTORY_BLOCKS="${MAX_HISTORY_BLOCKS:-8}"
EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-0.03}"
GPU_ID="${GPU_ID:-0}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${FEATURE_ROOT}/ragtruth_cept_transport_${RUN_TAG}_seed${SEED}}"
TEACHER_DIR="${RUN_ROOT}/teacher"
TRANSPORT_DIR="${RUN_ROOT}/transport"
ELIGIBLE_RESPONSE_IDS="${RUN_ROOT}/eligible_train_response_ids.jsonl"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

for required in \
  "${DATASET_DIR}/response.jsonl" \
  "${DATASET_DIR}/source_info.jsonl" \
  "${MODEL_PATH}/config.json" \
  "${ATTENTION_CACHE_ROOT}/train" \
  "${ATTENTION_CACHE_ROOT}/test"; do
  if [[ ! -e "${required}" ]]; then
    printf 'Missing CEPT transport input: %s\n' "${required}" >&2
    printf 'Both official train and test attention caches are required.\n' >&2
    exit 1
  fi
done

if [[ -e "${TRANSPORT_DIR}" ]]; then
  printf 'Transport output already exists; choose a new RUN_ROOT: %s\n' \
    "${TRANSPORT_DIR}" >&2
  exit 1
fi

# Fail before the expensive K/V teacher if the cached attention cannot be
# cryptographically bound to the exact observer model.  The legacy extractor
# already stored every model-directory root filename and SHA-256 digest.
"${PYTHON_BIN}" - "${ATTENTION_CACHE_ROOT}" "${MODEL_PATH}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

from counterfactual_grounding.observer_runtime import (
    observer_runtime_from_cache_manifest,
)

cache_root = Path(sys.argv[1]).resolve()
model_root = Path(sys.argv[2]).resolve()


def declared_inventory(manifest, *, split, field):
    if field not in manifest:
        return None
    values = manifest[field]
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise SystemExit(
            f"{split} attention cache {field} must be a list of non-empty strings"
        )
    if len(values) != len(set(values)):
        raise SystemExit(f"{split} attention cache has duplicate {field}")
    return values


# Validate only cheap cache metadata first.  Do not hash multi-GB model files
# when the cache is already known to be partial or ambiguously inventoried.
manifests = {}
runtimes = {}
for split in ("train", "test"):
    path = cache_root / split / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"attention cache manifest is absent: {path}")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise SystemExit(f"{split} attention cache manifest must be a JSON object")
    if str(manifest.get("state", "")).casefold() != "complete":
        raise SystemExit(
            f"{split} attention cache inventory is incomplete; resume extraction "
            "before running CEPT transport"
        )

    observed = sorted(item.name for item in path.parent.glob("attention_*.pt"))
    cache_file_names = declared_inventory(
        manifest, split=split, field="cache_file_names"
    )
    expected_files = declared_inventory(
        manifest, split=split, field="expected_files"
    )
    if cache_file_names is None and expected_files is None:
        raise SystemExit(
            f"{split} attention cache declares neither cache_file_names nor "
            "expected_files"
        )
    if (
        cache_file_names is not None
        and expected_files is not None
        and set(cache_file_names) != set(expected_files)
    ):
        raise SystemExit(
            f"{split} attention cache cache_file_names and expected_files disagree"
        )
    declared = cache_file_names if cache_file_names is not None else expected_files
    if not observed or len(declared) != len(observed) or set(declared) != set(observed):
        raise SystemExit(
            f"{split} attention cache declared inventory does not exactly match "
            "the observed attention-file membership"
        )

    for field in ("matched_samples", "cache_files"):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != len(observed):
            raise SystemExit(
                f"{split} attention cache {field} does not match the observed "
                f"file count ({len(observed)})"
            )
    declared_hashes = manifest.get("cache_files_sha256")
    if not isinstance(declared_hashes, dict) or set(map(str, declared_hashes)) != set(
        observed
    ):
        raise SystemExit(
            f"{split} attention cache has no exact cache_files_sha256 membership; "
            "re-extract the split before running CEPT transport"
        )
    if any(
        re.fullmatch(r"[0-9a-fA-F]{64}", str(digest)) is None
        for digest in declared_hashes.values()
    ):
        raise SystemExit(f"{split} attention cache has malformed cache_files_sha256")
    try:
        runtime, runtime_signature = observer_runtime_from_cache_manifest(manifest)
    except (TypeError, ValueError) as error:
        raise SystemExit(
            f"{split} attention cache observer runtime is missing/invalid: {error}; "
            "re-extract before running CEPT transport"
        ) from error
    required_runtime = {
        "transformers_version": "4.52.3",
        "torch_version": torch.__version__,
        "attn_implementation": "eager",
    }
    mismatches = {
        field: (runtime[field], expected)
        for field, expected in required_runtime.items()
        if runtime[field] != expected
    }
    if mismatches:
        raise SystemExit(
            f"{split} attention cache observer runtime {runtime} cannot be replayed "
            f"by the fixed teacher runtime {required_runtime}; mismatches={mismatches}. "
            "Re-extract both attention splits with transformers==4.52.3, the base "
            "torch runtime, and eager attention."
        )
    runtimes[split] = (runtime, runtime_signature)
    manifests[split] = manifest
    print(f"cache_manifest_metadata_{split}=complete files={len(observed)}")

if runtimes["train"] != runtimes["test"]:
    raise SystemExit(
        "train/test attention caches have different observer runtimes; re-extract "
        "both splits under one runtime"
    )


model_files = sorted(path for path in model_root.iterdir() if path.is_file())
inventory = []
current_hashes = {}

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for path in model_files:
    relative = path.relative_to(model_root).as_posix()
    digest = file_sha256(path)
    current_hashes[relative] = digest
    inventory.append({"path": relative, "size": path.stat().st_size, "sha256": digest})
payload = json.dumps(
    inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode()
current_signature = "sha256:" + hashlib.sha256(payload).hexdigest()

for split in ("train", "test"):
    manifest = manifests[split]
    explicit = manifest.get("model_source_signature")
    provenance = manifest.get("provenance")
    if explicit is None and isinstance(provenance, dict):
        explicit = provenance.get("model_source_signature")
    if explicit is not None:
        explicit = str(explicit)
        if len(explicit) == 64:
            explicit = "sha256:" + explicit
        if explicit != current_signature:
            raise SystemExit(
                f"{split} attention cache model signature disagrees with MODEL_PATH"
            )
        evidence = "explicit_model_source_signature"
    else:
        spec = manifest.get("attention_cache_spec")
        declared = spec.get("model_files_sha256") if isinstance(spec, dict) else None
        if not isinstance(declared, dict) or not declared:
            raise SystemExit(
                f"{split} attention cache has no model-byte provenance; "
                "re-extract it with run_ragtruth_extract_validate.sh"
            )
        if {str(key): str(value) for key, value in declared.items()} != current_hashes:
            raise SystemExit(
                f"{split} legacy attention-cache model files differ from MODEL_PATH"
            )
        evidence = "legacy_model_files_sha256_exact_inventory"
    print(f"cache_model_identity_{split}=verified evidence={evidence}")
PY

CACHE_DTYPE="$("${PYTHON_BIN}" - \
  "${ATTENTION_CACHE_ROOT}/train/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

from counterfactual_grounding.observer_runtime import (
    observer_runtime_from_cache_manifest,
)

manifest = json.loads(Path(sys.argv[1]).read_text())
runtime, _ = observer_runtime_from_cache_manifest(manifest)
print(runtime["dtype"])
PY
)"
DTYPE="${DTYPE:-${CACHE_DTYPE}}"
if [[ "${DTYPE}" != "${CACHE_DTYPE}" ]]; then
  printf 'Requested DTYPE=%s disagrees with attention-cache dtype=%s.\n' \
    "${DTYPE}" "${CACHE_DTYPE}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"

# Materialize the exact label-free teacher sampling frame from the already
# validated train-cache filenames.  This duplicates the upstream extractor's
# deterministic filename mapping without loading multi-GB attention tensors.
"${PYTHON_BIN}" - \
  "${ATTENTION_CACHE_ROOT}" \
  "${DATASET_DIR}/response.jsonl" \
  "${DATASET_DIR}/source_info.jsonl" \
  "${ELIGIBLE_RESPONSE_IDS}" <<'PY'
# CEPT_ELIGIBLE_FRAME_PYTHON
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

cache_root = Path(sys.argv[1]).resolve()
responses_path = Path(sys.argv[2]).resolve()
sources_path = Path(sys.argv[3]).resolve()
output_path = Path(sys.argv[4]).resolve()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def safe_sample_name(response_id):
    value = str(response_id)
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    return safe or "unknown"


def normalized_name(value):
    return "".join(character for character in str(value).casefold() if character.isalnum())


dataset_hashes = {
    "response.jsonl": file_sha256(responses_path),
    "source_info.jsonl": file_sha256(sources_path),
}
contracts = {}
manifests = {}
for split in ("train", "test"):
    manifest_path = cache_root / split / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest.get("attention_cache_spec")
    if not isinstance(spec, dict):
        raise SystemExit(f"{split} cache has no attention_cache_spec")
    declared_hashes = spec.get("dataset_files_sha256")
    if declared_hashes != dataset_hashes:
        raise SystemExit(
            f"{split} cache dataset hashes disagree with current RAGTruth files; "
            "re-extract before running the intervention teacher"
        )
    generator_selector = spec.get("generator_model")
    task_selector = spec.get("task_type")
    if not isinstance(generator_selector, str) or not generator_selector.strip():
        raise SystemExit(f"{split} cache has no generator_model selector")
    if not isinstance(task_selector, str) or not task_selector.strip():
        raise SystemExit(f"{split} cache has no task_type selector")
    contracts[split] = {
        "dataset_files_sha256": declared_hashes,
        "generator_model_selector": generator_selector,
        "task_type_selector": task_selector,
    }
    manifests[split] = (manifest_path, manifest)
if contracts["train"] != contracts["test"]:
    raise SystemExit("train/test cache dataset and generator/task selectors disagree")

source_rows = read_jsonl(sources_path)
sources = {str(row.get("source_id", "")): row for row in source_rows}
if not sources or len(sources) != len(source_rows) or "" in sources:
    raise SystemExit("source_info.jsonl has missing or duplicate source_id values")
response_rows = read_jsonl(responses_path)
response_ids = [str(row.get("id", "")) for row in response_rows]
if "" in response_ids or len(response_ids) != len(set(response_ids)):
    raise SystemExit("response.jsonl has missing or duplicate response IDs")

by_cache_file = {}
for row in response_rows:
    if str(row.get("split", "")).casefold() != "train":
        continue
    response_id = str(row["id"])
    filename = f"attention_{safe_sample_name(response_id)}.pt"
    by_cache_file.setdefault(filename, []).append(row)

train_manifest_path, train_manifest = manifests["train"]
observed = sorted(path.name for path in (cache_root / "train").glob("attention_*.pt"))
eligible = []
generator_selector = contracts["train"]["generator_model_selector"]
task_selector = contracts["train"]["task_type_selector"]
for filename in observed:
    candidates = by_cache_file.get(filename, [])
    if len(candidates) != 1:
        raise SystemExit(
            f"cache filename must map to exactly one train response: {filename}; "
            f"matches={len(candidates)}"
        )
    row = candidates[0]
    source_id = str(row.get("source_id", ""))
    source = sources.get(source_id)
    if source is None:
        raise SystemExit(f"cached response references absent source_id: {source_id}")
    if normalized_name(row.get("model", "")) != normalized_name(generator_selector):
        raise SystemExit(f"cached response violates generator selector: {row['id']}")
    task_type = str(source.get("task_type", ""))
    if task_selector.casefold() != "all" and task_type.casefold() != task_selector.casefold():
        raise SystemExit(f"cached response violates task selector: {row['id']}")
    eligible.append(
        {
            "schema": "cept-eligible-response-frame-v1",
            "record_type": "response",
            "response_id": str(row["id"]),
            "source_id": source_id,
            "official_split": "train",
            "generator_model": str(row.get("model", "unknown")),
            "task_type": task_type,
            "cache_file": filename,
        }
    )
if not eligible:
    raise SystemExit("verified train cache produced an empty eligible-response frame")
eligible.sort(key=lambda row: row["response_id"])
metadata = {
    "schema": "cept-eligible-response-frame-v1",
    "record_type": "metadata",
    "official_split": "train",
    "origin": "verified_train_attention_cache_filename_inventory",
    "eligible_response_count": len(eligible),
    "dataset_files_sha256": dataset_hashes,
    "generator_model_selector": generator_selector,
    "task_type_selector": task_selector,
    "cache_manifest_path": str(train_manifest_path),
    "cache_manifest_sha256": file_sha256(train_manifest_path),
}
content = "".join(
    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    for row in [metadata, *eligible]
)
if output_path.exists():
    if output_path.read_text(encoding="utf-8") != content:
        raise SystemExit(
            "existing eligible-response frame differs from the verified cache; "
            "choose a new RUN_ROOT"
        )
else:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
print(
    f"eligible_sampling_frame=verified count={len(eligible)} "
    f"generator={generator_selector} task={task_selector} path={output_path}"
)
PY

printf 'method=CEPT intervention-calibrated provenance transport\n'
printf 'student_input=attention_only teacher_signal=kv_intervention_plus_logprob\n'
printf 'run_root=%s graph_store=%s\n' "${RUN_ROOT}" "${GRAPH_STORE}"
printf 'teacher_samples=%s history_block=%s max_blocks=%s epochs=%s\n' \
  "${TEACHER_SAMPLES}" "${HISTORY_BLOCK_SIZE}" "${MAX_HISTORY_BLOCKS}" "${EPOCHS}"

TEACHER_COMPLETE=0
if [[ -f "${TEACHER_DIR}/manifest.json" ]] && \
   [[ -f "${TEACHER_DIR}/transport_teacher.jsonl" ]]; then
  if "${PYTHON_BIN}" - "${TEACHER_DIR}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if manifest.get("run_state") == "complete" else 1)
PY
  then
    TEACHER_COMPLETE=1
  fi
fi

if [[ "${TEACHER_COMPLETE}" == "0" ]]; then
  TEACHER_RESUME=0
  if [[ -f "${TEACHER_DIR}/manifest.json" ]]; then
    TEACHER_RESUME=1
  fi
  OUTPUT_DIR="${TEACHER_DIR}" \
  ELIGIBLE_RESPONSE_IDS="${ELIGIBLE_RESPONSE_IDS}" \
  NUM_SAMPLES="${TEACHER_SAMPLES}" \
  HISTORY_BLOCK_SIZE="${HISTORY_BLOCK_SIZE}" \
  MAX_HISTORY_BLOCKS="${MAX_HISTORY_BLOCKS}" \
  SEED="${SEED}" GPU_ID="${GPU_ID}" DTYPE="${DTYPE}" \
  RESUME="${TEACHER_RESUME}" \
  PYTHON_BIN="${PYTHON_BIN}" CEPT_VENV="${CEPT_VENV}" \
    bash ./run_ragtruth_cept_pilot.sh
else
  printf 'Reusing structured intervention teacher: %s\n' \
    "${TEACHER_DIR}/transport_teacher.jsonl"
fi

CEPT_PYTHON="${CEPT_VENV}/bin/python"
if [[ ! -x "${CEPT_PYTHON}" ]]; then
  printf 'CEPT Python was not created: %s\n' "${CEPT_PYTHON}" >&2
  exit 1
fi

MODEL_SOURCE_SIGNATURE="$("${CEPT_PYTHON}" - \
  "${TEACHER_DIR}/transport_teacher.jsonl" "${MODEL_PATH}" "${DTYPE}" <<'PY'
import json
import sys
from pathlib import Path

import torch
import transformers

from counterfactual_grounding.artifacts import source_inventory_signature
from counterfactual_grounding.observer_runtime import (
    observer_runtime_identity,
    parse_observer_runtime_identity,
)

path = Path(sys.argv[1])
first = json.loads(next(line for line in path.read_text().splitlines() if line.strip()))
teacher_runtime, teacher_runtime_signature = parse_observer_runtime_identity(
    first.get("observer_runtime"), first.get("observer_runtime_signature")
)
expected_runtime, expected_signature = observer_runtime_identity(
    transformers_version=transformers.__version__,
    torch_version=torch.__version__,
    dtype=sys.argv[3],
    attn_implementation="eager",
)
if teacher_runtime != expected_runtime or teacher_runtime_signature != expected_signature:
    raise SystemExit("reused teacher observer runtime disagrees with current runtime")
teacher_signature = first["model_source_signature"]
actual_signature = "sha256:" + source_inventory_signature(Path(sys.argv[2]))
if teacher_signature != actual_signature:
    raise SystemExit(
        "reused teacher model signature disagrees with current MODEL_PATH bytes"
    )
print(actual_signature)
PY
)"

for split in train test; do
  if [[ ! -f "${GRAPH_STORE}/${split}.index.jsonl" ]]; then
    "${CEPT_PYTHON}" -u -m counterfactual_grounding.cli build-graphs \
      --cache-root "${ATTENTION_CACHE_ROOT}" \
      --responses "${DATASET_DIR}/response.jsonl" \
      --sources "${DATASET_DIR}/source_info.jsonl" \
      --tokenizer "${MODEL_PATH}" \
      --model-source-signature "${MODEL_SOURCE_SIGNATURE}" \
      --output-dir "${GRAPH_STORE}" \
      --split "${split}"
  else
    printf 'Reusing canonical %s graph index: %s\n' \
      "${split}" "${GRAPH_STORE}/${split}.index.jsonl"
  fi
done

"${CEPT_PYTHON}" - \
  "${TEACHER_DIR}" "${GRAPH_STORE}" "${ATTENTION_CACHE_ROOT}" \
  "${DATASET_DIR}/response.jsonl" "${DATASET_DIR}/source_info.jsonl" \
  "${MODEL_PATH}" "${SEED}" "${TEACHER_SAMPLES}" \
  "${HISTORY_BLOCK_SIZE}" "${MAX_HISTORY_BLOCKS}" \
  "${ELIGIBLE_RESPONSE_IDS}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch
import transformers

from counterfactual_grounding.artifacts import file_sha256
from counterfactual_grounding.observer_runtime import (
    observer_runtime_identity,
    parse_observer_runtime_identity,
)

teacher = Path(sys.argv[1]).resolve()
graphs = Path(sys.argv[2]).resolve()
cache_root = Path(sys.argv[3]).resolve()
responses = Path(sys.argv[4]).resolve()
sources = Path(sys.argv[5]).resolve()
model = Path(sys.argv[6]).resolve()
seed = int(sys.argv[7])
teacher_samples = int(sys.argv[8])
history_block_size = int(sys.argv[9])
max_history_blocks = int(sys.argv[10])
eligible_response_ids = Path(sys.argv[11]).resolve()
manifest = json.loads((teacher / "manifest.json").read_text())
report = json.loads((teacher / "gate_report.json").read_text())
if manifest.get("run_state") != "complete":
    raise SystemExit("teacher manifest is not complete")
run_contract = manifest.get("run_contract", {})
contract_config = run_contract.get("config", {})
expected_contract = {
    "responses": str(responses),
    "sources": str(sources),
    "model": str(model),
    "eligible_response_ids": str(eligible_response_ids),
    "split": "train",
    "num_samples": teacher_samples,
    "seed": seed,
    "history_block_size": history_block_size,
    "max_history_blocks": max_history_blocks,
}
for field, expected in expected_contract.items():
    if contract_config.get(field) != expected:
        raise SystemExit(
            f"reused teacher contract mismatch for {field}: "
            f"stored={contract_config.get(field)!r} requested={expected!r}"
        )
expected_dataset = {
    "responses_sha256": file_sha256(responses),
    "sources_sha256": file_sha256(sources),
}
for field, expected in expected_dataset.items():
    if run_contract.get("dataset", {}).get(field) != expected:
        raise SystemExit(f"reused teacher dataset hash mismatch for {field}")
eligible_rows = [
    json.loads(line)
    for line in eligible_response_ids.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not eligible_rows or eligible_rows[0].get("record_type") != "metadata":
    raise SystemExit("eligible-response frame metadata is absent")
eligible_count = sum(row.get("record_type") == "response" for row in eligible_rows)
expected_sampling_frame = {
    "path": str(eligible_response_ids),
    "sha256": file_sha256(eligible_response_ids),
    "count": eligible_count,
}
if run_contract.get("sampling_frame") != expected_sampling_frame:
    raise SystemExit("reused teacher eligible-response frame contract mismatch")
estimand = run_contract.get("estimand_scope", {})
for field in ("generator_model_selector", "task_type_selector"):
    if estimand.get(field) != eligible_rows[0].get(field):
        raise SystemExit(f"reused teacher estimand scope mismatch for {field}")
expected_environment = {
    "transformers_version": transformers.__version__,
    "torch_version": torch.__version__,
}
for field, expected in expected_environment.items():
    if run_contract.get("environment", {}).get(field) != expected:
        raise SystemExit(f"reused teacher runtime mismatch for {field}")
package_root = Path.cwd() / "counterfactual_grounding"
current_implementation = {
    relative: file_sha256(package_root / relative)
    for relative in (
        "experiment.py",
        "teacher/counterfactuals.py",
        "teacher/mediation.py",
        "teacher/pilot.py",
        "observer_runtime.py",
        "transport/data.py",
    )
}
if run_contract.get("implementation_sha256") != current_implementation:
    raise SystemExit("reused teacher was produced by different CEPT source code")
gate1 = report.get("gate1", {})
required = (
    "self_patch_pass",
    "decomposition_pass",
    "first_token_causal_visibility_pass",
)
if not all(gate1.get(name) is True for name in required):
    raise SystemExit("teacher failed a core Gate-1 intervention contract")
artifact = manifest.get("artifacts", {}).get("transport_teacher", {})
teacher_path = teacher / str(artifact.get("path", ""))
if not teacher_path.is_file():
    raise SystemExit("structured transport teacher artifact is absent")
digest = hashlib.sha256(teacher_path.read_bytes()).hexdigest()
if digest != artifact.get("sha256"):
    raise SystemExit("structured transport teacher artifact hash mismatch")
first = json.loads(next(line for line in teacher_path.read_text().splitlines() if line.strip()))
teacher_runtime, teacher_runtime_signature = parse_observer_runtime_identity(
    first.get("observer_runtime"), first.get("observer_runtime_signature")
)
expected_teacher_runtime, expected_teacher_runtime_signature = observer_runtime_identity(
    transformers_version=transformers.__version__,
    torch_version=torch.__version__,
    dtype=contract_config.get("dtype"),
    attn_implementation="eager",
)
if (
    teacher_runtime != expected_teacher_runtime
    or teacher_runtime_signature != expected_teacher_runtime_signature
):
    raise SystemExit("teacher row observer runtime disagrees with Gate-1 runtime/config")
expected_model_signature = first.get("model_source_signature")
if not isinstance(expected_model_signature, str) or not expected_model_signature:
    raise SystemExit("teacher has no model-source signature")
if run_contract.get("model_source_signature") != expected_model_signature:
    raise SystemExit("teacher row and run contract model signatures disagree")
if not first.get("block_definitions"):
    raise SystemExit("teacher has no response-history block rescue targets")
if "direct_seed_rescue" not in first["events"][0]:
    raise SystemExit("teacher predates the graph-aligned direct-seed rescue protocol")
if "joint_seed_history_rescue" not in first["events"][0]:
    raise SystemExit("teacher predates the joint seed/history rescue protocol")
for split in ("train", "test"):
    graph_manifest = json.loads((graphs / f"{split}.manifest.json").read_text())
    if graph_manifest.get("schema") != "cept-canonical-graph-store-v2":
        raise SystemExit(f"canonical {split} graph-store schema is stale")
    if graph_manifest.get("graph_inventory_complete") is not True:
        raise SystemExit(f"canonical {split} graph inventory is partial")
    if graph_manifest.get("graph_schema") != "cept-prediction-event-graph-v2":
        raise SystemExit(f"canonical {split} graph schema is stale")
    if graph_manifest.get("deployment_seed_protocol") != (
        "numeric_digit_surface_preserving_v1_first_candidate"
    ):
        raise SystemExit(f"canonical {split} deployment seed protocol is stale")
    expected_manifest = {
        "cache_root": str(cache_root),
        "responses_sha256": file_sha256(responses),
        "sources_sha256": file_sha256(sources),
        "model_source_signature": expected_model_signature,
        "observer_runtime": teacher_runtime,
        "observer_runtime_signature": teacher_runtime_signature,
        "cache_content_identity_status": (
            "verified_extractor_cache_files_sha256"
        ),
        "cache_content_identity_evidence": "cache_files_sha256_exact_bytes",
    }
    for field, expected in expected_manifest.items():
        if graph_manifest.get(field) != expected:
            raise SystemExit(
                f"canonical {split} graph store has stale {field}; "
                "choose a new GRAPH_STORE"
            )
    first_index = json.loads(
        next(
            line
            for line in (graphs / f"{split}.index.jsonl").read_text().splitlines()
            if line.strip()
        )
    )
    content_inventory_signature = graph_manifest.get(
        "cache_content_inventory_signature"
    )
    if (
        not isinstance(content_inventory_signature, str)
        or not content_inventory_signature.startswith("sha256:")
        or len(content_inventory_signature) != 71
        or first_index.get("cache_content_inventory_signature")
        != content_inventory_signature
    ):
        raise SystemExit(
            f"canonical {split} graph/index cache-content inventory binding is "
            "missing or inconsistent; choose a new GRAPH_STORE"
        )
    if not first_index.get("model_source_signature"):
        raise SystemExit(
            f"canonical {split} index predates the model-source identity contract; "
            "choose a new GRAPH_STORE"
        )
    if not first_index.get("cache_model_identity_status"):
        raise SystemExit(
            f"canonical {split} index predates cache-model provenance tracking; "
            "choose a new GRAPH_STORE"
        )
    if (
        first_index.get("observer_runtime") != teacher_runtime
        or first_index.get("observer_runtime_signature")
        != teacher_runtime_signature
    ):
        raise SystemExit(
            f"canonical {split} index observer runtime disagrees with teacher; "
            "choose a new GRAPH_STORE"
        )
    if (
        first_index.get("cache_content_identity_status")
        != "verified_extractor_cache_files_sha256"
        or first_index.get("cache_content_identity_evidence")
        != "cache_files_sha256_exact_bytes"
        or first_index.get("extractor_declared_cache_sha256")
        != first_index.get("cache_sha256")
    ):
        raise SystemExit(
            f"canonical {split} index predates extractor cache-content binding; "
            "choose a new GRAPH_STORE"
        )
    if "deployment_seed_positions" not in first_index:
        raise SystemExit(
            f"canonical {split} index predates deployment-seed alignment; "
            "choose a new GRAPH_STORE"
        )
    if first_index["cache_model_identity_status"] not in {
        "verified_extractor_manifest",
        "verified_legacy_content_manifest",
    }:
        raise SystemExit(
            f"canonical {split} cache-model identity is unverified; choose a "
            "new GRAPH_STORE so byte-level provenance is rebuilt"
        )
print("preflight=teacher_and_canonical_graph_contracts_passed")
PY

"${CEPT_PYTHON}" -u -m counterfactual_grounding.cli train-transport \
  --graph-index "${GRAPH_STORE}/train.index.jsonl" \
  --teacher "${TEACHER_DIR}/transport_teacher.jsonl" \
  --score-index "${GRAPH_STORE}/test.index.jsonl" \
  --output-dir "${TRANSPORT_DIR}" \
  --device cuda:0 \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --variants true rewired mass_only one_hop no_residual

"${CEPT_PYTHON}" - "${TRANSPORT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
fidelity = json.loads((root / "teacher_fidelity.json").read_text())
gate2 = fidelity.get("gate2", {})
print("gate2=", gate2)
if gate2.get("claim_supported") is not True:
    raise SystemExit(
        "Gate 2 failed: true source-event incidence did not beat every "
        "pre-registered structural/residual/zero control or the teacher "
        "population was not identifiable. Frozen predictions were retained, "
        "but RAGTruth test labels were not opened."
    )
print("gate2=passed; opening labels only for frozen post-hoc evaluation")
PY

"${CEPT_PYTHON}" -u -m counterfactual_grounding.cli evaluate-transport \
  --predictions-dir "${TRANSPORT_DIR}" \
  --test-graph-index "${GRAPH_STORE}/test.index.jsonl" \
  --output "${TRANSPORT_DIR}/evaluation.json"

"${CEPT_PYTHON}" - "${TRANSPORT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
fidelity = json.loads((root / "teacher_fidelity.json").read_text())
evaluation = json.loads((root / "evaluation.json").read_text())
print("gate2=", fidelity["gate2"])
print("token=", evaluation["token"])
print("response=", evaluation["response"])
print("detection_claim=", evaluation["detection_claim"])
print("output=", root)
PY
