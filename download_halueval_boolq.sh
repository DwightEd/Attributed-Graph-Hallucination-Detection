#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DATA_ROOT="${DATA_ROOT:-/share/home/tm902089733300000/a903202310/lys/data}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
OFFLINE_ONLY="${OFFLINE_ONLY:-0}"

export DATA_ROOT FORCE_DOWNLOAD OFFLINE_ONLY

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DATA_ROOT = Path(os.environ["DATA_ROOT"]).expanduser().resolve()
FORCE = os.environ.get("FORCE_DOWNLOAD", "0") == "1"
OFFLINE = os.environ.get("OFFLINE_ONLY", "0") == "1"
if FORCE and OFFLINE:
    raise SystemExit("FORCE_DOWNLOAD=1 and OFFLINE_ONLY=1 are incompatible")
HALU_REVISION = "b7253db3cdaa0ab2c382f92b26b390109174f77e"
BOOLQ_REPOSITORY_REVISION = "90af34107399cc7a446b373dc4ee35b8001da7c2"

SOURCES = {
    "halueval_qa": {
        "path": DATA_ROOT / "HaluEval" / "qa_data.json",
        "url": (
            "https://raw.githubusercontent.com/RUCAIBox/HaluEval/"
            f"{HALU_REVISION}/data/qa_data.json"
        ),
        "expected_rows": 10000,
        "license": "MIT",
        "source_revision": HALU_REVISION,
    },
    "boolq_train": {
        "path": DATA_ROOT / "BoolQ" / "train.jsonl",
        "url": "https://storage.googleapis.com/boolq/train.jsonl",
        "expected_rows": 9427,
        "license": "CC BY-SA 3.0",
        "source_revision": None,
        "reference_repository_revision": BOOLQ_REPOSITORY_REVISION,
    },
    "boolq_dev": {
        "path": DATA_ROOT / "BoolQ" / "dev.jsonl",
        "url": "https://storage.googleapis.com/boolq/dev.jsonl",
        "expected_rows": 3270,
        "license": "CC BY-SA 3.0",
        "source_revision": None,
        "reference_repository_revision": BOOLQ_REPOSITORY_REVISION,
    },
}

# Literal counts are intentionally kept here so code review/tests can verify the
# exact full-dataset contract without importing this script.
EXPECTED_ROWS = {"halueval_qa": 10000, "boolq_train": 9427, "boolq_dev": 3270}
MANIFEST_PATH = DATA_ROOT / "hallucination_datasets" / "dataset_manifest.json"
if MANIFEST_PATH.is_file():
    previous_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
else:
    previous_manifest = {}


def download_to_temporary(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "dataset-audit/1.0"})
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
            if temporary.stat().st_size == 0:
                raise RuntimeError(f"Downloaded an empty file from {url}")
            return temporary
        except Exception as error:  # pragma: no cover - exercised on remote network
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def read_records(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Dataset is empty: {path}")
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON array: {path}")
        return [dict(row) for row in value]
    return [dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def validate(name: str, path: Path) -> int:
    records = read_records(path)
    expected = EXPECTED_ROWS[name]
    if len(records) != expected:
        raise ValueError(f"{name}: expected {expected} rows, found {len(records)}")
    if name == "halueval_qa":
        required = {"knowledge", "question", "right_answer", "hallucinated_answer"}
        for index, row in enumerate(records):
            if not required.issubset(row):
                raise ValueError(f"{name} row {index} lacks {required - set(row)}")
            if not all(isinstance(row[key], str) and row[key].strip() for key in required):
                raise ValueError(f"{name} row {index} contains an empty/non-string field")
            if row["right_answer"] == row["hallucinated_answer"]:
                raise ValueError(f"{name} row {index} has identical candidates")
    else:
        required = {"passage", "question", "answer"}
        for index, row in enumerate(records):
            if not required.issubset(row):
                raise ValueError(f"{name} row {index} lacks {required - set(row)}")
            if not isinstance(row["passage"], str) or not row["passage"].strip():
                raise ValueError(f"{name} row {index} has an invalid passage")
            if not isinstance(row["question"], str) or not row["question"].strip():
                raise ValueError(f"{name} row {index} has an invalid question")
            if not isinstance(row["answer"], bool):
                raise ValueError(f"{name} row {index} answer is not boolean")
    return len(records)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifest_entries = {}
for source_name, source in SOURCES.items():
    destination = source["path"]
    downloaded_temporary = None
    if FORCE or not destination.is_file():
        if OFFLINE:
            raise FileNotFoundError(f"Offline validation cannot find {destination}")
        print(f"Downloading {source_name} -> {destination}", flush=True)
        downloaded_temporary = download_to_temporary(str(source["url"]), destination)
        candidate = downloaded_temporary
    else:
        print(f"Reusing {source_name} -> {destination}", flush=True)
        candidate = destination
    try:
        rows = validate(source_name, candidate)
        content_sha256 = sha256(candidate)
        previous_entry = previous_manifest.get("datasets", {}).get(source_name, {})
        previous_sha256 = previous_entry.get("sha256")
        if previous_sha256 and content_sha256 != previous_sha256 and not FORCE:
            raise RuntimeError(
                f"{source_name} no longer matches the recorded SHA256 "
                f"{previous_sha256}; inspect the file and use FORCE_DOWNLOAD=1 "
                "only for an intentional update"
            )
        if downloaded_temporary is not None:
            os.replace(downloaded_temporary, destination)
    except Exception:
        if downloaded_temporary is not None:
            downloaded_temporary.unlink(missing_ok=True)
        raise
    manifest_entries[source_name] = {
        "path": str(destination),
        "source_url": source["url"],
        "source_revision": source["source_revision"],
        "reference_repository_revision": source.get("reference_repository_revision"),
        "license": source["license"],
        "rows": rows,
        "bytes": destination.stat().st_size,
        "sha256": content_sha256,
    }

manifest = {
    "schema_version": 1,
    "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    "data_root": str(DATA_ROOT),
    "datasets": manifest_entries,
}
manifest_path = MANIFEST_PATH
manifest_path.parent.mkdir(parents=True, exist_ok=True)
temporary_manifest = manifest_path.with_name(
    f".{manifest_path.name}.{os.getpid()}.tmp"
)
temporary_manifest.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary_manifest, manifest_path)
print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
print(f"DATASET_MANIFEST={manifest_path}", flush=True)
PY
