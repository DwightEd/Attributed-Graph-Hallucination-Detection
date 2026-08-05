"""Atomic, JSON-only CEPT pilot artifacts and provenance hashes."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch


def jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [jsonable(item) for item in value]
    return value


def atomic_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(
                jsonable(value),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_jsonl(path: str | Path, records: Sequence[Mapping[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        jsonable(record),
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_inventory_signature(path: str | Path) -> str:
    """Hash every file that defines a local model/tokenizer source.

    Weight filenames and sizes are not a model identity: two checkpoints can
    have the same inventory while containing different values.  Gate artifacts
    therefore pay the one-time sequential-read cost and bind themselves to the
    actual checkpoint bytes.
    """

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"model/tokenizer source is absent: {source}")
    if source.is_file():
        return file_sha256(source)
    inventory: list[dict[str, object]] = []
    for item in sorted(source.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(source).as_posix()
        row: dict[str, object] = {
            "path": relative,
            "size": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        inventory.append(row)
    payload = json.dumps(
        inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "atomic_json",
    "atomic_jsonl",
    "canonical_hash",
    "file_sha256",
    "jsonable",
    "source_inventory_signature",
]
