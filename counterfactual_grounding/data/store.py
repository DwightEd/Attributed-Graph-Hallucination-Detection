"""One-time canonical graph adaptation and atomic storage."""

from __future__ import annotations

import hashlib
import os
import struct
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from .graph import PredictionEventGraph
from .ragtruth import RagTruthTokenLayout

_CACHE_FIELDS = (
    "source_id",
    "response_id",
    "dataset_split",
    "attention_cache_fingerprint",
    "response_idx",
    "num_attention_layers",
    "num_attention_heads",
    "attention_floor",
    "token_ids",
    "attention_diagonal",
    "response_row_ptr",
    "response_column_indices",
    "response_values",
)


@dataclass(frozen=True)
class StoredPredictionGraph:
    path: Path
    sha256: str
    source_id: str
    response_id: str
    dataset_split: str
    available_events: int
    total_events: int


def adapt_legacy_cache_to_layout(
    cache_record: Mapping[str, object],
    layout: RagTruthTokenLayout,
    *,
    cache_sha256: str | None = None,
) -> dict[str, object]:
    """Attach audited E/Q/R segments through a strict cache-field whitelist."""

    missing = [field for field in _CACHE_FIELDS if field not in cache_record]
    if missing:
        raise ValueError("legacy attention cache is missing " + ", ".join(missing))
    cache_ids = torch.as_tensor(cache_record["token_ids"]).detach().cpu().long().flatten()
    if not torch.equal(cache_ids, layout.input_ids.detach().cpu().long().flatten()):
        raise RuntimeError(
            "RAGTruth tokenizer replay disagrees with attention-cache token_ids"
        )
    if int(cache_record["response_idx"]) != layout.response_idx:
        raise RuntimeError(
            "RAGTruth tokenizer replay disagrees with attention-cache response_idx"
        )
    # Labels and all unregistered diagnostics are intentionally absent even if
    # they were present in the source .pt mapping.
    adapted = {field: cache_record[field] for field in _CACHE_FIELDS}
    adapted["token_ids"] = cache_ids
    adapted["segment_ids"] = layout.segment_ids.detach().cpu().to(torch.int8)
    adapted["cache_dtype"] = str(
        cache_record.get(
            "cache_dtype",
            torch.as_tensor(cache_record["response_values"]).dtype,
        )
    ).removeprefix("torch.")
    adapted["attention_cache_sha256"] = (
        str(cache_record.get("attention_cache_sha256", ""))
        if cache_sha256 is None
        else cache_sha256
    )
    return adapted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_frame(
    update: Callable[[bytes | memoryview], None], value: bytes | memoryview
) -> None:
    update(len(value).to_bytes(8, byteorder="big", signed=False))
    update(value)


def _payload_content_sha256(payload: Mapping[str, object]) -> str:
    """Hash every persisted scalar and tensor without JSON-expanding tensors."""

    digest = hashlib.sha256()
    update = digest.update
    for key in sorted(payload):
        _update_frame(update, str(key).encode("utf-8"))
        value = payload[key]
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            _update_frame(update, b"tensor")
            _update_frame(update, str(tensor.dtype).encode("ascii"))
            _update_frame(
                update,
                b"".join(
                    int(size).to_bytes(8, byteorder="big", signed=True)
                    for size in tensor.shape
                ),
            )
            raw_bytes = memoryview(tensor.view(torch.uint8).numpy()).cast("B")
            _update_frame(update, raw_bytes)
        elif isinstance(value, str):
            _update_frame(update, b"str")
            _update_frame(update, value.encode("utf-8"))
        elif isinstance(value, bool):
            _update_frame(update, b"bool")
            _update_frame(update, b"1" if value else b"0")
        elif isinstance(value, int):
            _update_frame(update, b"int")
            _update_frame(update, str(value).encode("ascii"))
        elif isinstance(value, float):
            _update_frame(update, b"float")
            _update_frame(update, struct.pack(">d", value))
        else:
            raise TypeError(
                f"canonical graph field {key!r} has unsupported type {type(value)!r}"
            )
    return digest.hexdigest()


def _record(path: Path, graph: PredictionEventGraph) -> StoredPredictionGraph:
    return StoredPredictionGraph(
        path=path,
        sha256=_sha256(path),
        source_id=graph.source_id,
        response_id=graph.response_id,
        dataset_split=graph.dataset_split,
        available_events=int(graph.row_available.sum()),
        total_events=int(graph.row_available.numel()),
    )


def save_prediction_event_graph(
    graph: PredictionEventGraph, output_dir: str | Path
) -> StoredPredictionGraph:
    """Persist one reusable graph; repeated calls never create run copies."""

    if not graph.response_id:
        raise ValueError("canonical graphs require a non-empty response_id")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = graph.as_payload()
    content_sha256 = _payload_content_sha256(payload)
    filename = content_sha256 + ".graph.pt"
    destination = root / filename
    if destination.is_file():
        try:
            existing = torch.load(
                destination, map_location="cpu", weights_only=True, mmap=True
            )
        except TypeError:
            existing = torch.load(destination, map_location="cpu", weights_only=True)
        try:
            existing_sha256 = (
                _payload_content_sha256(existing)
                if isinstance(existing, Mapping)
                else None
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"invalid canonical graph already exists: {destination}"
            ) from error
        if existing_sha256 != content_sha256:
            raise RuntimeError(f"canonical graph identity collision: {destination}")
        return _record(destination, graph)

    temporary = root / f".{filename}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _record(destination, graph)


__all__ = [
    "StoredPredictionGraph",
    "adapt_legacy_cache_to_layout",
    "save_prediction_event_graph",
]
