"""Read the five-field public index and load its prepared attention graphs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from .data import load_graph
from .graph import AttentionGraph

INDEX_FIELDS = {"response_id", "pair_id", "split", "label", "graph_path"}
SPLITS = {"train", "validation", "test"}


def read_index(
    index_path: str | Path, *, split: str | None = None
) -> list[dict[str, object]]:
    """Read prepared/graphs/index.json, optionally keeping one logical split."""

    path = Path(index_path).expanduser().resolve()
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}: {split}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read graph index: {path}") from error
    if not isinstance(value, list):
        raise TypeError(f"graph index must be a JSON list: {path}")

    records: list[dict[str, object]] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != INDEX_FIELDS:
            raise ValueError(
                f"graph index row {position} must contain exactly: "
                + ", ".join(sorted(INDEX_FIELDS))
            )
        row_split = str(raw["split"])
        label = raw["label"]
        if row_split not in SPLITS:
            raise ValueError(f"graph index row {position} has invalid split: {row_split}")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError(f"graph index row {position} has non-binary label: {label}")
        graph_path = Path(str(raw["graph_path"])).expanduser()
        if not graph_path.is_absolute():
            graph_path = path.parent / graph_path
        record = dict(raw)
        record["graph_path"] = graph_path.resolve()
        if split is None or row_split == split:
            records.append(record)
    return records


def load_indexed_graph(
    record: Mapping[str, object],
    *,
    device: str | torch.device = "cpu",
    mmap: bool = True,
) -> AttentionGraph:
    """Load one indexed graph through the existing strict artifact loader."""

    if "graph_path" not in record:
        raise ValueError("indexed graph record has no graph_path")
    return load_graph(Path(str(record["graph_path"])), device=device, mmap=mmap)


__all__ = ["INDEX_FIELDS", "load_indexed_graph", "read_index"]
