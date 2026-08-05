"""Minimal evaluation-only index for prepared HaluEval attention graphs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from attention_graph.halueval import load_halueval_response_labels

from .artifacts import atomic_json

LABELED_GRAPH_INDEX_FIELDS = frozenset(
    {"response_id", "pair_id", "split", "label", "graph_path"}
)
_UNLABELED_GRAPH_INDEX_FIELDS = frozenset(
    {"response_id", "pair_id", "split", "graph_path"}
)
_SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


def _nonempty(value: object, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"labeled graph index requires non-empty {name}")
    return result


def build_labeled_graph_index(
    rows: Sequence[Mapping[str, object]], labels: Mapping[str, int]
) -> list[dict[str, object]]:
    """Join graph identity to evaluation labels without retaining audit fields."""

    if not rows:
        raise ValueError("labeled graph index requires at least one graph")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for position, row in enumerate(rows):
        if set(row) != _UNLABELED_GRAPH_INDEX_FIELDS:
            raise ValueError(
                f"unlabeled graph index row {position} must contain exactly "
                + ", ".join(sorted(_UNLABELED_GRAPH_INDEX_FIELDS))
            )
        response_id = _nonempty(row["response_id"], "response_id")
        if response_id in seen:
            raise ValueError(f"duplicate graph response_id: {response_id}")
        seen.add(response_id)
        split = _nonempty(row["split"], "split").casefold()
        if split not in _SPLIT_ORDER:
            raise ValueError(f"unsupported graph split: {split}")
        if response_id not in labels:
            raise ValueError(f"evaluation label is absent for graph: {response_id}")
        label = labels[response_id]
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError(f"graph label must be binary for: {response_id}")
        result.append(
            {
                "response_id": response_id,
                "pair_id": _nonempty(row["pair_id"], "pair_id"),
                "split": split,
                "label": int(label),
                "graph_path": _nonempty(row["graph_path"], "graph_path"),
            }
        )
    result.sort(
        key=lambda row: (_SPLIT_ORDER[str(row["split"])], str(row["response_id"]))
    )
    return result


def write_labeled_graph_index(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    labels: Mapping[str, int],
) -> list[dict[str, object]]:
    """Atomically write the five-field user-facing graph index."""

    index = build_labeled_graph_index(rows, labels)
    atomic_json(Path(path).expanduser().resolve(), index)
    return index


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def _is_artifact_index(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(row, Mapping)
            and "response_id" in row
            and "graph_path" in row
            and "dataset_split" in row
            and "num_nodes" in row
            and "label" not in row
            for row in value
        )
    )


def separate_artifact_index(graph_dir: str | Path) -> Path:
    """Move the builder's technical index away from the dataset index name."""

    directory = Path(graph_dir).expanduser().resolve()
    source = directory / "index.json"
    destination = directory / "artifact_index.json"
    if not source.is_file():
        if destination.is_file():
            return destination
        raise FileNotFoundError(f"prepared graph artifact index is absent: {source}")
    value = _read_json(source)
    if not _is_artifact_index(value):
        if (
            destination.is_file()
            and isinstance(value, list)
            and all(
                isinstance(row, Mapping) and set(row) == LABELED_GRAPH_INDEX_FIELDS
                for row in value
            )
        ):
            return destination
        raise ValueError(
            f"prepared graph technical index has unexpected fields: {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    return destination


def _require_frozen_scores(run_dir: Path) -> None:
    freeze_path = run_dir / "score_freeze.json"
    value = _read_json(freeze_path) if freeze_path.is_file() else None
    if (
        not isinstance(value, Mapping)
        or value.get("state") != "scores_frozen_before_label_read"
    ):
        raise ValueError("scores must be frozen before exporting a labeled graph index")


def _labels_path(run_dir: Path, explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    run_path = run_dir / "run.json"
    value = _read_json(run_path) if run_path.is_file() else None
    provenance = value.get("provenance") if isinstance(value, Mapping) else None
    raw = (
        provenance.get("evaluation_label_sidecar")
        if isinstance(provenance, Mapping)
        else None
    )
    if not raw:
        raise ValueError("run.json does not identify an evaluation label sidecar")
    return Path(str(raw)).expanduser().resolve()


def _rows_from_splits(run_dir: Path) -> list[dict[str, object]]:
    value = _read_json(run_dir / "splits.json")
    partitions = value.get("partitions") if isinstance(value, Mapping) else None
    if not isinstance(partitions, Mapping):
        raise TypeError("splits.json lacks a partitions mapping")
    rows: list[dict[str, object]] = []
    for split in _SPLIT_ORDER:
        records = partitions.get(split)
        if not isinstance(records, list):
            raise TypeError(f"splits.json lacks the {split} graph records")
        for position, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"{split} split row {position} must be an object")
            declared = str(record.get("partition", split)).strip().casefold()
            if declared != split:
                raise ValueError(
                    f"splits.json partition disagrees for {split} row {position}"
                )
            graph_path = Path(_nonempty(record.get("graph_path"), "graph_path"))
            if not graph_path.is_file():
                raise FileNotFoundError(f"prepared graph is absent: {graph_path}")
            rows.append(
                {
                    "response_id": _nonempty(record.get("response_id"), "response_id"),
                    "pair_id": _nonempty(record.get("pair_id"), "pair_id"),
                    "split": split,
                    "graph_path": str(graph_path.resolve()),
                }
            )
    return rows


def export_completed_run_graph_index(
    run_dir: str | Path, *, labels_path: str | Path | None = None
) -> dict[str, object]:
    """Create a minimal labeled index for a completed, score-frozen run."""

    root = Path(run_dir).expanduser().resolve()
    _require_frozen_scores(root)
    graph_dir = root / "prepared" / "graphs"
    artifact_index = separate_artifact_index(graph_dir)
    rows = _rows_from_splits(root)
    labels_file = _labels_path(root, labels_path)
    labels = load_halueval_response_labels(labels_file)
    destination = graph_dir / "index.json"
    index = write_labeled_graph_index(destination, rows, labels)
    return {
        "index": str(destination),
        "artifact_index": str(artifact_index),
        "samples": len(index),
        "split_counts": dict(Counter(str(row["split"]) for row in index)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a minimal evaluation-only HaluEval graph index"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_completed_run_graph_index(args.run_dir, labels_path=args.labels)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LABELED_GRAPH_INDEX_FIELDS",
    "build_labeled_graph_index",
    "export_completed_run_graph_index",
    "main",
    "separate_artifact_index",
    "write_labeled_graph_index",
]
