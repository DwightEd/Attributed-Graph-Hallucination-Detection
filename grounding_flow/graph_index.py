"""Minimal evaluation-only index for prepared HaluEval attention graphs."""

from __future__ import annotations

import argparse
import json
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


class _LabelSidecarNotRecorded(ValueError):
    pass


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
    seen_graph_paths: set[str] = set()
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
        graph_path = _nonempty(row["graph_path"], "graph_path")
        if graph_path in seen_graph_paths:
            raise ValueError(f"duplicate graph_path: {graph_path}")
        seen_graph_paths.add(graph_path)
        result.append(
            {
                "response_id": response_id,
                "pair_id": _nonempty(row["pair_id"], "pair_id"),
                "split": split,
                "label": int(label),
                "graph_path": graph_path,
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
    """Atomically preserve the builder index under its technical name."""

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
    # Copy atomically instead of renaming first: until the caller atomically
    # replaces ``index.json``, an interruption still leaves the legacy index
    # intact and the export can be retried without repair.
    atomic_json(destination, value)
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
        raise _LabelSidecarNotRecorded(
            "run.json does not identify an evaluation label sidecar"
        )
    return Path(str(raw)).expanduser().resolve()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except _LabelSidecarNotRecorded:
        return False
    return True


def _load_artifact_rows(graph_dir: Path) -> tuple[Path, list[Mapping[str, object]]]:
    candidates = (graph_dir / "artifact_index.json", graph_dir / "index.json")
    for path in candidates:
        if not path.is_file():
            continue
        value = _read_json(path)
        if _is_artifact_index(value):
            return path, list(value)
    raise ValueError(
        "prepared graph artifact index is absent or invalid: "
        + ", ".join(str(path) for path in candidates)
    )


def _resolve_artifact_graph(
    graph_dir: Path, record: Mapping[str, object]
) -> Path:
    root = graph_dir.resolve()
    raw = Path(_nonempty(record.get("graph_path"), "graph_path")).expanduser()
    cache_split = _nonempty(record.get("dataset_split"), "dataset_split").casefold()
    if cache_split not in {"train", "test"}:
        raise ValueError(f"unsupported physical graph split: {cache_split}")
    local = root / cache_split / raw.name
    if local.is_file():
        resolved = local.resolve()
        if not _is_within(resolved, root):
            raise ValueError(f"prepared graph path escapes graph directory: {local}")
        return resolved
    stored = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if stored.is_file():
        if not _is_within(stored, root):
            raise ValueError(f"prepared graph path escapes graph directory: {stored}")
        return stored
    matches = sorted(
        candidate.resolve()
        for candidate in root.rglob(raw.name)
        if candidate.is_file() and _is_within(candidate.resolve(), root)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"multiple prepared graphs match artifact: {raw.name}")
    raise FileNotFoundError(f"prepared graph is absent: {raw}")


def _rows_from_splits(
    run_dir: Path,
    graph_dir: Path,
    artifact_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    artifacts: dict[str, tuple[Path, str | None]] = {}
    artifact_paths: set[Path] = set()
    for artifact in artifact_rows:
        response_id = _nonempty(artifact.get("response_id"), "response_id")
        if response_id in artifacts:
            raise ValueError(f"duplicate artifact response_id: {response_id}")
        graph_path = _resolve_artifact_graph(graph_dir, artifact)
        if graph_path in artifact_paths:
            raise ValueError(f"duplicate artifact graph_path: {graph_path}")
        artifact_paths.add(graph_path)
        source_id = str(artifact.get("source_id", "")).strip() or None
        artifacts[response_id] = (graph_path, source_id)

    value = _read_json(run_dir / "splits.json")
    partitions = value.get("partitions") if isinstance(value, Mapping) else None
    if not isinstance(partitions, Mapping):
        raise TypeError("splits.json lacks a partitions mapping")
    rows: list[dict[str, object]] = []
    seen_responses: set[str] = set()
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
            response_id = _nonempty(record.get("response_id"), "response_id")
            if response_id in seen_responses:
                raise ValueError(f"duplicate split response_id: {response_id}")
            seen_responses.add(response_id)
            if response_id not in artifacts:
                raise ValueError(
                    f"split response_id is absent from artifact index: {response_id}"
                )
            artifact_path, source_id = artifacts[response_id]
            split_path = Path(_nonempty(record.get("graph_path"), "graph_path"))
            if split_path.name != artifact_path.name:
                raise ValueError(
                    f"split and artifact graph_path disagree for: {response_id}"
                )
            pair_id = _nonempty(record.get("pair_id"), "pair_id")
            if source_id is not None and pair_id != source_id:
                raise ValueError(
                    f"split pair_id and artifact source_id disagree for: {response_id}"
                )
            rows.append(
                {
                    "response_id": response_id,
                    "pair_id": pair_id,
                    "split": split,
                    "graph_path": str(artifact_path),
                }
            )
    if seen_responses != set(artifacts):
        missing = sorted(set(artifacts).difference(seen_responses))
        raise ValueError(
            "artifact graphs are absent from splits.json: " + ", ".join(missing[:10])
        )
    return rows


def export_completed_run_graph_index(
    run_dir: str | Path,
    *,
    labels_path: str | Path | None = None,
    halueval_qa_path: str | Path | None = None,
) -> dict[str, object]:
    """Create a minimal labeled index for a completed, score-frozen run."""

    root = Path(run_dir).expanduser().resolve()
    _require_frozen_scores(root)
    graph_dir = root / "prepared" / "graphs"
    _, artifact_rows = _load_artifact_rows(graph_dir)
    rows = _rows_from_splits(root, graph_dir, artifact_rows)
    try:
        labels_file = _labels_path(root, labels_path)
    except ValueError:
        if halueval_qa_path is None:
            raise
        labels_file = root / ".missing-evaluation-label-sidecar"
    if labels_file.is_file():
        labels = load_halueval_response_labels(labels_file)
        label_source = str(labels_file)
    elif halueval_qa_path is not None:
        from unsupervised_token_graph.data import load_halueval_qa

        qa_path = Path(halueval_qa_path).expanduser().resolve()
        if not qa_path.is_file():
            raise FileNotFoundError(f"HaluEval QA source is absent: {qa_path}")
        _, labels = load_halueval_qa(qa_path)
        label_source = str(qa_path)
    else:
        raise FileNotFoundError(
            f"evaluation label sidecar is absent: {labels_file}; "
            "provide --halueval-qa to deterministically rebuild labels"
        )
    # Finish every validation before replacing the legacy technical index.
    index = build_labeled_graph_index(rows, labels)
    artifact_index = separate_artifact_index(graph_dir)
    destination = graph_dir / "index.json"
    atomic_json(destination, index)
    return {
        "index": str(destination),
        "artifact_index": str(artifact_index),
        "samples": len(index),
        "split_counts": dict(Counter(str(row["split"]) for row in index)),
        "label_source": label_source,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a minimal evaluation-only HaluEval graph index"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--halueval-qa", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_completed_run_graph_index(
        args.run_dir,
        labels_path=args.labels,
        halueval_qa_path=args.halueval_qa,
    )
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
