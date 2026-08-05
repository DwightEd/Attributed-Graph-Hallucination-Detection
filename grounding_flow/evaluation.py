"""Label-free score controls and the explicit post-hoc label boundary."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from attention_graph.halueval import (
    evaluate_halueval_predictions,
    load_halueval_response_labels,
)

from .artifacts import atomic_json, file_sha256, iter_jsonl, read_jsonl
from .graph_index import write_labeled_graph_index


def _forbidden_evaluation_paths(value: object, prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).casefold()
            location = f"{prefix}.{key}" if prefix else str(key)
            if "label" in name or name in {"y", "y_token", "direction_score"}:
                found.add(location)
            found.update(_forbidden_evaluation_paths(child, location))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            location = f"{prefix}[{index}]"
            found.update(_forbidden_evaluation_paths(child, location))
    return found


def _fit_length_coefficients(
    train_predictions: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    if len(train_predictions) < 2:
        raise ValueError("length residual requires at least two training predictions")
    scores = np.asarray([float(row["score"]) for row in train_predictions])
    lengths = np.asarray(
        [math.log1p(float(row["response_tokens"])) for row in train_predictions]
    )
    if not np.isfinite(scores).all() or not np.isfinite(lengths).all():
        raise ValueError("length residual inputs must be finite")
    design = np.column_stack((np.ones(len(lengths)), lengths))
    coefficient, *_ = np.linalg.lstsq(design, scores, rcond=None)
    return {"intercept": float(coefficient[0]), "log1p_length": float(coefficient[1])}


def add_length_residual_scores(
    train_predictions: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    *,
    coefficients: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Remove the train-only linear response-length trend without labels."""

    fitted = (
        _fit_length_coefficients(train_predictions)
        if coefficients is None
        else {
            "intercept": float(coefficients["intercept"]),
            "log1p_length": float(coefficients["log1p_length"]),
        }
    )
    result: list[dict[str, object]] = []
    for prediction in predictions:
        row = dict(prediction)
        expected = fitted["intercept"] + fitted["log1p_length"] * math.log1p(
            float(row["response_tokens"])
        )
        row["length_residual_score"] = float(row["score"]) - expected
        result.append(row)
    return result, fitted


def freeze_prediction_files(
    output_dir: str | Path, prediction_paths: Sequence[str | Path]
) -> dict[str, object]:
    """Create the immutable score boundary that must precede all label reads."""

    output = Path(output_dir).expanduser().resolve()
    files: dict[str, object] = {}
    for raw_path in prediction_paths:
        path = Path(raw_path).expanduser().resolve()
        try:
            relative = path.relative_to(output).as_posix()
        except ValueError as error:
            raise ValueError("prediction files must live inside output_dir") from error
        forbidden: set[str] = set()
        record_count = 0
        for record in iter_jsonl(path):
            record_count += 1
            forbidden.update(_forbidden_evaluation_paths(record))
        if forbidden:
            raise ValueError(
                "prediction freeze is label-blind: " + ", ".join(sorted(forbidden))
            )
        files[relative] = {
            "sha256": file_sha256(path),
            "records": record_count,
            "bytes": path.stat().st_size,
        }
    artifact: dict[str, object] = {
        "schema": "grounding-flow-score-freeze-v1",
        "state": "scores_frozen_before_label_read",
        "files": files,
    }
    atomic_json(output / "score_freeze.json", artifact)
    return artifact


def _verify_frozen_file(output: Path, prediction_path: Path) -> dict[str, object]:
    freeze_path = output / "score_freeze.json"
    value = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("state") != "scores_frozen_before_label_read":
        raise ValueError("score_freeze.json is missing or invalid")
    relative = prediction_path.resolve().relative_to(output.resolve()).as_posix()
    files = value.get("files")
    if not isinstance(files, Mapping) or relative not in files:
        raise ValueError("prediction file is absent from score freeze")
    expected = files[relative]
    if not isinstance(expected, Mapping) or expected.get("sha256") != file_sha256(
        prediction_path
    ):
        raise ValueError("frozen prediction digest changed")
    return dict(expected)


def evaluate_frozen_halueval_predictions(
    *,
    output_dir: str | Path,
    prediction_path: str | Path,
    labels_path: str | Path,
    pair_by_response: Mapping[str, object],
    response_length_by_id: Mapping[str, object],
    seed: int,
    bootstrap_samples: int = 1_000,
    score_fields: Mapping[str, str] | None = None,
    graph_index_rows: Sequence[Mapping[str, object]] | None = None,
    graph_index_path: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate only a checksum-verified disk snapshot, never mutable scores."""

    if (graph_index_rows is None) != (graph_index_path is None):
        raise ValueError(
            "graph_index_rows and graph_index_path must be provided together"
        )
    output = Path(output_dir).expanduser().resolve()
    predictions_file = Path(prediction_path).expanduser().resolve()
    _verify_frozen_file(output, predictions_file)
    # This is intentionally the sole label read in the whole pipeline.
    all_labels = load_halueval_response_labels(Path(labels_path).expanduser().resolve())
    # Re-check and re-read after the label call so even a side effect cannot
    # replace the frozen in-memory score stream used for evaluation.
    _verify_frozen_file(output, predictions_file)
    rows = read_jsonl(predictions_file)
    response_ids = {str(row.get("response_id", "")) for row in rows}
    if "" in response_ids or len(response_ids) != len(rows):
        raise ValueError("frozen response predictions require unique response_id values")
    if not response_ids.issubset(all_labels):
        raise ValueError("evaluation labels do not cover the frozen test responses")
    labels = {response_id: all_labels[response_id] for response_id in response_ids}
    pairs = {str(key): value for key, value in pair_by_response.items()}
    lengths = {str(key): value for key, value in response_length_by_id.items()}
    if set(pairs) != response_ids or set(lengths) != response_ids:
        raise ValueError("test pair/length maps must exactly match frozen predictions")

    fields = {"primary": "score"} if score_fields is None else dict(score_fields)
    if not fields:
        raise ValueError("at least one score field is required")
    reports: dict[str, object] = {}
    for index, (name, field) in enumerate(fields.items()):
        converted = []
        for row in rows:
            if field not in row:
                raise ValueError(f"frozen prediction is missing score field {field}")
            converted.append(
                {"response_id": row["response_id"], "score": float(row[field])}
            )
        reports[name] = evaluate_halueval_predictions(
            converted,
            labels,
            pairs,
            response_length_by_id=lengths if index == 0 else None,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    if score_fields is None:
        result = dict(reports["primary"])
    else:
        primary_name = next(iter(fields))
        result = {
            "schema": "grounding-flow-halueval-evaluation-v1",
            "primary_score": primary_name,
            "primary": reports[primary_name],
            "variants": reports,
        }
    # Reuse the one evaluation-only label read. The graph artifacts remain
    # immutable and label-free; only the separate five-field index is written.
    if graph_index_rows is not None and graph_index_path is not None:
        write_labeled_graph_index(graph_index_path, graph_index_rows, all_labels)
    return result


__all__ = [
    "add_length_residual_scores",
    "evaluate_frozen_halueval_predictions",
    "freeze_prediction_files",
]
