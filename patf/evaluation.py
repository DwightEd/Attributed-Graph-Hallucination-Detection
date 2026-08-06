from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _auroc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        rank_sum += rank * sum(labels[index] for index in order[start:end])
        start = end
    return (
        rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def _metrics(records: list[dict[str, object]]) -> dict[str, float | int]:
    labels = [int(row["label"]) for row in records]
    scores = [float(row["topology_anomaly_score"]) for row in records]
    return {
        "samples": len(records),
        "positive_samples": sum(labels),
        "positive_fraction": sum(labels) / len(labels) if labels else 0.0,
        "auroc": _auroc(labels, scores),
        "average_precision": _average_precision(labels, scores),
    }


def _group_metrics(
    records: list[dict[str, object]], key: str
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        groups[str(row[key])].append(row)
    return {name: _metrics(rows) for name, rows in sorted(groups.items())}


def evaluate(
    prediction_path: str | Path,
    ragtruth_root: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    predictions = _read_jsonl(prediction_path)
    ragtruth_root = Path(ragtruth_root)
    responses = _read_jsonl(ragtruth_root / "response.jsonl")
    response_by_id = {str(row["id"]): row for row in responses}
    sources = {
        str(row["source_id"]): row
        for row in _read_jsonl(ragtruth_root / "source_info.jsonl")
    }

    evaluated: list[dict[str, object]] = []
    for prediction in predictions:
        response = response_by_id.get(str(prediction["sample_id"]))
        if response is None and prediction.get("original_idx") is not None:
            response = responses[int(prediction["original_idx"])]
        if response is None:
            raise KeyError(f"RAGTruth response not found: {prediction['sample_id']}")
        source = sources[str(prediction["source_id"])]
        evaluated.append(
            {
                **prediction,
                "label": int(bool(response.get("labels", []))),
                "task": str(source.get("task_type", "unknown")),
                "model": str(response.get("model", "unknown")),
            }
        )

    report = {
        "overall": _metrics(evaluated),
        "by_task": _group_metrics(evaluated, "task"),
        "by_model": _group_metrics(evaluated, "model"),
    }
    Path(output_path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
