"""Evaluation-only RAGTruth label join for frozen topology scores."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Mapping


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [
        dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _binary_auroc(labels: list[int], scores: list[float]) -> float:
    """Exact Mann-Whitney AUROC with average ranks for tied scores."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum_positive = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        rank_sum_positive += average_rank * sum(labels[index] for index in order[start:end])
        start = end
    u_statistic = rank_sum_positive - positives * (positives + 1) / 2.0
    return u_statistic / (positives * negatives)


def _binary_average_precision(labels: list[int], scores: list[float]) -> float:
    """Threshold-integrated AP matching the precision-recall step definition."""

    positives = sum(labels)
    if positives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    true_positive = false_positive = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        threshold = scores[order[start]]
        while end < len(order) and scores[order[end]] == threshold:
            end += 1
        for index in order[start:end]:
            if labels[index]:
                true_positive += 1
            else:
                false_positive += 1
        recall = true_positive / positives
        precision = true_positive / max(true_positive + false_positive, 1)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return average_precision


def _metrics(records: list[Mapping[str, object]]) -> dict[str, float | int]:
    labels = [int(record["label"]) for record in records]
    scores = [float(record["topology_anomaly_score"]) for record in records]
    prevalence = sum(labels) / len(labels) if labels else 0.0
    result: dict[str, float | int] = {
        "samples": len(records),
        "positive_samples": sum(labels),
        "positive_fraction": prevalence,
    }
    if len(set(labels)) < 2:
        result.update({"auroc": float("nan"), "average_precision": float("nan")})
        return result
    auroc = _binary_auroc(labels, scores)
    ap = _binary_average_precision(labels, scores)
    reversed_ap = _binary_average_precision(labels, [-score for score in scores])
    result.update(
        {
            "auroc": auroc,
            "average_precision": ap,
            "average_precision_lift": ap / max(prevalence, 1e-12),
            "orientation_free_auroc": max(auroc, 1.0 - auroc),
            "orientation_free_average_precision": max(ap, reversed_ap),
        }
    )
    return result


def join_ragtruth_scores(
    score_records: list[Mapping[str, object]],
    *,
    response_path: str | Path,
    source_path: str | Path,
) -> list[dict[str, object]]:
    """Join official labels only after scores have been persisted and frozen."""

    responses = _read_jsonl(response_path)
    sources = {str(row["source_id"]): row for row in _read_jsonl(source_path)}
    output = []
    seen: set[int] = set()
    for source_record in score_records:
        forbidden = [
            key
            for key in source_record
            if "label" in str(key).casefold() or "target" in str(key).casefold()
        ]
        if forbidden:
            raise ValueError(
                "frozen score records must remain label-free before evaluation: "
                + ", ".join(sorted(map(str, forbidden)))
            )
        original_idx = int(source_record["original_idx"])
        if original_idx in seen:
            raise ValueError(f"duplicate original_idx in score file: {original_idx}")
        seen.add(original_idx)
        if not 0 <= original_idx < len(responses):
            raise ValueError(f"original_idx out of range: {original_idx}")
        response = responses[original_idx]
        source_id = str(source_record["source_id"])
        if str(response.get("source_id")) != source_id:
            raise ValueError(f"source mismatch at original_idx={original_idx}")
        if source_id not in sources:
            raise ValueError(f"source metadata not found: {source_id}")
        labels = response.get("labels", [])
        if not isinstance(labels, list):
            raise ValueError("RAGTruth response labels must be a list")
        record = dict(source_record)
        record.update(
            {
                "label": int(bool(labels)),
                "annotation_count": len(labels),
                "task": str(sources[source_id].get("task_type", "unknown")),
                "model": str(response.get("model", "unknown")),
                "split": str(response.get("split", "unknown")),
            }
        )
        output.append(record)
    return output


def evaluate_ragtruth(
    score_path: str | Path,
    *,
    response_path: str | Path,
    source_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    scores = _read_jsonl(score_path)
    evaluated = join_ragtruth_scores(
        scores, response_path=response_path, source_path=source_path
    )
    report: dict[str, object] = {"overall": _metrics(evaluated), "strata": {}}
    for axis in ("task", "model", "split"):
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for record in evaluated:
            groups[str(record[axis])].append(record)
        report["strata"][axis] = {
            name: _metrics(group) for name, group in sorted(groups.items())
        }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return report
