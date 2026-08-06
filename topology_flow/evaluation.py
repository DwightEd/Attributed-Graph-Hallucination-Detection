"""Evaluation-only RAGTruth label join for frozen topology scores."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Mapping

_RESPONSE_ID_KEYS = ("response_id", "id", "sample_id", "uid")


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
        rank_sum_positive += average_rank * sum(
            labels[index] for index in order[start:end]
        )
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


def _response_id_lookup(
    responses: list[Mapping[str, object]],
) -> dict[str, tuple[int, Mapping[str, object]]]:
    """Index every explicit response identifier plus the stable row index."""

    lookup: dict[str, tuple[int, Mapping[str, object]]] = {}
    duplicates: set[str] = set()
    for index, response in enumerate(responses):
        identifiers = {str(index)}
        for key in _RESPONSE_ID_KEYS:
            value = response.get(key)
            if value is not None and str(value).strip():
                identifiers.add(str(value).strip())
        for identifier in identifiers:
            if identifier in lookup and lookup[identifier][0] != index:
                duplicates.add(identifier)
            else:
                lookup[identifier] = (index, response)
    for identifier in duplicates:
        lookup.pop(identifier, None)
    return lookup


def _resolve_response(
    score: Mapping[str, object],
    responses: list[Mapping[str, object]],
    lookup: Mapping[str, tuple[int, Mapping[str, object]]],
) -> tuple[int, Mapping[str, object], str]:
    """Resolve by original_idx, then cache response_id/sample_id, then row id."""

    original = score.get("original_idx")
    if original is not None:
        original_idx = int(original)
        if not 0 <= original_idx < len(responses):
            raise ValueError(f"original_idx out of range: {original_idx}")
        return original_idx, responses[original_idx], "original_idx"

    candidates = []
    for key in ("response_id", "sample_id"):
        value = score.get(key)
        if value is not None and str(value).strip():
            candidates.append(str(value).strip())
    for candidate in candidates:
        matched = lookup.get(candidate)
        if matched is not None:
            index, response = matched
            return index, response, "response_id"

    preview = ", ".join(candidates[:3]) or "<missing>"
    raise ValueError(
        "could not align frozen score to RAGTruth response.jsonl; "
        f"original_idx is absent and response/sample id did not match: {preview}. "
        "Inspect one raw attention cache response_id and one response.jsonl row."
    )


def join_ragtruth_scores(
    score_records: list[Mapping[str, object]],
    *,
    response_path: str | Path,
    source_path: str | Path,
) -> list[dict[str, object]]:
    """Join official labels only after scores have been persisted and frozen."""

    responses = _read_jsonl(response_path)
    response_lookup = _response_id_lookup(responses)
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
        response_index, response, join_key = _resolve_response(
            source_record, responses, response_lookup
        )
        if response_index in seen:
            raise ValueError(f"duplicate RAGTruth response in score file: {response_index}")
        seen.add(response_index)
        source_id = str(source_record["source_id"])
        if str(response.get("source_id")) != source_id:
            raise ValueError(
                f"source mismatch for response index={response_index}: "
                f"score={source_id!r} response={response.get('source_id')!r}"
            )
        if source_id not in sources:
            raise ValueError(f"source metadata not found: {source_id}")
        labels = response.get("labels", [])
        if not isinstance(labels, list):
            raise ValueError("RAGTruth response labels must be a list")
        record = dict(source_record)
        record.update(
            {
                "original_idx": response_index,
                "evaluation_join_key": join_key,
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
