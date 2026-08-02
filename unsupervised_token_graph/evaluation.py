"""Label-gated evaluation kept outside extraction, training, and profiling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _binary_auc(values: np.ndarray, labels: np.ndarray) -> float:
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("both evaluation label classes are required")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2.0
        position = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def summarize_feature_separation(
    records: Sequence[Mapping[str, object]],
    evaluation_labels: Mapping[str, int],
) -> dict[str, dict[str, float]]:
    """Join labels only after features have been produced and frozen."""

    if not records:
        raise ValueError("at least one feature record is required")
    missing = [
        record["example_id"]
        for record in records
        if record["example_id"] not in evaluation_labels
    ]
    if missing:
        raise ValueError(f"Missing evaluation labels for {missing[:3]}")
    feature_names = sorted(
        set.intersection(
            *(
                {
                    key
                    for key, value in record.items()
                    if key != "example_id"
                    and isinstance(value, (int, float, np.number))
                }
                for record in records
            )
        )
    )
    labels = np.asarray(
        [int(evaluation_labels[str(record["example_id"])]) for record in records]
    )
    summary: dict[str, dict[str, float]] = {}
    for name in feature_names:
        values = np.asarray([record[name] for record in records], dtype=float)
        if not np.isfinite(values).all():
            continue
        auc = _binary_auc(values, labels)
        summary[name] = {
            "median_label_0": float(np.median(values[labels == 0])),
            "median_label_1": float(np.median(values[labels == 1])),
            "auc": float(auc),
            "separability": float(max(auc, 1.0 - auc)),
        }
    return summary


def summarize_pattern_enrichment(
    records: Sequence[Mapping[str, object]],
    evaluation_labels: Mapping[str, int],
) -> dict[str, dict[str, float | int]]:
    """Measure pattern prevalence by class after unlabeled patterns are frozen."""

    if not records:
        raise ValueError("at least one pattern record is required")
    labels = []
    patterns = set()
    for record in records:
        example_id = str(record["example_id"])
        if example_id not in evaluation_labels:
            raise ValueError(f"Missing evaluation label for {example_id!r}")
        label = int(evaluation_labels[example_id])
        if label not in (0, 1):
            raise ValueError("evaluation labels must be binary")
        labels.append(label)
        patterns.update(str(pattern) for pattern in record.get("patterns", ()))
    class_totals = {
        label: sum(observed == label for observed in labels) for label in (0, 1)
    }
    if not class_totals[0] or not class_totals[1]:
        raise ValueError("both evaluation label classes are required")

    summary: dict[str, dict[str, float | int]] = {}
    for pattern in sorted(patterns):
        counts = {0: 0, 1: 0}
        for record, label in zip(records, labels):
            if pattern in record.get("patterns", ()):
                counts[label] += 1
        correct_prevalence = counts[0] / class_totals[0]
        error_prevalence = counts[1] / class_totals[1]
        summary[pattern] = {
            "correct_count": counts[0],
            "error_count": counts[1],
            "correct_prevalence": float(correct_prevalence),
            "error_prevalence": float(error_prevalence),
            "prevalence_gap": float(error_prevalence - correct_prevalence),
        }
    return summary


def summarize_paired_feature_deltas(
    records: Sequence[Mapping[str, object]],
    evaluation_labels: Mapping[str, int],
) -> dict[str, dict[str, float | int]]:
    """Summarize error-minus-correct feature changes within the same pair."""

    if not records:
        raise ValueError("at least one feature record is required")
    feature_names = sorted(
        set.intersection(
            *(
                {
                    key
                    for key, value in record.items()
                    if key not in {"example_id", "pair_id"}
                    and isinstance(value, (int, float, np.number))
                }
                for record in records
            )
        )
    )
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["pair_id"]), []).append(record)
    deltas = {name: [] for name in feature_names}
    for pair_records in grouped.values():
        by_label = {0: [], 1: []}
        for record in pair_records:
            example_id = str(record["example_id"])
            if example_id not in evaluation_labels:
                raise ValueError(f"Missing evaluation label for {example_id!r}")
            by_label[int(evaluation_labels[example_id])].append(record)
        if not by_label[0] or not by_label[1]:
            continue
        for name in feature_names:
            correct = np.mean([float(record[name]) for record in by_label[0]])
            error = np.mean([float(record[name]) for record in by_label[1]])
            deltas[name].append(float(error - correct))
    summary = {}
    for name, values in deltas.items():
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        summary[name] = {
            "evaluated_pairs": len(values),
            "mean_error_minus_correct": float(array.mean()),
            "median_error_minus_correct": float(np.median(array)),
            "error_higher_fraction": float(
                ((array > 0).sum() + 0.5 * (array == 0).sum()) / len(array)
            ),
        }
    return summary


def summarize_paired_ranking(
    records: Sequence[Mapping[str, object]],
    evaluation_labels: Mapping[str, int],
    *,
    score_field: str = "anomaly_score",
) -> dict[str, float | int | None]:
    """Measure whether the error candidate outranks its correct paired answer."""

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["pair_id"]), []).append(record)
    wins = ties = evaluated = 0
    for pair_records in grouped.values():
        correct_scores = []
        error_scores = []
        for record in pair_records:
            example_id = str(record["example_id"])
            if example_id not in evaluation_labels:
                continue
            destination = error_scores if evaluation_labels[example_id] == 1 else correct_scores
            destination.append(float(record[score_field]))
        if not correct_scores or not error_scores:
            continue
        evaluated += 1
        correct_score = float(np.mean(correct_scores))
        error_score = float(np.mean(error_scores))
        wins += int(error_score > correct_score)
        ties += int(error_score == correct_score)
    accuracy = None if not evaluated else float((wins + 0.5 * ties) / evaluated)
    return {
        "evaluated_pairs": evaluated,
        "wins": wins,
        "ties": ties,
        "paired_ranking_accuracy": accuracy,
    }
