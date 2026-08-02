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
    missing = [record["example_id"] for record in records if record["example_id"] not in evaluation_labels]
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
