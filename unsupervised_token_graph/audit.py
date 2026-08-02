"""Label-blind profiling of observable token-attention failure modes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


class UnsupervisedPatternProfiler:
    """Learn robust feature tails and name them without correctness labels."""

    def __init__(self, *, lower_quantile: float = 0.1, upper_quantile: float = 0.9):
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)

    def fit(self, records: Sequence[Mapping[str, object]]):
        if not records:
            raise ValueError("at least one reference record is required")
        numeric_names = sorted(
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
        self.thresholds_ = {}
        for name in numeric_names:
            values = np.asarray([record[name] for record in records], dtype=float)
            if np.isfinite(values).all():
                self.thresholds_[name] = {
                    "low": float(np.quantile(values, self.lower_quantile)),
                    "high": float(np.quantile(values, self.upper_quantile)),
                }
        return self

    def transform(
        self, records: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        if not hasattr(self, "thresholds_"):
            raise RuntimeError("fit must be called before transform")
        transformed = []
        for source_record in records:
            record = dict(source_record)
            patterns: list[str] = []
            self._append_low(record, "answer_to_passage_ratio", "evidence_neglect", patterns)
            self._append_low(record, "answer_to_question_ratio", "question_neglect", patterns)
            self._append_high(
                record,
                "answer_self_reliance",
                "answer_self_reinforcement",
                patterns,
            )
            self._append_high(record, "answer_attention_entropy", "diffuse_attention", patterns)
            self._append_high(record, "passage_head_disagreement", "head_disagreement", patterns)
            self._append_low(record, "passage_layer_drift", "late_layer_grounding_drop", patterns)
            self._append_low(record, "mean_answer_log_prob", "low_confidence", patterns)
            record["patterns"] = patterns or ["no_extreme_pattern"]
            transformed.append(record)
        return transformed

    def _append_low(self, record, feature, pattern, patterns):
        if feature in record and feature in self.thresholds_:
            if float(record[feature]) < self.thresholds_[feature]["low"]:
                patterns.append(pattern)

    def _append_high(self, record, feature, pattern, patterns):
        if feature in record and feature in self.thresholds_:
            if float(record[feature]) > self.thresholds_[feature]["high"]:
                patterns.append(pattern)
