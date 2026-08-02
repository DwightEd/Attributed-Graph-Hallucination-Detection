"""Label-blind profiling of observable token-attention failure modes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .data import read_evaluation_labels
from .evaluation import summarize_feature_separation, summarize_paired_ranking
from .scoring import RobustMahalanobisScorer


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


DEFAULT_SCORE_FEATURES = (
    "answer_to_passage_ratio",
    "answer_to_question_ratio",
    "answer_self_reliance",
    "answer_attention_entropy",
    "passage_head_disagreement",
    "passage_layer_drift",
    "answer_edge_density",
    "mean_answer_log_prob",
    "mean_answer_next_token_entropy",
)


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [
        dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _score_records(
    reference_records: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    feature_names = [
        name
        for name in DEFAULT_SCORE_FEATURES
        if all(name in record for record in reference_records)
        and all(name in record for record in records)
    ]
    if not feature_names:
        raise ValueError("No common anomaly-score features were extracted")
    reference_matrix = np.asarray(
        [[record[name] for name in feature_names] for record in reference_records],
        dtype=float,
    )
    matrix = np.asarray(
        [[record[name] for name in feature_names] for record in records],
        dtype=float,
    )
    scorer = RobustMahalanobisScorer().fit(reference_matrix)
    scores = scorer.score_samples(matrix)
    scored = []
    for record, score in zip(records, scores):
        output = dict(record)
        output["anomaly_score"] = float(score)
        scored.append(output)
    return scored, feature_names


def _render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Unsupervised token-graph reasoning-pattern audit",
        "",
        "> Pattern names come from unlabeled feature tails. Labels are loaded only in the evaluation section.",
        "",
        f"- Samples: {report['samples']}",
        f"- Reference samples: {report['reference_samples']}",
        f"- Score features: {', '.join(report['score_features'])}",
        "",
        "## Unlabeled pattern counts",
        "",
        "| Pattern | Count |",
        "|---|---:|",
    ]
    for name, count in report["pattern_counts"].items():
        lines.append(f"| `{name}` | {count} |")
    if "feature_separation" in report:
        lines.extend(
            [
                "",
                "## Evaluation-only feature separation",
                "",
                "| Feature | Median correct | Median error | AUROC | Separability |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        ordered = sorted(
            report["feature_separation"].items(),
            key=lambda item: item[1]["separability"],
            reverse=True,
        )
        for name, values in ordered:
            lines.append(
                f"| `{name}` | {values['median_label_0']:.4f} | "
                f"{values['median_label_1']:.4f} | {values['auc']:.4f} | "
                f"{values['separability']:.4f} |"
            )
    if "paired_ranking" in report:
        paired = report["paired_ranking"]
        lines.extend(
            [
                "",
                "## Paired ranking",
                "",
                f"- Evaluated pairs: {paired['evaluated_pairs']}",
                f"- Error candidate ranked higher: {paired['paired_ranking_accuracy']:.4f}",
            ]
        )
    return "\n".join(lines) + "\n"


def run_pattern_audit(
    features_path: str | Path,
    output_dir: str | Path,
    *,
    reference_features_path: str | Path | None = None,
    evaluation_labels_path: str | Path | None = None,
) -> dict[str, object]:
    """Fit patterns/scores first; optionally join labels only for evaluation."""

    records = _read_jsonl(features_path)
    reference_records = (
        _read_jsonl(reference_features_path) if reference_features_path else records
    )
    profiler = UnsupervisedPatternProfiler().fit(reference_records)
    profiled = profiler.transform(records)
    scored, score_features = _score_records(reference_records, profiled)
    pattern_counts = Counter(
        pattern for record in scored for pattern in record["patterns"]
    )
    report: dict[str, object] = {
        "samples": len(scored),
        "reference_samples": len(reference_records),
        "score_features": score_features,
        "pattern_counts": dict(sorted(pattern_counts.items())),
    }
    if evaluation_labels_path:
        labels = read_evaluation_labels(evaluation_labels_path)
        report["feature_separation"] = summarize_feature_separation(scored, labels)
        report["paired_ranking"] = summarize_paired_ranking(scored, labels)

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "pattern_records.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in scored)
        + "\n",
        encoding="utf-8",
    )
    (output_directory / "pattern_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "pattern_audit.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit unlabeled token-graph patterns.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--reference-features")
    parser.add_argument("--evaluation-labels")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_pattern_audit(
        args.features,
        args.output_dir,
        reference_features_path=args.reference_features,
        evaluation_labels_path=args.evaluation_labels,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
