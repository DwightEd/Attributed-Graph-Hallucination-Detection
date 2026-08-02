"""Label-blind profiling of observable token-attention failure modes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .data import read_evaluation_labels, read_prepared_examples
from .evaluation import (
    summarize_feature_separation,
    summarize_paired_feature_deltas,
    summarize_paired_ranking,
    summarize_pattern_enrichment,
)
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
            self._append_low_first(
                record,
                ("answer_to_passage_token_normalized", "answer_to_passage_ratio"),
                "evidence_neglect",
                patterns,
            )
            self._append_low_first(
                record,
                ("answer_to_question_token_normalized", "answer_to_question_ratio"),
                "question_neglect",
                patterns,
            )
            self._append_high_first(
                record,
                ("answer_to_prior_answer_token_normalized", "answer_self_reliance"),
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

    def _append_low_first(self, record, features, pattern, patterns):
        for feature in features:
            if feature in record and feature in self.thresholds_:
                self._append_low(record, feature, pattern, patterns)
                return

    def _append_high_first(self, record, features, pattern, patterns):
        for feature in features:
            if feature in record and feature in self.thresholds_:
                self._append_high(record, feature, pattern, patterns)
                return


DEFAULT_SCORE_FEATURES = (
    "answer_to_passage_token_normalized",
    "answer_to_question_token_normalized",
    "answer_to_prior_answer_token_normalized",
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
        "# Unsupervised token-graph support-routing pattern audit",
        "",
        "> Pattern names come from unlabeled feature tails. Labels are loaded "
        "only in the evaluation section.",
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
    if "pattern_enrichment" in report:
        lines.extend(
            [
                "",
                "## Evaluation-only pattern enrichment",
                "",
                "| Pattern | Correct prevalence | Error prevalence | Error - correct |",
                "|---|---:|---:|---:|",
            ]
        )
        ordered_patterns = sorted(
            report["pattern_enrichment"].items(),
            key=lambda item: item[1]["prevalence_gap"],
            reverse=True,
        )
        for name, values in ordered_patterns:
            lines.append(
                f"| `{name}` | {values['correct_prevalence']:.4f} | "
                f"{values['error_prevalence']:.4f} | "
                f"{values['prevalence_gap']:+.4f} |"
            )
    if "paired_ranking" in report:
        paired = report["paired_ranking"]
        accuracy = paired["paired_ranking_accuracy"]
        lines.extend(
            [
                "",
                "## Paired ranking",
                "",
                f"- Evaluated pairs: {paired['evaluated_pairs']}",
                "- Error candidate ranked higher: "
                + (f"{accuracy:.4f}" if accuracy is not None else "N/A (unpaired dataset)"),
            ]
        )
    if "paired_feature_deltas" in report and report["paired_feature_deltas"]:
        lines.extend(
            [
                "",
                "## Within-pair support-routing deltas",
                "",
                "Positive values mean error candidate minus correct candidate.",
                "",
                "| Feature | Pairs | Median delta | Error-higher fraction |",
                "|---|---:|---:|---:|",
            ]
        )
        ordered = sorted(
            report["paired_feature_deltas"].items(),
            key=lambda item: abs(item[1]["median_error_minus_correct"]),
            reverse=True,
        )
        for name, values in ordered:
            lines.append(
                f"| `{name}` | {values['evaluated_pairs']} | "
                f"{values['median_error_minus_correct']:+.4f} | "
                f"{values['error_higher_fraction']:.4f} |"
            )
    if report.get("top_error_examples"):
        lines.extend(
            [
                "",
                "## Evaluation-only top error examples",
                "",
                "Full passage/question/answer rows are in `evaluation_error_cases.jsonl`.",
                "",
                "| Example | Patterns | Anomaly score | Question | Answer |",
                "|---|---|---:|---|---|",
            ]
        )
        for example in report["top_error_examples"]:
            question = str(example["question"]).replace("|", "\\|").replace("\n", " ")[:100]
            answer = str(example["answer"]).replace("|", "\\|").replace("\n", " ")[:100]
            lines.append(
                f"| `{example['example_id']}` | {', '.join(example['patterns'])} | "
                f"{example['anomaly_score']:.4f} | {question} | {answer} |"
            )
    return "\n".join(lines) + "\n"


def _evaluation_case_records(scored, examples, labels):
    examples_by_id = {example.example_id: example for example in examples}
    cases = []
    for record in scored:
        example_id = str(record["example_id"])
        if example_id not in examples_by_id:
            raise ValueError(f"Missing prepared example for {example_id!r}")
        example = examples_by_id[example_id]
        label = int(labels[example_id])
        patterns = list(record.get("patterns", ()))
        has_extreme = patterns != ["no_extreme_pattern"]
        case_type = (
            "true_error_pattern"
            if label == 1 and has_extreme
            else "false_positive_pattern"
            if label == 0 and has_extreme
            else "missed_error_pattern"
            if label == 1
            else "correct_no_extreme_pattern"
        )
        cases.append(
            {
                **dict(record),
                "evaluation_label": label,
                "case_type": case_type,
                "passage": example.passage,
                "question": example.question,
                "answer": example.answer,
            }
        )
    return sorted(cases, key=lambda record: record["anomaly_score"], reverse=True)


def _paired_case_records(cases):
    grouped = {}
    for case in cases:
        grouped.setdefault(str(case["pair_id"]), []).append(case)
    output = []
    for pair_id, pair_cases in grouped.items():
        correct = [case for case in pair_cases if case["evaluation_label"] == 0]
        errors = [case for case in pair_cases if case["evaluation_label"] == 1]
        if not correct or not errors:
            continue
        feature_names = sorted(
            set.intersection(
                *(
                    {
                        key
                        for key, value in case.items()
                        if key
                        not in {
                            "evaluation_label",
                            "example_id",
                            "pair_id",
                        }
                        and isinstance(value, (int, float, np.number))
                    }
                    for case in pair_cases
                )
            )
        )
        deltas = {
            name: float(
                np.mean([case[name] for case in errors])
                - np.mean([case[name] for case in correct])
            )
            for name in feature_names
        }
        output.append(
            {
                "pair_id": pair_id,
                "passage": pair_cases[0]["passage"],
                "question": pair_cases[0]["question"],
                "correct_candidates": [
                    {
                        "example_id": case["example_id"],
                        "answer": case["answer"],
                        "patterns": case["patterns"],
                        "anomaly_score": case["anomaly_score"],
                    }
                    for case in correct
                ],
                "error_candidates": [
                    {
                        "example_id": case["example_id"],
                        "answer": case["answer"],
                        "patterns": case["patterns"],
                        "anomaly_score": case["anomaly_score"],
                    }
                    for case in errors
                ],
                "feature_deltas": deltas,
            }
        )
    return sorted(output, key=lambda record: record["pair_id"])


def run_pattern_audit(
    features_path: str | Path,
    output_dir: str | Path,
    *,
    reference_features_path: str | Path | None = None,
    evaluation_labels_path: str | Path | None = None,
    examples_path: str | Path | None = None,
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
        report["pattern_enrichment"] = summarize_pattern_enrichment(scored, labels)
        report["paired_feature_deltas"] = summarize_paired_feature_deltas(scored, labels)
        report["paired_ranking"] = summarize_paired_ranking(scored, labels)
        if examples_path:
            cases = _evaluation_case_records(
                scored, read_prepared_examples(examples_path), labels
            )
            error_cases = [case for case in cases if case["evaluation_label"] == 1]
            pair_cases = _paired_case_records(cases)
            report["top_error_examples"] = [
                {
                    "example_id": case["example_id"],
                    "patterns": case["patterns"],
                    "anomaly_score": case["anomaly_score"],
                    "question": case["question"],
                    "answer": case["answer"],
                }
                for case in error_cases[:10]
            ]
    elif examples_path:
        raise ValueError("--examples requires --evaluation-labels")

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "pattern_records.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in scored)
        + "\n",
        encoding="utf-8",
    )
    if evaluation_labels_path and examples_path:
        (output_directory / "evaluation_casebook.jsonl").write_text(
            "\n".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                for record in cases
            )
            + "\n",
            encoding="utf-8",
        )
        (output_directory / "evaluation_error_cases.jsonl").write_text(
            "\n".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                for record in error_cases
            )
            + "\n",
            encoding="utf-8",
        )
        (output_directory / "evaluation_pair_deltas.jsonl").write_text(
            "\n".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                for record in pair_cases
            )
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
    parser.add_argument("--examples", help="Prepared label-free P/Q/A rows for casebook output")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_pattern_audit(
        args.features,
        args.output_dir,
        reference_features_path=args.reference_features,
        evaluation_labels_path=args.evaluation_labels,
        examples_path=args.examples,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
