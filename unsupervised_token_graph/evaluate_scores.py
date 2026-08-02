"""Evaluation-only CLI for frozen unsupervised graph scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import read_evaluation_labels, read_json_records
from .evaluation import summarize_feature_separation, summarize_paired_ranking


def evaluate_score_file(
    scores_path: str | Path,
    evaluation_labels_path: str | Path,
    output_path: str | Path,
    *,
    split: str | None = "test",
) -> dict[str, object]:
    scores = read_json_records(scores_path)
    if split is not None:
        scores = [record for record in scores if record.get("split") == split]
        if not scores:
            raise ValueError(f"No frozen scores belong to split {split!r}")
    labels = read_evaluation_labels(evaluation_labels_path)
    observed_labels = {
        labels[str(record["example_id"])]
        for record in scores
        if str(record["example_id"]) in labels
    }
    report = {
        "samples": len(scores),
        "split": split or "all",
        "paired_ranking": summarize_paired_ranking(scores, labels),
    }
    if observed_labels == {0, 1}:
        report["score_separation"] = summarize_feature_separation(scores, labels)
    else:
        report["score_separation"] = {}
        report["evaluation_warning"] = (
            "AUROC is undefined because this label-blind split does not contain "
            "both correct and error examples"
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen unsupervised scores.")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--evaluation-labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default="test",
        help="Evaluate untouched test scores by default; 'all' is diagnostic only",
    )
    args = parser.parse_args(argv)
    report = evaluate_score_file(
        args.scores,
        args.evaluation_labels,
        args.output,
        split=None if args.split == "all" else args.split,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
