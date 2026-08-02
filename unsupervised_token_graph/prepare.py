"""CLI for preparing HaluEval-QA or generated BoolQ answers."""

from __future__ import annotations

import argparse
import json

from .data import (
    load_boolq_predictions,
    load_halueval_qa,
    write_prepared_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare label-separated passage/question/answer examples."
    )
    parser.add_argument("--dataset", choices=("halueval_qa", "boolq"), required=True)
    parser.add_argument("--input", required=True, help="Source dataset JSON/JSONL")
    parser.add_argument(
        "--predictions",
        help="BoolQ JSONL with id/model_answer; forbidden for HaluEval-QA",
    )
    parser.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="Prepare only BoolQ rows that have generated predictions",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset == "halueval_qa":
        if args.predictions or args.allow_missing_predictions:
            raise SystemExit("BoolQ prediction options are invalid for HaluEval-QA")
        examples, labels = load_halueval_qa(args.input)
    else:
        if not args.predictions:
            raise SystemExit("BoolQ requires --predictions with generated model answers")
        examples, labels = load_boolq_predictions(
            args.input,
            args.predictions,
            allow_missing=args.allow_missing_predictions,
        )
    paths = write_prepared_dataset(examples, labels, args.output_dir)
    print(
        json.dumps(
            {
                "examples": len(examples),
                "pairs": len({example.pair_id for example in examples}),
                "model_inputs": str(paths["examples"]),
                "evaluation_only_labels": str(paths["evaluation_labels"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
