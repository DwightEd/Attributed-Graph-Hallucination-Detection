"""Command line interface for Prompt-Anchored Topology Flow."""

from __future__ import annotations

import argparse
import json
import sys

from .config import CorruptionConfig, TopologyConfig
from .evaluation import evaluate_ragtruth
from .model import RankerConfig
from .pipeline import (
    extract_directory,
    score_directory,
    train_directory,
    validate_directory,
)

_NEW_COMMANDS = {"validate", "extract", "train", "score", "evaluate"}


def _topology(arguments: argparse.Namespace) -> TopologyConfig:
    return TopologyConfig(
        mass_cover=arguments.mass_cover,
        relay_discount=arguments.relay_discount,
        head_reducer=arguments.head_reducer,
    )


def _add_topology_options(
    parser: argparse.ArgumentParser, *, optional: bool = False
) -> None:
    parser.add_argument("--mass-cover", type=float, default=None if optional else 0.80)
    parser.add_argument(
        "--relay-discount", type=float, default=None if optional else 0.85
    )
    parser.add_argument(
        "--head-reducer",
        choices=("median", "mean"),
        default=None if optional else "median",
    )
    parser.add_argument("--no-recursive", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prompt-Anchored Topology Flow for label-free hallucination anomaly detection"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="check whether saved .pt attention graphs are compatible"
    )
    validate.add_argument("--input-dir", required=True)
    validate.add_argument("--limit", type=int)
    validate.add_argument("--output")
    validate.add_argument("--no-recursive", action="store_true")

    extract = subparsers.add_parser("extract", help="freeze label-free topology trajectories")
    extract.add_argument("--input-dir", required=True)
    extract.add_argument("--output", required=True)
    _add_topology_options(extract)

    train = subparsers.add_parser("train", help="train on controlled topology erosions")
    train.add_argument("--input-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument(
        "--corruption-mode",
        choices=("incidence", "collapse", "composite", "all"),
        default="all",
    )
    train.add_argument("--prompt-transfer", type=float, default=0.45)
    train.add_argument("--local-window", type=int, default=4)
    train.add_argument("--support-keep-fraction", type=float, default=0.55)
    train.add_argument("--concentration-power", type=float, default=1.8)
    train.add_argument("--hidden-dim", type=int, default=48)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--margin", type=float, default=0.5)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--validation-fraction", type=float, default=0.20)
    train.add_argument("--patience", type=int, default=12)
    train.add_argument("--seed", type=int, default=42)
    _add_topology_options(train)

    score = subparsers.add_parser(
        "score", help="score saved graphs; topology settings default to checkpoint values"
    )
    score.add_argument("--input-dir", required=True)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--output", required=True)
    _add_topology_options(score, optional=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="join RAGTruth labels after score freezing"
    )
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--responses", required=True)
    evaluate.add_argument("--sources", required=True)
    evaluate.add_argument("--output", required=True)
    return parser


def _optional_topology(arguments: argparse.Namespace) -> TopologyConfig | None:
    supplied = (
        arguments.mass_cover,
        arguments.relay_discount,
        arguments.head_reducer,
    )
    if all(value is None for value in supplied):
        return None
    if any(value is None for value in supplied):
        raise ValueError(
            "when overriding score topology, pass mass-cover, relay-discount and head-reducer together"
        )
    return _topology(arguments)


def main(argv: list[str] | None = None) -> int:
    arguments_list = list(sys.argv[1:] if argv is None else argv)
    if (
        arguments_list
        and arguments_list[0] not in _NEW_COMMANDS
        and arguments_list[0] not in {"-h", "--help"}
    ):
        from attention_graph.cli import main as legacy_main

        return int(legacy_main(arguments_list) or 0)

    parser = build_parser()
    arguments = parser.parse_args(arguments_list)
    if arguments.command == "validate":
        report = validate_directory(
            arguments.input_dir,
            recursive=not arguments.no_recursive,
            limit=arguments.limit,
        )
        rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        print(rendered)
        if arguments.output:
            from pathlib import Path

            destination = Path(arguments.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(rendered + "\n", encoding="utf-8")
        return 0
    if arguments.command == "evaluate":
        evaluate_ragtruth(
            arguments.scores,
            response_path=arguments.responses,
            source_path=arguments.sources,
            output_path=arguments.output,
        )
        return 0

    recursive = not arguments.no_recursive
    if arguments.command == "extract":
        extract_directory(
            arguments.input_dir,
            arguments.output,
            topology=_topology(arguments),
            recursive=recursive,
        )
        return 0
    if arguments.command == "train":
        train_directory(
            arguments.input_dir,
            arguments.output_dir,
            topology=_topology(arguments),
            corruption=CorruptionConfig(
                prompt_transfer=arguments.prompt_transfer,
                local_window=arguments.local_window,
                support_keep_fraction=arguments.support_keep_fraction,
                concentration_power=arguments.concentration_power,
                mode=arguments.corruption_mode,
            ),
            ranker=RankerConfig(
                hidden_dim=arguments.hidden_dim,
                epochs=arguments.epochs,
                learning_rate=arguments.learning_rate,
                margin=arguments.margin,
                batch_size=arguments.batch_size,
                validation_fraction=arguments.validation_fraction,
                patience=arguments.patience,
                seed=arguments.seed,
            ),
            recursive=recursive,
            device=arguments.device,
        )
        return 0
    if arguments.command == "score":
        score_directory(
            arguments.input_dir,
            arguments.checkpoint,
            arguments.output,
            topology=_optional_topology(arguments),
            recursive=recursive,
        )
        return 0
    raise AssertionError(f"unhandled command: {arguments.command}")
