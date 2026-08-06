"""Single command-line entrypoint for the staged CEPT implementation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .experiment import PilotConfig, run_pilot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CEPT counterfactual evidence-path Gate 0/1 experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pilot = commands.add_parser(
        "pilot",
        help="run label-blind RAGTruth layout and 2x2 K/V mediation gates",
    )
    pilot.add_argument("--responses", type=Path, required=True)
    pilot.add_argument("--sources", type=Path, required=True)
    pilot.add_argument("--model", type=Path, required=True)
    pilot.add_argument("--output-dir", type=Path, required=True)
    pilot.add_argument(
        "--eligible-response-ids",
        type=Path,
        required=True,
        help="verified train-cache response-ID frame emitted by the transport runner",
    )
    pilot.add_argument("--split", choices=("train",), default="train")
    pilot.add_argument("--num-samples", type=int, default=50)
    pilot.add_argument("--seed", type=int, default=42)
    pilot.add_argument("--device", default="cuda:0")
    pilot.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="float16",
    )
    pilot.add_argument("--max-sequence-tokens", type=int, default=2048)
    pilot.add_argument("--max-counterfactuals", type=int, default=2)
    pilot.add_argument(
        "--history-block-size",
        type=int,
        default=4,
        help="response-history K/V rescue block width (0 disables with max=0)",
    )
    pilot.add_argument("--max-history-blocks", type=int, default=8)
    pilot.add_argument(
        "--resume",
        action="store_true",
        help="resume verified per-sample shards from an interrupted identical run",
    )
    graphs = commands.add_parser(
        "build-graphs",
        help="build the one-time prediction-event graph store from formal cache files",
    )
    graphs.add_argument("--cache-root", type=Path, required=True)
    graphs.add_argument("--responses", type=Path, required=True)
    graphs.add_argument("--sources", type=Path, required=True)
    graphs.add_argument("--tokenizer", type=Path, required=True)
    graphs.add_argument("--output-dir", type=Path, required=True)
    graphs.add_argument("--split", choices=("train", "test"), default="train")
    graphs.add_argument("--limit", type=int)
    graphs.add_argument("--model-source-signature")
    transport = commands.add_parser(
        "train-transport",
        help="train causal provenance gates and structural controls without labels",
    )
    transport.add_argument("--graph-index", type=Path, required=True)
    transport.add_argument("--teacher", type=Path, required=True)
    transport.add_argument("--output-dir", type=Path, required=True)
    transport.add_argument("--score-index", type=Path)
    transport.add_argument("--device", default="cuda:0")
    transport.add_argument("--epochs", type=int, default=100)
    transport.add_argument("--learning-rate", type=float, default=0.03)
    transport.add_argument("--validation-fraction", type=float, default=0.2)
    transport.add_argument("--mechanism-holdout-fraction", type=float, default=0.2)
    transport.add_argument("--noise-floor", type=float, default=1e-4)
    transport.add_argument("--block-weight", type=float, default=1.0)
    transport.add_argument(
        "--allow-unidentifiable-pilot",
        action="store_true",
        help=(
            "continue after a failed teacher-population audit for an explicit "
            "non-claim pilot"
        ),
    )
    transport.add_argument("--seed", type=int, default=42)
    transport.add_argument(
        "--variants",
        nargs="+",
        choices=("true", "rewired", "mass_only", "one_hop", "no_residual"),
        default=("true", "rewired", "mass_only", "one_hop", "no_residual"),
    )
    evaluate = commands.add_parser(
        "evaluate-transport",
        help="read held-out labels only after frozen transport scoring",
    )
    evaluate.add_argument("--predictions-dir", type=Path, required=True)
    evaluate.add_argument("--test-graph-index", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "pilot":
        run_pilot(
            PilotConfig(
                responses=args.responses,
                sources=args.sources,
                model=args.model,
                output_dir=args.output_dir,
                eligible_response_ids=args.eligible_response_ids,
                split=args.split,
                num_samples=args.num_samples,
                seed=args.seed,
                device=args.device,
                dtype=args.dtype,
                max_sequence_tokens=args.max_sequence_tokens,
                max_counterfactuals=args.max_counterfactuals,
                history_block_size=args.history_block_size,
                max_history_blocks=args.max_history_blocks,
                resume=args.resume,
            )
        )
        return 0
    if args.command == "build-graphs":
        from .graph_build import build_legacy_prediction_graph_store

        build_legacy_prediction_graph_store(
            cache_root=args.cache_root,
            responses=args.responses,
            sources=args.sources,
            tokenizer_source=args.tokenizer,
            output_dir=args.output_dir,
            split=args.split,
            limit=args.limit,
            model_source_signature=args.model_source_signature,
        )
        return 0
    if args.command == "train-transport":
        from .transport.experiment import TransportTrainConfig, train_transport

        train_transport(
            TransportTrainConfig(
                graph_index=args.graph_index,
                teacher=args.teacher,
                output_dir=args.output_dir,
                score_index=args.score_index,
                device=args.device,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                validation_fraction=args.validation_fraction,
                mechanism_holdout_fraction=args.mechanism_holdout_fraction,
                noise_floor=args.noise_floor,
                block_weight=args.block_weight,
                allow_unidentifiable_pilot=args.allow_unidentifiable_pilot,
                seed=args.seed,
                variants=tuple(args.variants),
            )
        )
        return 0
    if args.command == "evaluate-transport":
        from .transport.experiment import evaluate_frozen_transport

        evaluate_frozen_transport(
            predictions_dir=args.predictions_dir,
            test_graph_index=args.test_graph_index,
            output=args.output,
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
