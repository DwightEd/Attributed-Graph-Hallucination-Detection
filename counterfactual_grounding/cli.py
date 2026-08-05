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
        default=0,
        help="0 disables optional block-rescue calls in the Gate-1 pilot",
    )
    pilot.add_argument("--max-history-blocks", type=int, default=0)
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
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
