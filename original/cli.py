"""Command-line entry points for the preserved upstream baseline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from attention_graph.data import audit_attention_cache

from .ragtruth_graph import prepare_original_graphs
from .train import run_training


def _build(args: argparse.Namespace) -> int:
    def progress(current: int, total: int) -> None:
        if current == 1 or current == total or current % 25 == 0:
            print(
                json.dumps(
                    {"event": "graph_progress", "current": current, "total": total}
                ),
                flush=True,
            )

    records = prepare_original_graphs(
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        tau=args.tau,
        splits=args.splits,
        device=args.device,
        resume=args.resume,
        limit=args.limit,
        require_complete_cache=args.require_complete_cache,
        progress_callback=progress,
    )
    split_counts = {
        split: sum(row["split"] == split for row in records) for split in args.splits
    }
    print(
        json.dumps(
            {
                "event": "graph_preparation_complete",
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "graphs": len(records),
                "split_counts": split_counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    summary = run_training(
        graph_root=args.graph_root,
        output_dir=args.output_dir,
        seeds=args.seeds,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        lr=args.lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        device=args.device,
        allow_partial_cache=args.allow_partial_cache,
    )
    print(
        json.dumps(
            {
                "event": "training_complete",
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "mean_test_auroc": summary["mean_test_auroc"],
                "mean_test_auprc": summary["mean_test_auprc"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _audit(args: argparse.Namespace) -> int:
    report = audit_attention_cache(args.cache_root, splits=args.splits)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if all(row["complete"] for row in report.values()) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the original attributed-graph RAGTruth baseline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="check formal cache completeness")
    audit.add_argument("--cache-root", type=Path, required=True)
    audit.add_argument(
        "--splits", nargs="+", choices=("train", "test"), default=("train", "test")
    )
    audit.set_defaults(handler=_audit)

    build = subparsers.add_parser("build", help="persist original-format token graphs")
    build.add_argument("--cache-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--tau", type=float, default=0.05)
    build.add_argument("--splits", nargs="+", choices=("train", "test"), default=("train", "test"))
    build.add_argument("--device", default="cuda")
    build.add_argument("--limit", type=int)
    build.add_argument("--require-complete-cache", action="store_true")
    build.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    build.set_defaults(handler=_build)

    train = subparsers.add_parser("train", help="run upstream supervised CHARM")
    train.add_argument("--graph-root", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--split-seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--gnn-layers", type=int, default=3)
    train.add_argument("--lr", type=float, default=0.0005)
    train.add_argument("--dropout", type=float, default=0.25)
    train.add_argument("--weight-decay", type=float, default=0.001)
    train.add_argument("--device", default="cuda")
    train.add_argument("--allow-partial-cache", action="store_true")
    train.set_defaults(handler=_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
