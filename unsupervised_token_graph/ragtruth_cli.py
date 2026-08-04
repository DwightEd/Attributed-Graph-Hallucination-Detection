"""Command line entry point for the GPU typed-token-graph experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .ragtruth_pipeline import (
    compact_attention_cache,
    evaluate_score_file,
    score_typed_autoencoder,
    train_typed_autoencoder,
    write_sentence_score_file,
)


def _add_device_store_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--residency",
        choices=("cuda", "stream"),
        default="cuda",
        help="keep compact graphs on GPU, or explicitly stream one graph at a time",
    )
    parser.add_argument(
        "--max-resident-gib",
        type=float,
        default=0.0,
        help="0 lets resident graphs use at most 45%% of currently free GPU memory",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unsupervised typed token graphs from cached RAGTruth attention."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    compact = commands.add_parser(
        "compact",
        help="GPU top-k graph construction from formal sparse CSR or legacy dense attention",
    )
    compact.add_argument("--attention-dir", required=True)
    compact.add_argument("--output-dir", required=True)
    compact.add_argument("--device", default="cuda:0")
    compact.add_argument("--prefix-top-k", type=int, default=8)
    compact.add_argument("--history-top-k", type=int, default=8)
    compact.add_argument("--query-block", type=int, default=64)
    compact.add_argument("--layer-chunk", type=int, default=2)
    compact.add_argument(
        "--storage-dtype", choices=("float16", "float32"), default="float16"
    )
    compact.add_argument("--limit", type=int)
    compact.add_argument("--resume", action="store_true")

    train = commands.add_parser("train", help="label-free typed-neighbourhood MAE")
    train.add_argument("--graph-dir", required=True)
    train.add_argument("--output-dir", required=True)
    _add_device_store_arguments(train)
    train.add_argument("--hidden-dim", type=int, default=192)
    train.add_argument("--num-layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--patience", type=int, default=6)
    train.add_argument("--max-nodes", type=int, default=12_000)
    train.add_argument("--max-edges", type=int, default=192_000)
    train.add_argument("--mask-ratio", type=float, default=0.20)
    train.add_argument("--neighborhood-weight", type=float, default=0.25)
    train.add_argument("--route-weight", type=float, default=0.10)
    train.add_argument("--train-fraction", type=float, default=0.70)
    train.add_argument("--validation-fraction", type=float, default=0.15)
    train.add_argument(
        "--split-policy", choices=("official", "source_random"), default="official",
        help="official holds out RAGTruth test; source_random is legacy exploratory mode",
    )
    train.add_argument(
        "--amp", choices=("none", "bfloat16", "float16"), default="bfloat16"
    )
    train.add_argument("--seed", type=int, default=42)

    score = commands.add_parser("score", help="score every held-out response token")
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--graph-dir", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--split-file")
    score.add_argument(
        "--partition", choices=("train", "validation", "test"), default="test"
    )
    _add_device_store_arguments(score)
    score.add_argument("--mask-stride", type=int, default=8)
    score.add_argument("--calibration-limit", type=int)
    score.add_argument("--calibration-max-tokens", type=int, default=200_000)
    score.add_argument(
        "--amp", choices=("none", "bfloat16", "float16"), default="bfloat16"
    )

    sentence = commands.add_parser(
        "sentences", help="pool frozen token scores into label-free sentence scores"
    )
    sentence.add_argument("--scores", required=True)
    sentence.add_argument("--attention-dir", required=True)
    sentence.add_argument("--graph-dir", required=True)
    sentence.add_argument("--responses", required=True)
    sentence.add_argument("--sources", required=True)
    sentence.add_argument("--tokenizer", required=True)
    sentence.add_argument("--output", required=True)
    sentence.add_argument("--top-fraction", type=float, default=0.20)

    evaluate = commands.add_parser(
        "evaluate", help="join cached labels after scores are frozen"
    )
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--attention-dir", required=True)
    evaluate.add_argument("--graph-dir")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--sentence-scores")
    evaluate.add_argument(
        "--label-shift",
        type=int,
        default=0,
        help="explicit cached-label correction; positive values move labels right",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compact":
        result = compact_attention_cache(
            args.attention_dir,
            args.output_dir,
            device=args.device,
            prefix_top_k=args.prefix_top_k,
            history_top_k=args.history_top_k,
            query_block=args.query_block,
            layer_chunk=args.layer_chunk,
            storage_dtype=args.storage_dtype,
            limit=args.limit,
            resume=args.resume,
        )
        print(
            f"compact complete: samples={result['samples']} nodes={result['nodes']} "
            f"edges={result['edges']} output={Path(args.output_dir).resolve()}"
        )
    elif args.command == "train":
        result = train_typed_autoencoder(
            args.graph_dir,
            args.output_dir,
            device=args.device,
            residency=args.residency,
            max_resident_gib=args.max_resident_gib,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            mask_ratio=args.mask_ratio,
            neighborhood_weight=args.neighborhood_weight,
            route_weight=args.route_weight,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            split_policy=args.split_policy,
            amp=args.amp,
            seed=args.seed,
        )
        print(
            f"training complete: best_validation={result['best_validation_loss']:.5f} "
            f"checkpoint={result['checkpoint']}"
        )
    elif args.command == "score":
        result = score_typed_autoencoder(
            args.checkpoint,
            args.graph_dir,
            args.output,
            split_file=args.split_file,
            partition=args.partition,
            device=args.device,
            residency=args.residency,
            max_resident_gib=args.max_resident_gib,
            mask_stride=args.mask_stride,
            calibration_limit=args.calibration_limit,
            calibration_max_tokens=args.calibration_max_tokens,
            amp=args.amp,
        )
        print(
            f"scoring complete: graphs={result['graphs']} tokens={result['tokens']} "
            f"output={result['scores']}"
        )
    elif args.command == "sentences":
        result = write_sentence_score_file(
            args.scores,
            args.attention_dir,
            args.graph_dir,
            args.responses,
            args.sources,
            args.tokenizer,
            args.output,
            top_fraction=args.top_fraction,
        )
        print(
            f"sentence scoring complete: responses={result['responses']} "
            f"sentences={result['sentences']} output={result['scores']}"
        )
    else:
        result = evaluate_score_file(
            args.scores,
            args.attention_dir,
            args.output,
            label_shift=args.label_shift,
            graph_dir=args.graph_dir,
            sentence_score_path=args.sentence_scores,
        )
        token = result["token"]
        sample = result["sample_max"]
        print(
            f"token: AUROC={token['auroc']:.4f} AUPRC={token['auprc']:.4f} "
            f"lift={token['auprc_lift']:.2f}x n={token['token_count']}"
        )
        print(
            f"sample-max: AUROC={sample['auroc']:.4f} "
            f"AUPRC={sample['auprc']:.4f} n={sample['sample_count']}"
        )
        if "sentence" in result:
            sentence = result["sentence"]
            print(
                f"sentence ({sentence['pooling']}): AUROC={sentence['auroc']:.4f} "
                f"AUPRC={sentence['auprc']:.4f} n={sentence['sentence_count']}"
            )
        for name, metrics in result["components"].items():
            print(
                f"{name}: AUROC={metrics['auroc']:.4f} "
                f"AUPRC={metrics['auprc']:.4f}"
            )
        print(f"report={Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
