from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate
from .train import ColaConfig, score, train


def main() -> int:
    parser = argparse.ArgumentParser(description="CoLA baseline for RAGTruth attention graphs")
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--subgraph-size", type=int, default=4)
    parser.add_argument("--test-rounds", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = ColaConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        subgraph_size=args.subgraph_size,
        test_rounds=args.test_rounds,
        seed=args.seed,
    )
    checkpoint = train(
        args.graph_root,
        output / "model",
        config=config,
        device=args.device,
    )
    _, response_scores = score(
        args.graph_root,
        checkpoint,
        output,
        device=args.device,
    )
    report = evaluate(
        args.graph_root,
        response_scores,
        output / "evaluation.json",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
