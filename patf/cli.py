from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import ExperimentConfig
from .experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt-Anchored Topology Flow")
    parser.add_argument("--config", required=True)
    parser.add_argument("--attention-root", required=True)
    parser.add_argument("--ragtruth-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    raw.update(
        attention_root=args.attention_root,
        ragtruth_root=args.ragtruth_root,
        output_dir=args.output_dir,
    )
    config = ExperimentConfig.from_mapping(raw)
    if args.device:
        config = replace(config, device=args.device)
    if args.epochs is not None:
        config = replace(
            config,
            training=replace(config.training, epochs=args.epochs),
        )
    if args.workers is not None:
        config = replace(
            config,
            runtime=replace(config.runtime, workers=args.workers),
        )
    if args.torch_threads is not None:
        config = replace(
            config,
            runtime=replace(config.runtime, torch_threads=args.torch_threads),
        )
    if args.no_resume:
        config = replace(config, resume=False)

    print(json.dumps(run_experiment(config), indent=2, sort_keys=True))
    return 0
