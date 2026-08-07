from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from attention_cache import discover_split
from patf.config import ExperimentConfig
from patf.evaluate import evaluate
from patf.features import prepare_features
from patf.train import score_features, train_ranker


def _write_json(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(config: ExperimentConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(config.to_dict(), output / "config.json")
    status_path = output / "status.json"

    _write_json({"stage": "train_features"}, status_path)
    train_features = prepare_features(
        discover_split(config.attention_root, "train"),
        output / "features" / "train",
        flow=config.flow,
        corruption=config.corruption,
        counterfactual=True,
        resume=config.resume,
        split="train",
        workers=config.runtime.workers,
        torch_threads=config.runtime.torch_threads,
    )

    _write_json({"stage": "training"}, status_path)
    checkpoint = train_ranker(
        train_features,
        output / "model",
        config=config.training,
        device=config.device,
    )

    _write_json({"stage": "test_features"}, status_path)
    test_features = prepare_features(
        discover_split(config.attention_root, "test"),
        output / "features" / "test",
        flow=config.flow,
        corruption=config.corruption,
        counterfactual=False,
        resume=config.resume,
        split="test",
        workers=config.runtime.workers,
        torch_threads=config.runtime.torch_threads,
    )

    _write_json({"stage": "scoring"}, status_path)
    prediction_path = output / "predictions.jsonl"
    score_features(test_features, checkpoint, prediction_path)

    _write_json({"stage": "evaluation"}, status_path)
    report = evaluate(
        prediction_path,
        config.ragtruth_root,
        output / "evaluation.json",
    )
    _write_json({"stage": "complete"}, status_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt-Anchored Topology Flow")
    parser.add_argument("--config", default="configs/patf.json")
    parser.add_argument("--attention-root", required=True)
    parser.add_argument("--ragtruth-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

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

    print(json.dumps(run(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
