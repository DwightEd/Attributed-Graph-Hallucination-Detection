from __future__ import annotations

import json
from pathlib import Path

from .config import ExperimentConfig
from .data import discover_split
from .evaluation import evaluate
from .features import prepare_features
from .trainer import score_features, train_ranker


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Run feature preparation, label-free training, scoring, and evaluation."""
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    train_features = prepare_features(
        discover_split(config.attention_root, "train"),
        output / "features" / "train",
        topology=config.topology,
        corruption=config.corruption,
        modes=config.corruption.modes,
        resume=config.resume,
        split="train",
    )
    checkpoint = train_ranker(
        train_features,
        output / "model",
        modes=config.corruption.modes,
        config=config.training,
        device=config.device,
    )
    test_features = prepare_features(
        discover_split(config.attention_root, "test"),
        output / "features" / "test",
        topology=config.topology,
        corruption=config.corruption,
        modes=(),
        resume=config.resume,
        split="test",
    )
    prediction_path = output / "predictions.jsonl"
    score_features(test_features, checkpoint, prediction_path)
    return evaluate(prediction_path, config.ragtruth_root, output / "evaluation.json")
