from __future__ import annotations

import json
import os
from pathlib import Path

from attention_cache import discover_split

from .config import ExperimentConfig
from .evaluation import evaluate
from .features import prepare_features
from .trainer import score_features, train_ranker


def _write_json(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Prepare topology trajectories, train the ranker, score, and evaluate."""
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(config.to_dict(), output / "config.json")
    status_path = output / "status.json"

    _write_json({"stage": "train_features"}, status_path)
    train_features = prepare_features(
        discover_split(config.attention_root, "train"),
        output / "features" / "train",
        topology=config.topology,
        corruption=config.corruption,
        modes=config.corruption.modes,
        resume=config.resume,
        split="train",
        workers=config.runtime.workers,
        torch_threads=config.runtime.torch_threads,
    )

    _write_json({"stage": "training"}, status_path)
    checkpoint = train_ranker(
        train_features,
        output / "model",
        modes=config.corruption.modes,
        config=config.training,
        device=config.device,
    )

    _write_json({"stage": "test_features"}, status_path)
    test_features = prepare_features(
        discover_split(config.attention_root, "test"),
        output / "features" / "test",
        topology=config.topology,
        corruption=config.corruption,
        modes=(),
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
