from __future__ import annotations

import json
from pathlib import Path

from patf.config import CorruptionConfig, FlowConfig, TrainConfig
from patf.evaluate import evaluate
from patf.features import load_feature, prepare_features
from patf.train import score_features, train_ranker

from .conftest import write_cache


def test_parallel_feature_train_score_and_evaluate(tmp_path: Path) -> None:
    train_dir = tmp_path / "attention" / "train"
    test_dir = tmp_path / "attention" / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    for index in range(6):
        write_cache(
            train_dir / f"attention_{index}.pt",
            sample_id=str(index),
            source_id=f"source-{index}",
            variant=index * 0.02,
        )
    for index in range(2):
        write_cache(
            test_dir / f"attention_{index}.pt",
            sample_id=str(100 + index),
            source_id=f"test-source-{index}",
            variant=index * 0.05,
        )

    flow = FlowConfig()
    corruption = CorruptionConfig()
    train_features = prepare_features(
        sorted(train_dir.glob("*.pt")),
        tmp_path / "features" / "train",
        flow=flow,
        corruption=corruption,
        counterfactual=True,
        resume=True,
        split="train",
        workers=2,
        torch_threads=1,
    )
    test_features = prepare_features(
        sorted(test_dir.glob("*.pt")),
        tmp_path / "features" / "test",
        flow=flow,
        corruption=corruption,
        counterfactual=False,
        resume=True,
        split="test",
        workers=2,
        torch_threads=1,
    )
    assert set(load_feature(train_features[0])["trajectories"]) == {
        "original", "eroded"
    }

    checkpoint = train_ranker(
        train_features,
        tmp_path / "model",
        config=TrainConfig(epochs=2, hidden_dim=8, batch_size=4, patience=2),
        device="cpu",
    )
    predictions = score_features(
        test_features, checkpoint, tmp_path / "predictions.jsonl"
    )
    assert len(predictions) == 2

    ragtruth = tmp_path / "ragtruth"
    ragtruth.mkdir()
    responses = [
        {"id": "100", "source_id": "test-source-0", "labels": [], "model": "m"},
        {"id": "101", "source_id": "test-source-1", "labels": [{"start": 0}], "model": "m"},
    ]
    sources = [
        {"source_id": "test-source-0", "task_type": "QA"},
        {"source_id": "test-source-1", "task_type": "QA"},
    ]
    (ragtruth / "response.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in responses), encoding="utf-8"
    )
    (ragtruth / "source_info.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8"
    )
    report = evaluate(
        tmp_path / "predictions.jsonl",
        ragtruth,
        tmp_path / "evaluation.json",
    )
    assert report["overall"]["samples"] == 2
