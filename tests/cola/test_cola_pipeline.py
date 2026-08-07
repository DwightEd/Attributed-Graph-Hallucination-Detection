from __future__ import annotations

from pathlib import Path

import torch

from baselines.cola.evaluate import evaluate
from baselines.cola.train import ColaConfig, score, train


def _write_graph(path: Path, split: str, label: int, offset: float) -> None:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 0], [2, 2, 3, 3, 4, 4, 4]],
        dtype=torch.long,
    )
    torch.save(
        {
            "schema": "original-ragtruth-attributed-graph-v2",
            "sample_id": path.stem,
            "source_id": path.stem,
            "split": split,
            "response_idx": 2,
            "x": torch.rand(5, 4) + offset + 0.1,
            "edge_index": edge_index,
            "response_label": label,
        },
        path,
    )


def test_cola_train_score_evaluate(tmp_path: Path) -> None:
    for split, count in (("train", 3), ("test", 2)):
        directory = tmp_path / "graphs" / split
        directory.mkdir(parents=True)
        for index in range(count):
            _write_graph(
                directory / f"{split}-{index}.graph.pt",
                split,
                int(split == "test" and index == 1),
                0.05 * index,
            )

    output = tmp_path / "out"
    checkpoint = train(
        tmp_path,
        output / "model",
        config=ColaConfig(epochs=1, batch_size=3, test_rounds=2),
        device="cpu",
    )
    token_scores, response_scores = score(
        tmp_path,
        checkpoint,
        output,
        device="cpu",
    )
    report = evaluate(tmp_path, response_scores, output / "evaluation.json")
    assert checkpoint.is_file()
    assert token_scores.is_file()
    assert response_scores.is_file()
    assert report["samples"] == 2
