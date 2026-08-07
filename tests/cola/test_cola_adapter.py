from __future__ import annotations

import random
from pathlib import Path

import torch

from baselines.cola.data import load_graph
from baselines.cola.sampler import build_batch, sample_subgraph
from baselines.cola.upstream.model import Model


def _write_graph(path: Path, split: str = "train") -> None:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [2, 2, 3, 3, 4, 4]],
        dtype=torch.long,
    )
    torch.save(
        {
            "schema": "original-ragtruth-attributed-graph-v2",
            "sample_id": path.stem,
            "source_id": "source",
            "split": split,
            "response_idx": 2,
            "x": torch.tensor(
                [
                    [1.0, 1.0, 2.0],
                    [1.0, 2.0, 1.0],
                    [2.0, 1.0, 1.0],
                    [1.0, 3.0, 2.0],
                    [3.0, 1.0, 2.0],
                ]
            ),
            "edge_index": edge_index,
            "response_label": 0,
        },
        path,
    )


def test_upstream_model_and_adapter_layout(tmp_path: Path) -> None:
    path = tmp_path / "sample.graph.pt"
    _write_graph(path)
    graph = load_graph(path)
    nodes = sample_subgraph(graph, 4, 4, random.Random(1))
    assert len(nodes) == 4
    assert nodes[-1] == 4

    features, adjacency = build_batch(
        graph,
        [2, 3, 4],
        subgraph_size=4,
        rng=random.Random(1),
        device=torch.device("cpu"),
    )
    assert features.shape == (3, 5, 3)
    assert adjacency.shape == (3, 5, 5)
    assert torch.all(features[:, -2] == 0)
    assert torch.all(adjacency[:, -1, -1] == 1)

    model = Model(3, 64, "prelu", 1, "avg")
    logits = model(features, adjacency)
    assert logits.shape == (6, 1)
