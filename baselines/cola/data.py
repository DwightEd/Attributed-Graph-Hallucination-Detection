from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch

GRAPH_SCHEMA = "original-ragtruth-attributed-graph-v2"


@dataclass(frozen=True)
class ColaGraph:
    path: Path
    sample_id: str
    source_id: str
    split: str
    response_idx: int
    features: torch.Tensor
    neighbors: tuple[tuple[int, ...], ...]
    degrees: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def response_nodes(self) -> range:
        return range(self.response_idx, self.num_nodes)

    def normalized_subgraph_adjacency(self, nodes: list[int]) -> torch.Tensor:
        size = len(nodes)
        adjacency = torch.zeros((size, size), dtype=torch.float32)
        positions: dict[int, list[int]] = {}
        for position, node in enumerate(nodes):
            positions.setdefault(node, []).append(position)
        for left, node in enumerate(nodes):
            degree_i = float(self.degrees[node])
            if degree_i <= 0:
                continue
            for neighbor in self.neighbors[node]:
                right_positions = positions.get(neighbor)
                if not right_positions:
                    continue
                degree_j = float(self.degrees[neighbor])
                if degree_j <= 0:
                    continue
                value = 1.0 / (degree_i * degree_j) ** 0.5
                for right in right_positions:
                    adjacency[left, right] = value
        adjacency += torch.eye(size)
        return adjacency


def _row_normalize(features: torch.Tensor) -> torch.Tensor:
    features = features.float()
    row_sum = features.sum(dim=1, keepdim=True)
    return torch.where(row_sum != 0, features / row_sum, torch.zeros_like(features))


def _undirected_neighbors(
    edge_index: torch.Tensor, num_nodes: int
) -> tuple[tuple[int, ...], ...]:
    sets = [set() for _ in range(num_nodes)]
    if edge_index.numel():
        source = edge_index[0].long().tolist()
        target = edge_index[1].long().tolist()
        for u, v in zip(source, target):
            if u == v:
                continue
            sets[u].add(v)
            sets[v].add(u)
    return tuple(tuple(sorted(values)) for values in sets)


def load_graph(path: str | Path) -> ColaGraph:
    path = Path(path)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping) or raw.get("schema") != GRAPH_SCHEMA:
        raise ValueError(f"unsupported attributed graph: {path}")

    features = torch.as_tensor(raw["x"], dtype=torch.float32)
    edge_index = torch.as_tensor(raw["edge_index"], dtype=torch.long)
    response_idx = int(raw["response_idx"])
    if features.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"invalid attributed graph tensors: {path}")
    if not 0 < response_idx < features.shape[0]:
        raise ValueError(f"invalid response_idx: {path}")

    neighbors = _undirected_neighbors(edge_index, int(features.shape[0]))
    degrees = torch.tensor([len(values) for values in neighbors], dtype=torch.float32)
    return ColaGraph(
        path=path,
        sample_id=str(raw.get("sample_id", raw.get("response_id", path.stem))),
        source_id=str(raw.get("source_id", "unknown")),
        split=str(raw.get("split", path.parent.name)),
        response_idx=response_idx,
        features=_row_normalize(features),
        neighbors=neighbors,
        degrees=degrees,
    )


def discover_graphs(graph_root: str | Path, split: str) -> list[Path]:
    directory = Path(graph_root) / "graphs" / split
    paths = sorted(directory.glob("*.graph.pt"))
    if not paths:
        raise FileNotFoundError(f"no *.graph.pt files in {directory}")
    return paths


def iter_graphs(paths: Iterable[Path]) -> Iterable[ColaGraph]:
    for path in paths:
        yield load_graph(path)
