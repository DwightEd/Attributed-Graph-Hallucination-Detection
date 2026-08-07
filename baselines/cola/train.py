from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import ColaGraph, discover_graphs, load_graph
from .sampler import build_batch
from .upstream.model import Model

CHECKPOINT_SCHEMA = "cola-ragtruth-v1"


@dataclass(frozen=True)
class ColaConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 1
    embedding_dim: int = 64
    epochs: int = 100
    batch_size: int = 300
    subgraph_size: int = 4
    readout: str = "avg"
    test_rounds: int = 256
    negsamp_ratio: int = 1


def _resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _atomic_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _batches(indices: list[int], batch_size: int) -> list[list[int]]:
    if not indices:
        return []
    if len(indices) == 1:
        return [indices * 2]
    batches = [
        indices[start : start + batch_size]
        for start in range(0, len(indices), batch_size)
    ]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


def _loss_function(negsamp_ratio: int, device: torch.device) -> nn.Module:
    return nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([negsamp_ratio], device=device),
    )


def _labels(batch_size: int, negsamp_ratio: int, device: torch.device) -> torch.Tensor:
    return torch.unsqueeze(
        torch.cat(
            (
                torch.ones(batch_size),
                torch.zeros(batch_size * negsamp_ratio),
            )
        ),
        1,
    ).to(device)


def train(
    graph_root: str | Path,
    output_dir: str | Path,
    *,
    config: ColaConfig,
    device: str = "cuda",
) -> Path:
    """Train upstream CoLA on RAGTruth train graphs without reading labels."""
    if config.negsamp_ratio != 1:
        raise ValueError("the upstream CoLA anomaly-score code assumes negsamp_ratio=1")
    if config.batch_size < 2:
        raise ValueError("CoLA requires batch_size >= 2 for negative context rotation")

    _set_seed(config.seed)
    device_obj = _resolve_device(device)
    train_paths = discover_graphs(graph_root, "train")
    first = load_graph(train_paths[0])
    model = Model(
        first.feature_dim,
        config.embedding_dim,
        "prelu",
        config.negsamp_ratio,
        config.readout,
    ).to(device_obj)
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    b_xent = _loss_function(config.negsamp_ratio, device_obj)
    rng = random.Random(config.seed)

    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "model.pt"
    history_path = output_dir / "history.json"
    best = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(config.epochs):
        model.train()
        graph_order = list(train_paths)
        rng.shuffle(graph_order)
        total_loss = 0.0
        total_targets = 0

        for graph_path in graph_order:
            graph = load_graph(graph_path)
            if graph.feature_dim != first.feature_dim:
                raise ValueError(f"feature dimension mismatch: {graph_path}")
            targets = list(graph.response_nodes)
            rng.shuffle(targets)
            for batch_targets in _batches(targets, config.batch_size):
                optimiser.zero_grad()
                batch_features, batch_adj = build_batch(
                    graph,
                    batch_targets,
                    subgraph_size=config.subgraph_size,
                    rng=rng,
                    device=device_obj,
                )
                logits = model(batch_features, batch_adj)
                loss_all = b_xent(
                    logits,
                    _labels(len(batch_targets), config.negsamp_ratio, device_obj),
                )
                loss = torch.mean(loss_all)
                loss.backward()
                optimiser.step()

                count = len(batch_targets)
                total_loss += float(loss.detach().cpu()) * count
                total_targets += count

        mean_loss = total_loss / max(total_targets, 1)
        row = {"epoch": epoch + 1, "train_loss": mean_loss, "targets": total_targets}
        history.append(row)
        _write_json(history, history_path)
        print(json.dumps(row, sort_keys=True), flush=True)

        if mean_loss < best:
            best = mean_loss
            best_epoch = epoch
            _atomic_save(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "state_dict": model.state_dict(),
                    "feature_dim": first.feature_dim,
                    "config": asdict(config),
                    "best_epoch": best_epoch,
                    "best_train_loss": best,
                },
                checkpoint_path,
            )

    return checkpoint_path


def load_model(path: str | Path, device: str = "cpu") -> tuple[Model, ColaConfig]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported CoLA checkpoint: {path}")
    config = ColaConfig(**checkpoint["config"])
    model = Model(
        int(checkpoint["feature_dim"]),
        config.embedding_dim,
        "prelu",
        config.negsamp_ratio,
        config.readout,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, config


def score_graph(
    graph: ColaGraph,
    model: Model,
    config: ColaConfig,
    *,
    device: torch.device,
    rng: random.Random,
) -> tuple[list[float], float]:
    """Use the upstream CoLA score and mean response-token aggregation."""
    targets = list(graph.response_nodes)
    rounds = torch.zeros((config.test_rounds, len(targets)), dtype=torch.float32)

    for round_index in range(config.test_rounds):
        order = list(range(len(targets)))
        rng.shuffle(order)
        for positions in _batches(order, config.batch_size):
            original_positions = positions[:1] if len(targets) == 1 else positions
            batch_targets = [targets[position] for position in positions]
            batch_features, batch_adj = build_batch(
                graph,
                batch_targets,
                subgraph_size=config.subgraph_size,
                rng=rng,
                device=device,
            )
            with torch.no_grad():
                logits = torch.squeeze(model(batch_features, batch_adj), dim=-1)
                logits = torch.sigmoid(logits)
            batch_size = len(batch_targets)
            anomaly = -(logits[:batch_size] - logits[batch_size : 2 * batch_size])
            if len(targets) == 1:
                rounds[round_index, 0] = anomaly.detach().cpu()[0]
            else:
                rounds[round_index, original_positions] = anomaly.detach().cpu()

    token_scores = rounds.mean(dim=0).tolist()
    response_score = float(rounds.mean())
    return [float(value) for value in token_scores], response_score


def score(
    graph_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> tuple[Path, Path]:
    device_obj = _resolve_device(device)
    model, config = load_model(checkpoint_path, str(device_obj))
    rng = random.Random(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / "test_token_scores.jsonl"
    response_path = output_dir / "test_response_scores.jsonl"

    token_rows: list[str] = []
    response_rows: list[str] = []
    test_paths = discover_graphs(graph_root, "test")
    for index, graph_path in enumerate(test_paths, 1):
        graph = load_graph(graph_path)
        token_scores, response_score = score_graph(
            graph,
            model,
            config,
            device=device_obj,
            rng=rng,
        )
        for offset, value in enumerate(token_scores):
            token_rows.append(
                json.dumps(
                    {
                        "sample_id": graph.sample_id,
                        "source_id": graph.source_id,
                        "token_index": graph.response_idx + offset,
                        "cola_anomaly_score": value,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        response_rows.append(
            json.dumps(
                {
                    "sample_id": graph.sample_id,
                    "source_id": graph.source_id,
                    "response_tokens": len(token_scores),
                    "cola_anomaly_score": response_score,
                },
                sort_keys=True,
            )
            + "\n"
        )
        print(f"[cola score] {index}/{len(test_paths)} {graph.sample_id}", flush=True)

    token_path.write_text("".join(token_rows), encoding="utf-8")
    response_path.write_text("".join(response_rows), encoding="utf-8")
    return token_path, response_path
