"""Runnable wrapper around the upstream supervised CHARM token classifier."""

from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import get_cosine_schedule_with_warmup

from train_charm_grid import CHARM

from .ragtruth_graph import (
    DATASET_SCHEMA,
    GRAPH_SCHEMA,
    UPSTREAM_COMMIT,
    _atomic_text,
    _atomic_torch_save,
    validate_original_graph,
)

TRAINING_SCHEMA = "original-charm-supervised-token-v1"


def _load_mapping(path: Path) -> Mapping[str, object]:
    graph = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(graph, Mapping):
        raise TypeError(f"graph must contain a mapping: {path}")
    return graph


def _graph_to_data(graph: Mapping[str, object], path: Path) -> Data:
    try:
        validate_original_graph(graph)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"invalid {GRAPH_SCHEMA} artifact {path}: {error}") from error
    x = torch.as_tensor(graph["x"]).detach().float()
    edge_index = torch.as_tensor(graph["edge_index"]).detach().long()
    edge_attr = torch.as_tensor(graph["edge_attr"]).detach().float()
    edge_mark = torch.as_tensor(graph["edge_mark"]).detach().float()
    labels = torch.as_tensor(graph["y_token"]).detach().float().flatten()
    response_idx = int(graph["response_idx"])
    node_count = int(x.shape[0])
    edge_count = int(edge_index.shape[1])

    # This deliberately preserves the upstream degree definition, including
    # its source-node normalization convention.
    degree = torch.zeros(node_count, dtype=torch.float32)
    if edge_count:
        degree.index_add_(0, edge_index[0], torch.ones(edge_count))
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_mark=edge_mark,
        y=labels,
        deg_out=degree,
        node_pos=torch.arange(node_count, dtype=torch.long),
        response_idx=torch.tensor(response_idx, dtype=torch.long),
    )


def _dataset_contract(graph_root: Path) -> tuple[Mapping[str, object], dict[str, list[Path]]]:
    manifest_path = graph_root / "manifest.json"
    index_path = graph_root / "index.jsonl"
    if not manifest_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("original graph dataset requires manifest.json and index.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"invalid original graph manifest: {manifest_path}")
    paths_by_split: dict[str, list[Path]] = {"train": [], "test": []}
    observed: set[Path] = set()
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise TypeError(f"index row {line_number} must contain an object")
        split = str(row.get("split", "")).strip().casefold()
        relative = Path(str(row.get("graph_path", "")))
        if split not in paths_by_split or relative.is_absolute():
            raise ValueError(f"invalid graph index row {line_number}")
        resolved = (graph_root / relative).resolve()
        if not resolved.is_relative_to(graph_root) or resolved in observed:
            raise ValueError(f"unsafe or duplicate graph path in index row {line_number}")
        if not resolved.is_file():
            raise FileNotFoundError(f"indexed graph is absent: {resolved}")
        observed.add(resolved)
        paths_by_split[split].append(resolved)
    return manifest, paths_by_split


def _load_split(paths: Sequence[Path], split: str) -> list[Data]:
    if not paths:
        raise ValueError(f"no indexed original attributed graphs found for {split}")
    graphs: list[Data] = []
    for path in paths:
        graph = _load_mapping(path)
        if str(graph.get("split", "")).strip().casefold() != split:
            raise ValueError(f"indexed graph split disagrees with graph payload: {path}")
        graphs.append(_graph_to_data(graph, path))
    return graphs


def _response_label(data: Data) -> int:
    return int(data.y[int(data.response_idx) :].any().item())


def _stratified_validation_split(
    graphs: Sequence[Data],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[Data], list[Data]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if len(graphs) < 4:
        raise ValueError("at least four train graphs are required")
    positive = [index for index, graph in enumerate(graphs) if _response_label(graph)]
    negative = [index for index, graph in enumerate(graphs) if not _response_label(graph)]
    if not positive or not negative:
        raise ValueError("train graphs must contain correct and hallucinated responses")
    validation_size = max(1, int(len(graphs) * validation_fraction))
    positive_count = int(len(positive) / len(graphs) * validation_size)
    negative_count = validation_size - positive_count
    if positive_count < 1 or negative_count < 1:
        raise ValueError("validation split is too small to preserve both response classes")
    rng = random.Random(seed)
    validation_indices = set(
        rng.sample(positive, min(positive_count, len(positive)))
        + rng.sample(negative, min(negative_count, len(negative)))
    )
    training = [graph for index, graph in enumerate(graphs) if index not in validation_indices]
    validation = [graph for index, graph in enumerate(graphs) if index in validation_indices]
    if not training or not validation:
        raise ValueError("train/validation split produced an empty partition")
    return training, validation


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _response_arrays(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probability = torch.sigmoid(model(batch))
            mask = batch.node_pos >= batch.response_idx[batch.batch]
            labels.append(batch.y[mask].detach().cpu().numpy())
            scores.append(probability[mask].detach().cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


def _metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
    labels, scores = _response_arrays(model, loader, device)
    both_classes = len(np.unique(labels)) == 2
    return {
        "samples": int(labels.size),
        "positive_count": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)) if both_classes else 0.5,
        "auprc": float(average_precision_score(labels, scores)) if both_classes else 0.0,
    }


def _positive_weight(graphs: Sequence[Data], device: torch.device) -> torch.Tensor:
    labels = torch.cat([graph.y[int(graph.response_idx) :] for graph in graphs])
    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())
    if positive == 0 or negative == 0:
        raise ValueError("response training tokens must contain both classes")
    return torch.tensor(min(negative / positive, 8.0), device=device)


def _train_seed(
    *,
    train_graphs: Sequence[Data],
    validation_graphs: Sequence[Data],
    test_graphs: Sequence[Data],
    output_dir: Path,
    seed: int,
    device: torch.device,
    hp: Mapping[str, object],
) -> dict[str, float | int]:
    _set_seed(seed)
    train_loader = DataLoader(train_graphs, batch_size=int(hp["batch_size"]), shuffle=True)
    validation_loader = DataLoader(
        validation_graphs, batch_size=int(hp["batch_size"]), shuffle=False
    )
    test_loader = DataLoader(test_graphs, batch_size=int(hp["batch_size"]), shuffle=False)
    node_dim = int(train_graphs[0].x.shape[1])
    edge_dim = int(train_graphs[0].edge_attr.shape[1])
    model = CHARM(node_dim, edge_dim, dict(hp)).to(device)
    optimizer = AdamW(
        model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"])
    )
    total_steps = int(hp["epochs"]) * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=_positive_weight(train_graphs, device)
    )

    best_state: Mapping[str, torch.Tensor] | None = None
    best_validation_auprc = -math.inf
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(hp["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch)
            mask = batch.node_pos >= batch.response_idx[batch.batch]
            loss = loss_function(logits[mask], batch.y[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach())
        validation = _metrics(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_loader)),
            "validation_auroc": float(validation["auroc"]),
            "validation_auprc": float(validation["auprc"]),
        }
        history.append(row)
        print(json.dumps({"event": "epoch", "seed": seed, **row}), flush=True)
        if float(validation["auprc"]) > best_validation_auprc:
            best_validation_auprc = float(validation["auprc"])
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(hp["patience"]):
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test = _metrics(model, test_loader, device)
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_text(
        output_dir / "history.json",
        json.dumps(history, indent=2, sort_keys=True) + "\n",
    )
    _atomic_torch_save(
        output_dir / "checkpoint.pt",
        {
            "schema": TRAINING_SCHEMA,
            "upstream_commit": UPSTREAM_COMMIT,
            "seed": seed,
            "hyperparameters": dict(hp),
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "model_state": best_state,
            "test_metrics": test,
        },
    )
    return {
        "seed": seed,
        "best_validation_auprc": best_validation_auprc,
        "epochs_ran": len(history),
        "test_samples": int(test["samples"]),
        "test_positive_count": int(test["positive_count"]),
        "test_auroc": float(test["auroc"]),
        "test_auprc": float(test["auprc"]),
    }


def run_training(
    graph_root: str | Path,
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    validation_fraction: float = 0.2,
    split_seed: int = 42,
    epochs: int = 50,
    patience: int = 5,
    batch_size: int = 32,
    hidden_dim: int = 128,
    gnn_layers: int = 3,
    lr: float = 0.0005,
    dropout: float = 0.25,
    weight_decay: float = 0.001,
    device: str = "cuda",
    allow_partial_cache: bool = False,
) -> dict[str, object]:
    """Train the upstream supervised token classifier on persisted graphs."""

    dataset_root = Path(graph_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"training output directory is not empty: {destination}")
    if not seeds or len({int(seed) for seed in seeds}) != len(seeds):
        raise ValueError("seeds must be a non-empty unique sequence")
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")

    graph_manifest, indexed_paths = _dataset_contract(dataset_root)
    graph_scope = str(graph_manifest.get("experiment_scope", "unrecorded"))
    if graph_scope != "official_complete_cache" and not allow_partial_cache:
        raise RuntimeError(
            "original CHARM training requires graph scope "
            f"official_complete_cache, got {graph_scope}; explicitly allow a "
            "partial-cache pilot to continue"
        )
    all_train = _load_split(indexed_paths["train"], "train")
    test_graphs = _load_split(indexed_paths["test"], "test")
    train_graphs, validation_graphs = _stratified_validation_split(
        all_train, validation_fraction=validation_fraction, seed=split_seed
    )
    hp: dict[str, object] = {
        "lr": float(lr),
        "scheduler": "cosine",
        "batch_size": int(batch_size),
        "dropout": float(dropout),
        "hidden_dim": int(hidden_dim),
        "gnn_layers": int(gnn_layers),
        "weight_decay": float(weight_decay),
        "norm": "layer",
        "residual_encoder": True,
        "residual_mp": True,
        "epochs": int(epochs),
        "patience": int(patience),
    }
    per_seed = [
        _train_seed(
            train_graphs=train_graphs,
            validation_graphs=validation_graphs,
            test_graphs=test_graphs,
            output_dir=destination / f"seed_{int(seed)}",
            seed=int(seed),
            device=requested_device,
            hp=hp,
        )
        for seed in seeds
    ]
    test_aurocs = np.asarray([row["test_auroc"] for row in per_seed], dtype=float)
    test_auprcs = np.asarray([row["test_auprc"] for row in per_seed], dtype=float)
    summary: dict[str, object] = {
        "schema": TRAINING_SCHEMA,
        "method": "upstream_CHARM_supervised_token_classification",
        "upstream_commit": UPSTREAM_COMMIT,
        "graph_root": str(dataset_root),
        "graph_experiment_scope": graph_scope,
        "partial_cache_explicitly_allowed": bool(allow_partial_cache),
        "graph_tau": graph_manifest.get("tau"),
        "seeds": [int(seed) for seed in seeds],
        "split_seed": int(split_seed),
        "train_graphs": len(train_graphs),
        "validation_graphs": len(validation_graphs),
        "test_graphs": len(test_graphs),
        "test_samples": int(per_seed[0]["test_samples"]),
        "hyperparameters": hp,
        "per_seed": per_seed,
        "mean_test_auroc": float(test_aurocs.mean()),
        "std_test_auroc": float(test_aurocs.std()),
        "mean_test_auprc": float(test_auprcs.mean()),
        "std_test_auprc": float(test_auprcs.std()),
    }
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        destination / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary
