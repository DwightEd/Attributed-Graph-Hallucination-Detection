"""Label-blind training utilities for the masked token-graph autoencoder."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
import random

import torch

from .model import (
    TokenGraphMaskedAutoencoder,
    make_answer_block_mask,
    masked_reconstruction_loss,
)


_FORBIDDEN_TRAINING_KEYS = {
    "candidate_role",
    "candidate_type",
    "gold_answer",
    "is_correct",
    "is_hallucinated",
    "label",
    "labels",
    "target",
    "y",
    "y_token",
}


def validate_label_free_graph(value, path: str = "graph") -> None:
    """Reject evaluation fields recursively before any optimizer sees a graph."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN_TRAINING_KEYS:
                raise ValueError(f"evaluation field {path}.{key} is forbidden in training")
            validate_label_free_graph(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            validate_label_free_graph(nested, f"{path}[{index}]")


def collate_token_graphs(graphs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Batch token graphs by concatenating nodes and offsetting causal edges."""

    if not graphs:
        raise ValueError("at least one graph is required")
    for graph in graphs:
        validate_label_free_graph(graph)
    node_counts = [len(torch.as_tensor(graph["x"])) for graph in graphs]
    offsets = [0]
    for count in node_counts:
        offsets.append(offsets[-1] + count)
    edge_indices = []
    for graph, offset in zip(graphs, offsets[:-1]):
        edge_indices.append(torch.as_tensor(graph["edge_index"], dtype=torch.long) + offset)
    return {
        "x": torch.cat([torch.as_tensor(graph["x"]).float() for graph in graphs]),
        "segment_ids": torch.cat(
            [torch.as_tensor(graph["segment_ids"]).long() for graph in graphs]
        ),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_attr": torch.cat(
            [torch.as_tensor(graph["edge_attr"]).float() for graph in graphs]
        ),
        "edge_mark": torch.cat(
            [torch.as_tensor(graph["edge_mark"]).float() for graph in graphs]
        ),
        "graph_ptr": torch.tensor(offsets, dtype=torch.long),
        "example_ids": [str(graph["example_id"]) for graph in graphs],
        "pair_ids": [str(graph.get("pair_id", graph["example_id"])) for graph in graphs],
    }


def _to_model_device(model, batch: Mapping[str, object]) -> dict[str, object]:
    device = next(model.parameters()).device
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _make_batch_mask(batch, mask_ratio, generator):
    mask = torch.zeros_like(batch["segment_ids"], dtype=torch.bool)
    pointers = batch["graph_ptr"].tolist()
    for start, end in zip(pointers[:-1], pointers[1:]):
        local_mask = make_answer_block_mask(
            batch["segment_ids"][start:end],
            mask_ratio=mask_ratio,
            generator=generator,
        )
        mask[start:end] = local_mask.to(mask.device)
    return mask


def reconstruction_step(
    model: TokenGraphMaskedAutoencoder,
    batch: Mapping[str, object],
    optimizer: torch.optim.Optimizer,
    *,
    mask_ratio: float,
    trim_fraction: float,
    generator: torch.Generator | None = None,
) -> float:
    """Optimize one batch using only masked answer-node reconstruction."""

    model.train()
    values = _to_model_device(model, batch)
    masked_nodes = _make_batch_mask(values, mask_ratio, generator)
    predictions = model(
        values["x"],
        values["edge_index"],
        values["edge_attr"],
        values["edge_mark"],
        masked_nodes,
    )
    loss = masked_reconstruction_loss(
        predictions,
        values["x"],
        masked_nodes,
        trim_fraction=trim_fraction,
        graph_ptr=values["graph_ptr"],
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def reconstruction_batch_loss(
    model: TokenGraphMaskedAutoencoder,
    batch: Mapping[str, object],
    *,
    mask_ratio: float,
    trim_fraction: float,
    generator: torch.Generator | None = None,
) -> float:
    """Measure the same label-free objective without changing model weights."""

    values = _to_model_device(model, batch)
    masked_nodes = _make_batch_mask(values, mask_ratio, generator)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        predictions = model(
            values["x"],
            values["edge_index"],
            values["edge_attr"],
            values["edge_mark"],
            masked_nodes,
        )
        loss = masked_reconstruction_loss(
            predictions,
            values["x"],
            masked_nodes,
            trim_fraction=trim_fraction,
            graph_ptr=values["graph_ptr"],
        )
    model.train(was_training)
    return float(loss.cpu())


def score_graph(
    model: TokenGraphMaskedAutoencoder,
    graph: Mapping[str, object],
    *,
    block_size: int = 4,
) -> dict[str, object]:
    """Cover every answer token with deterministic masks and average residuals."""

    if block_size < 1:
        raise ValueError("block_size must be positive")
    validate_label_free_graph(graph)
    batch = _to_model_device(model, collate_token_graphs([graph]))
    answer_indices = torch.nonzero(
        batch["segment_ids"] == 3, as_tuple=False
    ).flatten()
    if not len(answer_indices):
        raise ValueError("the graph has no answer tokens")
    was_training = model.training
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(answer_indices), block_size):
            selected = answer_indices[start : start + block_size]
            mask = torch.zeros(len(batch["x"]), dtype=torch.bool, device=batch["x"].device)
            mask[selected] = True
            prediction = model(
                batch["x"],
                batch["edge_index"],
                batch["edge_attr"],
                batch["edge_mark"],
                mask,
            )
            residual = (prediction[selected] - batch["x"][selected]).square().mean(dim=-1)
            scores.extend(float(value) for value in residual.cpu())
    model.train(was_training)
    return {
        "example_id": str(graph["example_id"]),
        "anomaly_score": float(sum(scores) / len(scores)),
        "answer_node_indices": [int(value) for value in answer_indices.cpu()],
        "answer_node_scores": scores,
    }


def _load_graph(path: str | Path) -> dict[str, object]:
    graph = torch.load(path, map_location="cpu", weights_only=True)
    validate_label_free_graph(graph)
    return graph


def _stable_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\x1f{value}".encode("utf-8")).hexdigest()


def _split_paths_by_pair(
    paths: Sequence[Path],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    if not 0.0 < validation_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("validation_fraction and test_fraction must be between zero and one")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below one")
    pair_to_paths: dict[str, list[Path]] = {}
    for path in paths:
        graph = _load_graph(path)
        pair_id = str(graph.get("pair_id", graph["example_id"]))
        pair_to_paths.setdefault(pair_id, []).append(path)
    if len(pair_to_paths) < 3:
        raise ValueError("at least three pair groups are required for train/validation/test")
    ordered = sorted(pair_to_paths, key=lambda value: _stable_order(value, seed))
    test_count = max(1, round(len(ordered) * test_fraction))
    validation_count = min(
        len(ordered) - test_count - 1,
        max(1, round(len(ordered) * validation_fraction)),
    )
    test_count = min(test_count, len(ordered) - validation_count - 1)
    test_pairs = set(ordered[:test_count])
    validation_pairs = set(ordered[test_count : test_count + validation_count])
    train_paths = [
        path
        for pair_id, pair_paths in pair_to_paths.items()
        if pair_id not in validation_pairs and pair_id not in test_pairs
        for path in pair_paths
    ]
    validation_paths = [
        path
        for pair_id, pair_paths in pair_to_paths.items()
        if pair_id in validation_pairs
        for path in pair_paths
    ]
    test_paths = [
        path
        for pair_id, pair_paths in pair_to_paths.items()
        if pair_id in test_pairs
        for path in pair_paths
    ]
    return sorted(train_paths), sorted(validation_paths), sorted(test_paths)


def _path_batches(paths: Sequence[Path], batch_size: int):
    for start in range(0, len(paths), batch_size):
        yield paths[start : start + batch_size]


def _discover_graph_paths(graph_dir: str | Path) -> list[Path]:
    """Use the extraction manifest so orphaned cache files cannot enter training."""

    directory = Path(graph_dir)
    manifest_path = directory.parent / "extraction_manifest.json"
    if not manifest_path.exists():
        return sorted(directory.glob("*.pt"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "complete":
        raise RuntimeError(f"Extraction manifest is not complete: {manifest_path}")
    file_names = manifest.get("graph_files")
    if not isinstance(file_names, list) or not file_names:
        raise RuntimeError(f"Extraction manifest has no graph_files: {manifest_path}")
    paths = []
    for file_name in file_names:
        candidate_name = Path(str(file_name))
        if candidate_name.name != str(file_name):
            raise RuntimeError(f"Unsafe graph filename in manifest: {file_name!r}")
        path = directory / candidate_name
        if not path.is_file():
            raise RuntimeError(f"Manifest graph is missing: {path}")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise RuntimeError("Extraction manifest contains duplicate graph files")
    return paths


def run_training(
    graph_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    mask_ratio: float,
    trim_fraction: float,
    validation_fraction: float,
    patience: int,
    seed: int,
    score_block_size: int,
    mask_incident_edge_attrs: bool = True,
    test_fraction: float = 0.2,
) -> dict[str, object]:
    """Train, early-stop, and score without ever opening an evaluation label file."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    paths = _discover_graph_paths(graph_dir)
    if not paths:
        raise ValueError(f"No .pt token graphs found in {graph_dir}")
    train_paths, validation_paths, test_paths = _split_paths_by_pair(
        paths,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    sample = _load_graph(train_paths[0])
    model = TokenGraphMaskedAutoencoder(
        node_dim=int(sample["x"].shape[1]),
        edge_dim=int(sample["edge_attr"].shape[1]),
        edge_mark_dim=int(sample["edge_mark"].shape[1]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        mask_incident_edge_attrs=mask_incident_edge_attrs,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_generator = torch.Generator().manual_seed(seed)
    randomizer = random.Random(seed)
    best_state = None
    best_validation_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        shuffled_paths = list(train_paths)
        randomizer.shuffle(shuffled_paths)
        train_loss_sum = 0.0
        train_graph_count = 0
        for path_batch in _path_batches(shuffled_paths, batch_size):
            batch = collate_token_graphs([_load_graph(path) for path in path_batch])
            batch_loss = reconstruction_step(
                    model,
                    batch,
                    optimizer,
                    mask_ratio=mask_ratio,
                    trim_fraction=trim_fraction,
                    generator=train_generator,
                )
            train_loss_sum += batch_loss * len(path_batch)
            train_graph_count += len(path_batch)
        validation_generator = torch.Generator().manual_seed(seed + 100_000)
        validation_loss_sum = 0.0
        validation_graph_count = 0
        for path_batch in _path_batches(validation_paths, batch_size):
            batch = collate_token_graphs([_load_graph(path) for path in path_batch])
            batch_loss = reconstruction_batch_loss(
                    model,
                    batch,
                    mask_ratio=mask_ratio,
                    trim_fraction=trim_fraction,
                    generator=validation_generator,
                )
            validation_loss_sum += batch_loss * len(path_batch)
            validation_graph_count += len(path_batch)
        train_loss = train_loss_sum / train_graph_count
        validation_loss = validation_loss_sum / validation_graph_count
        history.append(
            {
                "epoch": epoch,
                "train_reconstruction_loss": train_loss,
                "validation_reconstruction_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch}: train_reconstruction={train_loss:.6f} "
            f"validation_reconstruction={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    config = {
        "node_dim": int(sample["x"].shape[1]),
        "edge_dim": int(sample["edge_attr"].shape[1]),
        "edge_mark_dim": int(sample["edge_mark"].shape[1]),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "mask_ratio": mask_ratio,
        "trim_fraction": trim_fraction,
        "seed": seed,
        "mask_incident_edge_attrs": mask_incident_edge_attrs,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
    }
    torch.save(
        {"model_state": best_state, "model_config": config},
        output_directory / "best_model.pt",
    )
    split_lookup = {
        str(path): split
        for split, split_paths in (
            ("train", train_paths),
            ("validation", validation_paths),
            ("test", test_paths),
        )
        for path in split_paths
    }
    score_records = []
    for position, path in enumerate(paths, start=1):
        graph = _load_graph(path)
        score = score_graph(model, graph, block_size=score_block_size)
        score["pair_id"] = str(graph.get("pair_id", graph["example_id"]))
        score["split"] = split_lookup[str(path)]
        score_records.append(score)
        if position == 1 or position % 25 == 0 or position == len(paths):
            print(f"Scored {position}/{len(paths)} token graphs")
    (output_directory / "unsupervised_scores.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in score_records) + "\n",
        encoding="utf-8",
    )
    summary = {
        "state": "complete",
        "train_graphs": len(train_paths),
        "validation_graphs": len(validation_paths),
        "test_graphs": len(test_paths),
        "best_validation_reconstruction_loss": best_validation_loss,
        "epochs_ran": len(history),
        "model_config": config,
        "history": history,
        "scores": str(output_directory / "unsupervised_scores.jsonl"),
    }
    (output_directory / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a label-blind token-graph MAE.")
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--trim-fraction", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-block-size", type=int, default=1)
    parser.add_argument(
        "--keep-masked-edge-attrs",
        action="store_true",
        help="Ablation: allow incident attention values to remain visible",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_training(
        args.graph_dir,
        args.output_dir,
        device=args.device,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mask_ratio=args.mask_ratio,
        trim_fraction=args.trim_fraction,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        seed=args.seed,
        score_block_size=args.score_block_size,
        mask_incident_edge_attrs=not args.keep_masked_edge_attrs,
        test_fraction=args.test_fraction,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
