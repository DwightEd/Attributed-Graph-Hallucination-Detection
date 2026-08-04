"""Label-free training, calibration, and scoring for typed token graphs."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from .ragtruth_data import (
    CompactGraphStore,
    atomic_json,
    atomic_torch_save,
    autocast_context,
    budget_batches,
    collate_graphs,
    load_compact_manifest,
    load_graph_semantic_signature,
    make_answer_mask,
    new_generator,
    split_paths_by_official_split,
    split_paths_by_source,
)
from .typed_model import (
    TypedNeighborhoodAutoencoder,
    score_masked_tokens,
    typed_reconstruction_loss,
)


def _require_checkpoint_graph_semantics(
    checkpoint: Mapping[str, object], graph_dir: str | Path
) -> str:
    """Fail closed before a checkpoint sees graphs with different semantics."""

    expected = load_graph_semantic_signature(graph_dir)
    actual = str(checkpoint.get("graph_semantic_signature", "")).strip()
    if not actual:
        raise RuntimeError("checkpoint is missing its graph semantic signature")
    if actual != expected:
        raise RuntimeError(
            "checkpoint/graph semantic signature mismatch; retrain on this compact corpus"
        )
    return expected


def _model_from_graph(graph: Mapping[str, object], *, hidden_dim: int,
                      num_layers: int, dropout: float) -> tuple[TypedNeighborhoodAutoencoder, dict[str, int | float]]:
    config: dict[str, int | float] = {
        "node_dim": int(graph["x"].shape[1]),
        "edge_dim": int(graph["edge_attr"].shape[1]),
        "num_edge_types": 2, "hidden_dim": int(hidden_dim),
        "num_layers": int(num_layers), "dropout": float(dropout),
        "route_dim": int(graph["route_stats_target"].shape[-1]),
        "context_dim": int(graph["node_context"].shape[1]) if "node_context" in graph else 0,
    }
    return TypedNeighborhoodAutoencoder(**config), config


def _run_batch(model: TypedNeighborhoodAutoencoder, graphs: Sequence[Mapping[str, object]], *,
               mask_ratio: float, generator: torch.Generator, neighborhood_weight: float,
               route_weight: float, amp: str) -> tuple[torch.Tensor, int]:
    batch = collate_graphs(graphs)
    mask = make_answer_mask(batch["response_mask"], batch["graph_ptr"],
                            mask_ratio=mask_ratio, generator=generator)
    with autocast_context(next(model.parameters()).device, amp):
        outputs = model(batch, mask)
        loss = typed_reconstruction_loss(outputs, batch, mask,
                                         neighborhood_weight=neighborhood_weight,
                                         route_weight=route_weight)
    return loss, int(mask.sum())


def train_typed_autoencoder(
    graph_dir: str | Path, output_dir: str | Path, *, device: str = "cuda:0",
    residency: str = "cuda", max_resident_gib: float = 0.0, hidden_dim: int = 192,
    num_layers: int = 2, dropout: float = 0.1, learning_rate: float = 3e-4,
    weight_decay: float = 1e-4, epochs: int = 40, patience: int = 6,
    max_nodes: int = 12_000, max_edges: int = 192_000, mask_ratio: float = 0.20,
    neighborhood_weight: float = 0.25, route_weight: float = 0.10,
    train_fraction: float = 0.70, validation_fraction: float = 0.15,
    split_policy: str = "source_random", amp: str = "bfloat16", seed: int = 42,
) -> dict[str, object]:
    """Train with label-free validation and persist one concise best checkpoint."""

    if min(hidden_dim, num_layers, epochs, patience) < 1:
        raise ValueError("model dimensions, epochs, and patience must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("optimizer hyperparameters are invalid")
    if min(neighborhood_weight, route_weight) < 0:
        raise ValueError("loss weights cannot be negative")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    records = load_compact_manifest(graph_dir)
    graph_semantic_signature = load_graph_semantic_signature(graph_dir)
    if split_policy == "official":
        paths_by_split = split_paths_by_official_split(
            records, validation_fraction=validation_fraction, seed=seed
        )
    elif split_policy == "source_random":
        paths_by_split = split_paths_by_source(
            records, train_fraction=train_fraction,
            validation_fraction=validation_fraction, seed=seed,
        )
    else:
        raise ValueError("split_policy must be official or source_random")
    record_by_path = {Path(record["path"]): record for record in records}
    training_paths = set(paths_by_split["train"]) | set(paths_by_split["validation"])
    store = CompactGraphStore([record_by_path[path] for path in training_paths],
                              device=device, residency=residency,
                              max_resident_gib=max_resident_gib)
    requested = torch.device(device)
    sample = store.get(paths_by_split["train"][0])
    model, model_config = _model_from_graph(sample, hidden_dim=hidden_dim,
                                            num_layers=num_layers, dropout=dropout)
    model.to(requested)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp == "float16")
    train_generator, randomizer = new_generator(requested, seed), random.Random(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation, stale_epochs = math.inf, 0
    history: list[dict[str, float | int]] = []
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        store.close()
        raise RuntimeError("tqdm is required; install requirements-unsupervised-token-graph.txt") from error
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_paths = list(paths_by_split["train"])
            randomizer.shuffle(train_paths)
            train_batches = budget_batches(train_paths, record_by_path,
                                           max_nodes=max_nodes, max_edges=max_edges)
            train_total = 0.0
            train_masked = 0
            progress = tqdm(train_batches, desc=f"train epoch {epoch}/{epochs}", unit="batch", leave=False)
            for batch_paths in progress:
                loss, masked_count = _run_batch(
                    model, [store.get(path) for path in batch_paths], mask_ratio=mask_ratio,
                    generator=train_generator, neighborhood_weight=neighborhood_weight,
                    route_weight=route_weight, amp=amp,
                )
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                train_total += float(loss.detach()) * masked_count
                train_masked += masked_count
                progress.set_postfix(loss=f"{train_total / train_masked:.4f}")

            model.eval()
            validation_generator = new_generator(requested, seed + 100_000)
            validation_total = 0.0
            validation_masked = 0
            validation_batches = budget_batches(paths_by_split["validation"], record_by_path,
                                                max_nodes=max_nodes, max_edges=max_edges)
            with torch.no_grad():
                for batch_paths in validation_batches:
                    loss, masked_count = _run_batch(
                        model, [store.get(path) for path in batch_paths], mask_ratio=mask_ratio,
                        generator=validation_generator, neighborhood_weight=neighborhood_weight,
                        route_weight=route_weight, amp=amp,
                    )
                    validation_total += float(loss) * masked_count
                    validation_masked += masked_count
            train_loss, validation_loss = train_total / train_masked, validation_total / validation_masked
            history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
            print(f"epoch {epoch:03d}  train={train_loss:.5f}  validation={validation_loss:.5f}")
            if validation_loss < best_validation - 1e-6:
                best_validation = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"early stopping at epoch {epoch}; best={best_validation:.5f}")
                    break
    finally:
        store.close()
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    root = Path(graph_dir).resolve()
    split_payload = {name: [path.relative_to(root).as_posix() for path in paths]
                     for name, paths in paths_by_split.items()}
    atomic_torch_save(output / "best.pt", {
        "schema_version": "typed_neighborhood_mae_v1", "state_dict": best_state,
        "model_config": model_config,
        "score_config": {"neighborhood_weight": neighborhood_weight, "route_weight": route_weight},
        "graph_semantic_signature": graph_semantic_signature,
    })
    atomic_json(output / "splits.json", split_payload)
    atomic_json(output / "history.json", history)
    summary: dict[str, object] = {"checkpoint": str(output / "best.pt"),
                                  "best_validation_loss": best_validation,
                                  "epochs_ran": len(history),
                                  "graphs": {name: len(paths) for name, paths in paths_by_split.items()},
                                  "split_policy": split_policy,
                                  "residency": residency, "device": str(requested),
                                  "graph_semantic_signature": graph_semantic_signature}
    atomic_json(output / "training_summary.json", summary)
    return summary


@torch.no_grad()
def score_graph_strided(
    model: TypedNeighborhoodAutoencoder, graph: Mapping[str, object], *, mask_stride: int,
    neighborhood_weight: float, route_weight: float, amp: str,
) -> dict[str, torch.Tensor | str]:
    """Cover every response token in a fixed number of vectorised mask views."""

    if mask_stride < 1:
        raise ValueError("mask_stride must be positive")
    response = torch.as_tensor(graph["response_mask"], dtype=torch.bool)
    response_count = int(response.sum())
    if response_count < 1:
        raise ValueError("graph has no response tokens")
    local_rank = response.long().cumsum(0) - 1
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in (
        "token_idx", "scores", "node_scores", "neighborhood_scores", "route_scores")}
    model.eval()
    for view in range(min(mask_stride, response_count)):
        mask = response & (local_rank.remainder(min(mask_stride, response_count)) == view)
        with autocast_context(next(model.parameters()).device, amp):
            result = score_masked_tokens(model, graph, mask,
                                         neighborhood_weight=neighborhood_weight,
                                         route_weight=route_weight)
        for name, values in collected.items():
            values.append(result[name].detach())
    token_idx = torch.cat(collected["token_idx"])
    order = token_idx.argsort()
    return {"source_id": str(graph["source_id"]), "sample_id": str(graph["sample_id"]),
            **{name: torch.cat(values)[order] for name, values in collected.items()}}


def _component_matrix(result: Mapping[str, object]) -> torch.Tensor:
    return torch.stack((result["node_scores"], result["neighborhood_scores"],
                        result["route_scores"]), dim=1).float()


def _fit_robust_calibration(values: torch.Tensor) -> dict[str, list[float]]:
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("calibration values must have shape [tokens, 3]")
    median = values.median(dim=0).values
    mad = (values - median).abs().median(dim=0).values
    return {"median": [float(value) for value in median],
            "scale": [float(value) for value in (1.4826 * mad).clamp_min(1e-6)]}


def _resolve_split_paths(graph_dir: str | Path, split_file: str | Path, partition: str) -> list[Path]:
    payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
    if partition not in payload or not isinstance(payload[partition], list):
        raise ValueError(f"partition {partition!r} is absent from {split_file}")
    root = Path(graph_dir).resolve()
    paths = [(root / str(value)).resolve() for value in payload[partition]]
    if not paths or any(root not in path.parents or not path.is_file() for path in paths):
        raise RuntimeError(f"partition {partition!r} contains unsafe or missing paths")
    return paths


def score_typed_autoencoder(
    checkpoint_path: str | Path, graph_dir: str | Path, output_path: str | Path, *,
    split_file: str | Path | None = None, partition: str = "test", device: str = "cuda:0",
    residency: str = "cuda", max_resident_gib: float = 0.0, mask_stride: int = 8,
    calibration_limit: int | None = None, calibration_max_tokens: int = 200_000,
    amp: str = "bfloat16",
) -> dict[str, object]:
    """Calibrate on label-free train graphs, then score a held-out partition."""

    checkpoint_path = Path(checkpoint_path)
    split_file = Path(split_file) if split_file else checkpoint_path.parent / "splits.json"
    records = load_compact_manifest(graph_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    graph_semantic_signature = _require_checkpoint_graph_semantics(checkpoint, graph_dir)
    record_by_path = {Path(record["path"]): record for record in records}
    calibration_paths = _resolve_split_paths(graph_dir, split_file, "train")
    score_paths = _resolve_split_paths(graph_dir, split_file, partition)
    if calibration_limit is not None:
        if calibration_limit < 1:
            raise ValueError("calibration_limit must be positive")
        calibration_paths = calibration_paths[:calibration_limit]
    if calibration_max_tokens < 1:
        raise ValueError("calibration_max_tokens must be positive")
    if len(calibration_paths) > calibration_max_tokens:
        calibration_paths = calibration_paths[:calibration_max_tokens]
    selected_paths = set(calibration_paths) | set(score_paths)
    unknown = selected_paths.difference(record_by_path)
    if unknown:
        raise RuntimeError(f"split references graphs outside the manifest: {sorted(unknown)}")
    store = CompactGraphStore([record_by_path[path] for path in selected_paths], device=device,
                              residency=residency, max_resident_gib=max_resident_gib)
    model = TypedNeighborhoodAutoencoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    score_config = checkpoint.get("score_config", {})
    neighborhood_weight = float(score_config.get("neighborhood_weight", 0.25))
    route_weight = float(score_config.get("route_weight", 0.10))
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        store.close()
        raise RuntimeError("tqdm is required; install requirements-unsupervised-token-graph.txt") from error
    try:
        calibration_values = []
        per_graph_quota = max(1, calibration_max_tokens // len(calibration_paths))
        for path in tqdm(calibration_paths, desc="fit label-free residual calibration", unit="graph"):
            result = score_graph_strided(model, store.get(path), mask_stride=mask_stride,
                                         neighborhood_weight=neighborhood_weight,
                                         route_weight=route_weight, amp=amp)
            values = _component_matrix(result)
            if len(values) > per_graph_quota:
                selected = torch.linspace(0, len(values) - 1, per_graph_quota,
                                          device=values.device).round().long()
                values = values[selected]
            calibration_values.append(values)
        calibration_token_count = sum(len(values) for values in calibration_values)
        calibration = _fit_robust_calibration(torch.cat(calibration_values))
        del calibration_values
        median, scale = (torch.tensor(calibration[name], device=device)
                         for name in ("median", "scale"))
        component_weights = torch.tensor([1.0, neighborhood_weight, route_weight], device=device)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        token_count = 0
        with temporary.open("w", encoding="utf-8") as stream:
            for path in tqdm(score_paths, desc=f"score {partition} tokens", unit="graph"):
                result = score_graph_strided(model, store.get(path), mask_stride=mask_stride,
                                             neighborhood_weight=neighborhood_weight,
                                             route_weight=route_weight, amp=amp)
                components = _component_matrix(result)
                calibrated = ((components - median) / scale * component_weights).sum(dim=1)
                token_indices, component_rows, calibrated_scores = (
                    result["token_idx"].cpu().tolist(), components.cpu().tolist(), calibrated.cpu().tolist())
                rows = [json.dumps({"source_id": result["source_id"],
                                    "sample_id": result["sample_id"], "token_idx": int(token_idx),
                                    "score": float(score), "node_residual": float(component[0]),
                                    "neighborhood_residual": float(component[1]), "route_residual": float(component[2])},
                                   ensure_ascii=False, separators=(",", ":"))
                        for token_idx, component, score in zip(token_indices, component_rows, calibrated_scores)]
                stream.write("\n".join(rows) + "\n")
                token_count += len(rows)
        temporary.replace(output)
    finally:
        store.close()
    summary: dict[str, object] = {"scores": str(output_path), "partition": partition,
                                  "graphs": len(score_paths), "tokens": token_count,
                                  "mask_stride": mask_stride, "calibration_graphs": len(calibration_paths),
                                  "calibration_tokens": calibration_token_count,
                                  "calibration": calibration,
                                  "graph_semantic_signature": graph_semantic_signature,
                                  "component_weights": [1.0, neighborhood_weight, route_weight]}
    atomic_json(Path(output_path).with_suffix(".summary.json"), summary)
    return summary
