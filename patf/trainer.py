from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import torch

from .config import TrainConfig
from .features import load_feature
from .model import RobustScaler, TrajectoryRanker

CHECKPOINT_SCHEMA = "patf-ranker-v2"


def _group_split(
    groups: list[str], fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    unique = sorted(set(groups))
    if fraction <= 0 or len(unique) < 3:
        return torch.arange(len(groups)), torch.empty(0, dtype=torch.long)
    rng = random.Random(seed)
    rng.shuffle(unique)
    count = min(max(1, round(len(unique) * fraction)), len(unique) - 1)
    validation_groups = set(unique[:count])
    train = torch.tensor(
        [index for index, group in enumerate(groups) if group not in validation_groups]
    )
    validation = torch.tensor(
        [index for index, group in enumerate(groups) if group in validation_groups]
    )
    return train, validation


def _atomic_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _paired_tensors(
    feature_paths: list[Path], modes: tuple[str, ...]
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str], tuple[str, ...]]:
    originals: list[torch.Tensor] = []
    corrupted: list[torch.Tensor] = []
    groups: list[str] = []
    feature_names: tuple[str, ...] | None = None
    for path in feature_paths:
        record = load_feature(path)
        names = tuple(record["feature_names"])
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"feature contract mismatch: {path}")
        trajectories = record["trajectories"]
        clean = torch.as_tensor(trajectories["original"])
        group = str(record["source_id"])
        if group == "unknown":
            group = str(record["sample_id"])
        for mode in modes:
            originals.append(clean)
            corrupted.append(torch.as_tensor(trajectories[mode]))
            groups.append(group)
    if not originals:
        raise ValueError("training requires at least one original/counterfactual pair")
    return originals, corrupted, groups, feature_names or ()


def train_ranker(
    feature_paths: list[Path],
    output_dir: str | Path,
    *,
    modes: tuple[str, ...],
    config: TrainConfig,
    device: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    history_path = output_dir / "history.json"

    originals, corrupted, groups, feature_names = _paired_tensors(
        feature_paths, modes
    )
    train_index, validation_index = _group_split(
        groups, config.validation_fraction, config.seed
    )
    scaler = RobustScaler.fit([originals[index] for index in train_index.tolist()])
    original = torch.stack([scaler.transform(value) for value in originals]).to(device)
    counterfactual = torch.stack(
        [scaler.transform(value) for value in corrupted]
    ).to(device)
    train_index = train_index.to(device)
    validation_index = validation_index.to(device)

    torch.manual_seed(config.seed)
    model = TrajectoryRanker(len(feature_names), config.hidden_dim).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)

    best_loss = math.inf
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = train_index.cpu()[
            torch.randperm(len(train_index), generator=generator)
        ].to(device)
        losses: list[float] = []
        accuracies: list[float] = []
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            clean_score = model(original[index])
            corrupt_score = model(counterfactual[index])
            ranking = torch.relu(
                config.margin - (corrupt_score - clean_score)
            ).mean()
            loss = ranking + config.origin_regularization * clean_score.square().mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
            accuracies.append(float((corrupt_score > clean_score).float().mean()))

        model.eval()
        with torch.no_grad():
            if len(validation_index):
                clean_score = model(original[validation_index])
                corrupt_score = model(counterfactual[validation_index])
                validation_loss = float(
                    torch.relu(
                        config.margin - (corrupt_score - clean_score)
                    ).mean()
                )
                validation_accuracy = float(
                    (corrupt_score > clean_score).float().mean()
                )
            else:
                validation_loss = sum(losses) / len(losses)
                validation_accuracy = float("nan")

        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "train_pair_accuracy": sum(accuracies) / len(accuracies),
            "validation_loss": validation_loss,
            "validation_pair_accuracy": validation_accuracy,
        }
        history.append(row)
        _atomic_json(history, history_path)
        print(json.dumps(row, sort_keys=True), flush=True)

        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            stale = 0
            state = copy.deepcopy(model.state_dict())
            _atomic_save(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "state_dict": state,
                    "input_dim": model.input_dim,
                    "hidden_dim": model.hidden_dim,
                    "feature_names": feature_names,
                    "scaler_median": scaler.median,
                    "scaler_scale": scaler.scale,
                    "train_config": asdict(config),
                    "best_epoch": epoch,
                    "best_validation_loss": best_loss,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if len(validation_index) and stale >= config.patience:
                break
    return checkpoint_path


def load_ranker(
    path: str | Path,
) -> tuple[TrajectoryRanker, RobustScaler, tuple[str, ...]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported checkpoint: {path}")
    model = TrajectoryRanker(
        int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    scaler = RobustScaler(
        torch.as_tensor(checkpoint["scaler_median"]),
        torch.as_tensor(checkpoint["scaler_scale"]),
    )
    return model, scaler, tuple(checkpoint["feature_names"])


def score_features(
    feature_paths: list[Path],
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 256,
) -> list[dict[str, object]]:
    model, scaler, feature_names = load_ranker(checkpoint_path)
    model.eval()
    metadata: list[dict[str, object]] = []
    trajectories: list[torch.Tensor] = []
    for path in feature_paths:
        record = load_feature(path)
        if tuple(record["feature_names"]) != feature_names:
            raise ValueError(f"feature contract mismatch: {path}")
        trajectories.append(
            scaler.transform(torch.as_tensor(record["trajectories"]["original"]))
        )
        metadata.append(
            {
                "sample_id": str(record["sample_id"]),
                "source_id": str(record["source_id"]),
                "original_idx": record["original_idx"],
            }
        )

    values = torch.stack(trajectories)
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            scores.extend(model(values[start : start + batch_size]).tolist())
    records = [
        {**row, "topology_anomaly_score": float(score)}
        for row, score in zip(metadata, scores)
    ]
    Path(output_path).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    print(f"[score] samples={len(records)}", flush=True)
    return records
