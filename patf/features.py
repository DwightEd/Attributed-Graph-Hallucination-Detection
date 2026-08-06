from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch

from .config import CorruptionConfig, TopologyConfig
from .data import load_attention
from .topology import FEATURE_NAMES, extract_trajectories

FEATURE_SCHEMA = "patf-feature-v1"


def _save_atomic(record: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(record, temporary)
    os.replace(temporary, path)


def prepare_features(
    paths: Iterable[Path],
    output_dir: str | Path,
    *,
    topology: TopologyConfig,
    corruption: CorruptionConfig,
    modes: tuple[str, ...],
    resume: bool,
    split: str,
) -> list[Path]:
    paths = list(paths)
    output_dir = Path(output_dir)
    feature_paths: list[Path] = []
    feature_config = {
        "topology": asdict(topology),
        "corruption": asdict(corruption),
        "modes": modes,
    }

    for index, attention_path in enumerate(paths, 1):
        feature_path = output_dir / f"{attention_path.stem}.features.pt"
        feature_paths.append(feature_path)
        if resume and feature_path.is_file():
            cached = torch.load(feature_path, map_location="cpu", weights_only=True)
            if cached.get("feature_config") == feature_config:
                print(
                    f"[{split} features] {index}/{len(paths)} reuse {feature_path.name}",
                    flush=True,
                )
                continue

        print(f"[{split} features] {index}/{len(paths)} extract {attention_path.name}", flush=True)
        sample = load_attention(attention_path)
        trajectories = extract_trajectories(
            sample,
            topology=topology,
            corruption=corruption,
            modes=modes,
        )
        _save_atomic(
            {
                "schema": FEATURE_SCHEMA,
                "sample_id": sample.sample_id,
                "source_id": sample.source_id,
                "original_idx": sample.original_idx,
                "response_idx": sample.response_idx,
                "token_count": sample.token_count,
                "feature_names": FEATURE_NAMES,
                "feature_config": feature_config,
                "trajectories": trajectories,
            },
            feature_path,
        )
    return feature_paths


def load_feature(path: str | Path) -> dict[str, object]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("schema") != FEATURE_SCHEMA:
        raise ValueError(f"unsupported feature file: {path}")
    return record
