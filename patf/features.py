from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch

from attention_cache import load_attention

from .config import CorruptionConfig, TopologyConfig
from .topology import FEATURE_NAMES, extract_trajectories

FEATURE_SCHEMA = "patf-feature-v2"


def _source_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _save_atomic(record: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(record, temporary)
    os.replace(temporary, path)


def _is_current(
    feature_path: Path,
    *,
    feature_config: dict[str, object],
    source_signature: tuple[int, int],
) -> bool:
    if not feature_path.is_file():
        return False
    record = torch.load(feature_path, map_location="cpu", weights_only=True)
    return bool(
        record.get("schema") == FEATURE_SCHEMA
        and record.get("feature_config") == feature_config
        and tuple(record.get("source_signature", ())) == source_signature
    )


def _extract_one(
    attention_path: str,
    feature_path: str,
    topology: TopologyConfig,
    corruption: CorruptionConfig,
    modes: tuple[str, ...],
    feature_config: dict[str, object],
    torch_threads: int,
) -> tuple[str, float]:
    torch.set_num_threads(torch_threads)
    started = time.perf_counter()
    source = Path(attention_path)
    destination = Path(feature_path)
    sample = load_attention(source)
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
            "source_signature": _source_signature(source),
            "trajectories": trajectories,
        },
        destination,
    )
    return destination.name, time.perf_counter() - started


def prepare_features(
    paths: Iterable[Path],
    output_dir: str | Path,
    *,
    topology: TopologyConfig,
    corruption: CorruptionConfig,
    modes: tuple[str, ...],
    resume: bool,
    split: str,
    workers: int = 1,
    torch_threads: int = 1,
) -> list[Path]:
    """Extract sample features in parallel and cache each sample atomically."""
    paths = list(paths)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_config: dict[str, object] = {
        "topology": asdict(topology),
        "corruption": asdict(corruption),
        "modes": modes,
    }
    feature_paths = [
        output_dir / f"{attention_path.stem}.features.pt"
        for attention_path in paths
    ]

    pending: list[tuple[Path, Path]] = []
    reused = 0
    for attention_path, feature_path in zip(paths, feature_paths):
        current = resume and _is_current(
            feature_path,
            feature_config=feature_config,
            source_signature=_source_signature(attention_path),
        )
        if current:
            reused += 1
        else:
            pending.append((attention_path, feature_path))

    print(
        f"[{split} features] total={len(paths)} reuse={reused} "
        f"extract={len(pending)} workers={max(1, workers)}",
        flush=True,
    )
    if not pending:
        return feature_paths

    arguments = [
        (
            str(attention_path),
            str(feature_path),
            topology,
            corruption,
            modes,
            feature_config,
            max(1, torch_threads),
        )
        for attention_path, feature_path in pending
    ]

    if workers <= 1:
        for index, argument in enumerate(arguments, 1):
            name, elapsed = _extract_one(*argument)
            print(
                f"[{split} features] {index}/{len(arguments)} {name} "
                f"{elapsed:.1f}s",
                flush=True,
            )
        return feature_paths

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(arguments)),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(_extract_one, *argument): argument[0]
            for argument in arguments
        }
        for index, future in enumerate(as_completed(futures), 1):
            name, elapsed = future.result()
            print(
                f"[{split} features] {index}/{len(arguments)} {name} "
                f"{elapsed:.1f}s",
                flush=True,
            )
    return feature_paths


def load_feature(path: str | Path) -> dict[str, object]:
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("schema") != FEATURE_SCHEMA:
        raise ValueError(f"unsupported feature file: {path}")
    return record
