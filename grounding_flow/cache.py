"""Trajectory caching, streaming unsupervised fitting, and split scoring."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .artifacts import (
    atomic_json,
    atomic_torch_save,
    canonical_hash,
    file_sha256,
    torch_load,
)
from .data import FlowDataset, FlowPreparedRecord
from .experiment import (
    DetectorFitConfig,
    GroundingFlowDetector,
    TrajectoryRecord,
    trajectory_from_calibrated,
)
from .method import NullModelConfig, calibrate_against_null
from .state_model import fit_trajectory_projector, fit_two_state_hmm


ProgressCallback = Callable[[str, int, int], None]
MetricCallback = Callable[[str, int, float], None]


@dataclass(frozen=True)
class FlowCacheConfig:
    device: str
    evidence_segment_ids: tuple[int, ...]
    num_nulls: int
    null_config: NullModelConfig
    null_std_floor: float
    seed: int
    resume: bool


@dataclass(frozen=True)
class TrajectoryCacheEntry:
    response_id: str
    pair_id: str
    partition: str
    path: Path
    source_identity: str


def _stable_sample_seed(seed: int, response_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\x1f{response_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _source_identity(record: FlowPreparedRecord) -> str:
    def stat(path: Path) -> dict[str, object]:
        value = path.resolve().stat()
        return {
            "path": str(path.resolve()),
            "size": value.st_size,
            "mtime_ns": value.st_mtime_ns,
            "sha256": file_sha256(path.resolve()),
        }

    return canonical_hash(
        {
            "response_id": record.response_id,
            "pair_id": record.pair_id,
            "prepared_graph": stat(record.graph_record.graph_path),
            "legacy_graph": stat(record.legacy_graph_path),
            "segment_sha256": hashlib.sha256(
                record.segment_ids.detach().cpu().numpy().tobytes()
            ).hexdigest(),
            "token_sha256": hashlib.sha256(
                record.token_ids.detach().cpu().numpy().tobytes()
            ).hexdigest(),
            "shape": {
                "nodes": record.graph_record.num_nodes,
                "response_nodes": record.graph_record.num_response_nodes,
                "edges": record.graph_record.num_edges,
                "traces": record.graph_record.num_traces,
            },
        }
    )


def _trajectory_path(root: Path, record: FlowPreparedRecord) -> Path:
    name = hashlib.sha256(record.response_id.encode("utf-8")).hexdigest() + ".pt"
    return root / record.partition / name


def load_cached_trajectory(
    entry: TrajectoryCacheEntry, *, protocol_id: str
) -> TrajectoryRecord:
    loaded = torch_load(entry.path)
    if not isinstance(loaded, Mapping):
        raise ValueError("trajectory cache must contain a mapping")
    record = TrajectoryRecord.from_artifact(
        loaded,
        expected_protocol_id=protocol_id,
        expected_source_identity=entry.source_identity,
    )
    if (
        record.response_id != entry.response_id
        or record.pair_id != entry.pair_id
        or record.partition != entry.partition
    ):
        raise ValueError("trajectory cache identity disagrees with its index")
    return record


def build_trajectory_cache(
    partitions: Mapping[str, Sequence[FlowPreparedRecord]],
    *,
    output_dir: Path,
    protocol_id: str,
    config: FlowCacheConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, list[TrajectoryCacheEntry]]:
    """Compute each expensive null-calibrated trajectory once and atomically cache it."""

    total = sum(len(records) for records in partitions.values())
    current = 0
    cache_hits = 0
    result: dict[str, list[TrajectoryCacheEntry]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for partition in ("train", "validation", "test"):
        records = tuple(partitions[partition])
        dataset = FlowDataset(records)
        for local_index, prepared in enumerate(records):
            current += 1
            source_identity = _source_identity(prepared)
            path = _trajectory_path(output_dir / "trajectory_cache", prepared)
            entry = TrajectoryCacheEntry(
                response_id=prepared.response_id,
                pair_id=prepared.pair_id,
                partition=partition,
                path=path,
                source_identity=source_identity,
            )
            reusable = False
            if config.resume and path.is_file():
                try:
                    load_cached_trajectory(entry, protocol_id=protocol_id)
                    reusable = True
                    cache_hits += 1
                except (OSError, RuntimeError, TypeError, ValueError, KeyError):
                    reusable = False
            if not reusable:
                graph, segment_ids, identity = dataset[local_index]
                if identity.response_id != prepared.response_id:
                    raise RuntimeError("flow dataset returned the wrong prepared graph")
                show_null_progress = current == 1 or current == total or current % 25 == 0

                def null_progress(done: int, count: int) -> None:
                    if progress_callback is not None and show_null_progress:
                        progress_callback(
                            f"conditional_null[{current}/{total}]", done, count
                        )

                with torch.no_grad():
                    calibrated = calibrate_against_null(
                        graph,
                        segment_ids,
                        num_nulls=config.num_nulls,
                        seed=_stable_sample_seed(config.seed, prepared.response_id),
                        null_config=config.null_config,
                        evidence_segment_ids=config.evidence_segment_ids,
                        flow_device=config.device,
                        standard_deviation_floor=config.null_std_floor,
                        progress_callback=null_progress,
                    )
                trajectory = trajectory_from_calibrated(
                    response_id=prepared.response_id,
                    pair_id=prepared.pair_id,
                    partition=partition,
                    calibrated=calibrated,
                )
                atomic_torch_save(
                    path,
                    trajectory.to_artifact(
                        protocol_id=protocol_id,
                        source_identity=source_identity,
                    ),
                )
                del calibrated, trajectory, graph
                if torch.device(config.device).type == "cuda" and current % 25 == 0:
                    torch.cuda.empty_cache()
            result[partition].append(entry)
            if progress_callback is not None:
                progress_callback("trajectory_cache", current, total)
    atomic_json(
        output_dir / "trajectory_cache" / "index.json",
        {
            "schema": "grounding-flow-trajectory-index-v1",
            "protocol_id": protocol_id,
            "cache_hits": cache_hits,
            "count": total,
            "partitions": {
                name: [asdict(entry) for entry in entries]
                for name, entries in result.items()
            },
        },
    )
    return result


def _response_balanced_surfaces(
    entries: Sequence[TrajectoryCacheEntry],
    *,
    protocol_id: str,
    capacity: int,
    seed: int,
) -> list[np.ndarray]:
    """Bound PCA memory while giving each identifiable response equal mass."""

    generator = np.random.default_rng(seed)
    eligible: list[tuple[TrajectoryCacheEntry, int]] = []
    input_shape: tuple[int, ...] | None = None
    for entry in entries:
        record = load_cached_trajectory(entry, protocol_id=protocol_id)
        if not record.is_null_identifiable:
            continue
        surface = np.asarray(record.model_surface, dtype=np.float32)
        if input_shape is None:
            input_shape = tuple(surface.shape[1:])
        elif tuple(surface.shape[1:]) != input_shape:
            raise ValueError("trajectory caches disagree on layer/head dimensions")
        eligible.append((entry, len(surface)))
    if len(eligible) < 2:
        raise ValueError(
            "training requires at least two null-identifiable responses"
        )
    if len(eligible) > capacity:
        selected = np.sort(
            generator.choice(len(eligible), size=capacity, replace=False)
        )
        eligible = [eligible[index] for index in selected]
    per_response = min(
        max(length for _, length in eligible),
        max(1, capacity // len(eligible)),
    )
    balanced: list[np.ndarray] = []
    for entry, length in eligible:
        record = load_cached_trajectory(entry, protocol_id=protocol_id)
        surface = np.asarray(record.model_surface, dtype=np.float32)
        selected = generator.choice(
            length,
            size=per_response,
            replace=length < per_response,
        )
        balanced.append(surface[np.sort(selected)])
    if sum(len(surface) for surface in balanced) < 2:
        raise ValueError("PCA fitting requires at least two balanced token observations")
    return balanced


def fit_detector_from_cache(
    entries: Sequence[TrajectoryCacheEntry],
    *,
    protocol_id: str,
    config: DetectorFitConfig,
    progress_callback: ProgressCallback | None = None,
    metric_callback: MetricCallback | None = None,
) -> GroundingFlowDetector:
    """Fit PCA/HMM label-free while streaming large layer/head surfaces from disk."""

    config.validate()
    if len(entries) < 2 or any(entry.partition != "train" for entry in entries):
        raise ValueError("detector fitting accepts at least two training trajectories only")
    balanced_surfaces = _response_balanced_surfaces(
        entries,
        protocol_id=protocol_id,
        capacity=config.pca_fit_tokens,
        seed=config.seed,
    )
    projector = fit_trajectory_projector(
        balanced_surfaces,
        max_components=config.pca_components,
        max_fit_tokens=config.pca_fit_tokens,
        seed=config.seed,
    )
    if progress_callback is not None:
        progress_callback("pca_fit", 1, 1)
    projected: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    for current, entry in enumerate(entries, start=1):
        record = load_cached_trajectory(entry, protocol_id=protocol_id)
        if not record.is_null_identifiable:
            continue
        projected.append(projector.transform(record.model_surface))
        anchors.append(np.asarray(record.mechanism_anchor, dtype=np.float64))
        if progress_callback is not None:
            progress_callback("training_projection", current, len(entries))

    def hmm_progress(iteration: int, log_likelihood: float) -> None:
        if progress_callback is not None:
            progress_callback("hmm_em", iteration, config.hmm_iterations)
        if metric_callback is not None:
            metric_callback(
                "hmm_response_balanced_log_likelihood",
                iteration,
                log_likelihood,
            )

    if len(projected) < 2:
        raise ValueError(
            "detector fitting requires at least two null-identifiable train responses"
        )
    state_model = fit_two_state_hmm(
        projected,
        anchors,
        seed=config.seed,
        max_iterations=config.hmm_iterations,
        tolerance=config.hmm_tolerance,
        variance_floor=config.hmm_variance_floor,
        progress_callback=hmm_progress,
    )
    return GroundingFlowDetector(
        projector=projector,
        state_model=state_model,
        fit_config=config,
    )


def score_trajectory_cache(
    entries: Sequence[TrajectoryCacheEntry],
    *,
    protocol_id: str,
    detector: GroundingFlowDetector,
    token_output: Path,
    require_complete_pairs: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Score identifiable trajectories and return explicit coverage exclusions."""

    token_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = token_output.with_name(
        f".{token_output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    responses: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    incomplete_pair_ids: set[str] = set()
    if require_complete_pairs:
        pair_members: dict[str, list[TrajectoryRecord]] = {}
        for entry in entries:
            record = load_cached_trajectory(entry, protocol_id=protocol_id)
            pair_members.setdefault(record.pair_id, []).append(record)
        for pair_id, records in pair_members.items():
            if len(records) != 2:
                raise ValueError(f"HaluEval pair {pair_id} does not contain two responses")
            if sum(record.is_null_identifiable for record in records) != 2:
                incomplete_pair_ids.add(pair_id)
    try:
        with temporary.open("w", encoding="utf-8") as token_handle:
            for current, entry in enumerate(entries, start=1):
                record = load_cached_trajectory(entry, protocol_id=protocol_id)
                if not record.is_null_identifiable or record.pair_id in incomplete_pair_ids:
                    exclusion: dict[str, object] = {
                        "response_id": record.response_id,
                        "example_id": record.response_id,
                        "pair_id": record.pair_id,
                        "split": record.partition,
                        "response_tokens": len(record.response_token_indices),
                        "exclusion_reason": (
                            record.null_exclusion_reason
                            if not record.is_null_identifiable
                            else "incomplete_identifiable_pair"
                        ),
                        "null_calibration_status": record.calibration_status,
                        "null_swap_fraction": float(record.null_swap_fraction),
                        "null_effective_fraction": float(
                            record.null_effective_fraction
                        ),
                        "null_stable_model_fraction": float(
                            record.null_stable_model_fraction
                        ),
                        "null_samples": int(record.null_samples),
                        "unique_null_samples": int(record.null_samples),
                        "duplicate_null_draws": int(record.duplicate_null_draws),
                    }
                    exclusion.update(
                        {
                            f"mean_{name}": float(np.asarray(value).mean())
                            for name, value in record.raw_token_summary.items()
                        }
                    )
                    exclusions.append(exclusion)
                    if progress_callback is not None:
                        progress_callback(
                            f"score_{entry.partition}", current, len(entries)
                        )
                    continue
                response, tokens = detector.score(record)
                if "mean_ancestry" in response:
                    response["one_minus_mean_ancestry"] = 1.0 - float(
                        response["mean_ancestry"]
                    )
                responses.append(response)
                for token in tokens:
                    token_handle.write(
                        json.dumps(
                            token,
                            ensure_ascii=False,
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                if progress_callback is not None:
                    progress_callback(
                        f"score_{entry.partition}", current, len(entries)
                    )
        os.replace(temporary, token_output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return responses, exclusions


__all__ = [
    "FlowCacheConfig",
    "MetricCallback",
    "ProgressCallback",
    "TrajectoryCacheEntry",
    "build_trajectory_cache",
    "fit_detector_from_cache",
    "load_cached_trajectory",
    "score_trajectory_cache",
]
