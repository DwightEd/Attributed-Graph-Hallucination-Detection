"""Top-level orchestration for the label-gated HaluEval grounding-flow run."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .artifacts import atomic_json, atomic_jsonl, canonical_hash, file_sha256
from .cache import (
    FlowCacheConfig,
    MetricCallback,
    ProgressCallback,
    TrajectoryCacheEntry,
    build_trajectory_cache,
    fit_detector_from_cache,
    score_trajectory_cache,
)
from .data import FlowPreparedRecord, prepare_halueval_flow_records
from .evaluation import (
    add_length_residual_scores,
    evaluate_frozen_halueval_predictions,
    freeze_prediction_files,
)
from .experiment import DetectorFitConfig, GroundingFlowDetector
from .method import NullModelConfig

METHOD_SCHEMA = "grounding-flow-conditional-null-hmm-v1"


@dataclass(frozen=True)
class GroundingFlowRunConfig:
    extraction_dir: Path
    examples_path: Path
    output_dir: Path
    evaluation_labels_path: Path | None = None
    device: str = "cuda"
    conversion_chunk_edges: int = 8_192
    query_block: int = 32
    validation_fraction: float = 0.10
    test_fraction: float = 0.20
    split_seed: int = 42
    limit_pairs: int | None = None
    expected_candidates: int | None = None
    min_test_pair_coverage: float = 0.90
    fail_on_low_coverage: bool = False
    group_by_prompt: bool = True
    require_complete_cache: bool = True
    evidence_segment_ids: tuple[int, ...] = (1,)
    num_nulls: int = 32
    null_swaps_per_edge: int = 4
    null_lag_boundaries: tuple[int, ...] = (4, 8, 16, 32, 64, 128)
    null_max_attempt_factor: int = 12
    null_std_floor: float = 1e-6
    pca_components: int = 32
    pca_fit_tokens: int = 20_000
    hmm_iterations: int = 50
    hmm_tolerance: float = 1e-4
    hmm_variance_floor: float = 1e-4
    bootstrap_samples: int = 1_000
    seed: int = 42
    resume: bool = True
    skip_evaluation: bool = False

    def null_config(self) -> NullModelConfig:
        return NullModelConfig(
            swaps_per_edge=self.null_swaps_per_edge,
            lag_boundaries=self.null_lag_boundaries,
            max_null_attempt_factor=self.null_max_attempt_factor,
        )

    def detector_config(self) -> DetectorFitConfig:
        return DetectorFitConfig(
            pca_components=self.pca_components,
            pca_fit_tokens=self.pca_fit_tokens,
            hmm_iterations=self.hmm_iterations,
            hmm_tolerance=self.hmm_tolerance,
            hmm_variance_floor=self.hmm_variance_floor,
            seed=self.seed,
        )

    def cache_config(self) -> FlowCacheConfig:
        return FlowCacheConfig(
            device=self.device,
            evidence_segment_ids=self.evidence_segment_ids,
            num_nulls=self.num_nulls,
            null_config=self.null_config(),
            null_std_floor=self.null_std_floor,
            seed=self.seed,
            resume=self.resume,
        )

    def validate(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be in (0, 1)")
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError("split fractions must leave a training partition")
        if self.limit_pairs is not None and self.limit_pairs < 3:
            raise ValueError("limit_pairs must be at least three")
        if self.expected_candidates is not None and self.expected_candidates < 1:
            raise ValueError("expected_candidates must be positive or None")
        if not np.isfinite(self.min_test_pair_coverage) or not (
            0.0 <= self.min_test_pair_coverage <= 1.0
        ):
            raise ValueError("min_test_pair_coverage must be in [0, 1]")
        if self.conversion_chunk_edges < 1 or self.query_block < 1:
            raise ValueError("conversion chunks and query blocks must be positive")
        if self.num_nulls < 2 or self.null_std_floor <= 0.0:
            raise ValueError("null calibration needs at least two nulls and a positive floor")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if not self.evidence_segment_ids or any(
            value not in (0, 1, 2) for value in self.evidence_segment_ids
        ):
            raise ValueError("evidence_segment_ids must contain prompt segment ids")
        self.null_config().validate()
        self.detector_config().validate()
        if not self.skip_evaluation and self.evaluation_labels_path is None:
            raise ValueError("evaluation_labels_path is required unless evaluation is skipped")


def canonical_protocol_id(value: Mapping[str, object]) -> str:
    """Hash computation semantics, independent of output path and GPU identity."""

    return canonical_hash(value)


def _validate_output(output: Path, *, resume: bool) -> None:
    if not output.exists():
        return
    contents = list(output.iterdir())
    if not contents:
        return
    if not resume:
        raise FileExistsError(f"output directory is not empty: {output}")
    allowed = {
        "prepared",
        "trajectory_cache",
        "run_state.json",
        "splits.json",
        "run.log",
        "detector.json",
        "length_residual.json",
        "score_freeze.json",
        "evaluation.json",
        "identifiable_coverage_gate.json",
        "run.json",
        "train.response_predictions.jsonl",
        "validation.response_predictions.jsonl",
        "test.response_predictions.jsonl",
        "train.token_predictions.jsonl",
        "validation.token_predictions.jsonl",
        "test.token_predictions.jsonl",
        "train.calibration_exclusions.jsonl",
        "validation.calibration_exclusions.jsonl",
        "test.calibration_exclusions.jsonl",
    }
    unexpected = sorted(path.name for path in contents if path.name not in allowed)
    if unexpected:
        raise FileExistsError(
            "resume accepts only preparation/trajectory caches; found final artifacts: "
            + ", ".join(unexpected)
        )
    resumable_prefix = {"prepared", "trajectory_cache", "run.log", "splits.json"}
    tail = [path for path in contents if path.name not in resumable_prefix]
    if tail and not (output / "run_state.json").is_file():
        raise FileExistsError(
            "regenerable tail artifacts require run_state.json before resume"
        )


def _protocol_payload(
    config: GroundingFlowRunConfig,
    *,
    preparation_metadata: Mapping[str, object],
) -> dict[str, object]:
    extraction_manifest = config.extraction_dir.resolve() / "extraction_manifest.json"
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    return {
        "schema": METHOD_SCHEMA,
        "trajectory_implementation": {
            name: file_sha256(package_root / name)
            for name in ("method.py", "data.py", "experiment.py", "cache.py")
        },
        "upstream_graph_implementation": {
            name: file_sha256(project_root / "attention_graph" / name)
            for name in ("graph.py", "data.py", "halueval.py")
        },
        "input": preparation_metadata["input_protocol"],
        "extraction_manifest_sha256": file_sha256(extraction_manifest),
        "examples_sha256": file_sha256(config.examples_path.resolve()),
        "split": {
            "validation_fraction": config.validation_fraction,
            "test_fraction": config.test_fraction,
            "split_seed": config.split_seed,
            "limit_pairs": config.limit_pairs,
            "expected_candidates": config.expected_candidates,
            "group_by_prompt": config.group_by_prompt,
            "require_complete_cache": config.require_complete_cache,
        },
        "flow": {
            "evidence_segment_ids": config.evidence_segment_ids,
            "num_nulls": config.num_nulls,
            "null_swaps_per_edge": config.null_swaps_per_edge,
            "null_lag_boundaries": config.null_lag_boundaries,
            "null_max_attempt_factor": config.null_max_attempt_factor,
            "null_std_floor": config.null_std_floor,
            "null_seed": config.seed,
        },
        "detector": asdict(config.detector_config()),
        "evaluation_policy": {
            "min_test_pair_coverage": config.min_test_pair_coverage,
            "bootstrap_samples": config.bootstrap_samples,
            "skip_evaluation": config.skip_evaluation,
            # fail_on_low_coverage is deliberately a resumable reporting
            # control: it never changes trajectories, frozen scores, or
            # metric definitions, only whether a failed target stops before
            # the already-isolated evaluation boundary.
        },
    }


def _write_splits(
    output: Path,
    partitions: Mapping[str, Sequence[FlowPreparedRecord]],
    metadata: Mapping[str, object],
    protocol_id: str,
) -> None:
    atomic_json(
        output / "splits.json",
        {
            **dict(metadata),
            "protocol_id": protocol_id,
            "partitions": {
                name: [
                    {
                        "response_id": record.response_id,
                        "pair_id": record.pair_id,
                        "partition": record.partition,
                        "graph_path": str(record.graph_record.graph_path),
                        "legacy_graph_path": str(record.legacy_graph_path),
                        "response_tokens": record.response_tokens,
                    }
                    for record in records
                ]
                for name, records in partitions.items()
            },
        },
    )


def _write_run_state(
    path: Path, *, protocol_id: str, stage: str, state: str = "in_progress"
) -> None:
    atomic_json(
        path,
        {
            "schema": METHOD_SCHEMA,
            "state": state,
            "stage": stage,
            "protocol_id": protocol_id,
        },
    )


def _score_all_partitions(
    *,
    output: Path,
    protocol_id: str,
    cache_entries: Mapping[str, Sequence[TrajectoryCacheEntry]],
    detector: GroundingFlowDetector,
    progress_callback: ProgressCallback | None,
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, list[dict[str, object]]],
    dict[str, Path],
    dict[str, Path],
    dict[str, Path],
]:
    response_predictions: dict[str, list[dict[str, object]]] = {}
    exclusions: dict[str, list[dict[str, object]]] = {}
    token_paths: dict[str, Path] = {}
    exclusion_paths: dict[str, Path] = {}
    for partition in ("train", "validation", "test"):
        token_path = output / f"{partition}.token_predictions.jsonl"
        token_paths[partition] = token_path
        response_predictions[partition], exclusions[partition] = score_trajectory_cache(
            cache_entries[partition],
            protocol_id=protocol_id,
            detector=detector,
            token_output=token_path,
            require_complete_pairs=partition == "test",
            progress_callback=progress_callback,
        )
        exclusion_path = output / f"{partition}.calibration_exclusions.jsonl"
        exclusion_paths[partition] = exclusion_path
        atomic_jsonl(exclusion_path, exclusions[partition])
    train_with_residual, length_coefficients = add_length_residual_scores(
        response_predictions["train"], response_predictions["train"]
    )
    response_predictions["train"] = train_with_residual
    for partition in ("validation", "test"):
        response_predictions[partition], _ = add_length_residual_scores(
            response_predictions["train"],
            response_predictions[partition],
            coefficients=length_coefficients,
        )
    atomic_json(output / "length_residual.json", length_coefficients)
    response_paths: dict[str, Path] = {}
    for partition, predictions in response_predictions.items():
        path = output / f"{partition}.response_predictions.jsonl"
        response_paths[partition] = path
        atomic_jsonl(path, predictions)
    return (
        response_predictions,
        exclusions,
        response_paths,
        token_paths,
        exclusion_paths,
    )


def _calibration_summary(
    predictions: Mapping[str, Sequence[Mapping[str, object]]],
    exclusions: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for partition, rows in predictions.items():
        excluded = exclusions[partition]
        all_rows = [*rows, *excluded]
        status_counts: dict[str, int] = {}
        for row in all_rows:
            status = str(row["null_calibration_status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        summary[partition] = {
            "total_responses": len(all_rows),
            "scored_responses": len(rows),
            "excluded_responses": len(excluded),
            "scored_fraction": (
                float(len(rows) / len(all_rows)) if all_rows else 0.0
            ),
            "status_counts": status_counts,
            "exclusion_reason_counts": {
                reason: sum(
                    str(row["exclusion_reason"]) == reason for row in excluded
                )
                for reason in sorted(
                    {str(row["exclusion_reason"]) for row in excluded}
                )
            },
            "mean_effective_null_fraction": float(
                np.mean(
                    [float(row["null_effective_fraction"]) for row in all_rows]
                )
            ),
            "mean_stable_model_fraction": float(
                np.mean(
                    [
                        float(row["null_stable_model_fraction"])
                        for row in all_rows
                    ]
                )
            ),
            "mean_swap_acceptance_fraction": float(
                np.mean([float(row["null_swap_fraction"]) for row in all_rows])
            ),
        }
    return summary


def _test_pair_coverage_gate(
    test_records: Sequence[FlowPreparedRecord],
    scored_rows: Sequence[Mapping[str, object]],
    *,
    minimum: float,
    excluded_rows: Sequence[Mapping[str, object]] | None = None,
    fail_on_low_coverage: bool = False,
) -> dict[str, object]:
    pair_members: dict[str, set[str]] = {}
    for record in test_records:
        pair_members.setdefault(record.pair_id, set()).add(record.response_id)
    if not pair_members or any(len(members) != 2 for members in pair_members.values()):
        raise ValueError("test coverage requires exactly two responses per HaluEval pair")
    scored_ids = {str(row["response_id"]) for row in scored_rows}
    all_ids = set().union(*pair_members.values())
    if not scored_ids.issubset(all_ids):
        raise ValueError("scored test responses are absent from the prepared split")
    partial = [
        pair_id
        for pair_id, members in pair_members.items()
        if 0 < len(members & scored_ids) < 2
    ]
    if partial:
        raise ValueError("test scoring must retain or exclude whole HaluEval pairs")
    scored_pairs = sum(members.issubset(scored_ids) for members in pair_members.values())
    coverage = float(scored_pairs / len(pair_members))
    passed = coverage >= minimum
    excluded = tuple(excluded_rows or ())
    reason_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    response_lengths: list[float] = []
    if excluded_rows is not None:
        excluded_ids = [str(row.get("response_id", "")) for row in excluded]
        if (
            any(not response_id for response_id in excluded_ids)
            or len(set(excluded_ids)) != len(excluded_ids)
            or set(excluded_ids) != all_ids - scored_ids
        ):
            raise ValueError(
                "test coverage exclusions must exactly identify every unscored response"
            )
    for row in excluded:
        reason = str(row.get("exclusion_reason", "unknown"))
        status = str(row.get("null_calibration_status", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
        if "response_tokens" in row:
            response_lengths.append(float(row["response_tokens"]))
    length_summary: dict[str, object] = {"count": len(response_lengths)}
    if response_lengths:
        values = np.asarray(response_lengths, dtype=np.float64)
        if not np.isfinite(values).all() or bool((values < 0).any()):
            raise ValueError(
                "coverage exclusion response lengths must be finite and non-negative"
            )
        length_summary.update(
            {
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "mean": float(values.mean()),
                "maximum": float(values.max()),
            }
        )
    return {
        "schema": "grounding-flow-identifiable-coverage-v1",
        "total_test_pairs": len(pair_members),
        "scored_test_pairs": scored_pairs,
        "excluded_test_pairs": len(pair_members) - scored_pairs,
        "test_pair_coverage": coverage,
        "minimum_test_pair_coverage": float(minimum),
        "passed": passed,
        "coverage_target_met": passed,
        "evaluation_population": "null_identifiable_complete_test_pairs",
        "action": (
            "evaluate_configured_coverage"
            if passed
            else (
                "fail_before_label_read"
                if fail_on_low_coverage
                else "evaluate_identifiable_subset"
            )
        ),
        "eligible_for_configured_coverage_claim": passed,
        "exclusion_summary": {
            "response_reason_counts": reason_counts,
            "response_calibration_status_counts": status_counts,
            "response_tokens": length_summary,
        },
        "warning": (
            None
            if passed
            else (
                "conditional-null metrics cover only the identifiable complete-pair "
                "subset and are not eligible for the configured coverage claim"
            )
        ),
    }


def run_grounding_flow(
    config: GroundingFlowRunConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    metric_callback: MetricCallback | None = None,
) -> dict[str, object]:
    """Prepare, calibrate, train, freeze scores, and optionally evaluate."""

    config.validate()
    run_started = time.perf_counter()
    wall_time_seconds: dict[str, float] = {}
    output = config.output_dir.expanduser().resolve()
    _validate_output(output, resume=config.resume)
    extraction_dir = config.extraction_dir.expanduser().resolve()
    examples_path = config.examples_path.expanduser().resolve()
    if not extraction_dir.is_dir():
        raise NotADirectoryError(f"extraction directory does not exist: {extraction_dir}")
    if not examples_path.is_file():
        raise FileNotFoundError(f"label-free examples do not exist: {examples_path}")
    labels_path = (
        None
        if config.evaluation_labels_path is None
        else config.evaluation_labels_path.expanduser().resolve()
    )
    if not config.skip_evaluation and (labels_path is None or not labels_path.is_file()):
        raise FileNotFoundError(f"evaluation label sidecar does not exist: {labels_path}")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    output.mkdir(parents=True, exist_ok=True)

    stage_started = time.perf_counter()
    partitions, preparation_metadata = prepare_halueval_flow_records(
        extraction_dir=extraction_dir,
        examples_path=examples_path,
        output_dir=output,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.split_seed,
        limit_pairs=config.limit_pairs,
        expected_candidates=config.expected_candidates,
        group_by_prompt=config.group_by_prompt,
        require_complete_cache=config.require_complete_cache,
        conversion_device=config.device,
        conversion_chunk_edges=config.conversion_chunk_edges,
        query_block=config.query_block,
        resume=config.resume,
        progress_callback=progress_callback,
    )
    wall_time_seconds["prepare"] = time.perf_counter() - stage_started
    protocol_payload = _protocol_payload(
        config, preparation_metadata=preparation_metadata
    )
    protocol_id = canonical_protocol_id(protocol_payload)
    state_path = output / "run_state.json"
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(previous, Mapping) or previous.get("protocol_id") != protocol_id:
            raise ValueError("interrupted output uses a different grounding-flow protocol")
    _write_run_state(state_path, protocol_id=protocol_id, stage="prepared")
    _write_splits(output, partitions, preparation_metadata, protocol_id)

    stage_started = time.perf_counter()
    cache_entries = build_trajectory_cache(
        partitions,
        output_dir=output,
        protocol_id=protocol_id,
        config=config.cache_config(),
        progress_callback=progress_callback,
    )
    wall_time_seconds["trajectory_cache"] = time.perf_counter() - stage_started
    _write_run_state(state_path, protocol_id=protocol_id, stage="trajectories_cached")
    stage_started = time.perf_counter()
    detector = fit_detector_from_cache(
        cache_entries["train"],
        protocol_id=protocol_id,
        config=config.detector_config(),
        progress_callback=progress_callback,
        metric_callback=metric_callback,
    )
    wall_time_seconds["detector_fit"] = time.perf_counter() - stage_started
    detector_path = output / "detector.json"
    atomic_json(detector_path, detector.to_dict())
    _write_run_state(state_path, protocol_id=protocol_id, stage="detector_fitted")

    stage_started = time.perf_counter()
    (
        response_predictions,
        calibration_exclusions,
        response_paths,
        token_paths,
        exclusion_paths,
    ) = _score_all_partitions(
        output=output,
        protocol_id=protocol_id,
        cache_entries=cache_entries,
        detector=detector,
        progress_callback=progress_callback,
    )
    wall_time_seconds["score"] = time.perf_counter() - stage_started
    _write_run_state(state_path, protocol_id=protocol_id, stage="scores_written")
    coverage_gate = _test_pair_coverage_gate(
        partitions["test"],
        response_predictions["test"],
        minimum=config.min_test_pair_coverage,
        excluded_rows=calibration_exclusions["test"],
        fail_on_low_coverage=config.fail_on_low_coverage,
    )
    atomic_json(output / "identifiable_coverage_gate.json", coverage_gate)
    if coverage_gate["action"] == "fail_before_label_read":
        _write_run_state(
            state_path, protocol_id=protocol_id, stage="coverage_gate_failed"
        )
        raise RuntimeError(
            "identifiable test-pair coverage is below the configured minimum: "
            f"observed={coverage_gate['test_pair_coverage']:.4f} "
            f"minimum={config.min_test_pair_coverage:.4f}"
        )
    freeze = freeze_prediction_files(
        output,
        [
            *(
                path
                for path in (
                    *response_paths.values(),
                    *token_paths.values(),
                    *exclusion_paths.values(),
                )
                if path.stat().st_size > 0
            ),
        ],
    )
    _write_run_state(state_path, protocol_id=protocol_id, stage="scores_frozen")

    evaluation_path: Path | None = None
    evaluation: dict[str, object] | None = None
    graph_dataset_index = output / "prepared" / "graphs" / "index.json"
    if not config.skip_evaluation:
        stage_started = time.perf_counter()
        if labels_path is None:  # pragma: no cover - config validation guarantees this
            raise RuntimeError("evaluation labels unexpectedly unavailable")
        test_records = partitions["test"]
        scored_test_ids = {
            str(row["response_id"]) for row in response_predictions["test"]
        }
        if not scored_test_ids:
            raise ValueError("test split has no null-identifiable responses to evaluate")
        evaluation = evaluate_frozen_halueval_predictions(
            output_dir=output,
            prediction_path=response_paths["test"],
            labels_path=labels_path,
            pair_by_response={
                record.response_id: record.pair_id
                for record in test_records
                if record.response_id in scored_test_ids
            },
            response_length_by_id={
                record.response_id: record.response_tokens
                for record in test_records
                if record.response_id in scored_test_ids
            },
            seed=config.seed,
            bootstrap_samples=config.bootstrap_samples,
            score_fields={
                "detached_state_mean": "score",
                "inverse_detached_state_mean": "inverse_detached_probability",
                "detached_state_max": "max_detached_probability",
                "detached_state_top10pct": "top_10_percent_mean",
                "length_residual": "length_residual_score",
                "raw_one_minus_ancestry": "one_minus_mean_ancestry",
                "raw_debt": "mean_debt",
                "raw_unknown_mass": "mean_unknown",
            },
            graph_index_rows=[
                {
                    "response_id": record.response_id,
                    "pair_id": record.pair_id,
                    "split": partition,
                    "graph_path": str(record.graph_record.graph_path),
                }
                for partition in ("train", "validation", "test")
                for record in partitions[partition]
            ],
            graph_index_path=graph_dataset_index,
        )
        evaluation["coverage_context"] = coverage_gate
        evaluation_path = output / "evaluation.json"
        atomic_json(evaluation_path, evaluation)
        wall_time_seconds["evaluation"] = time.perf_counter() - stage_started
        _write_run_state(state_path, protocol_id=protocol_id, stage="evaluated")

    primary = {} if evaluation is None else dict(evaluation["primary"])
    wall_time_seconds["total"] = time.perf_counter() - run_started
    low_coverage = not bool(coverage_gate["coverage_target_met"])
    result: dict[str, object] = {
        "schema": METHOD_SCHEMA,
        "status": "complete",
        "experiment_scope": (
            "low_identifiability_subset_pilot"
            if low_coverage
            else preparation_metadata["scope"]
        ),
        "input_scope": preparation_metadata["scope"],
        "labels_read_during": "never" if config.skip_evaluation else "evaluation_only",
        "method": {
            "graph_use": "layer-head evidence-provenance transport",
            "null": "target-relation-lag-conditioned RR source identity rewiring",
            "unsupervised_fit": "training-only PCA plus free-weight two-state diagonal Gaussian HMM",
            "hallucination_prevalence_assumption": "none",
            "neural_message_passing": False,
            "masked_reconstruction": False,
        },
        "output_dir": str(output),
        "protocol_id": protocol_id,
        "partition_counts": {
            name: len(records) for name, records in partitions.items()
        },
        "scored_partition_counts": {
            name: len(rows) for name, rows in response_predictions.items()
        },
        "detector": str(detector_path),
        "splits": str(output / "splits.json"),
        "score_freeze": str(output / "score_freeze.json"),
        "graph_dataset_index": (
            str(graph_dataset_index) if graph_dataset_index.is_file() else None
        ),
        "response_predictions": {
            name: str(path) for name, path in response_paths.items()
        },
        "token_predictions": {name: str(path) for name, path in token_paths.items()},
        "calibration_exclusions": {
            name: str(path) for name, path in exclusion_paths.items()
        },
        "evaluation": str(evaluation_path) if evaluation_path is not None else None,
        "core_metrics": {
            key: primary[key]
            for key in ("auroc", "average_precision", "paired_accuracy", "positive_fraction")
            if key in primary
        },
        "core_metrics_population": coverage_gate["evaluation_population"],
        "warnings": ([coverage_gate["warning"]] if low_coverage else []),
        "wall_time_seconds": wall_time_seconds,
        "identifiable_coverage_gate": coverage_gate,
        "training": {
            "pca_output_dimension": detector.projector.output_dimension,
            "pca_explained_variance_ratio": detector.projector.explained_variance_ratio.tolist(),
            "hmm_iterations_ran": len(detector.state_model.log_likelihood_history),
            "hmm_log_likelihood_history": list(
                detector.state_model.log_likelihood_history
            ),
            "fit_weighting": detector.state_model.fit_weighting,
            "state_occupancy": detector.state_model.state_occupancy.tolist(),
            "state_mechanism_anchor": detector.state_model.state_anchor.tolist(),
            "detached_state": detector.state_model.detached_state,
            "orientation_margin": detector.state_model.orientation_margin,
        },
        "null_calibration_coverage": _calibration_summary(
            response_predictions, calibration_exclusions
        ),
        "configuration": asdict(config),
        "protocol": protocol_payload,
        "score_freeze_manifest": freeze,
        "provenance": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            # Deliberately path-only: no label-derived hash/count/statistic.
            "evaluation_label_sidecar": str(labels_path) if labels_path else None,
        },
    }
    atomic_json(output / "run.json", result)
    _write_run_state(
        state_path,
        protocol_id=protocol_id,
        stage="complete",
        state="complete",
    )
    return result


__all__ = [
    "GroundingFlowRunConfig",
    "canonical_protocol_id",
    "run_grounding_flow",
]
