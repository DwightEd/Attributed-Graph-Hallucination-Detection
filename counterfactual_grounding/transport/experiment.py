"""Train, score, and separately evaluate causal provenance transport."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..artifacts import atomic_json, atomic_jsonl, canonical_hash, file_sha256
from ..data.graph import Segment
from ..observer_runtime import parse_observer_runtime_identity
from .data import (
    EffectCalibration,
    TeacherTargets,
    TransportTeacher,
    align_teacher_to_graph,
    build_teacher_targets,
    fit_effect_calibration,
    read_transport_teachers,
)
from .model import (
    BLOCK_RANKING_MARGIN,
    CausalProvenanceTransport,
    TransportOutput,
    response_risk,
    transport_loss,
)

REGISTERED_VARIANTS = (
    "true",
    "rewired",
    "mass_only",
    "one_hop",
    "no_residual",
)
VERIFIED_CACHE_MODEL_IDENTITY_STATUSES = frozenset(
    {"verified_extractor_manifest", "verified_legacy_content_manifest"}
)
STRUCTURAL_DETECTION_CONTROLS = frozenset(
    {"rewired", "mass_only", "one_hop", "no_residual", "ungated_true"}
)
REQUIRED_DETECTION_CONTROLS_BY_ENDPOINT = {
    "response": STRUCTURAL_DETECTION_CONTROLS
    | {
        "response_length",
        "negative_response_length",
        "response_evidence_token_count",
        "negative_response_evidence_token_count",
        "response_query_token_count",
        "negative_response_query_token_count",
        "response_seed_count",
        "negative_response_seed_count",
        "response_unknown_mass",
        "negative_response_unknown_mass",
    },
    "token": STRUCTURAL_DETECTION_CONTROLS
    | {
        "token_position",
        "negative_token_position",
        "token_unknown_mass",
        "negative_token_unknown_mass",
        "token_response_length",
        "negative_token_response_length",
        "token_evidence_token_count",
        "negative_token_evidence_token_count",
        "token_query_token_count",
        "negative_token_query_token_count",
        "token_seed_count",
        "negative_token_seed_count",
        "token_response_unknown_mass",
        "negative_token_response_unknown_mass",
    },
}
REQUIRED_DETECTION_CONTROLS = frozenset(
    set().union(*REQUIRED_DETECTION_CONTROLS_BY_ENDPOINT.values())
)
ELIGIBLE_POPULATION_SCOPE = (
    "verified_attention_cache_inventory_intersect_"
    "numeric_digit_surface_preserving_identifiable_subset"
)


def _positive_ci_lower(report: object, *path: str) -> bool:
    current = report
    for field in path:
        if not isinstance(current, Mapping):
            return False
        current = current.get(field)
    return (
        isinstance(current, (int, float))
        and not isinstance(current, bool)
        and math.isfinite(float(current))
        and float(current) > 0.0
    )


def _detection_claim_report(
    paired_detection: Mapping[str, object],
    absolute_detection: Mapping[str, object],
    population: Mapping[str, object],
) -> dict[str, object]:
    """Apply the pre-registered relative, absolute, and population gates."""

    missing = {
        endpoint: sorted(
            control
            for control in required
            if not isinstance(paired_detection.get(control), Mapping)
            or endpoint not in paired_detection[control]
        )
        for endpoint, required in REQUIRED_DETECTION_CONTROLS_BY_ENDPOINT.items()
    }
    structure_supported: dict[str, bool] = {}
    absolute_supported: dict[str, bool] = {}
    for endpoint in ("token", "response"):
        required = REQUIRED_DETECTION_CONTROLS_BY_ENDPOINT[endpoint]
        structure_supported[endpoint] = not missing[endpoint] and all(
            _positive_ci_lower(
                paired_detection.get(control),
                endpoint,
                metric,
                "ci95_low",
            )
            for control in sorted(required)
            for metric in ("auroc_delta", "average_precision_delta")
        )
        endpoint_population = population.get(endpoint)
        absolute_supported[endpoint] = bool(
            isinstance(endpoint_population, Mapping)
            and endpoint_population.get("pass") is True
            and _positive_ci_lower(
                absolute_detection.get(endpoint),
                "auroc_minus_0_5",
                "ci95_low",
            )
            and _positive_ci_lower(
                absolute_detection.get(endpoint),
                "average_precision_minus_prevalence",
                "ci95_low",
            )
        )
    token_supported = bool(
        structure_supported["token"] and absolute_supported["token"]
    )
    response_supported = bool(
        structure_supported["response"] and absolute_supported["response"]
    )
    return {
        "token_supported": token_supported,
        "response_supported": response_supported,
        "structure_superiority_supported": structure_supported,
        "registered_control_superiority_supported": structure_supported,
        "absolute_detection_supported": absolute_supported,
        "population": population,
        "absolute_source_bootstrap": absolute_detection,
        "claim_supported": token_supported and response_supported,
        "missing_controls": missing,
        "decision_rule": (
            "for token and response endpoints separately, source-cluster "
            "bootstrap true-minus-control AUROC/AP lower bounds must exceed zero "
            "against every registered structural and nuisance control; the true "
            "score must also meet "
            "minimum source/class counts and have AUROC-0.5 and AP-prevalence "
            "bootstrap lower bounds above zero"
        ),
    }


def _variant_modes(variant: str) -> tuple[str, str]:
    if variant == "no_residual":
        return "true", "none"
    if variant in {"true", "rewired", "mass_only", "one_hop"}:
        return variant, "learned"
    raise ValueError(f"unsupported transport variant: {variant}")


@dataclass(frozen=True)
class TransportTrainConfig:
    graph_index: Path
    teacher: Path
    output_dir: Path
    score_index: Path | None = None
    device: str = "cuda:0"
    epochs: int = 100
    learning_rate: float = 0.03
    validation_fraction: float = 0.2
    mechanism_holdout_fraction: float = 0.2
    noise_floor: float = 1e-4
    block_weight: float = 1.0
    allow_unidentifiable_pilot: bool = False
    minimum_mechanism_sources: int = 20
    minimum_positive_sources: int = 10
    minimum_positive_events: int = 50
    minimum_target_variance: float = 1e-6
    minimum_block_events: int = 25
    minimum_block_pairs: int = 50
    minimum_rewire_ry_changed_fraction: float = 0.10
    minimum_rewire_source_coverage: float = 0.80
    minimum_detection_sources: int = 20
    minimum_detection_positive_sources: int = 5
    minimum_detection_negative_sources: int = 5
    minimum_detection_positive_samples: int = 20
    minimum_detection_negative_samples: int = 20
    seed: int = 42
    variants: tuple[str, ...] = REGISTERED_VARIANTS

    def validate(self) -> None:
        for path, name in (
            (self.graph_index, "graph index"),
            (self.teacher, "teacher"),
        ):
            if not path.expanduser().is_file():
                raise FileNotFoundError(f"transport {name} is absent: {path}")
        if self.score_index is not None and not self.score_index.expanduser().is_file():
            raise FileNotFoundError(
                f"transport score index is absent: {self.score_index}"
            )
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1)")
        if not 0.0 < self.mechanism_holdout_fraction < 1.0:
            raise ValueError("mechanism_holdout_fraction must lie in (0, 1)")
        if self.validation_fraction + self.mechanism_holdout_fraction >= 1.0:
            raise TypeError(
                "validation and mechanism holdout fractions must sum to less than 1"
            )
        if self.noise_floor <= 0 or not math.isfinite(self.noise_floor):
            raise ValueError("noise_floor must be finite and positive")
        if self.block_weight < 0 or not math.isfinite(self.block_weight):
            raise ValueError("block_weight must be finite and non-negative")
        if (
            self.minimum_mechanism_sources < 2
            or self.minimum_positive_sources < 2
            or self.minimum_positive_events < 2
            or self.minimum_block_events < 2
            or self.minimum_block_pairs < 2
            or self.minimum_target_variance <= 0
            or not math.isfinite(self.minimum_target_variance)
            or not 0 < self.minimum_rewire_ry_changed_fraction <= 1
            or not 0 < self.minimum_rewire_source_coverage <= 1
            or self.minimum_detection_sources < 2
            or self.minimum_detection_positive_sources < 2
            or self.minimum_detection_negative_sources < 2
            or self.minimum_detection_positive_samples < 2
            or self.minimum_detection_negative_samples < 2
            or self.minimum_detection_positive_sources
            > self.minimum_detection_sources
            or self.minimum_detection_negative_sources
            > self.minimum_detection_sources
        ):
            raise ValueError("transport Gate-2 population minima are invalid")
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("transport variants must be non-empty and unique")
        unknown = set(self.variants).difference(REGISTERED_VARIANTS)
        if unknown or "true" not in self.variants:
            raise ValueError(f"unsupported transport variants: {sorted(unknown)}")


@dataclass(frozen=True)
class _GraphIndexRow:
    source_id: str
    response_id: str
    official_split: str
    task_type: str
    generator_model: str
    graph_path: Path
    graph_sha256: str
    model_source_signature: str
    cache_model_source_signature: str | None
    cache_model_identity_status: str
    observer_runtime: Mapping[str, str]
    observer_runtime_signature: str
    cache_path: Path | None
    cache_sha256: str | None
    extractor_declared_cache_sha256: str | None
    cache_content_identity_status: str
    cache_content_identity_evidence: str
    cache_content_inventory_signature: str
    deployment_seed_protocol: str
    deployment_seed_positions: tuple[int, ...]


@dataclass(frozen=True)
class _ExampleRef:
    row: _GraphIndexRow
    teacher: TransportTeacher
    graph: Mapping[str, object]


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"graph index row is not an object: {path}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"graph index is empty: {path}")
    return rows


def _resolve_index_path(raw: object, *, index: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = index.expanduser().resolve().parent / path
    return path.resolve()


def _graph_index(path: Path) -> tuple[_GraphIndexRow, ...]:
    index = path.expanduser().resolve()
    rows: list[_GraphIndexRow] = []
    for raw in _read_jsonl(index):
        if raw.get("schema") != "cept-canonical-graph-index-v2":
            raise ValueError("unsupported canonical graph index schema")
        source_id = str(raw.get("source_id", "")).strip()
        response_id = str(raw.get("response_id", "")).strip()
        split = str(raw.get("official_split", "")).casefold()
        task_type = str(raw.get("task_type", "")).strip()
        generator_model = str(raw.get("generator_model", "")).strip()
        graph_path = _resolve_index_path(raw.get("graph_path"), index=index)
        graph_sha256 = str(raw.get("graph_sha256", "")).strip()
        model_source_signature = str(raw.get("model_source_signature", "")).strip()
        cache_model_source_signature = (
            str(raw["cache_model_source_signature"]).strip()
            if raw.get("cache_model_source_signature")
            else None
        )
        cache_model_identity_status = str(
            raw.get("cache_model_identity_status", "")
        ).strip()
        try:
            observer_runtime, observer_runtime_signature = (
                parse_observer_runtime_identity(
                    raw.get("observer_runtime"),
                    raw.get("observer_runtime_signature"),
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "canonical graph index has missing or invalid observer runtime identity"
            ) from error
        cache_path = (
            _resolve_index_path(raw["cache_path"], index=index)
            if raw.get("cache_path")
            else None
        )
        cache_sha256 = str(raw.get("cache_sha256", "")).strip() or None
        extractor_declared_cache_sha256 = (
            str(raw.get("extractor_declared_cache_sha256", "")).strip() or None
        )
        cache_content_identity_status = str(
            raw.get("cache_content_identity_status", "")
        ).strip()
        cache_content_identity_evidence = str(
            raw.get("cache_content_identity_evidence", "")
        ).strip()
        cache_content_inventory_signature = str(
            raw.get("cache_content_inventory_signature", "")
        ).strip()
        deployment_seed_protocol = str(
            raw.get("deployment_seed_protocol", "")
        ).strip()
        raw_seed_positions = raw.get("deployment_seed_positions")
        if not isinstance(raw_seed_positions, Sequence) or isinstance(
            raw_seed_positions, (str, bytes)
        ):
            raise TypeError(
                "canonical graph index lacks deployment_seed_positions"
            )
        deployment_seed_positions = tuple(int(value) for value in raw_seed_positions)
        if (
            not source_id
            or not response_id
            or split not in {"train", "test"}
            or not task_type
            or not generator_model
            or len(graph_sha256) != 64
            or not model_source_signature
            or cache_model_identity_status
            not in (
                VERIFIED_CACHE_MODEL_IDENTITY_STATUSES
                | {"unverified_legacy_manifest"}
            )
            or deployment_seed_protocol
            != "numeric_digit_surface_preserving_v1_first_candidate"
            or len(set(deployment_seed_positions)) != len(deployment_seed_positions)
            or any(position < 0 for position in deployment_seed_positions)
        ):
            raise ValueError("canonical graph index contains an invalid identity/split")
        if not graph_path.is_file():
            raise FileNotFoundError(
                f"canonical prediction graph is absent: {graph_path}"
            )
        rows.append(
            _GraphIndexRow(
                source_id,
                response_id,
                split,
                task_type,
                generator_model,
                graph_path,
                graph_sha256,
                model_source_signature,
                cache_model_source_signature,
                cache_model_identity_status,
                observer_runtime,
                observer_runtime_signature,
                cache_path,
                cache_sha256,
                extractor_declared_cache_sha256,
                cache_content_identity_status,
                cache_content_identity_evidence,
                cache_content_inventory_signature,
                deployment_seed_protocol,
                deployment_seed_positions,
            )
        )
    identities = [(row.source_id, row.response_id) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("canonical graph index contains duplicate identities")
    return tuple(rows)


def _cache_estimand_scope(
    rows: Sequence[_GraphIndexRow],
) -> dict[str, object]:
    """Describe the exact cache population represented by an index."""

    if not rows:
        raise ValueError("cache estimand scope requires graph-index rows")
    splits = {row.official_split for row in rows}
    if len(splits) != 1:
        raise ValueError("cache estimand scope cannot mix official splits")
    return {
        "official_split": next(iter(splits)),
        "cache_inventory_samples": len(rows),
        "source_count": len({row.source_id for row in rows}),
        "generator_models": dict(
            sorted(Counter(row.generator_model for row in rows).items())
        ),
        "task_types": dict(sorted(Counter(row.task_type for row in rows).items())),
    }


def _require_official_split(
    rows: Sequence[_GraphIndexRow], *, expected: str, context: str
) -> None:
    observed = sorted({row.official_split for row in rows})
    if observed != [expected]:
        raise ValueError(
            f"{context} requires only official {expected} rows; observed={observed}"
        )


def _load_graph(row: _GraphIndexRow) -> Mapping[str, object]:
    path = row.graph_path
    if file_sha256(path) != row.graph_sha256:
        raise RuntimeError(
            f"canonical graph file hash disagrees with its index: {path}"
        )
    try:
        graph = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        graph = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(graph, Mapping):
        raise TypeError(f"canonical graph is not a mapping: {path}")
    forbidden = [
        str(key)
        for key in graph
        if "label" in str(key).casefold() or str(key).casefold() == "y_token"
    ]
    if forbidden:
        raise ValueError(f"canonical training graph leaked labels: {path}: {forbidden}")
    payload_identity = (
        str(graph.get("source_id", "")),
        str(graph.get("response_id", "")),
        str(graph.get("dataset_split", "")),
    )
    if payload_identity != (row.source_id, row.response_id, row.official_split):
        raise ValueError(
            f"canonical graph payload identity disagrees with index: {path}"
        )
    if (
        row.cache_sha256 is not None
        and str(graph.get("attention_cache_sha256", "")) != row.cache_sha256
    ):
        raise RuntimeError(
            f"canonical graph payload cache hash disagrees with index: {path}"
        )
    return graph


def _require_verified_cache_identity(
    row: _GraphIndexRow, *, context: str
) -> None:
    if (
        row.cache_model_identity_status
        not in VERIFIED_CACHE_MODEL_IDENTITY_STATUSES
        or row.cache_model_source_signature != row.model_source_signature
    ):
        raise RuntimeError(
            f"{context} has unverified cache-model identity: {row.response_id}"
        )
    if (
        row.cache_content_identity_status
        != "verified_extractor_cache_files_sha256"
        or row.cache_content_identity_evidence
        != "cache_files_sha256_exact_bytes"
        or row.extractor_declared_cache_sha256 != row.cache_sha256
        or not row.cache_content_inventory_signature.startswith("sha256:")
        or len(row.cache_content_inventory_signature) != 71
    ):
        raise RuntimeError(
            f"{context} has unverified cache content identity: {row.response_id}"
        )


def _join_training_data(
    index: Path, teacher_path: Path
) -> tuple[tuple[_ExampleRef, ...], tuple[_GraphIndexRow, ...]]:
    rows = _graph_index(index)
    _require_official_split(rows, expected="train", context="transport training")
    by_identity = {(row.source_id, row.response_id): row for row in rows}
    teachers = read_transport_teachers(teacher_path)
    expected_runtime = teachers[0].observer_runtime
    expected_runtime_signature = teachers[0].observer_runtime_signature
    expected_model_source_signature = teachers[0].model_source_signature
    for teacher in teachers:
        if (
            teacher.observer_runtime_signature != expected_runtime_signature
            or dict(teacher.observer_runtime) != dict(expected_runtime)
        ):
            raise RuntimeError("transport teachers have heterogeneous observer runtimes")
    for row in rows:
        _require_verified_cache_identity(row, context="transport training graph")
        if row.model_source_signature != expected_model_source_signature:
            raise RuntimeError(
                "transport training graph model source differs from teacher"
            )
        if (
            row.observer_runtime_signature != expected_runtime_signature
            or dict(row.observer_runtime) != dict(expected_runtime)
        ):
            raise RuntimeError(
                "teacher/attention observer runtime mismatch: "
                f"{(row.source_id, row.response_id)}"
            )
    refs: list[_ExampleRef] = []
    for teacher in teachers:
        identity = (teacher.source_id, teacher.response_id)
        if identity not in by_identity:
            raise ValueError(f"transport teacher has no canonical graph: {identity}")
        row = by_identity[identity]
        if teacher.model_source_signature != row.model_source_signature:
            raise RuntimeError(
                f"teacher/attention model source signature mismatch: {identity}"
            )
        if teacher.changed_evidence_positions != row.deployment_seed_positions:
            raise RuntimeError(
                "teacher and deployed intervention-seed operator disagree: "
                f"{identity}"
            )
        graph = _load_graph(row)
        align_teacher_to_graph(teacher, graph)
        refs.append(_ExampleRef(row=row, teacher=teacher, graph=graph))
    return tuple(refs), rows


def _split_sources(
    refs: Sequence[_ExampleRef],
    *,
    validation_fraction: float,
    mechanism_fraction: float,
    minimum_mechanism_sources: int,
    seed: int,
) -> tuple[tuple[_ExampleRef, ...], tuple[_ExampleRef, ...], tuple[_ExampleRef, ...]]:
    sources = sorted({ref.row.source_id for ref in refs})
    if len(sources) < minimum_mechanism_sources + 2:
        raise ValueError(
            "transport training has too few independent source groups for the "
            "pre-registered mechanism holdout"
        )
    ranked = sorted(
        sources,
        key=lambda source: hashlib.sha256(f"{seed}\0{source}".encode()).hexdigest(),
    )
    validation_count = min(
        len(sources) - minimum_mechanism_sources - 1,
        max(1, round(validation_fraction * len(sources))),
    )
    mechanism_count = min(
        len(sources) - validation_count - 1,
        max(minimum_mechanism_sources, round(mechanism_fraction * len(sources))),
    )
    validation_sources = set(ranked[:validation_count])
    mechanism_sources = set(
        ranked[validation_count : validation_count + mechanism_count]
    )
    train = tuple(
        ref
        for ref in refs
        if ref.row.source_id not in validation_sources | mechanism_sources
    )
    validation = tuple(ref for ref in refs if ref.row.source_id in validation_sources)
    mechanism = tuple(ref for ref in refs if ref.row.source_id in mechanism_sources)
    if not train or not validation or not mechanism:
        raise RuntimeError("three-way source-group split produced an empty partition")
    return train, validation, mechanism


def _model_shape(ref: _ExampleRef) -> tuple[int, int]:
    graph = ref.graph
    return int(graph["num_layers"]), int(graph["num_heads"])


def _preflight_score_rows(
    rows: Sequence[_GraphIndexRow],
    *,
    expected_model_source_signature: str,
    expected_observer_runtime: Mapping[str, str],
    expected_observer_runtime_signature: str,
    expected_shape: tuple[int, int],
) -> None:
    """Validate every frozen-score artifact before any GPU training starts."""

    signatures = {row.model_source_signature for row in rows}
    if signatures != {expected_model_source_signature}:
        raise RuntimeError(
            "score graphs and intervention teacher use different model sources"
        )
    runtime_signatures = {row.observer_runtime_signature for row in rows}
    if runtime_signatures != {expected_observer_runtime_signature} or any(
        dict(row.observer_runtime) != dict(expected_observer_runtime) for row in rows
    ):
        raise RuntimeError(
            "score observer runtime mismatch with intervention teacher"
        )
    for row in rows:
        _require_verified_cache_identity(row, context="transport score graph")
        if (
            row.cache_path is None
            or row.cache_sha256 is None
            or not row.cache_path.is_file()
        ):
            raise FileNotFoundError(
                f"score row lacks an existing bound attention cache: {row.response_id}"
            )
        if file_sha256(row.cache_path) != row.cache_sha256:
            raise RuntimeError(
                f"score attention cache hash disagrees with index: {row.cache_path}"
            )
        graph = _load_graph(row)
        shape = (int(graph["num_layers"]), int(graph["num_heads"]))
        if shape != expected_shape:
            raise ValueError(
                "score graph layer/head shape disagrees with training graphs"
            )
        seeds = torch.tensor(row.deployment_seed_positions, dtype=torch.long)
        if seeds.numel():
            segment = torch.as_tensor(graph["segment_ids"]).long().flatten()
            if bool((seeds >= segment.numel()).any()) or bool(
                (segment.index_select(0, seeds) != int(Segment.EVIDENCE)).any()
            ):
                raise RuntimeError(
                    f"score deployment seed is outside evidence: {row.response_id}"
                )


def _example_loss(
    model: CausalProvenanceTransport,
    ref: _ExampleRef,
    *,
    variant: str,
    calibration: EffectCalibration,
    block_weight: float,
) -> torch.Tensor:
    graph = ref.graph
    align_teacher_to_graph(ref.teacher, graph)
    incidence, residual_mode = _variant_modes(variant)
    output = model(
        graph,
        seed_positions=ref.teacher.changed_evidence_positions,
        block_positions=ref.teacher.block_positions,
        incidence=incidence,
        residual_mode=residual_mode,
    )
    targets = build_teacher_targets(ref.teacher, calibration=calibration)
    return transport_loss(output, targets, block_weight=block_weight)


def _partition_loss(
    model: CausalProvenanceTransport,
    refs: Sequence[_ExampleRef],
    *,
    variant: str,
    calibration: EffectCalibration,
    block_weight: float,
) -> float:
    values = [
        float(
            _example_loss(
                model,
                ref,
                variant=variant,
                calibration=calibration,
                block_weight=block_weight,
            )
        )
        for ref in refs
    ]
    return sum(values) / len(values)


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _rank(values: torch.Tensor) -> torch.Tensor:
    unique, inverse, counts = torch.unique(
        values, sorted=True, return_inverse=True, return_counts=True
    )
    del unique
    end = counts.cumsum(0).to(torch.float64)
    start = end - counts.to(torch.float64)
    average = (start + end - 1.0) / 2.0
    return average.index_select(0, inverse)


def _spearman(prediction: Sequence[float], target: Sequence[float]) -> float:
    if len(prediction) < 2 or len(set(prediction)) < 2 or len(set(target)) < 2:
        return 0.0
    x = _rank(torch.tensor(prediction, dtype=torch.float64))
    y = _rank(torch.tensor(target, dtype=torch.float64))
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x * y).sum() / denominator.clamp_min(1e-12))


def _variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(torch.tensor(values, dtype=torch.float64).var(unbiased=False))


def _informative_block_pairs(
    targets: TeacherTargets,
    available: torch.Tensor,
    *,
    prediction: torch.Tensor | None = None,
) -> tuple[list[float], int]:
    """Return within-event block margins or ranking outcomes.

    Cross-event block magnitudes are not comparable: K/V rescue scale changes
    with the predicted token.  A pair is therefore informative only when two
    blocks for the same prediction event have distinct teacher effects.  With
    ``prediction=None`` the returned values are absolute teacher margins;
    otherwise they are pairwise accuracies in ``{0, .5, 1}``.
    """

    block = torch.as_tensor(targets.block).float().cpu()
    mask = torch.as_tensor(targets.block_mask).bool().cpu()
    available = torch.as_tensor(available).bool().flatten().cpu()
    predicted = (
        torch.as_tensor(prediction).float().cpu()
        if prediction is not None
        else None
    )
    if block.shape != mask.shape or block.ndim != 2:
        raise ValueError("teacher block values/mask must be aligned matrices")
    if block.shape[1] != available.numel():
        raise ValueError("teacher block events disagree with availability mask")
    if predicted is not None and predicted.shape != block.shape:
        raise ValueError("predicted and teacher block matrices must align")

    values: list[float] = []
    informative_events = 0
    for event in range(block.shape[1]):
        if not bool(available[event]):
            continue
        indices = torch.nonzero(mask[:, event]).flatten().tolist()
        event_pairs = 0
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1 :]:
                target_delta = float(block[left, event] - block[right, event])
                if abs(target_delta) <= BLOCK_RANKING_MARGIN:
                    continue
                event_pairs += 1
                if predicted is None:
                    values.append(abs(target_delta))
                    continue
                prediction_delta = float(
                    predicted[left, event] - predicted[right, event]
                )
                if abs(prediction_delta) <= 1e-12:
                    values.append(0.5)
                else:
                    values.append(
                        float(prediction_delta * target_delta > 0.0)
                    )
        informative_events += int(event_pairs > 0)
    return values, informative_events


def _teacher_population_audit(
    refs: Sequence[_ExampleRef],
    *,
    calibration: EffectCalibration,
    config: TransportTrainConfig,
) -> dict[str, object]:
    """Pre-registered label-free identifiability checks for teacher targets."""

    positive_events = 0
    null_events = 0
    contradictory_events = 0
    excluded_unreliable_positive_events = 0
    positive_sources: set[str] = set()
    support_values: list[float] = []
    history_values: list[float] = []
    block_pair_margins: list[float] = []
    block_events = 0
    history_nonzero = 0
    block_sources: set[str] = set()
    for ref in refs:
        targets = build_teacher_targets(ref.teacher, calibration=calibration)
        available = torch.as_tensor(ref.graph["row_available"]).bool().flatten()
        positive = targets.positive_mask & available
        null = targets.null_mask & available
        contradictory = targets.contradictory_mask & available
        excluded_unreliable_positive = available & ~(
            positive | null | contradictory
        )
        positive_events += int(positive.sum())
        null_events += int(null.sum())
        contradictory_events += int(contradictory.sum())
        excluded_unreliable_positive_events += int(
            excluded_unreliable_positive.sum()
        )
        if bool(positive.any()):
            positive_sources.add(ref.row.source_id)
        support_values.extend(targets.support[positive].tolist())
        history_values.extend(targets.history[positive].tolist())
        history_nonzero += int((targets.history[positive] > 0).sum())
        pair_margins, informative_events = _informative_block_pairs(
            targets, available
        )
        block_pair_margins.extend(pair_margins)
        block_events += informative_events
        if pair_margins:
            block_sources.add(ref.row.source_id)
    denominator = positive_events + null_events
    checks = {
        "mechanism_sources_at_least_minimum": (
            len({ref.row.source_id for ref in refs})
            >= config.minimum_mechanism_sources
        ),
        "reliable_positive_sources_at_least_minimum": (
            len(positive_sources) >= config.minimum_positive_sources
        ),
        "reliable_positive_events_at_least_minimum": (
            positive_events >= config.minimum_positive_events
        ),
        "positive_fraction_between_0_01_and_0_95": (
            denominator > 0 and 0.01 <= positive_events / denominator <= 0.95
        ),
        "support_target_variance_at_least_minimum": (
            _variance(support_values) >= config.minimum_target_variance
        ),
        "history_nonzero_events_at_least_half_minimum": (
            history_nonzero >= max(2, config.minimum_positive_events // 2)
        ),
        "history_target_variance_at_least_minimum": (
            _variance(history_values) >= config.minimum_target_variance
        ),
        "informative_block_sources_at_least_minimum": (
            len(block_sources) >= config.minimum_positive_sources
        ),
        "informative_block_events_at_least_minimum": (
            block_events >= config.minimum_block_events
        ),
        "informative_block_pairs_at_least_minimum": (
            len(block_pair_margins) >= config.minimum_block_pairs
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "sources": len({ref.row.source_id for ref in refs}),
        "positive_sources": len(positive_sources),
        "positive_events": positive_events,
        "null_events": null_events,
        "contradictory_negative_events_excluded": contradictory_events,
        "unreliable_positive_events_excluded": (
            excluded_unreliable_positive_events
        ),
        "positive_fraction": positive_events / max(1, denominator),
        "support_target_variance": _variance(support_values),
        "history_nonzero_events": history_nonzero,
        "history_target_variance": _variance(history_values),
        "block_sources": len(block_sources),
        "informative_block_events": block_events,
        "informative_block_pairs": len(block_pair_margins),
        "block_pair_margin_mean": (
            sum(block_pair_margins) / max(1, len(block_pair_margins))
        ),
        "block_pair_margin_variance": _variance(block_pair_margins),
    }


def _teacher_fidelity(
    model: CausalProvenanceTransport,
    refs: Sequence[_ExampleRef],
    *,
    variant: str,
    calibration: EffectCalibration,
) -> dict[str, object]:
    incidence, residual_mode = _variant_modes(variant)
    support_pred: list[float] = []
    support_true: list[float] = []
    history_pred: list[float] = []
    history_true: list[float] = []
    positive_support_pred: list[float] = []
    positive_support_true: list[float] = []
    positive_history_pred: list[float] = []
    positive_history_true: list[float] = []
    block_pred: list[float] = []
    block_true: list[float] = []
    with torch.no_grad():
        for ref in refs:
            graph = ref.graph
            output = model(
                graph,
                seed_positions=ref.teacher.changed_evidence_positions,
                block_positions=ref.teacher.block_positions,
                incidence=incidence,
                residual_mode=residual_mode,
            )
            targets = build_teacher_targets(ref.teacher, calibration=calibration)
            event_mask = output.row_available.detach().cpu().bool() & (
                targets.event_weight > 0
            )
            support_pred.extend(
                output.support_lower.detach().cpu()[event_mask].tolist()
            )
            support_true.extend(targets.support[event_mask].tolist())
            history_pred.extend(
                output.history_lower.detach().cpu()[event_mask].tolist()
            )
            history_true.extend(targets.history[event_mask].tolist())
            positive_mask = output.row_available.detach().cpu().bool() & (
                targets.positive_mask
            )
            positive_support_pred.extend(
                output.support_lower.detach().cpu()[positive_mask].tolist()
            )
            positive_support_true.extend(targets.support[positive_mask].tolist())
            positive_history_pred.extend(
                output.history_lower.detach().cpu()[positive_mask].tolist()
            )
            positive_history_true.extend(targets.history[positive_mask].tolist())
            block_mask = targets.block_mask & event_mask.unsqueeze(0)
            block_pred.extend(output.block.detach().cpu()[block_mask].tolist())
            block_true.extend(targets.block[block_mask].tolist())

    def metrics(
        prediction: Sequence[float], target: Sequence[float]
    ) -> dict[str, float | int]:
        if not prediction:
            return {"samples": 0, "mae": 0.0, "spearman": 0.0}
        absolute = [abs(a - b) for a, b in zip(prediction, target, strict=True)]
        return {
            "samples": len(prediction),
            "mae": sum(absolute) / len(absolute),
            "spearman": _spearman(prediction, target),
        }

    support = metrics(support_pred, support_true)
    history = metrics(history_pred, history_true)
    positive_support = metrics(positive_support_pred, positive_support_true)
    positive_history = metrics(positive_history_pred, positive_history_true)
    block = metrics(block_pred, block_true)
    aggregate_mae = (
        float(support["mae"]) + float(history["mae"]) + float(block["mae"])
    ) / 3.0
    return {
        "variant": variant,
        "support": support,
        "history": history,
        "reliable_positive_support": positive_support,
        "reliable_positive_history": positive_history,
        "block": block,
        "aggregate_mae": aggregate_mae,
    }


def _component_fidelity_bootstrap(
    model: CausalProvenanceTransport,
    refs: Sequence[_ExampleRef],
    *,
    calibration: EffectCalibration,
    config: TransportTrainConfig,
    seed: int,
    draws: int = 1000,
) -> dict[str, object]:
    """Source-cluster bootstrap of intervention-teacher fidelity.

    Support/history are evaluated across every source with a reliable-positive
    target, including sources that contribute only one event.  History-block
    fidelity is evaluated only through within-event pair rankings because raw
    rescue magnitudes from different predicted tokens are not comparable.
    """

    grouped: dict[str, dict[str, tuple[list[float], list[float]]]] = {
        name: {} for name in ("support", "history")
    }
    block_pairs_by_source: dict[str, list[float]] = {}
    block_events_by_source: dict[str, int] = {}

    def extend(
        component: str,
        source: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        values = grouped[component].setdefault(source, ([], []))
        values[0].extend(prediction.detach().cpu()[mask].tolist())
        values[1].extend(target[mask].tolist())

    model.eval()
    with torch.no_grad():
        for ref in refs:
            output = model(
                ref.graph,
                seed_positions=ref.teacher.changed_evidence_positions,
                block_positions=ref.teacher.block_positions,
                incidence="true",
                residual_mode="learned",
            )
            targets = build_teacher_targets(ref.teacher, calibration=calibration)
            available = output.row_available.detach().cpu().bool()
            positive = targets.positive_mask & available
            extend(
                "support",
                ref.row.source_id,
                output.support_lower,
                targets.support,
                positive,
            )
            extend(
                "history",
                ref.row.source_id,
                output.history_lower,
                targets.history,
                positive,
            )
            pair_outcomes, informative_events = _informative_block_pairs(
                targets,
                available,
                prediction=output.block.detach().cpu(),
            )
            block_pairs_by_source.setdefault(ref.row.source_id, []).extend(
                pair_outcomes
            )
            block_events_by_source[ref.row.source_id] = (
                block_events_by_source.get(ref.row.source_id, 0)
                + informative_events
            )

    reports: dict[str, object] = {}
    for component_index, (component, by_source) in enumerate(grouped.items()):
        sources = sorted(
            source
            for source, (_, target) in by_source.items()
            if target
        )
        all_prediction = [
            value for source in sources for value in by_source[source][0]
        ]
        all_target = [value for source in sources for value in by_source[source][1]]
        generator = random.Random(seed + component_index)
        samples: list[float] = []
        attempts = 0
        while len(samples) < draws and attempts < draws * 3 and sources:
            attempts += 1
            selected = [generator.choice(sources) for _ in sources]
            prediction = [
                value for source in selected for value in by_source[source][0]
            ]
            target = [value for source in selected for value in by_source[source][1]]
            if len(set(target)) < 2 or len(set(prediction)) < 2:
                continue
            samples.append(_spearman(prediction, target))
        values = torch.tensor(samples, dtype=torch.float64)
        reports[component] = {
            "sources_with_targets": len(sources),
            "samples": len(all_target),
            "point_spearman": _spearman(all_prediction, all_target),
            "bootstrap_draws": len(samples),
            "ci95_low": (
                float(torch.quantile(values, 0.025)) if samples else 0.0
            ),
            "ci95_high": (
                float(torch.quantile(values, 0.975)) if samples else 0.0
            ),
            "pass": (
                len(sources) >= config.minimum_positive_sources
                and len(samples) >= draws // 2
                and bool(samples)
                and float(torch.quantile(values, 0.025)) > 0.0
            ),
        }

    block_sources = sorted(
        source for source, outcomes in block_pairs_by_source.items() if outcomes
    )
    block_outcomes = [
        outcome
        for source in block_sources
        for outcome in block_pairs_by_source[source]
    ]
    block_events = sum(block_events_by_source.get(source, 0) for source in block_sources)
    generator = random.Random(seed + len(grouped))
    block_samples: list[float] = []
    if block_sources:
        for _ in range(draws):
            selected = [generator.choice(block_sources) for _ in block_sources]
            outcomes = [
                outcome
                for source in selected
                for outcome in block_pairs_by_source[source]
            ]
            if outcomes:
                block_samples.append(sum(outcomes) / len(outcomes))
    block_values = torch.tensor(block_samples, dtype=torch.float64)
    block_report = {
        "estimand": "within_event_pairwise_ranking_accuracy",
        "sources_with_informative_pairs": len(block_sources),
        "informative_events": block_events,
        "informative_pairs": len(block_outcomes),
        "point_pair_accuracy": (
            sum(block_outcomes) / len(block_outcomes) if block_outcomes else 0.0
        ),
        "bootstrap_draws": len(block_samples),
        "ci95_low": (
            float(torch.quantile(block_values, 0.025)) if block_samples else 0.0
        ),
        "ci95_high": (
            float(torch.quantile(block_values, 0.975)) if block_samples else 0.0
        ),
        "pass": (
            len(block_sources) >= config.minimum_positive_sources
            and block_events >= config.minimum_block_events
            and len(block_outcomes) >= config.minimum_block_pairs
            and len(block_samples) >= draws // 2
            and bool(block_samples)
            and float(torch.quantile(block_values, 0.025)) > 0.5
        ),
    }
    reports["block"] = block_report
    reports["pass"] = all(
        isinstance(reports[name], Mapping)
        and reports[name].get("pass") is True
        for name in ("support", "history", "block")
    )
    return reports


def _train_variant(
    config: TransportTrainConfig,
    *,
    variant: str,
    train: Sequence[_ExampleRef],
    validation: Sequence[_ExampleRef],
    num_layers: int,
    num_heads: int,
    calibration: EffectCalibration,
) -> tuple[CausalProvenanceTransport, dict[str, object]]:
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for transport training but is unavailable"
        )
    model = CausalProvenanceTransport(num_layers=num_layers, num_heads=num_heads).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    order = list(train)
    for epoch in range(1, config.epochs + 1):
        random.Random(config.seed + epoch).shuffle(order)
        model.train()
        train_losses: list[float] = []
        for ref in order:
            optimizer.zero_grad(set_to_none=True)
            loss = _example_loss(
                model,
                ref,
                variant=variant,
                calibration=calibration,
                block_weight=config.block_weight,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = _partition_loss(
                model,
                validation,
                variant=variant,
                calibration=calibration,
                block_weight=config.block_weight,
            )
        train_loss = sum(train_losses) / len(train_losses)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            json.dumps(
                {
                    "event": "transport_epoch",
                    "variant": variant,
                    "epoch": epoch,
                    "epochs": config.epochs,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            ),
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("transport training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint = {
        "schema": "cept-layer-provenance-transport-checkpoint-v2",
        "variant": variant,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "config": asdict(config),
    }
    checkpoint_path = config.output_dir / f"{variant}.pt"
    _atomic_torch_save(checkpoint_path, checkpoint)
    fidelity = _teacher_fidelity(
        model,
        validation,
        variant=variant,
        calibration=calibration,
    )
    return model, {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "teacher_fidelity": fidelity,
    }


def _source_validation_losses(
    model: CausalProvenanceTransport,
    refs: Sequence[_ExampleRef],
    *,
    variant: str,
    calibration: EffectCalibration,
    block_weight: float,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    model.eval()
    with torch.no_grad():
        for ref in refs:
            grouped.setdefault(ref.row.source_id, []).append(
                float(
                    _example_loss(
                        model,
                        ref,
                        variant=variant,
                        calibration=calibration,
                        block_weight=block_weight,
                    )
                )
            )
    return {source: sum(values) / len(values) for source, values in grouped.items()}


def _zero_source_losses(
    refs: Sequence[_ExampleRef],
    *,
    calibration: EffectCalibration,
    block_weight: float,
) -> dict[str, float]:
    """Constant-zero baseline under the exact same masks and balanced loss."""

    grouped: dict[str, list[float]] = {}
    for ref in refs:
        targets = build_teacher_targets(ref.teacher, calibration=calibration)
        available = torch.as_tensor(ref.graph["row_available"]).bool().flatten()
        zeros = torch.zeros_like(targets.support)
        output = TransportOutput(
            direct=zeros,
            history_lower=zeros,
            history_upper=zeros,
            support_lower=zeros,
            support_upper=zeros,
            block=torch.zeros_like(targets.block),
            row_available=available,
            unsupported_history_lower=zeros,
            unsupported_history_upper=zeros,
        )
        grouped.setdefault(ref.row.source_id, []).append(
            float(transport_loss(output, targets, block_weight=block_weight))
        )
    return {source: sum(values) / len(values) for source, values in grouped.items()}


def _paired_bootstrap_improvement(
    primary: Mapping[str, float],
    control: Mapping[str, float],
    *,
    seed: int,
    draws: int = 2000,
) -> dict[str, float | int]:
    sources = sorted(set(primary) & set(control))
    if set(sources) != set(primary) or set(sources) != set(control) or not sources:
        raise ValueError(
            "paired source losses do not share an identical non-empty domain"
        )
    differences = torch.tensor(
        [control[source] - primary[source] for source in sources],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(sources), (draws, len(sources)), generator=generator)
    means = differences.index_select(0, indices.flatten()).view(draws, -1).mean(1)
    return {
        "sources": len(sources),
        "mean_control_minus_true": float(differences.mean()),
        "ci95_low": float(torch.quantile(means, 0.025)),
        "ci95_high": float(torch.quantile(means, 0.975)),
    }


def _informativeness_report(
    token_records: Sequence[Mapping[str, object]],
    response_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    available_tokens = [row for row in token_records if row["row_available"]]
    scorable_responses = [
        row for row in response_records if int(row["scorable_events"]) > 0
    ]
    token_scores = [float(row["score"]) for row in available_tokens]
    interval_widths = [
        max(0.0, float(row["score_upper"]) - float(row["score"]))
        for row in available_tokens
    ]
    score_variance = _variance(token_scores)
    token_coverage = len(available_tokens) / max(1, len(token_records))
    zero_score_fraction = sum(score <= 1e-8 for score in token_scores) / max(
        1, len(token_scores)
    )
    mean_interval_width = sum(interval_widths) / max(1, len(interval_widths))
    unknown_correlation = _spearman(
        token_scores,
        [float(row["unknown_mass_mean"]) for row in available_tokens],
    )
    length_correlation = _spearman(
        [float(row["routing_risk_score"]) for row in scorable_responses],
        [float(row["response_token_count"]) for row in scorable_responses],
    )
    evidence_count_correlation = _spearman(
        [float(row["routing_risk_score"]) for row in scorable_responses],
        [float(row["evidence_token_count"]) for row in scorable_responses],
    )
    checks = {
        "token_coverage_at_least_0_90": token_coverage >= 0.90,
        "zero_score_fraction_below_0_50": zero_score_fraction < 0.50,
        "score_variance_at_least_1e_6": score_variance >= 1e-6,
        "mean_interval_width_at_most_0_25": mean_interval_width <= 0.25,
        "unknown_mass_abs_spearman_below_0_50": abs(unknown_correlation) < 0.50,
        "response_length_abs_spearman_below_0_50": abs(length_correlation) < 0.50,
        "evidence_count_abs_spearman_below_0_50": (
            abs(evidence_count_correlation) < 0.50
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "available_tokens": len(available_tokens),
        "total_tokens": len(token_records),
        "token_coverage": token_coverage,
        "scorable_responses": len(scorable_responses),
        "total_responses": len(response_records),
        "zero_score_fraction": zero_score_fraction,
        "score_variance": score_variance,
        "mean_interval_width": mean_interval_width,
        "unknown_mass_spearman": unknown_correlation,
        "response_length_spearman": length_correlation,
        "evidence_count_spearman": evidence_count_correlation,
    }


def _mechanism_informativeness(
    model: CausalProvenanceTransport,
    refs: Sequence[_ExampleRef],
) -> dict[str, object]:
    """Audit deployed scores on the untouched train-side mechanism holdout."""

    token_records: list[dict[str, object]] = []
    response_records: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for ref in refs:
            graph = ref.graph
            seeds = torch.tensor(
                ref.row.deployment_seed_positions, dtype=torch.long
            )
            output = model(
                graph,
                seed_positions=seeds,
                incidence="true",
                residual_mode="learned",
            )
            risks = output.token_risk.detach().cpu()
            available = output.row_available.detach().cpu()
            score_upper = output.unsupported_history_upper.detach().cpu()
            unknown = torch.as_tensor(graph["unknown_mass"]).float().mean(dim=1)
            segment = torch.as_tensor(graph["segment_ids"]).long().flatten()
            target_positions = torch.as_tensor(
                graph["target_token_positions"]
            ).long()
            evidence_count = int((segment == int(Segment.EVIDENCE)).sum())
            response_count = int(target_positions.numel())
            for offset in range(response_count):
                token_records.append(
                    {
                        "score": float(risks[offset]),
                        "score_upper": float(score_upper[offset]),
                        "row_available": bool(available[offset]),
                        "unknown_mass_mean": float(unknown[offset]),
                    }
                )
            response_records.append(
                {
                    "routing_risk_score": response_risk(risks, available),
                    "scorable_events": int(available.sum()),
                    "response_token_count": response_count,
                    "evidence_token_count": evidence_count,
                }
            )
    report = _informativeness_report(token_records, response_records)
    report["scope"] = "untouched_train_side_mechanism_holdout"
    return report


def _rewire_identifiability_audit(
    model: CausalProvenanceTransport,
    refs: Sequence[_ExampleRef],
    *,
    config: TransportTrainConfig,
) -> dict[str, object]:
    """Require the registered RY control to actually alter held-out graphs."""

    by_source: dict[str, list[int]] = {}
    changed_total = 0
    trace_total = 0
    for ref in refs:
        report = model.rewire_diagnostics(ref.graph)
        changed = int(report["ry_changed_trace_view_pairs"])
        traces = int(report["ry_trace_view_pairs"])
        values = by_source.setdefault(ref.row.source_id, [0, 0])
        values[0] += changed
        values[1] += traces
        changed_total += changed
        trace_total += traces
    source_fractions = {
        source: changed / traces
        for source, (changed, traces) in by_source.items()
        if traces > 0
    }
    ordered = sorted(source_fractions.values())
    median = (
        float(torch.tensor(ordered, dtype=torch.float64).median())
        if ordered
        else 0.0
    )
    overall = changed_total / max(1, trace_total)
    source_coverage = len(source_fractions) / max(1, len(by_source))
    source_pass_fraction = sum(
        value >= config.minimum_rewire_ry_changed_fraction
        for value in source_fractions.values()
    ) / max(1, len(source_fractions))
    checks = {
        "ry_trace_pairs_present": trace_total > 0,
        "overall_ry_changed_fraction_at_least_minimum": (
            overall >= config.minimum_rewire_ry_changed_fraction
        ),
        "median_source_ry_changed_fraction_at_least_minimum": (
            median >= config.minimum_rewire_ry_changed_fraction
        ),
        "sources_with_ry_at_least_minimum_coverage": (
            source_coverage >= config.minimum_rewire_source_coverage
        ),
        "at_least_half_ry_sources_pass_change_minimum": (
            source_pass_fraction >= 0.50
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "ry_trace_view_pairs": trace_total,
        "ry_changed_trace_view_pairs": changed_total,
        "overall_ry_changed_trace_fraction": overall,
        "sources": len(by_source),
        "sources_with_ry": len(source_fractions),
        "source_coverage": source_coverage,
        "median_source_ry_changed_trace_fraction": median,
        "source_pass_fraction": source_pass_fraction,
    }


def _ungated_control_from(
    trained_true: CausalProvenanceTransport,
) -> CausalProvenanceTransport:
    """Hold learned residual persistence fixed while removing relation gates."""

    control = CausalProvenanceTransport(
        num_layers=trained_true.num_layers,
        num_heads=trained_true.num_heads,
    ).to(trained_true.raw_relation_channel_gates.device)
    with torch.no_grad():
        control.raw_residual_persistence.copy_(
            trained_true.raw_residual_persistence
        )
    control.eval()
    return control


def _available_unknown_mass_mean(
    unknown_mass_mean: torch.Tensor,
    row_available: torch.Tensor,
    *,
    unscorable_default: float = 1.0,
) -> float:
    """Pool coverage only over the prediction rows used by the risk score."""

    values = torch.as_tensor(unknown_mass_mean).float().flatten()
    available = torch.as_tensor(row_available).bool().flatten()
    if values.numel() != available.numel() or values.numel() == 0:
        raise ValueError("unknown-mass values and row availability must align")
    if not math.isfinite(unscorable_default) or not 0.0 <= unscorable_default <= 1.0:
        raise ValueError("unscorable unknown-mass default must lie in [0, 1]")
    if bool(available.any()):
        return float(values[available].mean())
    return float(unscorable_default)


def _score_graphs(
    model: CausalProvenanceTransport,
    index: Path,
    output_dir: Path,
    *,
    variant: str,
    rows: Sequence[_GraphIndexRow],
) -> dict[str, object]:
    token_records: list[dict[str, object]] = []
    response_records: list[dict[str, object]] = []
    model.eval()
    raw_attention = CausalProvenanceTransport(
        num_layers=model.num_layers, num_heads=model.num_heads
    ).to(model.raw_relation_channel_gates.device)
    # The ungated deployment control must differ only in relation/head gates.
    # Keeping the learned persistence coefficients avoids giving the trained
    # model an advantage that could actually come from a second parameter
    # change rather than intervention-calibrated channel weighting.
    with torch.no_grad():
        raw_attention.raw_residual_persistence.copy_(
            model.raw_residual_persistence
        )
    raw_attention.eval()
    incidence, residual_mode = _variant_modes(variant)
    with torch.no_grad():
        for current, row in enumerate(rows, start=1):
            graph = _load_graph(row)
            if (
                int(graph["num_layers"]) != model.num_layers
                or int(graph["num_heads"]) != model.num_heads
            ):
                raise ValueError(
                    "score graph layer/head shape disagrees with checkpoint"
                )
            segment = torch.as_tensor(graph["segment_ids"]).long().flatten()
            seeds = torch.tensor(row.deployment_seed_positions, dtype=torch.long)
            if seeds.numel() and bool(
                (segment.index_select(0, seeds) != int(Segment.EVIDENCE)).any()
            ):
                raise RuntimeError(
                    "deployment intervention seed lies outside evidence segment"
                )
            target_positions = torch.as_tensor(graph["target_token_positions"]).long()
            event_count = int(target_positions.numel())
            if seeds.numel():
                transport_output = model(
                    graph,
                    seed_positions=seeds,
                    incidence=incidence,
                    residual_mode=residual_mode,
                )
                observational = raw_attention(
                    graph,
                    seed_positions=seeds,
                    incidence=incidence,
                    residual_mode=residual_mode,
                )
                risks = transport_output.token_risk.detach().cpu()
                observational_risks = observational.token_risk.detach().cpu()
                score_upper = transport_output.unsupported_history_upper.detach().cpu()
                evidence_deficit = (
                    transport_output.overall_evidence_deficit_lower.detach().cpu()
                )
                support_lower = transport_output.support_lower.detach().cpu()
                support_upper = transport_output.support_upper.detach().cpu()
                direct_support = transport_output.direct.detach().cpu()
                history_support = transport_output.history_lower.detach().cpu()
                observational_support = observational.support_upper.detach().cpu()
                row_available = transport_output.row_available.detach().cpu()
            else:
                risks = torch.zeros(event_count)
                observational_risks = torch.zeros(event_count)
                score_upper = torch.ones(event_count)
                evidence_deficit = torch.zeros(event_count)
                support_lower = torch.zeros(event_count)
                support_upper = torch.ones(event_count)
                direct_support = torch.zeros(event_count)
                history_support = torch.zeros(event_count)
                observational_support = torch.ones(event_count)
                row_available = torch.zeros(event_count, dtype=torch.bool)
            unknown_mass_mean = (
                torch.as_tensor(graph["unknown_mass"]).float().mean(dim=1)
            )
            evidence_token_count = int((segment == int(Segment.EVIDENCE)).sum())
            query_token_count = int((segment == int(Segment.QUERY)).sum())
            seed_count = int(seeds.numel())
            response_unknown_mass = _available_unknown_mass_mean(
                unknown_mass_mean, row_available
            )
            for offset, token_position in enumerate(target_positions.tolist()):
                token_records.append(
                    {
                        "schema": "cept-layer-provenance-token-score-v2",
                        "source_id": row.source_id,
                        "response_id": row.response_id,
                        "sample_id": row.response_id,
                        "official_split": row.official_split,
                        "token_idx": int(token_position),
                        "response_token_offset": offset,
                        "score": float(risks[offset]),
                        "ungated_score": float(observational_risks[offset]),
                        "score_upper": float(score_upper[offset]),
                        "overall_evidence_deficit_lower": float(evidence_deficit[offset]),
                        "support_lower": float(support_lower[offset]),
                        "support_upper": float(support_upper[offset]),
                        "direct_support": float(direct_support[offset]),
                        "history_support_lower": float(history_support[offset]),
                        "observational_support_upper": float(observational_support[offset]),
                        "causal_provenance_gap": float(
                            observational_support[offset] - support_upper[offset]
                        ),
                        "unknown_mass_mean": float(unknown_mass_mean[offset]),
                        "evidence_token_count": evidence_token_count,
                        "query_token_count": query_token_count,
                        "deployment_seed_count": seed_count,
                        "response_unknown_mass_mean": response_unknown_mass,
                        "response_token_count": int(target_positions.numel()),
                        "row_available": bool(row_available[offset]),
                        "deployment_seed_available": bool(seeds.numel()),
                    }
                )
            response_records.append(
                {
                    "schema": "cept-layer-provenance-response-score-v2",
                    "source_id": row.source_id,
                    "response_id": row.response_id,
                    "sample_id": row.response_id,
                    "official_split": row.official_split,
                    "routing_risk_score": response_risk(
                        risks, row_available
                    ),
                    "ungated_routing_risk_score": response_risk(
                        observational_risks,
                        row_available,
                    ),
                    "scorable_events": int(row_available.sum()),
                    "total_events": int(row_available.numel()),
                    "deployment_seed_available": bool(seeds.numel()),
                    "evidence_token_count": evidence_token_count,
                    "query_token_count": query_token_count,
                    "deployment_seed_count": seed_count,
                    "response_unknown_mass_mean": response_unknown_mass,
                    "response_token_count": int(target_positions.numel()),
                    "pooling": "top_10_percent_cvar_available_events",
                }
            )
            print(
                json.dumps(
                    {
                        "event": "transport_score",
                        "variant": variant,
                        "current": current,
                        "total": len(rows),
                    }
                ),
                flush=True,
            )
    token_path = output_dir / f"{variant}.token_predictions.jsonl"
    response_path = output_dir / f"{variant}.response_predictions.jsonl"
    atomic_jsonl(token_path, token_records)
    atomic_jsonl(response_path, response_records)
    label_free_informativeness = _informativeness_report(
        token_records, response_records
    )
    return {
        "score_index": str(index.expanduser().resolve()),
        "score_index_sha256": file_sha256(index),
        "variant": variant,
        "estimand_scope": {
            "eligible_population": ELIGIBLE_POPULATION_SCOPE,
            "score_cache": _cache_estimand_scope(rows),
        },
        "responses": len(response_records),
        "tokens": len(token_records),
        "token_predictions": str(token_path.resolve()),
        "response_predictions": str(response_path.resolve()),
        "prediction_sha256": canonical_hash(
            {"response": response_records, "token": token_records}
        ),
        "label_free_informativeness": label_free_informativeness,
    }


def train_transport(config: TransportTrainConfig) -> dict[str, object]:
    """Fit every pre-registered structural control without reading labels."""

    config.validate()
    output = config.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"transport output already exists; choose a new output directory: {output}"
        )
    refs, training_index_rows = _join_training_data(
        config.graph_index, config.teacher
    )
    model_source_signatures = {ref.row.model_source_signature for ref in refs}
    teacher_source_signatures = {
        ref.teacher.model_source_signature for ref in refs
    }
    model_runtime_signatures = {ref.teacher.model_signature for ref in refs}
    tokenizer_signatures = {ref.teacher.tokenizer_signature for ref in refs}
    observer_runtime_signatures = {
        ref.teacher.observer_runtime_signature for ref in refs
    }
    if (
        len(model_source_signatures) != 1
        or teacher_source_signatures != model_source_signatures
        or len(model_runtime_signatures) != 1
        or len(tokenizer_signatures) != 1
        or len(observer_runtime_signatures) != 1
    ):
        raise RuntimeError(
            "transport training teacher/graph model identities are heterogeneous"
        )
    expected_model_source_signature = next(iter(model_source_signatures))
    expected_observer_runtime = refs[0].teacher.observer_runtime
    expected_observer_runtime_signature = refs[0].teacher.observer_runtime_signature
    if any(
        dict(ref.teacher.observer_runtime) != dict(expected_observer_runtime)
        for ref in refs[1:]
    ):
        raise RuntimeError("transport teacher observer runtimes are heterogeneous")
    num_layers, num_heads = _model_shape(refs[0])
    for ref in refs[1:]:
        if _model_shape(ref) != (num_layers, num_heads):
            raise ValueError(
                "transport training graphs have inconsistent layer/head shapes"
            )
    score_rows: tuple[_GraphIndexRow, ...] | None = None
    if config.score_index is not None:
        score_rows = _graph_index(config.score_index)
        _require_official_split(
            score_rows, expected="test", context="transport frozen scoring"
        )
        _preflight_score_rows(
            score_rows,
            expected_model_source_signature=expected_model_source_signature,
            expected_observer_runtime=expected_observer_runtime,
            expected_observer_runtime_signature=(
                expected_observer_runtime_signature
            ),
            expected_shape=(num_layers, num_heads),
        )
    train, validation, mechanism_holdout = _split_sources(
        refs,
        validation_fraction=config.validation_fraction,
        mechanism_fraction=config.mechanism_holdout_fraction,
        minimum_mechanism_sources=config.minimum_mechanism_sources,
        seed=config.seed,
    )
    calibration = fit_effect_calibration(
        [ref.teacher for ref in train],
        availability_by_identity={
            (ref.teacher.source_id, ref.teacher.response_id): tuple(
                bool(value)
                for value in torch.as_tensor(ref.graph["row_available"])
                .bool()
                .tolist()
            )
            for ref in train
        },
        minimum_null_floor=config.noise_floor,
    )
    teacher_population = _teacher_population_audit(
        mechanism_holdout,
        calibration=calibration,
        config=config,
    )
    if (
        teacher_population.get("pass") is not True
        and not config.allow_unidentifiable_pilot
    ):
        checks = teacher_population.get("checks")
        failed_checks = (
            sorted(name for name, passed in checks.items() if passed is not True)
            if isinstance(checks, Mapping)
            else ["teacher_population_audit_invalid"]
        )
        raise RuntimeError(
            "transport teacher population is not identifiable; refusing to start "
            f"model training. failed_checks={failed_checks}. Use "
            "--allow-unidentifiable-pilot only for an explicitly non-claim pilot."
        )
    output.mkdir(parents=True, exist_ok=False)
    variant_reports: dict[str, object] = {}
    trained_models: dict[str, CausalProvenanceTransport] = {}
    for variant in config.variants:
        model, report = _train_variant(
            config,
            variant=variant,
            train=train,
            validation=validation,
            num_layers=num_layers,
            num_heads=num_heads,
            calibration=calibration,
        )
        variant_reports[variant] = report
        trained_models[variant] = model
    for variant, model in trained_models.items():
        report = variant_reports[variant]
        if not isinstance(report, dict):
            raise TypeError("internal transport variant report is not mutable")
        report["mechanism_holdout_teacher_fidelity"] = _teacher_fidelity(
            model,
            mechanism_holdout,
            variant=variant,
            calibration=calibration,
        )
    rewire_identifiability = _rewire_identifiability_audit(
        trained_models["true"], mechanism_holdout, config=config
    )
    ungated_true = _ungated_control_from(trained_models["true"])
    variant_reports["ungated_true"] = {
        "trained": False,
        "relation_gates": "fixed_equal_to_one",
        "residual_persistence_copied_from": "true",
        "residual_persistence": (
            ungated_true.residual_persistence().detach().cpu().tolist()
        ),
        "mechanism_holdout_teacher_fidelity": _teacher_fidelity(
            ungated_true,
            mechanism_holdout,
            variant="true",
            calibration=calibration,
        ),
    }
    source_losses = {
        variant: _source_validation_losses(
            trained_models[variant],
            mechanism_holdout,
            variant=variant,
            calibration=calibration,
            block_weight=config.block_weight,
        )
        for variant in config.variants
    }
    source_losses["ungated_true"] = _source_validation_losses(
        ungated_true,
        mechanism_holdout,
        variant="true",
        calibration=calibration,
        block_weight=config.block_weight,
    )
    source_losses["constant_zero"] = _zero_source_losses(
        mechanism_holdout,
        calibration=calibration,
        block_weight=config.block_weight,
    )
    paired_controls = {
        variant: _paired_bootstrap_improvement(
            source_losses["true"],
            source_losses[variant],
            seed=config.seed + index + 1,
        )
        for index, variant in enumerate(config.variants)
        if variant != "true"
    }
    paired_controls["ungated_true"] = _paired_bootstrap_improvement(
        source_losses["true"],
        source_losses["ungated_true"],
        seed=config.seed + len(config.variants) + 1,
    )
    paired_controls["constant_zero"] = _paired_bootstrap_improvement(
        source_losses["true"],
        source_losses["constant_zero"],
        seed=config.seed + len(config.variants) + 2,
    )
    topology_control_names = ("rewired", "mass_only")
    topology_controls = [
        paired_controls[name]
        for name in topology_control_names
        if name in paired_controls
    ]
    recursion_control = paired_controls.get("one_hop")
    residual_control = paired_controls.get("no_residual")
    calibration_control = paired_controls["ungated_true"]
    nontriviality_control = paired_controls["constant_zero"]
    topology_supported = all(
        name in paired_controls for name in topology_control_names
    ) and all(
        float(report["ci95_low"]) > 0.0 for report in topology_controls
    ) and rewire_identifiability.get("pass") is True
    recursion_supported = bool(
        recursion_control and float(recursion_control["ci95_low"]) > 0.0
    )
    residual_supported = bool(
        residual_control and float(residual_control["ci95_low"]) > 0.0
    )
    calibration_supported = float(calibration_control["ci95_low"]) > 0.0
    nontriviality_supported = float(nontriviality_control["ci95_low"]) > 0.0
    true_report = variant_reports.get("true")
    if not isinstance(true_report, Mapping):
        raise TypeError("true transport report is absent or invalid")
    true_fidelity = true_report.get("mechanism_holdout_teacher_fidelity")
    if not isinstance(true_fidelity, Mapping):
        raise TypeError("true mechanism-holdout fidelity is absent or invalid")
    component_fidelity = _component_fidelity_bootstrap(
        trained_models["true"],
        mechanism_holdout,
        calibration=calibration,
        config=config,
        seed=config.seed + 10_000,
    )
    component_fidelity_supported = component_fidelity.get("pass") is True
    mechanism_informativeness = _mechanism_informativeness(
        trained_models["true"], mechanism_holdout
    )
    gate2 = {
        "cache_model_identity_verified": all(
            ref.row.cache_model_identity_status
            in VERIFIED_CACHE_MODEL_IDENTITY_STATUSES
            and ref.row.cache_model_source_signature == ref.row.model_source_signature
            for ref in refs
        ),
        "graph_supported": topology_supported,
        "rewire_control_identifiable": (
            rewire_identifiability.get("pass") is True
        ),
        "recursion_supported": recursion_supported,
        "residual_supported": residual_supported,
        "intervention_calibration_supported": calibration_supported,
        "constant_zero_baseline_beaten": nontriviality_supported,
        "teacher_population_supported": teacher_population.get("pass") is True,
        "component_fidelity_supported": component_fidelity_supported,
        "informativeness_supported": mechanism_informativeness.get("pass") is True,
        "claim_supported": False,
        "paired_source_bootstrap": paired_controls,
        "teacher_population": teacher_population,
        "component_fidelity": component_fidelity,
        "rewire_identifiability": rewire_identifiability,
        "label_free_informativeness": mechanism_informativeness,
        "decision_rule": (
            "on the untouched mechanism holdout, source-bootstrap 95% CI for "
            "control-minus-true loss must be above zero for rewired, mass-only, "
            "one-hop, no-residual, ungated-true, and constant-zero controls; "
            "teacher positive-target coverage, component fidelity, and deployed "
            "score informativeness must pass on that same untouched train-side "
            "mechanism holdout"
        ),
    }
    scoring: dict[str, object] | None = None
    if config.score_index is not None and score_rows is not None:
        score_cache_identity_verified = all(
            row.cache_model_identity_status
            in VERIFIED_CACHE_MODEL_IDENTITY_STATUSES
            and row.cache_model_source_signature == row.model_source_signature
            for row in score_rows
        )
        gate2["score_cache_model_identity_verified"] = score_cache_identity_verified
        gate2["cache_model_identity_verified"] = bool(
            gate2["cache_model_identity_verified"] and score_cache_identity_verified
        )
        scoring = {
            variant: _score_graphs(
                trained_models[variant],
                config.score_index,
                output,
                variant=variant,
                rows=score_rows,
            )
            for variant in config.variants
        }
        gate2["official_test_label_free_diagnostics"] = scoring["true"].get(
            "label_free_informativeness"
        )
        gate2["claim_supported"] = bool(
            gate2["cache_model_identity_verified"]
            and topology_supported
            and recursion_supported
            and residual_supported
            and calibration_supported
            and nontriviality_supported
            and gate2["teacher_population_supported"]
            and component_fidelity_supported
            and gate2["informativeness_supported"]
        )
    estimand_scope: dict[str, object] = {
        "dataset": "RAGTruth",
        "eligible_population": ELIGIBLE_POPULATION_SCOPE,
        "training_cache": _cache_estimand_scope(training_index_rows),
        "teacher_graph_subset": _cache_estimand_scope(
            [ref.row for ref in refs]
        ),
    }
    if score_rows is not None:
        estimand_scope["scoring_cache"] = _cache_estimand_scope(score_rows)
    manifest = {
        "schema": "cept-layer-provenance-transport-run-v2",
        "state": "complete",
        "labels_read_during_training_or_scoring": False,
        "estimand_scope": estimand_scope,
        "config": asdict(config),
        "graph_index_sha256": file_sha256(config.graph_index),
        "teacher_sha256": file_sha256(config.teacher),
        "observer_runtime": expected_observer_runtime,
        "observer_runtime_signature": expected_observer_runtime_signature,
        "implementation_sha256": {
            **{
                name: file_sha256(Path(__file__).with_name(name))
                for name in ("data.py", "model.py", "experiment.py")
            },
            "observer_runtime.py": file_sha256(
                Path(__file__).resolve().parents[1] / "observer_runtime.py"
            ),
        },
        "num_layers": num_layers,
        "num_heads": num_heads,
        "effect_calibration": asdict(calibration),
        "learned_parameters_per_variant": {
            variant: (
                3 * num_layers * num_heads
                if variant == "no_residual"
                else 3 * num_layers * num_heads + num_layers
            )
            for variant in config.variants
        },
        "split": {
            "policy": "sha256_source_group_train_selection_mechanism_holdout",
            "train_samples": len(train),
            "selection_validation_samples": len(validation),
            "mechanism_holdout_samples": len(mechanism_holdout),
            "train_sources": len({ref.row.source_id for ref in train}),
            "selection_validation_sources": len(
                {ref.row.source_id for ref in validation}
            ),
            "mechanism_holdout_sources": len(
                {ref.row.source_id for ref in mechanism_holdout}
            ),
            "pairwise_source_overlap": 0,
        },
        "variants": variant_reports,
        "gate2": gate2,
        "scoring": scoring,
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        output / "teacher_fidelity.json", {"variants": variant_reports, "gate2": gate2}
    )
    return manifest


def evaluate_frozen_transport(
    *, predictions_dir: Path, test_graph_index: Path, output: Path
) -> dict[str, object]:
    """Only this function imports the held-out label reader."""

    root = predictions_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"frozen transport manifest is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "cept-layer-provenance-transport-run-v2"
        or manifest.get("state") != "complete"
        or manifest.get("labels_read_during_training_or_scoring") is not False
    ):
        raise RuntimeError(
            "frozen transport run contract is stale/incomplete or label-tainted; "
            "held-out labels remain closed"
        )
    gate2 = manifest.get("gate2")
    if not isinstance(gate2, Mapping) or gate2.get("claim_supported") is not True:
        raise RuntimeError(
            "Gate 2 did not support the graph claim; held-out hallucination "
            "labels remain closed"
        )
    scoring = manifest.get("scoring")
    if not isinstance(scoring, Mapping) or not scoring:
        raise RuntimeError(
            "frozen transport run has no predictions; held-out labels remain closed"
        )
    requested_index_sha256 = file_sha256(test_graph_index)
    rows = _graph_index(test_graph_index)
    _require_official_split(
        rows, expected="test", context="transport post-hoc evaluation"
    )
    evaluation_cache_scope = _cache_estimand_scope(rows)
    expected_score_estimand_scope = {
        "eligible_population": ELIGIBLE_POPULATION_SCOPE,
        "score_cache": evaluation_cache_scope,
    }
    cache_paths = [row.cache_path for row in rows]
    if any(path is None or not path.is_file() for path in cache_paths):
        raise FileNotFoundError(
            "test graph index lacks an existing attention cache path"
        )
    for row in rows:
        assert row.cache_path is not None
        if row.cache_sha256 is None or file_sha256(row.cache_path) != row.cache_sha256:
            raise RuntimeError(
                f"evaluation attention cache hash differs from frozen index: "
                f"{row.cache_path}"
            )
    index_response_domain = {(row.source_id, row.response_id) for row in rows}
    prediction_hashes: dict[str, str] = {}
    response_score_maps: dict[str, dict[tuple[str, str], float]] = {}
    token_score_maps: dict[str, dict[tuple[str, str, int], float]] = {}
    variant_counts: dict[str, dict[str, int]] = {}
    response_nuisance_maps: dict[str, dict[tuple[str, str], float]] = {
        "response_length": {},
        "negative_response_length": {},
        "response_evidence_token_count": {},
        "negative_response_evidence_token_count": {},
        "response_query_token_count": {},
        "negative_response_query_token_count": {},
        "response_seed_count": {},
        "negative_response_seed_count": {},
        "response_unknown_mass": {},
        "negative_response_unknown_mass": {},
    }
    token_nuisance_maps: dict[str, dict[tuple[str, str, int], float]] = {
        "token_position": {},
        "negative_token_position": {},
        "token_unknown_mass": {},
        "negative_token_unknown_mass": {},
        "token_response_length": {},
        "negative_token_response_length": {},
        "token_evidence_token_count": {},
        "negative_token_evidence_token_count": {},
        "token_query_token_count": {},
        "negative_token_query_token_count": {},
        "token_seed_count": {},
        "negative_token_seed_count": {},
        "token_response_unknown_mass": {},
        "negative_token_response_unknown_mass": {},
    }
    for variant, raw_report in scoring.items():
        if not isinstance(raw_report, Mapping):
            raise TypeError(f"invalid frozen scoring report for {variant}")
        if raw_report.get("score_index_sha256") != requested_index_sha256:
            raise RuntimeError(
                f"evaluation graph index differs from frozen scoring index: {variant}"
            )
        token_path = Path(str(raw_report["token_predictions"])).resolve()
        response_path = Path(str(raw_report["response_predictions"])).resolve()
        for path in (token_path, response_path):
            if path.parent != root or not path.is_file():
                raise FileNotFoundError(
                    f"frozen {variant} prediction is absent/outside its run: {path}"
                )
        token_records = list(_read_jsonl(token_path))
        response_records = list(_read_jsonl(response_path))
        if (
            raw_report.get("variant") != variant
            or raw_report.get("tokens") != len(token_records)
            or raw_report.get("responses") != len(response_records)
        ):
            raise RuntimeError(
                f"frozen transport prediction inventory mismatch: {variant}"
            )
        observed_hash = canonical_hash(
            {"response": response_records, "token": token_records}
        )
        if observed_hash != raw_report.get("prediction_sha256"):
            raise RuntimeError(f"frozen transport prediction hash mismatch: {variant}")
        if raw_report.get("estimand_scope") != expected_score_estimand_scope:
            raise RuntimeError(
                f"frozen transport estimand scope mismatch: {variant}"
            )
        variant_name = str(variant)
        prediction_hashes[variant_name] = observed_hash

        response_scores: dict[tuple[str, str], float] = {}
        ungated_response_scores: dict[tuple[str, str], float] = {}
        response_seen: set[tuple[str, str]] = set()
        response_lengths: dict[tuple[str, str], int] = {}
        response_context: dict[
            tuple[str, str], tuple[int, int, int, float]
        ] = {}
        response_total = 0
        for record in response_records:
            if record.get("schema") != "cept-layer-provenance-response-score-v2":
                raise ValueError(f"invalid frozen response schema: {variant}")
            identity = (str(record["source_id"]), str(record["response_id"]))
            if identity in response_seen or identity not in index_response_domain:
                raise ValueError(f"duplicate response prediction: {identity}")
            response_seen.add(identity)
            response_total += 1
            scorable_events = int(record.get("scorable_events", -1))
            total_events = int(record.get("total_events", -1))
            response_length = int(record.get("response_token_count", -1))
            evidence_token_count = int(record.get("evidence_token_count", -1))
            query_token_count = int(record.get("query_token_count", -1))
            seed_count = int(record.get("deployment_seed_count", -1))
            response_unknown_mass = float(
                record.get("response_unknown_mass_mean", math.nan)
            )
            routing_score = float(record["routing_risk_score"])
            ungated_score = float(record["ungated_routing_risk_score"])
            if (
                scorable_events < 0
                or total_events < 0
                or scorable_events > total_events
                or response_length <= 0
                or total_events != response_length
                or evidence_token_count < 0
                or query_token_count < 0
                or seed_count < 0
                or not math.isfinite(response_unknown_mass)
                or not 0.0 <= response_unknown_mass <= 1.0
                or not math.isfinite(routing_score)
                or not math.isfinite(ungated_score)
            ):
                raise ValueError(f"invalid frozen response values: {identity}")
            response_lengths[identity] = response_length
            response_context[identity] = (
                evidence_token_count,
                query_token_count,
                seed_count,
                response_unknown_mass,
            )
            if scorable_events > 0:
                response_scores[identity] = routing_score
                if variant == "true":
                    ungated_response_scores[identity] = ungated_score
                    response_nuisance_maps["response_length"][identity] = (
                        float(response_length)
                    )
                    response_nuisance_maps["negative_response_length"][identity] = (
                        -float(response_length)
                    )
                    response_nuisance_maps[
                        "response_evidence_token_count"
                    ][identity] = float(evidence_token_count)
                    response_nuisance_maps[
                        "negative_response_evidence_token_count"
                    ][identity] = -float(evidence_token_count)
                    response_nuisance_maps[
                        "response_query_token_count"
                    ][identity] = float(query_token_count)
                    response_nuisance_maps[
                        "negative_response_query_token_count"
                    ][identity] = -float(query_token_count)
                    response_nuisance_maps["response_seed_count"][identity] = (
                        float(seed_count)
                    )
                    response_nuisance_maps[
                        "negative_response_seed_count"
                    ][identity] = -float(seed_count)
                    response_nuisance_maps["response_unknown_mass"][identity] = (
                        response_unknown_mass
                    )
                    response_nuisance_maps[
                        "negative_response_unknown_mass"
                    ][identity] = -response_unknown_mass
        if response_seen != index_response_domain:
            raise ValueError(
                f"frozen response prediction inventory differs from index: {variant}"
            )

        token_scores: dict[tuple[str, str, int], float] = {}
        ungated_token_scores: dict[tuple[str, str, int], float] = {}
        token_seen: set[tuple[str, str, int]] = set()
        token_offsets: dict[tuple[str, str], set[int]] = {}
        token_total = 0
        unavailable_tokens = 0
        for record in token_records:
            if record.get("schema") != "cept-layer-provenance-token-score-v2":
                raise ValueError(f"invalid frozen token schema: {variant}")
            token_total += 1
            identity = (str(record["source_id"]), str(record["response_id"]))
            response_length = int(record.get("response_token_count", -1))
            response_offset = int(record.get("response_token_offset", -1))
            key = (*identity, int(record["token_idx"]))
            row_available = record.get("row_available")
            score = float(record["score"])
            ungated_score = float(record["ungated_score"])
            unknown_mass = float(record["unknown_mass_mean"])
            evidence_token_count = int(record.get("evidence_token_count", -1))
            query_token_count = int(record.get("query_token_count", -1))
            seed_count = int(record.get("deployment_seed_count", -1))
            response_unknown_mass = float(
                record.get("response_unknown_mass_mean", math.nan)
            )
            expected_context = response_context.get(identity)
            context_matches = (
                expected_context is not None
                and evidence_token_count == expected_context[0]
                and query_token_count == expected_context[1]
                and seed_count == expected_context[2]
                and math.isclose(
                    response_unknown_mass,
                    expected_context[3],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            if (
                identity not in index_response_domain
                or key in token_seen
                or not isinstance(row_available, bool)
                or response_length != response_lengths.get(identity)
                or not context_matches
                or response_offset < 0
                or response_offset >= response_length
                or not math.isfinite(score)
                or not math.isfinite(ungated_score)
                or not math.isfinite(unknown_mass)
                or not 0.0 <= unknown_mass <= 1.0
            ):
                raise ValueError(f"invalid frozen token prediction: {key}")
            token_seen.add(key)
            offsets = token_offsets.setdefault(identity, set())
            if response_offset in offsets:
                raise ValueError(
                    f"duplicate frozen response-token offset: {identity} {response_offset}"
                )
            offsets.add(response_offset)
            if not row_available:
                unavailable_tokens += 1
                continue
            token_scores[key] = score
            if variant == "true":
                ungated_token_scores[key] = ungated_score
                normalized_position = float(response_offset) / max(
                    1.0, float(response_length - 1)
                )
                token_nuisance_maps["token_position"][key] = normalized_position
                token_nuisance_maps["negative_token_position"][key] = (
                    -normalized_position
                )
                token_nuisance_maps["token_unknown_mass"][key] = unknown_mass
                token_nuisance_maps["negative_token_unknown_mass"][key] = (
                    -unknown_mass
                )
                token_nuisance_maps["token_response_length"][key] = float(
                    response_length
                )
                token_nuisance_maps["negative_token_response_length"][key] = (
                    -float(response_length)
                )
                token_nuisance_maps["token_evidence_token_count"][key] = float(
                    evidence_token_count
                )
                token_nuisance_maps[
                    "negative_token_evidence_token_count"
                ][key] = -float(evidence_token_count)
                token_nuisance_maps["token_query_token_count"][key] = float(
                    query_token_count
                )
                token_nuisance_maps[
                    "negative_token_query_token_count"
                ][key] = -float(query_token_count)
                token_nuisance_maps["token_seed_count"][key] = float(seed_count)
                token_nuisance_maps["negative_token_seed_count"][key] = -float(
                    seed_count
                )
                token_nuisance_maps["token_response_unknown_mass"][key] = (
                    response_unknown_mass
                )
                token_nuisance_maps[
                    "negative_token_response_unknown_mass"
                ][key] = -response_unknown_mass
        if set(token_offsets) != index_response_domain or any(
            offsets != set(range(response_lengths[identity]))
            for identity, offsets in token_offsets.items()
        ):
            raise ValueError(
                f"frozen token prediction inventory differs from index: {variant}"
            )
        variant_counts[variant_name] = {
            "response_total": response_total,
            "unavailable_tokens": unavailable_tokens,
        }
        response_score_maps[variant_name] = response_scores
        token_score_maps[variant_name] = token_scores
        if variant == "true":
            response_score_maps["ungated_true"] = ungated_response_scores
            token_score_maps["ungated_true"] = ungated_token_scores
            variant_counts["ungated_true"] = dict(variant_counts[variant_name])
    if "true" not in response_score_maps:
        raise ValueError("frozen evaluation requires the true-incidence variant")

    structural_variants = set(response_score_maps).difference({"ungated_true"})
    for variant in structural_variants:
        if (
            set(response_score_maps[variant]) != set(response_score_maps["true"])
            or set(token_score_maps[variant]) != set(token_score_maps["true"])
        ):
            raise ValueError(
                f"frozen structural variants have different score domains: {variant}"
            )
    if (
        set(response_score_maps["ungated_true"])
        != set(response_score_maps["true"])
        or set(token_score_maps["ungated_true"]) != set(token_score_maps["true"])
    ):
        raise ValueError("frozen ungated control has a different score domain")

    for name, scores in response_nuisance_maps.items():
        if set(scores) != set(response_score_maps["true"]):
            raise ValueError(f"response nuisance domain differs from true: {name}")
        response_score_maps[name] = scores
    for name, scores in token_nuisance_maps.items():
        if set(scores) != set(token_score_maps["true"]):
            raise ValueError(f"token nuisance domain differs from true: {name}")
        token_score_maps[name] = scores

    manifest_estimand_scope = manifest.get("estimand_scope")
    if not isinstance(manifest_estimand_scope, Mapping):
        raise TypeError("frozen transport run has no structured estimand scope")
    evaluated_estimand_scope = dict(manifest_estimand_scope)
    if (
        evaluated_estimand_scope.get("eligible_population")
        != ELIGIBLE_POPULATION_SCOPE
        or evaluated_estimand_scope.get("scoring_cache")
        != evaluation_cache_scope
    ):
        raise RuntimeError(
            "frozen transport run estimand scope differs from evaluation index"
        )
    evaluated_estimand_scope["evaluation_cache"] = evaluation_cache_scope

    # Prediction paths, hashes, schemas, identities, finite values, and every
    # paired score domain are frozen and validated before this sole label read.
    from attention_graph.evaluate import evaluate_binary_scores, load_evaluation_labels

    labels = load_evaluation_labels([path for path in cache_paths if path is not None])
    if not set(response_score_maps["true"]).issubset(labels.response_labels):
        raise ValueError("response predictions contain identities absent from labels")
    if not set(token_score_maps["true"]).issubset(labels.token_labels):
        raise ValueError("token predictions contain identities absent from labels")

    def safe_binary_scores(
        label_map: Mapping[tuple[object, ...], int],
        score_map: Mapping[tuple[object, ...], float],
    ) -> dict[str, object]:
        keys = sorted(score_map)
        target = [int(label_map[key]) for key in keys]
        positives = sum(target)
        if not target or positives in {0, len(target)}:
            return {
                "metric_status": (
                    "undefined_empty_domain"
                    if not target
                    else "undefined_single_class_domain"
                ),
                "samples": len(target),
                "positive_count": positives,
                "positive_fraction": positives / max(1, len(target)),
                "auroc": None,
                "average_precision": None,
            }
        return {
            "metric_status": "defined",
            **evaluate_binary_scores(target, [score_map[key] for key in keys]),
        }

    variant_metrics: dict[str, object] = {}
    for variant, counts in variant_counts.items():
        response_scores = response_score_maps[variant]
        token_scores = token_score_maps[variant]
        variant_metrics[variant] = {
            "response": {
                **safe_binary_scores(labels.response_labels, response_scores),
                "coverage": len(response_scores) / len(labels.response_labels),
                "unscorable_responses": counts["response_total"]
                - len(response_scores),
                "scope": "numeric_digit_surface_preserving_identifiable_subset",
            },
            "token": {
                **safe_binary_scores(labels.token_labels, token_scores),
                "coverage": len(token_scores) / len(labels.token_labels),
                "unavailable_prediction_rows": counts["unavailable_tokens"],
                "scope": (
                    "numeric_digit_surface_preserving_identifiable_subset;"
                    "scorable_prediction_events_only"
                ),
            },
        }

    nuisance_metrics: dict[str, object] = {}
    for name, scores in response_nuisance_maps.items():
        nuisance_metrics[name] = {
            "endpoint": "response",
            "orientation": "as_named_no_label_selection",
            **safe_binary_scores(labels.response_labels, scores),
        }
    for name, scores in token_nuisance_maps.items():
        nuisance_metrics[name] = {
            "endpoint": "token",
            "orientation": "as_named_no_label_selection",
            **safe_binary_scores(labels.token_labels, scores),
        }

    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, Mapping):
        raise TypeError("transport manifest has no valid frozen training config")
    base_seed = int(manifest_config.get("seed", 42))
    detection_minima = {
        "sources": int(manifest_config.get("minimum_detection_sources", 20)),
        "positive_sources": int(
            manifest_config.get("minimum_detection_positive_sources", 5)
        ),
        "negative_sources": int(
            manifest_config.get("minimum_detection_negative_sources", 5)
        ),
        "positive_samples": int(
            manifest_config.get("minimum_detection_positive_samples", 20)
        ),
        "negative_samples": int(
            manifest_config.get("minimum_detection_negative_samples", 20)
        ),
    }

    def source_groups(
        score_map: Mapping[tuple[object, ...], float],
    ) -> dict[str, list[tuple[object, ...]]]:
        grouped: dict[str, list[tuple[object, ...]]] = {}
        for key in score_map:
            grouped.setdefault(str(key[0]), []).append(key)
        return grouped

    def detection_population(
        label_map: Mapping[tuple[object, ...], int],
        score_map: Mapping[tuple[object, ...], float],
    ) -> dict[str, object]:
        grouped = source_groups(score_map)
        positive_samples = sum(int(label_map[key]) == 1 for key in score_map)
        negative_samples = len(score_map) - positive_samples
        positive_sources = sum(
            any(int(label_map[key]) == 1 for key in keys)
            for keys in grouped.values()
        )
        negative_sources = sum(
            any(int(label_map[key]) == 0 for key in keys)
            for keys in grouped.values()
        )
        checks = {
            "sources_at_least_minimum": len(grouped) >= detection_minima["sources"],
            "positive_sources_at_least_minimum": (
                positive_sources >= detection_minima["positive_sources"]
            ),
            "negative_sources_at_least_minimum": (
                negative_sources >= detection_minima["negative_sources"]
            ),
            "positive_samples_at_least_minimum": (
                positive_samples >= detection_minima["positive_samples"]
            ),
            "negative_samples_at_least_minimum": (
                negative_samples >= detection_minima["negative_samples"]
            ),
        }
        return {
            "pass": all(checks.values()),
            "checks": checks,
            "minimum": detection_minima,
            "sources": len(grouped),
            "positive_sources": positive_sources,
            "negative_sources": negative_sources,
            "samples": len(score_map),
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "prevalence": positive_samples / max(1, len(score_map)),
        }

    def paired_source_bootstrap(
        label_map: Mapping[tuple[object, ...], int],
        primary: Mapping[tuple[object, ...], float],
        control: Mapping[tuple[object, ...], float],
        *,
        seed: int,
        draws: int = 2000,
    ) -> dict[str, object]:
        if set(primary) != set(control):
            raise ValueError("paired frozen variants have different score domains")
        keys_by_source: dict[str, list[tuple[object, ...]]] = {}
        for key in primary:
            keys_by_source.setdefault(str(key[0]), []).append(key)
        sources = sorted(keys_by_source)
        if len(sources) < 2:
            raise ValueError("source bootstrap requires at least two source groups")

        def delta(keys: Sequence[tuple[object, ...]]) -> tuple[float, float]:
            target = [int(label_map[key]) for key in keys]
            primary_metrics = evaluate_binary_scores(
                target, [primary[key] for key in keys]
            )
            control_metrics = evaluate_binary_scores(
                target, [control[key] for key in keys]
            )
            return (
                float(primary_metrics["auroc"]) - float(control_metrics["auroc"]),
                float(primary_metrics["average_precision"])
                - float(control_metrics["average_precision"]),
            )

        all_keys = [key for source in sources for key in keys_by_source[source]]
        point_auroc, point_aupr = delta(all_keys)
        generator = random.Random(seed)
        bootstrapped: list[tuple[float, float]] = []
        attempts = 0
        while len(bootstrapped) < draws and attempts < draws * 5:
            attempts += 1
            sampled = [generator.choice(sources) for _ in sources]
            sampled_keys = [
                key for source in sampled for key in keys_by_source[source]
            ]
            sampled_labels = {int(label_map[key]) for key in sampled_keys}
            if len(sampled_labels) < 2:
                continue
            bootstrapped.append(delta(sampled_keys))
        if len(bootstrapped) < max(20, draws // 2):
            raise RuntimeError("too few valid source-cluster bootstrap draws")
        samples = torch.tensor(bootstrapped, dtype=torch.float64)
        return {
            "sources": len(sources),
            "valid_draws": len(bootstrapped),
            "auroc_delta": {
                "point": point_auroc,
                "ci95_low": float(torch.quantile(samples[:, 0], 0.025)),
                "ci95_high": float(torch.quantile(samples[:, 0], 0.975)),
            },
            "average_precision_delta": {
                "point": point_aupr,
                "ci95_low": float(torch.quantile(samples[:, 1], 0.025)),
                "ci95_high": float(torch.quantile(samples[:, 1], 0.975)),
            },
        }

    def absolute_source_bootstrap(
        label_map: Mapping[tuple[object, ...], int],
        score_map: Mapping[tuple[object, ...], float],
        *,
        seed: int,
        draws: int = 2000,
    ) -> dict[str, object]:
        grouped = source_groups(score_map)
        sources = sorted(grouped)
        if len(sources) < 2:
            raise ValueError("source bootstrap requires at least two source groups")

        def excess(keys: Sequence[tuple[object, ...]]) -> tuple[float, float]:
            target = [int(label_map[key]) for key in keys]
            metrics = evaluate_binary_scores(
                target, [score_map[key] for key in keys]
            )
            prevalence = sum(target) / len(target)
            return (
                float(metrics["auroc"]) - 0.5,
                float(metrics["average_precision"]) - prevalence,
            )

        all_keys = [key for source in sources for key in grouped[source]]
        point_auroc, point_aupr = excess(all_keys)
        generator = random.Random(seed)
        bootstrapped: list[tuple[float, float]] = []
        attempts = 0
        while len(bootstrapped) < draws and attempts < draws * 5:
            attempts += 1
            sampled = [generator.choice(sources) for _ in sources]
            sampled_keys = [key for source in sampled for key in grouped[source]]
            if len({int(label_map[key]) for key in sampled_keys}) < 2:
                continue
            bootstrapped.append(excess(sampled_keys))
        if len(bootstrapped) < max(20, draws // 2):
            raise RuntimeError(
                "too few valid absolute source-cluster bootstrap draws"
            )
        samples = torch.tensor(bootstrapped, dtype=torch.float64)
        return {
            "sources": len(sources),
            "valid_draws": len(bootstrapped),
            "auroc_minus_0_5": {
                "point": point_auroc,
                "ci95_low": float(torch.quantile(samples[:, 0], 0.025)),
                "ci95_high": float(torch.quantile(samples[:, 0], 0.975)),
            },
            "average_precision_minus_prevalence": {
                "point": point_aupr,
                "ci95_low": float(torch.quantile(samples[:, 1], 0.025)),
                "ci95_high": float(torch.quantile(samples[:, 1], 0.975)),
            },
        }

    detection_population_reports = {
        "token": detection_population(
            labels.token_labels, token_score_maps["true"]
        ),
        "response": detection_population(
            labels.response_labels, response_score_maps["true"]
        ),
    }

    def skipped_population_bootstrap(endpoint: str) -> dict[str, object]:
        return {
            "status": "skipped_population_gate_failed",
            "endpoint": endpoint,
            "population": detection_population_reports[endpoint],
        }

    paired_detection: dict[str, object] = {}
    all_controls = sorted(
        (set(response_score_maps) | set(token_score_maps)).difference({"true"})
    )
    for index, control in enumerate(all_controls):
        control_report: dict[str, object] = {}
        if control in response_score_maps:
            if set(response_score_maps["true"]) != set(
                response_score_maps[control]
            ):
                raise ValueError(
                    "paired response variants have different score domains"
                )
            control_report["response"] = (
                paired_source_bootstrap(
                    labels.response_labels,
                    response_score_maps["true"],
                    response_score_maps[control],
                    seed=base_seed + 100 + index,
                )
                if detection_population_reports["response"]["pass"]
                else skipped_population_bootstrap("response")
            )
        if control in token_score_maps:
            if set(token_score_maps["true"]) != set(token_score_maps[control]):
                raise ValueError(
                    "paired token variants have different score domains"
                )
            control_report["token"] = (
                paired_source_bootstrap(
                    labels.token_labels,
                    token_score_maps["true"],
                    token_score_maps[control],
                    seed=base_seed + 200 + index,
                )
                if detection_population_reports["token"]["pass"]
                else skipped_population_bootstrap("token")
            )
        paired_detection[control] = control_report
    absolute_detection = {
        "token": (
            absolute_source_bootstrap(
                labels.token_labels,
                token_score_maps["true"],
                seed=base_seed + 20_001,
            )
            if detection_population_reports["token"]["pass"]
            else skipped_population_bootstrap("token")
        ),
        "response": (
            absolute_source_bootstrap(
                labels.response_labels,
                response_score_maps["true"],
                seed=base_seed + 20_002,
            )
            if detection_population_reports["response"]["pass"]
            else skipped_population_bootstrap("response")
        ),
    }

    detection_claim = _detection_claim_report(
        paired_detection,
        absolute_detection,
        detection_population_reports,
    )
    result = {
        "artifact_type": "cept-layer-provenance-posthoc-evaluation-v2",
        "primary_variant": "true",
        "estimand_scope": evaluated_estimand_scope,
        **variant_metrics["true"],
        "variants": variant_metrics,
        "nuisance_controls": nuisance_metrics,
        "paired_source_bootstrap_true_minus_control": paired_detection,
        "detection_claim": detection_claim,
        "prediction_sha256": prediction_hashes,
        "labels_read_during": "posthoc_evaluation_only",
    }
    atomic_json(output, result)
    return result


__all__ = [
    "REGISTERED_VARIANTS",
    "TransportTrainConfig",
    "evaluate_frozen_transport",
    "train_transport",
]
