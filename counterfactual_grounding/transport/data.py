"""Strict alignment between intervention teachers and canonical graphs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from ..artifacts import canonical_hash
from ..observer_runtime import parse_observer_runtime_identity
from ..teacher.pilot import Gate1Pair, Gate1Record


@dataclass(frozen=True)
class TransportEventTarget:
    target_position: int
    predictor_position: int
    target_token_id: int
    total_effect: float
    direct_effect: float
    history_mediated_effect: float
    alternate_history_effect: float
    interaction: float
    decomposition_residual: float
    direct_seed_rescue: float
    joint_seed_history_rescue: float
    representation_residual: float
    seed_history_interaction: float
    self_patch_error: float
    block_rescue: Mapping[str, float]


@dataclass(frozen=True)
class TransportTeacher:
    response_id: str
    source_id: str
    official_split: str
    model_signature: str
    model_source_signature: str
    tokenizer_signature: str
    observer_runtime: Mapping[str, str]
    observer_runtime_signature: str
    factual_input_ids_sha256: str
    response_idx: int
    changed_evidence_positions: tuple[int, ...]
    block_positions: Mapping[str, tuple[int, ...]]
    events: tuple[TransportEventTarget, ...]


@dataclass(frozen=True)
class AlignedTransportExample:
    teacher: TransportTeacher
    graph: Mapping[str, object]


@dataclass(frozen=True)
class TeacherTargets:
    support: torch.Tensor
    history: torch.Tensor
    block: torch.Tensor
    event_weight: torch.Tensor
    positive_mask: torch.Tensor
    null_mask: torch.Tensor
    contradictory_mask: torch.Tensor
    block_mask: torch.Tensor
    block_ids: tuple[str, ...]


@dataclass(frozen=True)
class EffectCalibration:
    null_floor: float
    support_scale: float
    history_scale: float
    block_scale: float

    def validate(self) -> None:
        values = (
            self.null_floor,
            self.support_scale,
            self.history_scale,
            self.block_scale,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("effect calibration values must be finite")
        if self.null_floor <= 0:
            raise ValueError("effect calibration null_floor must be positive")
        if (
            self.support_scale <= self.null_floor
            or self.history_scale <= self.null_floor
            or self.block_scale <= self.null_floor
        ):
            raise ValueError("effect calibration scales must exceed the null floor")


def _reject_labels(value: object, *, path: str = "teacher") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key)
            if "label" in name.casefold() or name.casefold() == "y_token":
                raise ValueError(
                    f"transport teacher contains forbidden label field: {path}.{name}"
                )
            _reject_labels(nested, path=f"{path}.{name}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_labels(nested, path=f"{path}[{index}]")


def _finite(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite scalar")
    return number


def _integer_tuple(
    value: object, name: str, *, allow_empty: bool = False
) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of integer positions")
    try:
        positions = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain integer positions") from error
    if not allow_empty and not positions:
        raise ValueError(f"{name} must not be empty")
    if len(set(positions)) != len(positions) or tuple(sorted(positions)) != positions:
        raise ValueError(f"{name} must be sorted and unique")
    return positions


def make_transport_teacher_record(
    record: Gate1Record,
    pair: Gate1Pair,
    *,
    changed_evidence_positions: Sequence[int],
    official_split: str,
    model_signature: str,
    model_source_signature: str,
    tokenizer_signature: str,
    observer_runtime: Mapping[str, object],
    observer_runtime_signature: str,
) -> dict[str, object]:
    """Serialize one structured, label-free intervention target.

    Unlike the flat ``effects.jsonl`` compatibility artifact, this record keeps
    the exact evidence intervention and response-history block endpoints needed
    to identify graph paths.
    """

    if official_split != "train":
        raise ValueError("transport teachers may only be generated from train data")
    if (
        record.sample_id != pair.sample_id
        or record.source_id != pair.source_id
        or record.response_idx != int(pair.factual["response_idx"])
    ):
        raise ValueError("Gate-1 record and pair identities do not align")
    factual_ids = torch.as_tensor(pair.factual["input_ids"], dtype=torch.long).flatten()
    counterfactual_ids = torch.as_tensor(
        pair.counterfactual["input_ids"], dtype=torch.long
    ).flatten()
    if factual_ids.shape != counterfactual_ids.shape:
        raise ValueError("transport teacher requires equal-token counterfactuals")
    changed = tuple(
        int(item)
        for item in torch.nonzero(factual_ids != counterfactual_ids).flatten().tolist()
    )
    declared_changed = tuple(int(item) for item in changed_evidence_positions)
    if (
        tuple(sorted(set(declared_changed))) != declared_changed
        or changed != declared_changed
    ):
        raise ValueError(
            "declared changed evidence positions disagree with token differences"
        )
    if not changed or any(
        position < 0 or position >= record.response_idx for position in changed
    ):
        raise ValueError(
            "counterfactual changes must be non-empty and precede the response"
        )
    evidence_positions = {
        int(item)
        for item in torch.as_tensor(
            pair.factual.get("evidence_token_positions"), dtype=torch.long
        )
        .flatten()
        .tolist()
    }
    if not evidence_positions or not set(changed).issubset(evidence_positions):
        raise ValueError(
            "changed positions must be a subset of registered evidence tokens"
        )
    if not model_signature or not model_source_signature or not tokenizer_signature:
        raise ValueError("transport teacher requires model and tokenizer signatures")
    normalized_runtime, normalized_runtime_signature = (
        parse_observer_runtime_identity(
            observer_runtime, observer_runtime_signature
        )
    )

    block_definitions = []
    block_ids = set(pair.rescue_blocks)
    for block_id, raw_positions in pair.rescue_blocks.items():
        positions = tuple(
            int(item)
            for item in torch.as_tensor(raw_positions, dtype=torch.long)
            .flatten()
            .tolist()
        )
        if not block_id or not positions or tuple(sorted(set(positions))) != positions:
            raise ValueError(
                "history block ids and positions must be non-empty, sorted, unique"
            )
        if any(
            position < record.response_idx or position >= factual_ids.numel() - 1
            for position in positions
        ):
            raise ValueError(
                "history rescue block lies outside usable response history"
            )
        block_definitions.append({"block_id": block_id, "positions": list(positions)})

    events: list[dict[str, object]] = []
    for effect in record.token_effects:
        if set(effect.block_rescue) != block_ids:
            raise ValueError("Gate-1 effect and pair rescue blocks disagree")
        events.append(
            {
                "target_position": effect.target_position,
                "predictor_position": effect.predictor_position,
                "target_token_id": effect.target_token_id,
                "total_effect": effect.total,
                "direct_effect": effect.direct,
                "history_mediated_effect": effect.mediated,
                "alternate_history_effect": effect.alternate,
                "interaction": effect.interaction,
                "decomposition_residual": effect.contract_residual,
                "direct_seed_rescue": effect.direct_seed_rescue,
                "joint_seed_history_rescue": effect.joint_seed_history_rescue,
                "representation_residual": effect.representation_residual,
                "seed_history_interaction": effect.seed_history_interaction,
                "self_patch_error": effect.self_patch_error,
                "block_rescue": dict(effect.block_rescue),
            }
        )
    payload: dict[str, object] = {
        "schema": "cept-causal-transport-teacher-v2",
        "response_id": record.sample_id,
        "source_id": record.source_id,
        "official_split": official_split,
        "model_signature": model_signature,
        "model_source_signature": model_source_signature,
        "tokenizer_signature": tokenizer_signature,
        "observer_runtime": normalized_runtime,
        "observer_runtime_signature": normalized_runtime_signature,
        "factual_input_ids_sha256": canonical_hash(factual_ids.tolist()),
        "response_idx": record.response_idx,
        "counterfactual_protocol": "numeric_digit_surface_preserving_v1",
        "changed_evidence_positions": list(changed),
        "block_definitions": block_definitions,
        "events": events,
    }
    _reject_labels(payload)
    return payload


def parse_transport_teacher(value: Mapping[str, object]) -> TransportTeacher:
    _reject_labels(value)
    if value.get("schema") != "cept-causal-transport-teacher-v2":
        raise ValueError("unsupported transport teacher schema")
    response_id = str(value.get("response_id", "")).strip()
    source_id = str(value.get("source_id", "")).strip()
    official_split = str(value.get("official_split", "")).casefold()
    model_signature = str(value.get("model_signature", "")).strip()
    model_source_signature = str(value.get("model_source_signature", "")).strip()
    tokenizer_signature = str(value.get("tokenizer_signature", "")).strip()
    observer_runtime, observer_runtime_signature = parse_observer_runtime_identity(
        value.get("observer_runtime"), value.get("observer_runtime_signature")
    )
    factual_input_ids_sha256 = str(value.get("factual_input_ids_sha256", "")).strip()
    response_idx = int(value.get("response_idx", -1))
    if (
        not response_id
        or not source_id
        or official_split != "train"
        or response_idx <= 0
        or not model_signature
        or not model_source_signature
        or not tokenizer_signature
        or not factual_input_ids_sha256
    ):
        raise ValueError("transport teacher has an invalid identity/split/response_idx")
    changed = _integer_tuple(
        value.get("changed_evidence_positions"), "changed_evidence_positions"
    )
    if any(position < 0 or position >= response_idx for position in changed):
        raise ValueError("changed evidence positions must precede response_idx")

    raw_blocks = value.get("block_definitions")
    if not isinstance(raw_blocks, Sequence) or isinstance(raw_blocks, (str, bytes)):
        raise TypeError("block_definitions must be a sequence")
    blocks: dict[str, tuple[int, ...]] = {}
    for raw in raw_blocks:
        if not isinstance(raw, Mapping):
            raise TypeError("each block definition must be an object")
        block_id = str(raw.get("block_id", "")).strip()
        if not block_id or block_id in blocks:
            raise ValueError("history block ids must be non-empty and unique")
        positions = _integer_tuple(raw.get("positions"), f"block {block_id} positions")
        if any(position < response_idx for position in positions):
            raise ValueError("history block positions must lie in the response")
        blocks[block_id] = positions

    raw_events = value.get("events")
    if (
        not isinstance(raw_events, Sequence)
        or isinstance(raw_events, (str, bytes))
        or not raw_events
    ):
        raise ValueError("transport teacher events must be a non-empty sequence")
    events: list[TransportEventTarget] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise TypeError("each transport event must be an object")
        block_rescue = raw.get("block_rescue")
        if not isinstance(block_rescue, Mapping) or set(block_rescue) != set(blocks):
            raise ValueError("event block_rescue keys disagree with block definitions")
        event = TransportEventTarget(
            target_position=int(raw["target_position"]),
            predictor_position=int(raw["predictor_position"]),
            target_token_id=int(raw["target_token_id"]),
            total_effect=_finite(raw["total_effect"], "total_effect"),
            direct_effect=_finite(raw["direct_effect"], "direct_effect"),
            history_mediated_effect=_finite(
                raw["history_mediated_effect"], "history_mediated_effect"
            ),
            alternate_history_effect=_finite(
                raw["alternate_history_effect"], "alternate_history_effect"
            ),
            interaction=_finite(raw["interaction"], "interaction"),
            decomposition_residual=_finite(
                raw["decomposition_residual"], "decomposition_residual"
            ),
            direct_seed_rescue=_finite(raw["direct_seed_rescue"], "direct_seed_rescue"),
            joint_seed_history_rescue=_finite(
                raw["joint_seed_history_rescue"], "joint_seed_history_rescue"
            ),
            representation_residual=_finite(
                raw["representation_residual"], "representation_residual"
            ),
            seed_history_interaction=_finite(
                raw["seed_history_interaction"], "seed_history_interaction"
            ),
            self_patch_error=_finite(raw["self_patch_error"], "self_patch_error"),
            block_rescue={
                str(key): _finite(item, f"block_rescue[{key}]")
                for key, item in block_rescue.items()
            },
        )
        identities = {
            "total=direct+history+residual": (
                event.total_effect,
                event.direct_effect
                + event.history_mediated_effect
                + event.decomposition_residual,
            ),
            "interaction=history-alternate": (
                event.interaction,
                event.history_mediated_effect - event.alternate_history_effect,
            ),
            "representation=total-joint": (
                event.representation_residual,
                event.total_effect - event.joint_seed_history_rescue,
            ),
            "seed-history interaction=joint-seed-alternate": (
                event.seed_history_interaction,
                event.joint_seed_history_rescue
                - event.direct_seed_rescue
                - event.alternate_history_effect,
            ),
        }
        for name, (observed, expected) in identities.items():
            tolerance = 1e-5 * (1.0 + max(abs(observed), abs(expected)))
            if abs(observed - expected) > tolerance:
                raise ValueError(
                    f"transport teacher violates algebraic identity: {name}"
                )
        expected_position = response_idx + index
        if (
            event.target_position != expected_position
            or event.predictor_position != expected_position - 1
        ):
            raise ValueError(
                "teacher target/predictor positions are not contiguous next-token events"
            )
        events.append(event)
    return TransportTeacher(
        response_id=response_id,
        source_id=source_id,
        official_split=official_split,
        model_signature=model_signature,
        model_source_signature=model_source_signature,
        tokenizer_signature=tokenizer_signature,
        observer_runtime=observer_runtime,
        observer_runtime_signature=observer_runtime_signature,
        factual_input_ids_sha256=factual_input_ids_sha256,
        response_idx=response_idx,
        changed_evidence_positions=changed,
        block_positions=blocks,
        events=tuple(events),
    )


def align_teacher_to_graph(
    teacher: TransportTeacher, graph: Mapping[str, object]
) -> AlignedTransportExample:
    if graph.get("schema") != "cept-prediction-event-graph-v2":
        raise ValueError("unsupported prediction-event graph schema")
    identities = (
        (teacher.response_id, str(graph.get("response_id", ""))),
        (teacher.source_id, str(graph.get("source_id", ""))),
        (teacher.official_split, str(graph.get("dataset_split", ""))),
    )
    if any(expected != observed for expected, observed in identities):
        raise ValueError("transport teacher and graph identity/split mismatch")
    observed = {
        "target": torch.as_tensor(graph["target_token_positions"]).long().flatten(),
        "predictor": torch.as_tensor(graph["predictor_positions"]).long().flatten(),
        "token": torch.as_tensor(graph["target_token_ids"]).long().flatten(),
    }
    expected = {
        "target": torch.tensor([event.target_position for event in teacher.events]),
        "predictor": torch.tensor(
            [event.predictor_position for event in teacher.events]
        ),
        "token": torch.tensor([event.target_token_id for event in teacher.events]),
    }
    for name, expected_value in expected.items():
        if not torch.equal(observed[name].cpu(), expected_value):
            raise ValueError(f"transport teacher/graph {name} alignment failed")
    graph_input_hash = canonical_hash(
        torch.as_tensor(graph["token_ids"]).long().flatten().tolist()
    )
    if graph_input_hash != teacher.factual_input_ids_sha256:
        raise ValueError("transport teacher/graph full factual input hash mismatch")
    return AlignedTransportExample(teacher=teacher, graph=graph)


def build_teacher_targets(
    teacher: TransportTeacher, *, calibration: EffectCalibration
) -> TeacherTargets:
    calibration.validate()
    total = torch.tensor(
        [event.total_effect for event in teacher.events], dtype=torch.float32
    )
    direction = torch.sign(total)
    positive_total = total > calibration.null_floor
    null_effect = total.abs() <= calibration.null_floor
    contradictory = total < -calibration.null_floor
    # Patching an evidence seed can change earlier response representations and
    # therefore measures total recursive seed support, not a pure direct path.
    support_raw = torch.relu(
        direction
        * torch.tensor(
            [event.direct_seed_rescue for event in teacher.events],
            dtype=torch.float32,
        )
    )
    history_raw = torch.relu(
        direction
        * torch.tensor(
            [event.alternate_history_effect for event in teacher.events],
            dtype=torch.float32,
        )
    )
    joint_raw = torch.relu(
        direction
        * torch.tensor(
            [event.joint_seed_history_rescue for event in teacher.events],
            dtype=torch.float32,
        )
    )
    support = (
        (support_raw - calibration.null_floor)
        / (calibration.support_scale - calibration.null_floor)
    ).clamp(0.0, 1.0)
    history = (
        (history_raw - calibration.null_floor)
        / (calibration.history_scale - calibration.null_floor)
    ).clamp(0.0, 1.0)
    support = torch.where(positive_total, support, torch.zeros_like(support))
    history = torch.where(
        positive_total, torch.minimum(history, support), torch.zeros_like(history)
    )

    representation_residual = torch.tensor(
        [abs(event.representation_residual) for event in teacher.events],
        dtype=torch.float32,
    )
    seed_history_interaction = torch.tensor(
        [abs(event.seed_history_interaction) for event in teacher.events],
        dtype=torch.float32,
    )
    representation_ratio = representation_residual / (
        total.abs() + calibration.null_floor
    )
    interaction_ratio = seed_history_interaction / (joint_raw + calibration.null_floor)
    reliable_positive = (
        positive_total
        & (support_raw > calibration.null_floor)
        & (joint_raw > calibration.null_floor)
        & (representation_ratio <= 1.0)
        & (interaction_ratio <= 1.0)
    )
    positive_weight = 1.0 / (1.0 + representation_ratio + interaction_ratio)
    event_weight = torch.where(
        positive_total,
        torch.where(
            reliable_positive, positive_weight, torch.zeros_like(positive_weight)
        ),
        torch.where(
            null_effect,
            torch.ones_like(positive_weight),
            torch.zeros_like(positive_weight),
        ),
    )

    block_ids = tuple(teacher.block_positions)
    block = torch.zeros((len(block_ids), len(teacher.events)), dtype=torch.float32)
    block_mask = torch.zeros_like(block, dtype=torch.bool)
    for event_index, event in enumerate(teacher.events):
        raw = torch.relu(
            direction[event_index]
            * torch.tensor(
                [event.block_rescue[block_id] for block_id in block_ids],
                dtype=torch.float32,
            )
        )
        block[:, event_index] = (
            (raw - calibration.null_floor)
            / (calibration.block_scale - calibration.null_floor)
        ).clamp(0.0, 1.0)
        for block_index, block_id in enumerate(block_ids):
            block_mask[block_index, event_index] = bool(
                reliable_positive[event_index]
            ) and any(
                position < event.target_position
                for position in teacher.block_positions[block_id]
            )
    return TeacherTargets(
        support=support,
        history=history,
        block=block,
        event_weight=event_weight,
        positive_mask=reliable_positive,
        null_mask=null_effect,
        contradictory_mask=contradictory,
        block_mask=block_mask,
        block_ids=block_ids,
    )


def _quantile_or_default(
    values: Sequence[float], *, quantile: float, default: float
) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return default
    return float(torch.quantile(torch.tensor(finite, dtype=torch.float64), quantile))


def fit_effect_calibration(
    teachers: Sequence[TransportTeacher],
    *,
    availability_by_identity: Mapping[tuple[str, str], Sequence[bool]],
    minimum_null_floor: float,
) -> EffectCalibration:
    """Fit label-free scales only on training events observable by the student."""

    if not teachers or minimum_null_floor <= 0 or not math.isfinite(minimum_null_floor):
        raise ValueError("calibration requires teachers and a positive minimum null")
    teacher_identities = [(teacher.source_id, teacher.response_id) for teacher in teachers]
    if len(set(teacher_identities)) != len(teacher_identities):
        raise ValueError("calibration teachers contain duplicate identities")
    if set(availability_by_identity) != set(teacher_identities):
        raise ValueError(
            "calibration availability identities must exactly match the teachers"
        )
    available_events: list[TransportEventTarget] = []
    for teacher in teachers:
        identity = (teacher.source_id, teacher.response_id)
        availability = tuple(bool(value) for value in availability_by_identity[identity])
        if len(availability) != len(teacher.events):
            raise ValueError(
                "calibration availability length does not match teacher events for "
                f"source_id={teacher.source_id!r} response_id={teacher.response_id!r}"
            )
        available_events.extend(
            event for event, is_available in zip(teacher.events, availability, strict=True)
            if is_available
        )
    if not available_events:
        raise ValueError("calibration has no graph-observable training events")

    self_errors = [event.self_patch_error for event in available_events]
    null_floor = max(
        minimum_null_floor,
        _quantile_or_default(self_errors, quantile=0.95, default=minimum_null_floor),
    )

    def reliable_positive(event: TransportEventTarget) -> bool:
        if (
            event.total_effect <= null_floor
            or event.direct_seed_rescue <= null_floor
            or event.joint_seed_history_rescue <= null_floor
        ):
            return False
        representation_ratio = abs(event.representation_residual) / (
            abs(event.total_effect) + null_floor
        )
        interaction_ratio = abs(event.seed_history_interaction) / (
            abs(event.joint_seed_history_rescue) + null_floor
        )
        return representation_ratio <= 1.0 and interaction_ratio <= 1.0

    positive_support = [
        event.direct_seed_rescue
        for event in available_events
        if reliable_positive(event)
    ]
    positive_history = [
        event.alternate_history_effect
        for event in available_events
        if reliable_positive(event) and event.alternate_history_effect > null_floor
    ]
    support_scale = _quantile_or_default(
        positive_support, quantile=0.90, default=10.0 * null_floor
    )
    history_scale = _quantile_or_default(
        positive_history, quantile=0.90, default=10.0 * null_floor
    )
    positive_blocks: list[float] = []
    for event in available_events:
        if not reliable_positive(event):
            continue
        positive_blocks.extend(
            value for value in event.block_rescue.values() if value > null_floor
        )
    block_scale = _quantile_or_default(
        positive_blocks, quantile=0.90, default=10.0 * null_floor
    )
    calibration = EffectCalibration(
        null_floor=null_floor,
        support_scale=max(support_scale, null_floor * (1.0 + 1e-3)),
        history_scale=max(history_scale, null_floor * (1.0 + 1e-3)),
        block_scale=max(block_scale, null_floor * (1.0 + 1e-3)),
    )
    calibration.validate()
    return calibration


def read_transport_teachers(path: str | Path) -> tuple[TransportTeacher, ...]:
    source = Path(path).expanduser().resolve()
    teachers: list[TransportTeacher] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise TypeError("record is not an object")
                teachers.append(parse_transport_teacher(raw))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid transport teacher at {source}:{line_number}"
                ) from error
    if not teachers:
        raise ValueError(f"transport teacher file is empty: {source}")
    identities = [(item.source_id, item.response_id) for item in teachers]
    if len(set(identities)) != len(identities):
        raise ValueError("transport teacher file contains duplicate identities")
    return tuple(teachers)


__all__ = [
    "AlignedTransportExample",
    "EffectCalibration",
    "TeacherTargets",
    "TransportEventTarget",
    "TransportTeacher",
    "align_teacher_to_graph",
    "build_teacher_targets",
    "fit_effect_calibration",
    "make_transport_teacher_record",
    "parse_transport_teacher",
    "read_transport_teachers",
]
