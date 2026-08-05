"""Build prediction-event graphs from the formal sparse attention cache."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

import torch


class Segment(IntEnum):
    EVIDENCE = 0
    QUERY = 1
    RESPONSE = 2


class Relation(IntEnum):
    EY = 0
    QY = 1
    RY = 2


_RELATION_BY_SEGMENT = {
    Segment.EVIDENCE: Relation.EY,
    Segment.QUERY: Relation.QY,
    Segment.RESPONSE: Relation.RY,
}
_CACHE_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}
_SOURCE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FLOAT32_SUM_SLACK = 8.0 * float(torch.finfo(torch.float32).eps)


@dataclass(frozen=True)
class PredictionEventGraph:
    """Portable graph whose targets are next-token prediction events."""

    source_id: str
    response_id: str
    dataset_split: str
    attention_cache_fingerprint: str
    attention_cache_sha256: str
    cache_dtype: str
    attention_floor: float
    num_layers: int
    num_heads: int
    token_ids: torch.Tensor
    segment_ids: torch.Tensor
    target_token_ids: torch.Tensor
    target_token_positions: torch.Tensor
    predictor_positions: torch.Tensor
    row_available: torch.Tensor
    edge_index: torch.Tensor
    edge_relation: torch.Tensor
    trace_edge_id: torch.Tensor
    trace_channel: torch.Tensor
    trace_value: torch.Tensor
    retained_mass_by_relation: torch.Tensor
    unknown_mass: torch.Tensor
    rounding_excess_mass: torch.Tensor

    @property
    def num_channels(self) -> int:
        return self.num_layers * self.num_heads

    def as_payload(self) -> dict[str, object]:
        """Return the explicit on-disk schema without evaluation labels."""

        return {
            "schema": "cept-prediction-event-graph-v2",
            "source_id": self.source_id,
            "response_id": self.response_id,
            "dataset_split": self.dataset_split,
            "attention_cache_fingerprint": self.attention_cache_fingerprint,
            "attention_cache_sha256": self.attention_cache_sha256,
            "cache_dtype": self.cache_dtype,
            "attention_floor": self.attention_floor,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "token_ids": self.token_ids,
            "segment_ids": self.segment_ids,
            "target_token_ids": self.target_token_ids,
            "target_token_positions": self.target_token_positions,
            "predictor_positions": self.predictor_positions,
            "row_available": self.row_available,
            "edge_index": self.edge_index,
            "edge_relation": self.edge_relation,
            "trace_edge_id": self.trace_edge_id,
            "trace_channel": self.trace_channel,
            "trace_value": self.trace_value,
            "retained_mass_by_relation": self.retained_mass_by_relation,
            "unknown_mass": self.unknown_mass,
            "rounding_excess_mass": self.rounding_excess_mass,
        }


def _reject_labels(record: Mapping[str, object]) -> None:
    forbidden = [
        str(key)
        for key in record
        if "label" in str(key).casefold() or str(key).casefold() == "y_token"
    ]
    if forbidden:
        raise ValueError(
            "prediction graphs must be label-blind; forbidden fields: "
            + ", ".join(sorted(forbidden))
        )


def _vector(record: Mapping[str, object], name: str, dtype: torch.dtype) -> torch.Tensor:
    if name not in record:
        raise ValueError(f"attention record is missing {name}")
    value = torch.as_tensor(record[name]).detach().cpu()
    if value.ndim != 1:
        raise ValueError(f"attention record {name} must be one-dimensional")
    return value.to(dtype)


def _storage_dtype(record: Mapping[str, object]) -> tuple[str, torch.dtype]:
    if "cache_dtype" in record:
        name = str(record["cache_dtype"]).casefold().removeprefix("torch.")
        if name not in _CACHE_DTYPES:
            raise ValueError("cache_dtype must be float16, bfloat16, or float32")
        dtype = _CACHE_DTYPES[name]
    else:
        dtype = torch.as_tensor(record["response_values"]).dtype
        if dtype not in set(_CACHE_DTYPES.values()):
            raise ValueError("response_values must use a supported floating dtype")
    return str(dtype).removeprefix("torch."), dtype


def _unit_roundoff(dtype: torch.dtype) -> float:
    return 0.5 * float(torch.finfo(dtype).eps)


def _half_smallest_subnormal(dtype: torch.dtype) -> float:
    info = torch.finfo(dtype)
    return 0.5 * float(info.tiny) * float(info.eps)


def _mass_rounding_tolerance(row_length: int, storage_dtype: torch.dtype) -> float:
    bounds: list[tuple[float, float]] = []
    for source_dtype in _SOURCE_DTYPES:
        source_roundoff = 0.0 if source_dtype == torch.float32 else _unit_roundoff(source_dtype)
        source_subnormal = (
            0.0
            if source_dtype == torch.float32
            else _half_smallest_subnormal(source_dtype)
        )
        if source_dtype == storage_dtype:
            storage_roundoff = 0.0
            storage_subnormal = 0.0
        else:
            storage_roundoff = _unit_roundoff(storage_dtype)
            storage_subnormal = _half_smallest_subnormal(storage_dtype)
        relative = (
            (1.0 + _FLOAT32_SUM_SLACK)
            * (1.0 + source_roundoff)
            * (1.0 + storage_roundoff)
            - 1.0
        )
        absolute = source_subnormal * (1.0 + storage_roundoff) + storage_subnormal
        bounds.append((relative, absolute))
    return max(value[0] for value in bounds) + (row_length + 1) * max(
        value[1] for value in bounds
    )


def validate_unlabeled_record(record: Mapping[str, object]) -> None:
    """Validate the formal cache subset required by prediction-event graphs."""

    _reject_labels(record)
    required_fields = (
        "source_id",
        "response_idx",
        "num_attention_layers",
        "num_attention_heads",
        "attention_floor",
        "token_ids",
        "segment_ids",
        "attention_diagonal",
        "response_row_ptr",
        "response_column_indices",
        "response_values",
    )
    missing = [name for name in required_fields if name not in record]
    if missing:
        raise ValueError("attention record is missing " + ", ".join(missing))
    source_id = str(record["source_id"]).strip()
    response_id = str(record.get("response_id", record.get("sample_id", ""))).strip()
    fingerprint = str(record.get("attention_cache_fingerprint", "")).strip()
    dataset_split = str(record.get("dataset_split", record.get("split", ""))).casefold()
    if not source_id or not response_id or not fingerprint:
        raise ValueError(
            "source_id, response_id, and attention_cache_fingerprint must be non-empty"
        )
    if dataset_split not in {"train", "test"}:
        raise ValueError("attention record split must be train or test")
    token_ids = _vector(record, "token_ids", torch.long)
    segment_ids = _vector(record, "segment_ids", torch.int8)
    if token_ids.numel() != segment_ids.numel():
        raise ValueError("segment_ids must align one-to-one with token_ids")
    token_count = int(token_ids.numel())
    response_idx = int(record["response_idx"])
    if not 0 < response_idx < token_count:
        raise ValueError("response_idx must split a non-empty prompt and response")
    valid_segments = torch.tensor([int(item) for item in Segment], dtype=torch.int8)
    if bool((~torch.isin(segment_ids, valid_segments)).any()):
        raise ValueError("segment_ids contain an unsupported segment")

    layers = int(record["num_attention_layers"])
    heads = int(record["num_attention_heads"])
    if layers <= 0 or heads <= 0:
        raise ValueError("attention layer/head counts must be positive")
    diagonal = torch.as_tensor(record.get("attention_diagonal")).detach().cpu()
    if diagonal.shape != (layers, heads, token_count):
        raise ValueError("attention_diagonal has an inconsistent shape")
    row_ptr = _vector(record, "response_row_ptr", torch.long)
    columns = _vector(record, "response_column_indices", torch.long)
    values = _vector(record, "response_values", torch.float32)
    _, storage_dtype = _storage_dtype(record)
    if torch.as_tensor(record["response_values"]).dtype != storage_dtype:
        raise ValueError("response_values dtype disagrees with cache_dtype")
    if torch.as_tensor(record["attention_diagonal"]).dtype != storage_dtype:
        raise ValueError("attention_diagonal dtype disagrees with cache_dtype")
    response_tokens = token_count - response_idx
    expected_rows = layers * heads * response_tokens
    if row_ptr.numel() != expected_rows + 1:
        raise ValueError("response_row_ptr has an inconsistent row count")
    if (
        int(row_ptr[0]) != 0
        or int(row_ptr[-1]) != values.numel()
        or bool((row_ptr[1:] < row_ptr[:-1]).any())
        or columns.numel() != values.numel()
    ):
        raise ValueError("response attention CSR arrays are inconsistent")
    floor = float(record["attention_floor"])
    if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError("attention_floor must be finite in (0, 1]")
    if not bool(torch.isfinite(diagonal).all()) or not bool(torch.isfinite(values).all()):
        raise ValueError("attention values must be finite")
    if bool(((diagonal < 0) | (diagonal > 1)).any()) or bool(
        ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("attention values must lie in [0, 1]")
    quantized_floor = torch.tensor(floor, dtype=storage_dtype)
    lower_floor = float(
        torch.nextafter(
            quantized_floor,
            torch.tensor(float("-inf"), dtype=storage_dtype),
        )
    )
    if bool((values < lower_floor).any()):
        raise ValueError("retained response attention falls below the cache floor")


def build_prediction_event_graph(
    record: Mapping[str, object],
) -> PredictionEventGraph:
    """Shift response query rows onto the tokens they actually predict.

    The legacy cache stores query rows at response positions.  Row ``q`` is
    therefore assigned to the event for token ``q + 1``.  The first response
    event is retained with an unavailable row and all attention mass unknown.
    """

    validate_unlabeled_record(record)
    token_ids = _vector(record, "token_ids", torch.long)
    segment_ids = _vector(record, "segment_ids", torch.int8)
    diagonal = torch.as_tensor(record["attention_diagonal"]).detach().cpu().float()
    row_ptr = _vector(record, "response_row_ptr", torch.long)
    columns = _vector(record, "response_column_indices", torch.long)
    values = _vector(record, "response_values", torch.float32)
    cache_dtype, storage_dtype = _storage_dtype(record)
    response_idx = int(record["response_idx"])
    token_count = int(token_ids.numel())
    response_tokens = token_count - response_idx
    layers = int(record["num_attention_layers"])
    heads = int(record["num_attention_heads"])
    channels = layers * heads

    target_positions = torch.arange(response_idx, token_count, dtype=torch.long)
    predictor_positions = target_positions - 1
    row_available = predictor_positions >= response_idx
    retained = torch.zeros((response_tokens, channels, 3), dtype=torch.float32)
    unknown = torch.ones((response_tokens, channels), dtype=torch.float32)
    rounding_excess = torch.zeros_like(unknown)

    edge_ids: dict[tuple[int, int], int] = {}
    edge_sources: list[int] = []
    edge_events: list[int] = []
    edge_relations: list[int] = []
    trace_edges: list[int] = []
    trace_channels: list[int] = []
    trace_values: list[float] = []

    def add_trace(source: int, event: int, channel: int, value: float) -> None:
        segment = Segment(int(segment_ids[source]))
        relation = _RELATION_BY_SEGMENT[segment]
        key = (source, event)
        edge_id = edge_ids.get(key)
        if edge_id is None:
            edge_id = len(edge_sources)
            edge_ids[key] = edge_id
            edge_sources.append(source)
            edge_events.append(event)
            edge_relations.append(int(relation))
        elif edge_relations[edge_id] != int(relation):
            raise RuntimeError("one source/event pair acquired conflicting relations")
        trace_edges.append(edge_id)
        trace_channels.append(channel)
        trace_values.append(value)
        retained[event, channel, int(relation)] += value

    for event in range(response_tokens):
        if not bool(row_available[event]):
            continue
        predictor = int(predictor_positions[event])
        response_row = predictor - response_idx
        for channel in range(channels):
            row_id = channel * response_tokens + response_row
            start = int(row_ptr[row_id])
            end = int(row_ptr[row_id + 1])
            row_columns = columns[start:end]
            if bool(((row_columns < 0) | (row_columns >= predictor)).any()):
                raise ValueError(
                    "cached response row contains a negative/future source column"
                )
            if row_columns.numel() > 1 and bool(
                (row_columns[1:] <= row_columns[:-1]).any()
            ):
                raise ValueError(
                    "cached response row source columns must be sorted and unique"
                )
            for entry in range(start, end):
                source = int(columns[entry])
                add_trace(source, event, channel, float(values[entry]))
            layer, head = divmod(channel, heads)
            diagonal_value = float(diagonal[layer, head, predictor])
            if diagonal_value:
                add_trace(predictor, event, channel, diagonal_value)
            observed = float(retained[event, channel].sum())
            tolerance = _mass_rounding_tolerance(end - start, storage_dtype)
            if observed > 1.0 + tolerance:
                raise ValueError(
                    "retained attention plus diagonal exceeds the dtype-aware row mass bound"
                )
            rounding_excess[event, channel] = max(0.0, observed - 1.0)
            unknown[event, channel] = max(0.0, 1.0 - observed)

    edge_index = (
        torch.tensor([edge_sources, edge_events], dtype=torch.long)
        if edge_sources
        else torch.empty((2, 0), dtype=torch.long)
    )
    return PredictionEventGraph(
        source_id=str(record["source_id"]),
        response_id=str(record.get("response_id", record.get("sample_id", ""))),
        dataset_split=str(record.get("dataset_split", record.get("split", ""))),
        attention_cache_fingerprint=str(record.get("attention_cache_fingerprint", "")),
        attention_cache_sha256=str(record.get("attention_cache_sha256", "")),
        cache_dtype=cache_dtype,
        attention_floor=float(record["attention_floor"]),
        num_layers=layers,
        num_heads=heads,
        token_ids=token_ids,
        segment_ids=segment_ids,
        target_token_ids=token_ids[response_idx:].clone(),
        target_token_positions=target_positions,
        predictor_positions=predictor_positions,
        row_available=row_available,
        edge_index=edge_index,
        edge_relation=torch.tensor(edge_relations, dtype=torch.int8),
        trace_edge_id=torch.tensor(trace_edges, dtype=torch.long),
        trace_channel=torch.tensor(trace_channels, dtype=torch.long),
        trace_value=torch.tensor(trace_values, dtype=torch.float32),
        retained_mass_by_relation=retained,
        unknown_mass=unknown,
        rounding_excess_mass=rounding_excess,
    )


__all__ = [
    "PredictionEventGraph",
    "Relation",
    "Segment",
    "build_prediction_event_graph",
    "validate_unlabeled_record",
]
