"""Attention-only graph construction with ordered layer/head channels.

Each response-attention token pair is one directed RP or RR edge.  The edge's
per-layer/per-head values remain sparse instead of being collapsed to summary
statistics or expanded into a potentially huge ``edges x channels`` matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Mapping

import torch


FORMAL_CACHE_SCHEMA = "ragtruth-all-layers-all-heads-sparse-response-csr-v1"
RP = 0
RR = 1


@dataclass(frozen=True)
class GraphBuildConfig:
    """Controls graph support selection without changing cached attention."""

    selection: str = "threshold"
    threshold: float | None = None
    top_k: int = 8
    max_edges_per_target: int | None = None
    query_block: int = 64

    def validate(self, *, attention_floor: float) -> None:
        if self.selection not in {"threshold", "global_topk", "typed_topk"}:
            raise ValueError(
                "selection must be threshold, global_topk, or typed_topk"
            )
        if self.top_k < 1 or self.query_block < 1:
            raise ValueError("top_k and query_block must be positive")
        if self.max_edges_per_target is not None and self.max_edges_per_target < 1:
            raise ValueError("max_edges_per_target must be positive when provided")
        threshold = attention_floor if self.threshold is None else self.threshold
        if not math.isfinite(threshold) or not attention_floor <= threshold <= 1.0:
            raise ValueError("threshold must be finite and at least the cache floor")


@dataclass(frozen=True)
class AttentionGraph:
    """One attention-attributed token graph.

    ``trace_*`` is a COO tensor over pair edges and layer/head channels:
    ``trace_value[t]`` is the retained attention for edge
    ``trace_edge_id[t]`` in flattened channel ``trace_channel[t]``.
    Missing channels are cache-censored (below ``attention_floor``), not zeros.
    """

    source_id: str
    response_id: str
    sample_id: str
    response_idx: int
    num_layers: int
    num_heads: int
    attention_floor: float
    node_attr: torch.Tensor
    node_context: torch.Tensor
    response_mask: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_score: torch.Tensor
    trace_edge_id: torch.Tensor
    trace_channel: torch.Tensor
    trace_value: torch.Tensor
    token_ids: torch.Tensor
    build_config: GraphBuildConfig

    @property
    def num_nodes(self) -> int:
        return int(self.node_attr.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_channels(self) -> int:
        return self.num_layers * self.num_heads

    def to(self, device: str | torch.device) -> "AttentionGraph":
        requested = torch.device(device)
        values: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(requested) if isinstance(value, torch.Tensor) else value
        return AttentionGraph(**values)


def _as_int(value: object, name: str) -> int:
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    return int(tensor.item())


def _reject_labels(sample: Mapping[str, object]) -> None:
    forbidden = [
        str(key)
        for key in sample
        if str(key).casefold() in {"y", "y_token", "hallucination_labels"}
        or "label" in str(key).casefold()
    ]
    if forbidden:
        raise ValueError(
            "graph construction is label-blind; remove evaluation fields first: "
            + ", ".join(sorted(forbidden))
        )


def _segmented_topk(
    score: torch.Tensor, group: torch.Tensor, *, top_k: int
) -> torch.Tensor:
    if not len(score):
        return torch.empty(0, dtype=torch.long, device=score.device)
    by_score = torch.argsort(score, descending=True, stable=True)
    order = by_score[torch.argsort(group[by_score], stable=True)]
    grouped = group[order]
    position = torch.arange(len(order), device=score.device)
    starts = torch.where(
        torch.cat(
            (
                torch.ones(1, dtype=torch.bool, device=score.device),
                grouped[1:] != grouped[:-1],
            )
        ),
        position,
        torch.zeros_like(position),
    )
    rank = position - torch.cummax(starts, dim=0).values
    return order[rank < top_k]


def _selected_pairs(
    *,
    score: torch.Tensor,
    maximum: torch.Tensor,
    target: torch.Tensor,
    relation: torch.Tensor,
    config: GraphBuildConfig,
    attention_floor: float,
) -> torch.Tensor:
    if config.selection == "threshold":
        threshold = attention_floor if config.threshold is None else config.threshold
        selected = torch.nonzero(maximum >= threshold, as_tuple=False).flatten()
        if config.max_edges_per_target is not None and len(selected):
            local = _segmented_topk(
                score[selected], target[selected], top_k=config.max_edges_per_target
            )
            selected = selected[local]
        return selected
    group = target if config.selection == "global_topk" else target * 2 + relation
    return _segmented_topk(score, group, top_k=config.top_k)


def _validate_and_move(
    sample: Mapping[str, object], device: torch.device
) -> tuple[int, int, int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if str(sample.get("attention_cache_schema", "")) != FORMAL_CACHE_SCHEMA:
        raise ValueError("only the formal sparse RAGTruth attention cache is supported")
    layers = _as_int(sample["num_attention_layers"], "num_attention_layers")
    heads = _as_int(sample["num_attention_heads"], "num_attention_heads")
    diagonal = torch.as_tensor(sample["attention_diagonal"]).to(
        device=device, dtype=torch.float32
    )
    if diagonal.ndim != 3 or tuple(diagonal.shape[:2]) != (layers, heads):
        raise ValueError("attention_diagonal must have shape [layers, heads, tokens]")
    token_count = int(diagonal.shape[2])
    response_idx = _as_int(sample["response_idx"], "response_idx")
    if not 0 < response_idx < token_count:
        raise ValueError("response_idx must split non-empty prompt and response")
    row_ptr = torch.as_tensor(sample["response_row_ptr"]).to(
        device=device, dtype=torch.long
    ).flatten()
    columns = torch.as_tensor(sample["response_column_indices"]).to(
        device=device, dtype=torch.long
    ).flatten()
    values = torch.as_tensor(sample["response_values"]).to(
        device=device, dtype=torch.float32
    ).flatten()
    expected_rows = layers * heads * (token_count - response_idx)
    valid_ptr = (
        row_ptr.numel() == expected_rows + 1
        and int(row_ptr[0]) == 0
        and int(row_ptr[-1]) == values.numel()
        and not bool((row_ptr[1:] < row_ptr[:-1]).any())
    )
    if not valid_ptr or columns.numel() != values.numel():
        raise ValueError("response attention CSR arrays are inconsistent")
    if not bool(torch.isfinite(diagonal).all()) or bool(
        ((diagonal < 0) | (diagonal > 1)).any()
    ):
        raise ValueError("attention diagonal must be finite in [0, 1]")
    if not bool(torch.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
        raise ValueError("response attention values must be finite in [0, 1]")
    return layers, heads, token_count, response_idx, diagonal, row_ptr, columns, values


def build_attention_graph(
    sample: Mapping[str, object],
    config: GraphBuildConfig | None = None,
    *,
    device: str | torch.device | None = None,
) -> AttentionGraph:
    """Build one RP/RR graph without labels or hand-crafted routing features."""

    _reject_labels(sample)
    requested = torch.device("cpu" if device is None else device)
    (
        layers,
        heads,
        token_count,
        response_idx,
        diagonal,
        row_ptr,
        columns,
        values,
    ) = _validate_and_move(sample, requested)
    attention_floor = float(sample["attention_floor"])
    if not math.isfinite(attention_floor) or not 0.0 < attention_floor <= 1.0:
        raise ValueError("attention_floor must be finite in (0, 1]")
    build_config = GraphBuildConfig() if config is None else config
    build_config.validate(attention_floor=attention_floor)

    response_tokens = token_count - response_idx
    channels = layers * heads
    channel_ids = torch.arange(channels, device=requested)
    edge_sources: list[torch.Tensor] = []
    edge_targets: list[torch.Tensor] = []
    edge_types: list[torch.Tensor] = []
    edge_scores: list[torch.Tensor] = []
    trace_edges: list[torch.Tensor] = []
    trace_channels: list[torch.Tensor] = []
    trace_values: list[torch.Tensor] = []
    edge_offset = 0

    for start in range(0, response_tokens, build_config.query_block):
        end = min(start + build_config.query_block, response_tokens)
        queries = torch.arange(start, end, device=requested)
        row_ids = (channel_ids[:, None] * response_tokens + queries).reshape(-1)
        starts = row_ptr[row_ids]
        lengths = row_ptr[row_ids + 1] - starts
        entry_count = int(lengths.sum())
        if entry_count == 0:
            continue
        repeated_starts = torch.repeat_interleave(starts, lengths)
        repeated_prefix = torch.repeat_interleave(
            torch.cumsum(lengths, dim=0) - lengths, lengths
        )
        positions = repeated_starts + torch.arange(
            entry_count, device=requested
        ) - repeated_prefix
        entry_rows = torch.repeat_interleave(row_ids, lengths)
        source = columns[positions]
        observed = values[positions]
        target = response_idx + entry_rows.remainder(response_tokens)
        channel = entry_rows // response_tokens
        if bool((source < 0).any()) or bool((source >= target).any()):
            raise ValueError("response attention columns violate causal ordering")

        pair_key, inverse = torch.unique(
            target * token_count + source, sorted=True, return_inverse=True
        )
        pair_count = len(pair_key)
        pair_sum = torch.zeros(pair_count, dtype=torch.float32, device=requested)
        pair_sum.index_add_(0, inverse, observed)
        pair_max = torch.full(
            (pair_count,), -torch.inf, dtype=torch.float32, device=requested
        )
        pair_max.scatter_reduce_(0, inverse, observed, reduce="amax", include_self=True)
        pair_source = pair_key.remainder(token_count)
        pair_target = pair_key // token_count
        relation = (pair_source >= response_idx).long()
        pair_score = pair_sum / float(channels)
        selected = _selected_pairs(
            score=pair_score,
            maximum=pair_max,
            target=pair_target,
            relation=relation,
            config=build_config,
            attention_floor=attention_floor,
        )
        if not len(selected):
            continue

        local_to_edge = torch.full(
            (pair_count,), -1, dtype=torch.long, device=requested
        )
        local_to_edge[selected] = torch.arange(
            edge_offset, edge_offset + len(selected), device=requested
        )
        entry_edge = local_to_edge[inverse]
        keep_trace = entry_edge >= 0
        edge_sources.append(pair_source[selected])
        edge_targets.append(pair_target[selected])
        edge_types.append(relation[selected])
        edge_scores.append(pair_score[selected])
        trace_edges.append(entry_edge[keep_trace])
        trace_channels.append(channel[keep_trace])
        trace_values.append(observed[keep_trace])
        edge_offset += len(selected)

    if edge_sources:
        edge_index = torch.stack(
            (torch.cat(edge_sources), torch.cat(edge_targets)), dim=0
        )
        edge_type = torch.cat(edge_types)
        edge_score = torch.cat(edge_scores)
        trace_edge_id = torch.cat(trace_edges)
        trace_channel = torch.cat(trace_channels)
        trace_value = torch.cat(trace_values)
        trace_key = trace_edge_id * channels + trace_channel
        trace_order = torch.argsort(trace_key, stable=True)
        trace_edge_id = trace_edge_id[trace_order]
        trace_channel = trace_channel[trace_order]
        trace_value = trace_value[trace_order]
        sorted_key = trace_key[trace_order]
        if sorted_key.numel() > 1 and bool(
            (sorted_key[1:] == sorted_key[:-1]).any()
        ):
            raise ValueError(
                "formal cache contains duplicate edge/channel trace entries"
            )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=requested)
        edge_type = torch.empty(0, dtype=torch.long, device=requested)
        edge_score = torch.empty(0, dtype=torch.float32, device=requested)
        trace_edge_id = torch.empty(0, dtype=torch.long, device=requested)
        trace_channel = torch.empty(0, dtype=torch.long, device=requested)
        trace_value = torch.empty(0, dtype=torch.float32, device=requested)

    response_mask = torch.arange(token_count, device=requested) >= response_idx
    position = torch.arange(token_count, dtype=torch.float32, device=requested)
    node_context = torch.stack(
        (
            position / max(token_count - 1, 1),
            (~response_mask).float(),
            response_mask.float(),
        ),
        dim=1,
    )
    node_attr = diagonal.permute(2, 0, 1).reshape(token_count, channels)
    token_ids = torch.as_tensor(sample["token_ids"]).to(
        device=requested, dtype=torch.long
    ).flatten()
    if token_ids.numel() != token_count:
        raise ValueError("token_ids must contain one id per node")
    response_id = str(sample.get("response_id", sample.get("sample_id", "")))
    sample_id = str(sample.get("sample_id", response_id))
    if not response_id or not sample_id:
        raise ValueError("response_id and sample_id must be non-empty")
    return AttentionGraph(
        source_id=str(sample["source_id"]),
        response_id=response_id,
        sample_id=sample_id,
        response_idx=response_idx,
        num_layers=layers,
        num_heads=heads,
        attention_floor=attention_floor,
        node_attr=node_attr,
        node_context=node_context,
        response_mask=response_mask,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_score=edge_score,
        trace_edge_id=trace_edge_id,
        trace_channel=trace_channel,
        trace_value=trace_value,
        token_ids=token_ids,
        build_config=build_config,
    )
