"""GPU-friendly, label-blind typed token graphs from legacy RAGTruth caches.

The legacy cache stores a dense attention tensor per response.  This module
never materialises a corpus-wide dense tensor: one cache record is memory
mapped, moved to the requested device, and immediately compressed into a
causal top-k COO graph.  ``top_k`` is a *combined* causal budget by default;
passing ``top_k_prefix`` and/or ``top_k_history`` requests independent relation
budgets when that is wanted for an ablation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import torch


_REQUIRED = frozenset({"source_id", "original_idx", "response_idx", "attention"})
_OPTIONAL = frozenset({"token_ids", "hidden"})
_FORBIDDEN_MARKERS = ("label", "target")
_ROUTE_DIM = 4  # edge count, mean weight, weight variance, mean relative span


def _as_scalar(value: object, name: str) -> int:
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    return int(tensor.item())


def _to_device(value: object, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach()
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device=device, non_blocking=device.type == "cuda")


def _require_label_blind_input(sample: Mapping[str, object]) -> None:
    forbidden = [
        str(key)
        for key in sample
        if str(key).casefold() in {"y", "y_token"}
        or any(marker in str(key).casefold() for marker in _FORBIDDEN_MARKERS)
    ]
    if forbidden:
        raise ValueError(
            "graph construction is label-blind; remove evaluation fields first: "
            + ", ".join(sorted(forbidden))
        )


def _validate_attention_sample(sample: Mapping[str, object]) -> tuple[torch.Tensor, int, int]:
    missing = sorted(_REQUIRED.difference(sample))
    if missing:
        raise ValueError(f"attention sample is missing required fields: {missing}")
    attention = torch.as_tensor(sample["attention"])
    if attention.ndim != 4 or attention.shape[-1] != attention.shape[-2]:
        raise ValueError("attention must have shape [layers, heads, tokens, tokens]")
    layers, heads, token_count, _ = attention.shape
    if layers < 1 or heads < 1 or token_count < 2:
        raise ValueError("attention requires at least one layer/head and two tokens")
    response_idx = _as_scalar(sample["response_idx"], "response_idx")
    if not 0 < response_idx < token_count:
        raise ValueError("response_idx must split a non-empty prefix and response")
    if "token_ids" in sample and torch.as_tensor(sample["token_ids"]).numel() != token_count:
        raise ValueError("token_ids must have one value per attention token")
    if "hidden" in sample:
        hidden = torch.as_tensor(sample["hidden"])
        valid = hidden.ndim == 2 and hidden.shape[0] == token_count
        valid |= hidden.ndim == 3 and hidden.shape[-2] == token_count
        if not valid:
            raise ValueError("hidden must have shape [tokens, dim] or [layers, tokens, dim]")
    return attention, response_idx, token_count


def load_attention_sample(
    path: str | Path,
    *,
    device: str | torch.device,
    mmap: bool = True,
    include_labels: bool = False,
) -> dict[str, object]:
    """Load one legacy record with a strict field whitelist.

    ``mmap=True`` requires a PyTorch version supporting memory-mapped
    ``torch.load``.  It intentionally raises instead of silently falling back
    to an eager CPU load, because the latter can exhaust host memory.
    """

    requested_device = torch.device(device)
    try:
        loaded = torch.load(
            Path(path), map_location="cpu", weights_only=True, mmap=bool(mmap)
        )
    except TypeError as error:
        if mmap:
            raise RuntimeError(
                "mmap=True requires a PyTorch build with torch.load(..., mmap=...); "
                "upgrade PyTorch or explicitly request mmap=False"
            ) from error
        raise
    if not isinstance(loaded, Mapping):
        raise ValueError(f"legacy attention cache must contain a mapping: {path}")

    unknown_required = _REQUIRED.difference(loaded)
    if unknown_required:
        raise ValueError(f"attention sample is missing required fields: {sorted(unknown_required)}")
    # A whitelist, rather than a blacklist, prevents newly added metadata from
    # accidentally carrying a target into graph training.
    record: dict[str, object] = {
        "source_id": str(loaded["source_id"]),
        "original_idx": _as_scalar(loaded["original_idx"], "original_idx"),
        "response_idx": _as_scalar(loaded["response_idx"], "response_idx"),
        # Preserve the storage dtype.  In the production path ``device=cpu``
        # keeps this tensor memory-mapped; query/layer tiles are copied to the
        # GPU later instead of materialising the full LxHxNxN tensor in RAM.
        "attention": _to_device(loaded["attention"], device=requested_device),
    }
    for name in _OPTIONAL:
        if name not in loaded:
            continue
        dtype = torch.long if name == "token_ids" else None
        record[name] = _to_device(loaded[name], device=requested_device, dtype=dtype)
    if include_labels and "hallucination_labels" in loaded:
        record["hallucination_labels"] = _to_device(
            loaded["hallucination_labels"], device=requested_device
        )
    _validate_attention_sample(record)
    return record


def _aggregate_query_tile(
    attention: torch.Tensor,
    query_start: int,
    query_end: int,
    *,
    device: torch.device,
    layer_chunk: int,
) -> torch.Tensor:
    """Average one QxN tile over layers/heads with bounded GPU memory."""

    layers, heads = attention.shape[:2]
    aggregate = torch.zeros(
        (query_end - query_start, attention.shape[-1]),
        device=device,
        dtype=torch.float32,
    )
    for layer_start in range(0, layers, layer_chunk):
        tile = attention[
            layer_start : layer_start + layer_chunk, :, query_start:query_end, :
        ].to(device=device, non_blocking=device.type == "cuda")
        aggregate.add_(tile.sum(dim=(0, 1), dtype=torch.float32))
        del tile
    return aggregate / float(layers * heads)


def _query_features(
    aggregate: torch.Tensor,
    query_ids: torch.Tensor,
    *,
    response_idx: int,
) -> torch.Tensor:
    """Return six attention-routing features for a query block."""

    keys = torch.arange(aggregate.shape[1], device=aggregate.device)
    causal = keys.unsqueeze(0) < query_ids.unsqueeze(1)
    causal_attention = aggregate.masked_fill(~causal, 0.0)
    total = causal_attention.sum(dim=1, keepdim=True)
    probabilities = causal_attention / total.clamp_min(torch.finfo(torch.float32).eps)
    entropy = -(
        probabilities * probabilities.clamp_min(torch.finfo(torch.float32).eps).log()
    ).sum(dim=1)
    entropy /= query_ids.clamp_min(1).float().log().clamp_min(1.0)
    prefix_mass = probabilities[:, :response_idx].sum(dim=1)
    history = (keys.unsqueeze(0) >= response_idx) & causal
    history_mass = probabilities.masked_fill(~history, 0.0).sum(dim=1)
    diagonal = aggregate.gather(1, query_ids[:, None]).squeeze(1)
    return torch.stack(
        (
            diagonal,
            prefix_mass,
            history_mass,
            entropy,
            probabilities.amax(dim=1),
            total.squeeze(1).clamp_min(0.0).log1p(),
        ),
        dim=1,
    )


def _select_block_edges(
    aggregate: torch.Tensor,
    query_ids: torch.Tensor,
    *,
    response_idx: int,
    top_k: int,
    top_k_prefix: int | None,
    top_k_history: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a query block in one tensor operation per relation."""

    token_count = aggregate.shape[1]
    keys = torch.arange(token_count, device=aggregate.device)
    causal = keys.unsqueeze(0) < query_ids.unsqueeze(1)
    separate = top_k_prefix is not None or top_k_history is not None
    prefix_budget = top_k if top_k_prefix is None else top_k_prefix
    history_budget = top_k if top_k_history is None else top_k_history
    relation_specs = (
        (
            causal & (keys.unsqueeze(0) < response_idx),
            prefix_budget,
            0,
        ),
        (
            causal & (keys.unsqueeze(0) >= response_idx),
            history_budget,
            1,
        ),
    ) if separate else ((causal, top_k, None),)

    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    relations: list[torch.Tensor] = []
    for valid, budget, relation_type in relation_specs:
        values = aggregate.masked_fill(~valid, -torch.inf)
        top_values, top_sources = torch.topk(
            values, k=min(int(budget), token_count), dim=1
        )
        present = torch.isfinite(top_values)
        source = top_sources[present]
        target = query_ids[:, None].expand_as(top_sources)[present]
        if not len(source):
            continue
        relation = (
            (source >= response_idx).long()
            if relation_type is None
            else torch.full_like(source, relation_type)
        )
        sources.append(source)
        targets.append(target)
        relations.append(relation)
    if not sources:
        empty = torch.empty(0, dtype=torch.long, device=aggregate.device)
        return empty, empty, empty
    return torch.cat(sources), torch.cat(targets), torch.cat(relations)


def _edge_attr_from_query_block(
    attention: torch.Tensor,
    *,
    query_start: int,
    query_end: int,
    source: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    layer_chunk: int,
) -> torch.Tensor:
    """Gather only selected edges while re-reading bounded attention tiles."""

    layers, heads = attention.shape[:2]
    edge_count = len(source)
    if edge_count == 0:
        return torch.empty((0, 8), device=device, dtype=torch.float32)
    local_target = target - query_start
    channel_sum = torch.zeros(edge_count, device=device)
    channel_square_sum = torch.zeros_like(channel_sum)
    channel_max = torch.full_like(channel_sum, -torch.inf)
    layer_sum = torch.zeros_like(channel_sum)
    layer_square_sum = torch.zeros_like(channel_sum)
    head_sum = torch.zeros((heads, edge_count), device=device)
    early_sum = torch.zeros_like(channel_sum)
    late_sum = torch.zeros_like(channel_sum)
    early_count = max(1, layers // 4)
    late_start = layers - early_count

    for layer_start in range(0, layers, layer_chunk):
        layer_end = min(layer_start + layer_chunk, layers)
        tile = attention[layer_start:layer_end, :, query_start:query_end, :].to(
            device=device, non_blocking=device.type == "cuda"
        )
        traces = tile[:, :, local_target, source].float()
        del tile
        channel_sum.add_(traces.sum(dim=(0, 1)))
        channel_square_sum.add_(traces.square().sum(dim=(0, 1)))
        channel_max = torch.maximum(channel_max, traces.amax(dim=(0, 1)))
        per_layer = traces.mean(dim=1)
        layer_sum.add_(per_layer.sum(dim=0))
        layer_square_sum.add_(per_layer.square().sum(dim=0))
        head_sum.add_(traces.sum(dim=0))
        early_overlap = max(0, min(layer_end, early_count) - layer_start)
        if early_overlap:
            early_sum.add_(per_layer[:early_overlap].sum(dim=0))
        late_offset = max(0, late_start - layer_start)
        if late_offset < len(per_layer):
            late_sum.add_(per_layer[late_offset:].sum(dim=0))

    channel_count = float(layers * heads)
    mean = channel_sum / channel_count
    std = (channel_square_sum / channel_count - mean.square()).clamp_min(0).sqrt()
    layer_mean = layer_sum / float(layers)
    layer_std = (layer_square_sum / float(layers) - layer_mean.square()).clamp_min(0).sqrt()
    head_mean = head_sum / float(layers)
    head_std = head_mean.std(dim=0, unbiased=False)
    return torch.stack(
        (
            mean,
            channel_max,
            std,
            early_sum / float(early_count),
            late_sum / float(early_count),
            layer_std,
            head_std,
            (target - source).float() / target.clamp_min(1).float(),
        ),
        dim=1,
    )


def _neighbour_targets(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_attr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted typed-neighbour moments and route statistics via scatter adds."""

    node_count, feature_dim = x.shape
    source, target = edge_index
    group = target * 2 + edge_type
    group_count = node_count * 2
    weight = edge_attr[:, 0].clamp_min(0.0)
    count = torch.zeros(group_count, device=x.device, dtype=x.dtype)
    weight_sum = torch.zeros_like(count)
    weight_square_sum = torch.zeros_like(count)
    span_sum = torch.zeros_like(count)
    count.index_add_(0, group, torch.ones_like(weight))
    weight_sum.index_add_(0, group, weight)
    weight_square_sum.index_add_(0, group, weight.square())
    span = (target - source).to(x.dtype) / target.clamp_min(1).to(x.dtype)
    span_sum.index_add_(0, group, span)
    denominator = weight_sum.clamp_min(torch.finfo(x.dtype).eps)
    summed = torch.zeros(group_count, feature_dim, device=x.device, dtype=x.dtype)
    summed.index_add_(0, group, x[source] * weight.unsqueeze(1))
    mean = summed / denominator.unsqueeze(1)
    second = torch.zeros_like(summed)
    second.index_add_(0, group, x[source].square() * weight.unsqueeze(1))
    variance = (second / denominator.unsqueeze(1) - mean.square()).clamp_min(0.0)
    mean_weight = weight_sum / count.clamp_min(1.0)
    variance_weight = (weight_square_sum / count.clamp_min(1.0) - mean_weight.square()).clamp_min(0.0)
    routes = torch.stack(
        (count, mean_weight, variance_weight, span_sum / count.clamp_min(1.0)), dim=1
    )
    log_variance = variance.clamp_min(torch.finfo(x.dtype).eps).log()
    log_variance = torch.where(count[:, None] > 0, log_variance, 0.0)
    return (
        mean.reshape(node_count, 2, feature_dim),
        log_variance.reshape(node_count, 2, feature_dim),
        routes.reshape(node_count, 2, _ROUTE_DIM),
    )


def build_compact_topk_graph(
    sample: Mapping[str, object],
    *,
    top_k: int = 8,
    device: str | torch.device | None = None,
    top_k_prefix: int | None = None,
    top_k_history: int | None = None,
    query_block: int = 64,
    layer_chunk: int = 2,
    use_hidden: bool = False,
    hidden_projection_dim: int = 64,
) -> dict[str, object]:
    """Build a compact, directed token graph directly on the requested device.

    Only response tokens are queries.  The graph is strictly causal
    (``source < target``), and ``edge_type`` distinguishes prefix-to-response
    (0) from response-history-to-response (1).  No ground-truth label is read.
    """

    _require_label_blind_input(sample)
    attention, response_idx, token_count = _validate_attention_sample(sample)
    requested = attention.device if device is None else torch.device(device)
    if top_k < 1 or query_block < 1 or layer_chunk < 1:
        raise ValueError("top_k, query_block, and layer_chunk must be positive")
    if hidden_projection_dim < 1:
        raise ValueError("hidden_projection_dim must be positive")
    for name, value in (("top_k_prefix", top_k_prefix), ("top_k_history", top_k_history)):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive when provided")

    feature_blocks: list[torch.Tensor] = []
    edge_sources: list[torch.Tensor] = []
    edge_targets: list[torch.Tensor] = []
    edge_types: list[torch.Tensor] = []
    edge_attributes: list[torch.Tensor] = []
    for query_start in range(0, token_count, query_block):
        query_end = min(query_start + query_block, token_count)
        query_ids = torch.arange(query_start, query_end, device=requested)
        aggregate = _aggregate_query_tile(
            attention,
            query_start,
            query_end,
            device=requested,
            layer_chunk=layer_chunk,
        )
        feature_blocks.append(
            _query_features(aggregate, query_ids, response_idx=response_idx)
        )
        answer_start = max(query_start, response_idx)
        if answer_start < query_end:
            row_offset = answer_start - query_start
            answer_queries = query_ids[row_offset:]
            source, target, relation = _select_block_edges(
                aggregate[row_offset:],
                answer_queries,
                response_idx=response_idx,
                top_k=top_k,
                top_k_prefix=top_k_prefix,
                top_k_history=top_k_history,
            )
            if len(source):
                edge_sources.append(source)
                edge_targets.append(target)
                edge_types.append(relation)
                edge_attributes.append(
                    _edge_attr_from_query_block(
                        attention,
                        query_start=query_start,
                        query_end=query_end,
                        source=source,
                        target=target,
                        device=requested,
                        layer_chunk=layer_chunk,
                    )
                )
        del aggregate

    x = torch.cat(feature_blocks, dim=0)
    if not edge_sources:
        raise ValueError("response queries contain no strictly causal attention edges")
    source = torch.cat(edge_sources)
    target = torch.cat(edge_targets)
    edge_type = torch.cat(edge_types)
    edge_attr = torch.cat(edge_attributes)
    edge_index = torch.stack((source, target), dim=0)

    positions = torch.arange(token_count, device=requested, dtype=torch.float32)
    if use_hidden and "hidden" in sample:
        hidden = _to_device(sample["hidden"], device=requested, dtype=torch.float32)
        if hidden.ndim == 3:
            hidden = hidden[-1]
        output_dim = min(hidden.shape[1], hidden_projection_dim)
        indices = torch.linspace(
            0, hidden.shape[1] - 1, output_dim, device=requested
        ).round().long()
        hidden = hidden[:, indices]
        hidden = (hidden - hidden.mean(dim=1, keepdim=True)) / hidden.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        x = torch.cat((x, hidden), dim=1)
    response_mask = torch.arange(token_count, device=requested) >= response_idx
    node_context = torch.stack(
        (
            positions / max(token_count - 1, 1),
            (~response_mask).to(torch.float32),
            response_mask.to(torch.float32),
        ),
        dim=1,
    )

    neighbour_mean, neighbour_log_variance, route_stats = _neighbour_targets(
        x, edge_index, edge_type, edge_attr
    )
    graph: dict[str, object] = {
        "schema_version": "ragtruth_typed_topk_v2",
        "source_id": str(sample["source_id"]),
        "original_idx": _as_scalar(sample["original_idx"], "original_idx"),
        "response_idx": response_idx,
        "x": x,
        "node_context": node_context,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_type": edge_type,
        "response_mask": response_mask,
        "neighbor_mean_target": neighbour_mean,
        "neighbor_log_variance_target": neighbour_log_variance,
        "route_stats_target": route_stats,
        "graph_config": {
            "top_k": int(top_k),
            "top_k_prefix": None if top_k_prefix is None else int(top_k_prefix),
            "top_k_history": None if top_k_history is None else int(top_k_history),
            "query_block": int(query_block),
            "layer_chunk": int(layer_chunk),
            "use_hidden": bool(use_hidden and "hidden" in sample),
            "hidden_projection_dim": int(hidden_projection_dim),
        },
    }
    if "token_ids" in sample:
        graph["token_ids"] = _to_device(sample["token_ids"], device=requested, dtype=torch.long)
    return graph
