"""GPU-friendly, label-blind typed token graphs from RAGTruth caches.

The formal cache is retained response-attention in CSR form; its arrays move
to the requested GPU once and are processed in query blocks without ever
forming an ``N x N`` tensor.  The legacy dense format remains supported via
layer/query tiles.  ``top_k`` is a combined causal budget unless independent
prefix/history budgets are requested explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import torch

FORMAL_SPARSE_CSR_SCHEMA = "ragtruth-all-layers-all-heads-sparse-response-csr-v1"
_REQUIRED = frozenset({"source_id", "original_idx", "response_idx", "attention"})
_OPTIONAL = frozenset({"token_ids", "hidden"})
_FORMAL_REQUIRED = frozenset({
    "attention_cache_schema", "attention_cache_fingerprint", "cache_dtype",
    "source_id", "response_id", "response_idx", "input_policy", "was_truncated",
    "token_ids", "num_attention_layers", "num_attention_heads",
    "attention_diagonal", "response_row_ptr", "response_column_indices",
    "response_values", "attention_floor",
})
_FORBIDDEN_MARKERS = ("label", "target")
_ROUTE_DIM = 4  # edge count, mean weight, weight variance, mean relative span
_POSSIBLE_SOURCE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_CACHE_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}
_FLOAT32_SOFTMAX_SUM_SLACK = 8.0 * float(torch.finfo(torch.float32).eps)


def _as_scalar(value: object, name: str) -> int:
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    return int(tensor.item())


def _cache_dtype(value: object) -> tuple[str, torch.dtype]:
    name = str(value).strip().casefold().removeprefix("torch.")
    if name not in _CACHE_DTYPES:
        raise ValueError("formal cache_dtype must be float16, bfloat16, or float32")
    dtype = _CACHE_DTYPES[name]
    return str(dtype).removeprefix("torch."), dtype


def _strict_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1 or tensor.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean scalar")
    return bool(tensor.item())


def _validate_formal_storage_contract(sample: Mapping[str, object]) -> tuple[str, torch.dtype]:
    """Check raw dtype metadata before any device transfer or index cast."""

    fingerprint = str(sample["attention_cache_fingerprint"]).strip()
    input_policy = str(sample["input_policy"]).strip()
    if not fingerprint or not input_policy:
        raise ValueError("formal fingerprint and input_policy must be non-empty")
    if _strict_bool(sample["was_truncated"], "was_truncated"):
        raise ValueError("formal cache was truncated; full-context attention is required")
    cache_name, cache_dtype = _cache_dtype(sample["cache_dtype"])
    exact_dtypes = {
        "token_ids": torch.int64,
        "response_row_ptr": torch.int64,
        "response_column_indices": torch.int32,
        "attention_diagonal": cache_dtype,
        "response_values": cache_dtype,
    }
    for name, expected in exact_dtypes.items():
        actual = torch.as_tensor(sample[name]).dtype
        if actual != expected:
            raise ValueError(
                f"formal {name} dtype must be {expected}; got {actual} "
                f"for cache_dtype={cache_name}"
            )
    return cache_name, cache_dtype


def _attention_mass_rounding_tolerance(
    row_lengths: torch.Tensor, *, storage_dtype: torch.dtype
) -> torch.Tensor:
    """Conservative v1-cache mass tolerance when observer dtype is unknown."""

    if storage_dtype not in _POSSIBLE_SOURCE_DTYPES:
        raise TypeError("formal sparse attention storage dtype must be fp16, bf16, or fp32")
    relative_bounds = []
    absolute_bounds = []
    for source_dtype in _POSSIBLE_SOURCE_DTYPES:
        source_roundoff = 0.0 if source_dtype == torch.float32 else 0.5 * float(torch.finfo(source_dtype).eps)
        source_subnormal = 0.0 if source_dtype == torch.float32 else 0.5 * float(torch.finfo(source_dtype).tiny) * float(torch.finfo(source_dtype).eps)
        storage_roundoff = 0.0 if source_dtype == storage_dtype else 0.5 * float(torch.finfo(storage_dtype).eps)
        storage_subnormal = 0.0 if source_dtype == storage_dtype else 0.5 * float(torch.finfo(storage_dtype).tiny) * float(torch.finfo(storage_dtype).eps)
        relative_bounds.append((1.0 + _FLOAT32_SOFTMAX_SUM_SLACK) * (1.0 + source_roundoff) * (1.0 + storage_roundoff) - 1.0)
        absolute_bounds.append(source_subnormal * (1.0 + storage_roundoff) + storage_subnormal)
    return (
        torch.as_tensor(row_lengths, device=row_lengths.device, dtype=torch.float32) + 1.0
    ) * max(absolute_bounds) + max(relative_bounds)


def _to_device(value: object, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach()
    return tensor.to(
        device=device,
        dtype=tensor.dtype if dtype is None else dtype,
        non_blocking=device.type == "cuda",
    )


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


def _validate_formal_sparse_sample(sample: Mapping[str, object]) -> tuple[int, int, int]:
    """Validate the persisted CSR contract without expanding it to dense form."""

    missing = sorted(_FORMAL_REQUIRED.difference(sample))
    if missing:
        raise ValueError(f"formal sparse attention cache is missing fields: {missing}")
    if str(sample["attention_cache_schema"]) != FORMAL_SPARSE_CSR_SCHEMA:
        raise ValueError("unsupported formal attention_cache_schema")
    _validate_formal_storage_contract(sample)
    if not str(sample["response_id"]):
        raise ValueError("formal sparse cache response_id must be non-empty")
    diagonal = torch.as_tensor(sample["attention_diagonal"])
    layers = _as_scalar(sample["num_attention_layers"], "num_attention_layers")
    heads = _as_scalar(sample["num_attention_heads"], "num_attention_heads")
    if layers < 1 or heads < 1 or diagonal.ndim != 3 or tuple(diagonal.shape[:2]) != (layers, heads):
        raise ValueError("formal attention_diagonal must have shape [layers, heads, tokens]")
    token_count = int(diagonal.shape[2])
    response_idx = _as_scalar(sample["response_idx"], "response_idx")
    if not 0 < response_idx < token_count:
        raise ValueError("response_idx must split a non-empty prefix and response")
    if torch.as_tensor(sample["token_ids"]).numel() != token_count:
        raise ValueError("token_ids must have one value per attention token")
    row_ptr = torch.as_tensor(sample["response_row_ptr"]).flatten()
    columns = torch.as_tensor(sample["response_column_indices"]).flatten()
    values = torch.as_tensor(sample["response_values"]).flatten()
    rows = layers * heads * (token_count - response_idx)
    if row_ptr.numel() != rows + 1 or int(row_ptr[0]) != 0 or bool((row_ptr[1:] < row_ptr[:-1]).any()) or int(row_ptr[-1]) != values.numel():
        raise ValueError("formal sparse CSR row pointers are invalid")
    if columns.numel() != values.numel():
        raise ValueError("formal sparse CSR values and columns do not align")
    floor = float(sample["attention_floor"])
    if not 0.0 < floor <= 1.0 or not math.isfinite(floor):
        raise ValueError("formal sparse attention_floor must be finite in (0, 1]")
    # Do not expand CSR entries on CPU here: this function is called directly
    # after mmap loading.  Per-entry causal/order/mass checks run on GPU in
    # query blocks inside the builder below.
    return layers, heads, token_count


def load_attention_sample(
    path: str | Path,
    *,
    device: str | torch.device,
    mmap: bool = True,
    include_labels: bool = False,
) -> dict[str, object]:
    """Load one formal sparse-CSR or legacy dense record via a strict whitelist.

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
        raise TypeError(f"attention cache must contain a mapping: {path}")

    if "attention_cache_schema" in loaded:
        if str(loaded["attention_cache_schema"]) != FORMAL_SPARSE_CSR_SCHEMA:
            raise ValueError(
                f"unsupported formal attention_cache_schema: {loaded['attention_cache_schema']!r}"
            )
        missing = sorted(_FORMAL_REQUIRED.difference(loaded))
        if missing:
            raise ValueError(f"formal sparse attention cache is missing fields: {missing}")
        _validate_formal_sparse_sample(loaded)
        cache_dtype_name, _ = _cache_dtype(loaded["cache_dtype"])
        formal: dict[str, object] = {
            "cache_format": "formal_sparse_csr",
            "attention_cache_schema": FORMAL_SPARSE_CSR_SCHEMA,
            "attention_cache_fingerprint": str(loaded["attention_cache_fingerprint"]),
            "cache_dtype": cache_dtype_name,
            "input_policy": str(loaded["input_policy"]),
            "was_truncated": _strict_bool(loaded["was_truncated"], "was_truncated"),
            "source_id": str(loaded["source_id"]),
            "response_id": str(loaded["response_id"]),
            "sample_id": str(loaded["response_id"]),
            "response_idx": _as_scalar(loaded["response_idx"], "response_idx"),
            "num_attention_layers": _as_scalar(loaded["num_attention_layers"], "num_attention_layers"),
            "num_attention_heads": _as_scalar(loaded["num_attention_heads"], "num_attention_heads"),
            "attention_floor": float(loaded["attention_floor"]),
            "token_ids": _to_device(loaded["token_ids"], device=requested_device, dtype=torch.long),
            "attention_diagonal": _to_device(loaded["attention_diagonal"], device=requested_device),
            "response_row_ptr": _to_device(loaded["response_row_ptr"], device=requested_device, dtype=torch.long),
            "response_column_indices": _to_device(loaded["response_column_indices"], device=requested_device),
            "response_values": _to_device(loaded["response_values"], device=requested_device),
        }
        if "split" in loaded:
            dataset_split = str(loaded["split"]).strip().casefold()
            if dataset_split not in {"train", "test"}:
                raise ValueError("formal RAGTruth split must be train or test")
            formal["dataset_split"] = dataset_split
        if include_labels and "y_token" in loaded:
            formal["y_token"] = _to_device(loaded["y_token"], device=requested_device)
        return formal

    unknown_required = _REQUIRED.difference(loaded)
    if unknown_required:
        raise ValueError(f"attention sample is missing required fields: {sorted(unknown_required)}")
    # A whitelist, rather than a blacklist, prevents newly added metadata from
    # accidentally carrying a target into graph training.
    record: dict[str, object] = {
        "cache_format": "legacy_dense",
        "source_id": str(loaded["source_id"]),
        "original_idx": _as_scalar(loaded["original_idx"], "original_idx"),
        "sample_id": str(_as_scalar(loaded["original_idx"], "original_idx")),
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
    early_count = max(1, layers // 2)
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


def _segmented_topk(
    score: torch.Tensor,
    target: torch.Tensor,
    relation: torch.Tensor,
    *,
    top_k: int,
    top_k_prefix: int | None,
    top_k_history: int | None,
) -> torch.Tensor:
    """Top-k retained COO pairs per target/relation without a dense QxN view."""

    separate = top_k_prefix is not None or top_k_history is not None
    group = target * 2 + relation if separate else target
    # ``torch.unique`` supplies source-ascending pair ids; stable score sorting
    # makes ties deterministic before stable grouping.
    by_score = torch.argsort(score, descending=True, stable=True)
    order = by_score[torch.argsort(group[by_score], stable=True)]
    grouped = group[order]
    positions = torch.arange(len(order), device=score.device)
    start = torch.where(
        torch.cat((torch.ones(1, device=score.device, dtype=torch.bool), grouped[1:] != grouped[:-1])),
        positions,
        torch.zeros_like(positions),
    )
    rank = positions - torch.cummax(start, dim=0).values
    if separate:
        prefix_budget = top_k if top_k_prefix is None else top_k_prefix
        history_budget = top_k if top_k_history is None else top_k_history
        budget = torch.where(relation[order] == 0, prefix_budget, history_budget)
        keep = rank < budget
    else:
        keep = rank < top_k
    return order[keep]


def _validate_formal_sparse_block(
    *,
    diagonal: torch.Tensor,
    row_ids: torch.Tensor,
    lengths: torch.Tensor,
    entry_rows: torch.Tensor,
    source: torch.Tensor,
    values: torch.Tensor,
    response_idx: int,
    response_tokens: int,
    values_dtype: torch.dtype,
    attention_floor: float,
) -> None:
    """Validate one CSR query block on GPU without materialising a dense view."""

    if not bool(torch.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
        raise ValueError("formal sparse attention values must be finite in [0, 1]")
    stored_floor = torch.tensor(
        attention_floor, dtype=values_dtype, device=values.device
    ).float()
    if bool((values < stored_floor).any()):
        raise ValueError("formal sparse attention values fall below their quantized floor")
    target = response_idx + entry_rows.remainder(response_tokens)
    if bool((source < 0).any()) or bool((source >= target).any()):
        raise ValueError("formal sparse columns violate causal ordering")
    same_row = entry_rows[1:] == entry_rows[:-1]
    if bool((same_row & (source[1:] <= source[:-1])).any()):
        raise ValueError("formal sparse columns must be strictly sorted per row")
    local_rows = torch.repeat_interleave(torch.arange(len(row_ids), device=values.device), lengths)
    retained = torch.zeros(len(row_ids), dtype=torch.float32, device=values.device)
    retained.index_add_(0, local_rows, values)
    channel = row_ids // response_tokens
    centers = response_idx + row_ids.remainder(response_tokens)
    diagonal_rows = diagonal.reshape(-1, diagonal.shape[-1])[channel, centers]
    tolerance = _attention_mass_rounding_tolerance(
        lengths, storage_dtype=values_dtype
    )
    if bool((retained + diagonal_rows > 1.0 + tolerance).any()):
        raise ValueError("formal sparse retained and diagonal attention mass exceeds one")


def _formal_edge_attr(
    inverse: torch.Tensor,
    selected_pairs: torch.Tensor,
    values: torch.Tensor,
    layer: torch.Tensor,
    *,
    layers: int,
    heads: int,
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Summarise selected retained pairs from sparse raw CSR entries."""

    edge_count = len(selected_pairs)
    if not edge_count:
        return torch.empty((0, 8), dtype=torch.float32, device=values.device)
    selected_id = torch.full((int(inverse.max()) + 1,), -1, dtype=torch.long, device=values.device)
    selected_id[selected_pairs] = torch.arange(edge_count, device=values.device)
    edge = selected_id[inverse]
    present = edge >= 0
    edge, observed, layer = edge[present], values[present].float(), layer[present]
    count = torch.zeros(edge_count, device=values.device)
    summed = torch.zeros_like(count)
    squared = torch.zeros_like(count)
    maximum = torch.full_like(count, -torch.inf)
    count.index_add_(0, edge, torch.ones_like(observed))
    summed.index_add_(0, edge, observed)
    squared.index_add_(0, edge, observed.square())
    maximum.scatter_reduce_(0, edge, observed, reduce="amax", include_self=True)
    early_count = max(1, layers // 2)
    late_start = layers - early_count
    early = layer < early_count
    late = layer >= late_start
    early_sum = torch.zeros_like(count)
    late_sum = torch.zeros_like(count)
    early_sum.index_add_(0, edge[early], observed[early])
    late_sum.index_add_(0, edge[late], observed[late])
    layer_pair = edge * layers + layer
    active_layers = torch.unique(layer_pair).remainder(layers)
    layer_counts = torch.zeros(edge_count, device=values.device)
    layer_counts.index_add_(0, torch.unique(layer_pair) // layers, torch.ones_like(active_layers, dtype=torch.float32))
    channel_count = float(layers * heads)
    mean = summed / channel_count
    return torch.stack(
        (
            mean,
            maximum,
            (squared / channel_count - mean.square()).clamp_min(0.0).sqrt(),
            early_sum / float(early_count * heads),
            late_sum / float(early_count * heads),
            count / channel_count,
            layer_counts / float(layers),
            (target - source).float() / target.clamp_min(1).float(),
        ),
        dim=1,
    )


def _build_formal_sparse_graph(
    sample: Mapping[str, object],
    *,
    top_k: int,
    device: torch.device,
    top_k_prefix: int | None,
    top_k_history: int | None,
    query_block: int,
    use_hidden: bool,
) -> dict[str, object]:
    """Build a typed graph from retained CSR entries, never dense attention."""

    if use_hidden:
        raise ValueError("formal sparse CSR caches do not contain hidden states")
    layers, heads, token_count = _validate_formal_sparse_sample(sample)
    response_idx = _as_scalar(sample["response_idx"], "response_idx")
    diagonal = _to_device(sample["attention_diagonal"], device=device, dtype=torch.float32)
    row_ptr = _to_device(sample["response_row_ptr"], device=device, dtype=torch.long)
    columns = _to_device(sample["response_column_indices"], device=device, dtype=torch.long)
    values = _to_device(sample["response_values"], device=device, dtype=torch.float32)
    values_dtype = torch.as_tensor(sample["response_values"]).dtype
    if not bool(torch.isfinite(diagonal).all()) or bool(((diagonal < 0) | (diagonal > 1)).any()):
        raise ValueError("formal attention_diagonal must be finite in [0, 1]")
    response_tokens = token_count - response_idx
    channels = layers * heads
    x = torch.zeros((token_count, 6), dtype=torch.float32, device=device)
    x[:, 0] = diagonal.mean(dim=(0, 1))
    selected_sources: list[torch.Tensor] = []
    selected_targets: list[torch.Tensor] = []
    selected_relations: list[torch.Tensor] = []
    selected_attributes: list[torch.Tensor] = []
    channel_ids = torch.arange(channels, device=device)

    for start in range(0, response_tokens, query_block):
        end = min(start + query_block, response_tokens)
        local_queries = torch.arange(start, end, device=device)
        row_ids = (channel_ids[:, None] * response_tokens + local_queries).reshape(-1)
        row_starts = row_ptr[row_ids]
        lengths = row_ptr[row_ids + 1] - row_starts
        entry_count = int(lengths.sum())
        if not entry_count:
            continue
        repeated_starts = torch.repeat_interleave(row_starts, lengths)
        repeated_prefix = torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        entry_positions = repeated_starts + torch.arange(entry_count, device=device) - repeated_prefix
        entry_rows = torch.repeat_interleave(row_ids, lengths)
        source = columns[entry_positions]
        observed = values[entry_positions]
        _validate_formal_sparse_block(
            diagonal=diagonal, row_ids=row_ids,
            lengths=lengths, entry_rows=entry_rows, source=source,
            values=observed, response_idx=response_idx,
            response_tokens=response_tokens, values_dtype=values_dtype,
            attention_floor=float(sample["attention_floor"]),
        )
        target = response_idx + entry_rows.remainder(response_tokens)
        channel = entry_rows // response_tokens
        layer = channel // heads
        pair_key = target * token_count + source
        pair_key, inverse = torch.unique(pair_key, sorted=True, return_inverse=True)
        pair_count = len(pair_key)
        lower_sum = torch.zeros(pair_count, device=device)
        lower_sum.index_add_(0, inverse, observed)
        score = lower_sum / float(channels)
        pair_target = pair_key // token_count
        pair_source = pair_key.remainder(token_count)
        relation = (pair_source >= response_idx).long()

        retained_total = torch.zeros(end - start, device=device)
        local_target = pair_target - (response_idx + start)
        retained_total.index_add_(0, local_target, score)
        prefix_total = torch.zeros_like(retained_total)
        history_total = torch.zeros_like(retained_total)
        prefix_total.index_add_(0, local_target[relation == 0], score[relation == 0])
        history_total.index_add_(0, local_target[relation == 1], score[relation == 1])
        probability = score / retained_total[local_target].clamp_min(torch.finfo(score.dtype).eps)
        entropy = torch.zeros_like(retained_total)
        entropy.index_add_(0, local_target, -probability * probability.clamp_min(torch.finfo(score.dtype).eps).log())
        maximum = torch.zeros_like(retained_total)
        maximum.scatter_reduce_(0, local_target, probability, reduce="amax", include_self=True)
        norm = (response_idx + local_queries).clamp_min(1).float().log().clamp_min(1.0)
        target_ids = response_idx + local_queries
        x[target_ids, 1] = prefix_total / retained_total.clamp_min(torch.finfo(score.dtype).eps)
        x[target_ids, 2] = history_total / retained_total.clamp_min(torch.finfo(score.dtype).eps)
        x[target_ids, 3] = entropy / norm
        x[target_ids, 4] = maximum
        x[target_ids, 5] = retained_total.log1p()

        selected = _segmented_topk(score, pair_target, relation, top_k=top_k,
                                   top_k_prefix=top_k_prefix, top_k_history=top_k_history)
        if len(selected):
            edge_source = pair_source[selected]
            edge_target = pair_target[selected]
            selected_sources.append(edge_source)
            selected_targets.append(edge_target)
            selected_relations.append(relation[selected])
            selected_attributes.append(_formal_edge_attr(
                inverse, selected, observed, layer, layers=layers, heads=heads,
                source=edge_source, target=edge_target,
            ))

    if selected_sources:
        source = torch.cat(selected_sources)
        target = torch.cat(selected_targets)
        edge_type = torch.cat(selected_relations)
        edge_attr = torch.cat(selected_attributes)
    else:
        source = target = edge_type = torch.empty(0, dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 8), dtype=torch.float32, device=device)
    edge_index = torch.stack((source, target), dim=0)
    response_mask = torch.arange(token_count, device=device) >= response_idx
    positions = torch.arange(token_count, device=device, dtype=torch.float32)
    node_context = torch.stack((positions / max(token_count - 1, 1),
                                (~response_mask).float(), response_mask.float()), dim=1)
    neighbour_mean, neighbour_log_variance, route_stats = _neighbour_targets(x, edge_index, edge_type, edge_attr)
    return {
        "schema_version": "ragtruth_typed_topk_v3",
        "source_id": str(sample["source_id"]),
        "response_id": str(sample["response_id"]),
        "sample_id": str(sample["sample_id"]),
        "response_idx": response_idx,
        "x": x, "node_context": node_context, "edge_index": edge_index,
        "edge_attr": edge_attr, "edge_type": edge_type, "response_mask": response_mask,
        "neighbor_mean_target": neighbour_mean,
        "neighbor_log_variance_target": neighbour_log_variance,
        "route_stats_target": route_stats,
        "token_ids": _to_device(sample["token_ids"], device=device, dtype=torch.long),
        "graph_config": {
            "cache_format": "formal_sparse_csr", "attention_cache_schema": FORMAL_SPARSE_CSR_SCHEMA,
            "selection_scope": "retained_only", "attention_floor": float(sample["attention_floor"]),
            "layers": layers, "heads": heads, "top_k": int(top_k),
            "top_k_prefix": top_k_prefix, "top_k_history": top_k_history,
            "query_block": int(query_block), "feature_semantics": "retained_csr_lower_bound",
            "raw_identity": {
                "attention_cache_fingerprint": str(sample["attention_cache_fingerprint"]),
                "cache_dtype": str(sample["cache_dtype"]),
                "input_policy": str(sample["input_policy"]),
                "was_truncated": bool(sample["was_truncated"]),
                "response_idx": response_idx, "token_count": token_count,
                "layers": layers, "heads": heads,
                "attention_floor": float(sample["attention_floor"]),
            },
        },
    }


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
    cache_format = str(sample.get("cache_format", "legacy_dense"))
    if cache_format == "formal_sparse_csr":
        requested = torch.device(device) if device is not None else torch.as_tensor(sample["attention_diagonal"]).device
        if top_k < 1 or query_block < 1 or layer_chunk < 1:
            raise ValueError("top_k, query_block, and layer_chunk must be positive")
        for name, value in (("top_k_prefix", top_k_prefix), ("top_k_history", top_k_history)):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when provided")
        return _build_formal_sparse_graph(
            sample, top_k=top_k, device=requested, top_k_prefix=top_k_prefix,
            top_k_history=top_k_history, query_block=query_block, use_hidden=use_hidden,
        )
    if cache_format != "legacy_dense":
        raise ValueError(f"unsupported attention cache format: {cache_format!r}")
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
        "schema_version": "ragtruth_typed_topk_v3",
        "sample_id": str(sample.get("sample_id", _as_scalar(sample["original_idx"], "original_idx"))),
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
            "cache_format": "legacy_dense",
            "selection_scope": "full_attention",
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
