"""Relation-aware masked autoencoder for sparse attention graphs.

The encoder consumes sparse ``(edge, layer/head, value)`` traces directly.  It
never materialises an ``edges x channels`` tensor: retained trace entries are
encoded independently and reduced onto their pair edge with ``index_add_``.
Cache-censored channels are therefore never confused with observed zeroes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .graph import AttentionGraph, RP, RR


def _causal_edge_type(
    graph: AttentionGraph, edge_ids: torch.Tensor | None = None
) -> torch.Tensor:
    """Immutable RP/RR domain derived from token positions, not model labels."""

    source = graph.edge_index[0] if edge_ids is None else graph.edge_index[0, edge_ids]
    return (source >= graph.response_idx).long()


@dataclass(frozen=True)
class MaskedGraphView:
    """Masks defining one self-supervised view of an attention graph."""

    graph: AttentionGraph
    visible_edge_mask: torch.Tensor
    masked_edge_ids: torch.Tensor
    node_mask: torch.Tensor
    channel_keep_mask: torch.Tensor


@dataclass(frozen=True)
class ReconstructionLosses:
    """Individual label-free reconstruction objectives and their sum."""

    support: torch.Tensor
    weight: torch.Tensor
    distribution: torch.Tensor
    node: torch.Tensor
    total: torch.Tensor


def _randperm(
    count: int, *, device: torch.device, generator: torch.Generator | None
) -> torch.Tensor:
    if generator is None:
        return torch.randperm(count, device=device)
    random_device = torch.device(generator.device)
    return torch.randperm(count, device=random_device, generator=generator).to(device)


def make_masked_view(
    graph: AttentionGraph,
    *,
    edge_mask_rate: float,
    node_mask_rate: float,
    channel_drop_rate: float = 0.0,
    generator: torch.Generator | None = None,
) -> MaskedGraphView:
    """Create a relation-stratified view without deleting a relation group.

    Edges are grouped by ``(target, RP/RR)``.  Singleton groups remain intact;
    every larger group retains at least one edge.  Only response-node features
    are masked, because prompt nodes provide the grounding context.
    """

    for name, rate in (
        ("edge_mask_rate", edge_mask_rate),
        ("node_mask_rate", node_mask_rate),
        ("channel_drop_rate", channel_drop_rate),
    ):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    device = graph.edge_index.device
    visible = torch.ones(graph.num_edges, dtype=torch.bool, device=device)
    if edge_mask_rate > 0.0 and graph.num_edges:
        group = graph.edge_index[1] * 2 + _causal_edge_type(graph)
        random_order = _randperm(
            graph.num_edges, device=device, generator=generator
        )
        order = random_order[torch.argsort(group[random_order], stable=True)]
        ordered_group = group[order]
        _groups, counts = torch.unique_consecutive(
            ordered_group, return_counts=True
        )
        repeated_count = torch.repeat_interleave(counts, counts)
        position = torch.arange(graph.num_edges, device=device)
        starts = torch.where(
            torch.cat(
                (
                    torch.ones(1, dtype=torch.bool, device=device),
                    ordered_group[1:] != ordered_group[:-1],
                )
            ),
            position,
            torch.zeros_like(position),
        )
        rank = position - torch.cummax(starts, dim=0).values
        mask_count = torch.round(
            repeated_count.to(torch.float32) * edge_mask_rate
        ).long()
        mask_count = torch.maximum(mask_count, torch.ones_like(mask_count))
        mask_count = torch.minimum(mask_count, repeated_count - 1)
        chosen = order[rank < mask_count]
        visible[chosen] = False
    masked_edge_ids = torch.nonzero(~visible, as_tuple=False).flatten()

    node_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
    response_nodes = torch.nonzero(graph.response_mask, as_tuple=False).flatten()
    if node_mask_rate > 0.0 and response_nodes.numel():
        count = max(1, int(round(response_nodes.numel() * node_mask_rate)))
        count = min(count, int(response_nodes.numel()))
        chosen = response_nodes[
            _randperm(int(response_nodes.numel()), device=device, generator=generator)[
                :count
            ]
        ]
        node_mask[chosen] = True

    channel_keep = torch.ones(graph.num_channels, dtype=torch.bool, device=device)
    if channel_drop_rate > 0.0 and graph.num_channels:
        drop_count = int(round(graph.num_channels * channel_drop_rate))
        drop_count = min(max(1, drop_count), max(graph.num_channels - 1, 0))
        if drop_count:
            dropped = _randperm(
                graph.num_channels, device=device, generator=generator
            )[:drop_count]
            channel_keep[dropped] = False

    return MaskedGraphView(
        graph=graph,
        visible_edge_mask=visible,
        masked_edge_ids=masked_edge_ids,
        node_mask=node_mask,
        channel_keep_mask=channel_keep,
    )


class _RelationMessageLayer(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.source = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.edge = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.self_update = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return self.norm(hidden + self.dropout(F.gelu(self.self_update(hidden))))
        source, target = edge_index
        message = self.source(hidden[source]) + self.edge(edge_embedding)
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, target, message)
        degree = torch.zeros(hidden.shape[0], dtype=hidden.dtype, device=hidden.device)
        degree.index_add_(0, target, torch.ones_like(target, dtype=hidden.dtype))
        aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(1)
        update = F.gelu(self.self_update(hidden) + aggregate)
        return self.norm(hidden + self.dropout(update))


class RelationAwareMaskGAE(nn.Module):
    """Sparse relation-aware masked graph autoencoder.

    Layer/head identity is learned through an ordered channel embedding shared
    by node and edge encoders.  RP and RR are distinct learned relations in
    both message passing and reconstruction.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        embedding_dim: int = 128,
        message_passing_steps: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(num_layers, num_heads, embedding_dim) < 1 or message_passing_steps < 0:
            raise ValueError(
                "model dimensions must be positive and message_passing_steps cannot be negative"
            )
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_channels = num_layers * num_heads
        self.embedding_dim = embedding_dim

        self.channel_embedding = nn.Embedding(self.num_channels, embedding_dim)
        self.relation_embedding = nn.Embedding(2, embedding_dim)
        self.context_encoder = nn.Linear(3, embedding_dim)
        self.node_value_scale = nn.Parameter(torch.ones(embedding_dim))
        self.node_mask_token = nn.Parameter(torch.zeros(embedding_dim))
        # Per-edge scalar magnitude complements two sparse channel projections
        # below.  Keeping this projection linear lets the encoder aggregate
        # millions of traces without materialising a traces x embedding tensor.
        self.trace_value_encoder = nn.Linear(1, embedding_dim)
        self.message_layers = nn.ModuleList(
            _RelationMessageLayer(embedding_dim, dropout)
            for _ in range(message_passing_steps)
        )

        pair_dim = embedding_dim * 4
        self.support_decoder = nn.Sequential(
            nn.Linear(pair_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, 1)
        )
        self.weight_decoder = nn.Sequential(
            nn.Linear(pair_dim + embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )
        self.distribution_entry_decoder = nn.Sequential(
            nn.Linear(pair_dim + embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )
        self.other_decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
        )
        self.node_decoder = nn.Linear(embedding_dim, self.num_channels)

    def _node_embedding(
        self, graph: AttentionGraph, view: MaskedGraphView
    ) -> torch.Tensor:
        if graph.num_channels != self.num_channels:
            raise ValueError("graph and model layer/head dimensions differ")
        channel_basis = self.channel_embedding.weight
        keep = view.channel_keep_mask.to(graph.node_attr.dtype)
        weighted = graph.node_attr * keep.unsqueeze(0)
        denominator = keep.sum().clamp_min(1.0)
        node = (weighted @ channel_basis) / denominator
        node = node * self.node_value_scale + self.context_encoder(graph.node_context)
        if bool(view.node_mask.any()):
            node = torch.where(view.node_mask.unsqueeze(1), self.node_mask_token, node)
        return node

    def _visible_edge_embedding(
        self, graph: AttentionGraph, view: MaskedGraphView
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visible_ids = torch.nonzero(view.visible_edge_mask, as_tuple=False).flatten()
        if visible_ids.numel() == 0:
            empty = graph.node_attr.new_empty((0, self.embedding_dim))
            return graph.edge_index[:, visible_ids], graph.edge_type[visible_ids], empty

        trace_visible = (
            view.visible_edge_mask[graph.trace_edge_id]
            & view.channel_keep_mask[graph.trace_channel]
        )
        trace_edge = graph.trace_edge_id[trace_visible]
        trace_channel = graph.trace_channel[trace_visible]
        trace_value = graph.trace_value[trace_visible]

        edge_count = torch.bincount(
            trace_edge, minlength=graph.num_edges
        ).to(dtype=graph.node_attr.dtype)
        edge_mass = torch.bincount(
            trace_edge,
            weights=trace_value.to(graph.node_attr.dtype),
            minlength=graph.num_edges,
        )
        if trace_value.numel():
            indices = torch.stack((trace_edge, trace_channel))
            shape = (graph.num_edges, graph.num_channels)
            weighted = torch.sparse_coo_tensor(
                indices,
                trace_value.to(graph.node_attr.dtype),
                shape,
                device=graph.node_attr.device,
                check_invariants=False,
                is_coalesced=True,
            )
            presence = torch.sparse_coo_tensor(
                indices,
                torch.ones_like(trace_value, dtype=graph.node_attr.dtype),
                shape,
                device=graph.node_attr.device,
                check_invariants=False,
                is_coalesced=True,
            )
            denominator = edge_count.clamp_min(1.0).unsqueeze(1)
            channel_signal = (
                torch.sparse.mm(weighted, self.channel_embedding.weight)
                + torch.sparse.mm(presence, self.channel_embedding.weight)
            ) / denominator
            magnitude = self.trace_value_encoder(
                (edge_mass / edge_count.clamp_min(1.0)).unsqueeze(1)
            )
            edge_all = channel_signal + magnitude
        else:
            edge_all = graph.node_attr.new_zeros(
                (graph.num_edges, self.embedding_dim)
            )
        edge = edge_all[visible_ids]
        relation = graph.edge_type[visible_ids]
        edge = edge + self.relation_embedding(relation)
        return graph.edge_index[:, visible_ids], relation, edge

    def encode(
        self, graph: AttentionGraph, view: MaskedGraphView
    ) -> torch.Tensor:
        hidden = self._node_embedding(graph, view)
        if not self.message_layers:
            return hidden
        edge_index, _edge_type, edge_embedding = self._visible_edge_embedding(graph, view)
        for layer in self.message_layers:
            hidden = layer(hidden, edge_index, edge_embedding)
        return hidden

    def graph_embedding(
        self, hidden: torch.Tensor, graph: AttentionGraph
    ) -> torch.Tensor:
        response = hidden[graph.response_mask]
        if response.shape[0] == 0:
            raise ValueError("graph has no response nodes")
        return response.mean(dim=0)

    def forward(
        self, graph: AttentionGraph, view: MaskedGraphView
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(graph, view)
        return hidden, self.graph_embedding(hidden, graph)

    def _pair_features(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        source_h, target_h = hidden[source], hidden[target]
        return torch.cat(
            (
                source_h,
                target_h,
                source_h * target_h,
                self.relation_embedding(edge_type),
            ),
            dim=1,
        )

    def decode_support(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        return self.support_decoder(
            self._pair_features(hidden, edge_index, edge_type)
        ).squeeze(1)

    def decode_weight(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        channel: torch.Tensor,
    ) -> torch.Tensor:
        pair = self._pair_features(hidden, edge_index, edge_type)
        features = torch.cat((pair, self.channel_embedding(channel)), dim=1)
        return self.weight_decoder(features).squeeze(1)

    def decode_distribution_entry(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        channel: torch.Tensor,
    ) -> torch.Tensor:
        pair = self._pair_features(hidden, edge_index, edge_type)
        features = torch.cat((pair, self.channel_embedding(channel)), dim=1)
        return self.distribution_entry_decoder(features).squeeze(1)


def sample_support_negatives(
    graph: AttentionGraph,
    positive_edge_ids: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample absent causal pairs with the positive's target and relation."""

    device = graph.edge_index.device
    if positive_edge_ids.numel() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    targets = graph.edge_index[1, positive_edge_ids]
    relations = graph.edge_type[positive_edge_ids]
    causal_relations = (
        graph.edge_index[0, positive_edge_ids] >= graph.response_idx
    ).long()
    domain_start = torch.where(
        causal_relations == RP,
        torch.zeros_like(targets),
        torch.full_like(targets, graph.response_idx),
    )
    domain_size = torch.where(
        causal_relations == RP,
        torch.full_like(targets, graph.response_idx),
        targets - graph.response_idx,
    )
    causal_edge_relation = (
        graph.edge_index[0] >= graph.response_idx
    ).long()
    all_group = graph.edge_index[1] * 2 + causal_edge_relation
    _unique, inverse, counts = torch.unique(
        all_group, sorted=True, return_inverse=True, return_counts=True
    )
    occupied_count = counts[inverse[positive_edge_ids]]
    free_count = domain_size - occupied_count
    if bool((free_count <= 0).any()):
        raise ValueError(
            "cannot sample an absent edge for a saturated (target, relation) group"
        )
    random_device = torch.device(generator.device) if generator is not None else device
    random = torch.rand(
        positive_edge_ids.numel(),
        generator=generator,
        device=random_device,
    ).to(device)
    free_rank = torch.floor(random * free_count.to(random.dtype)).long()

    # Map a uniformly sampled rank among missing sources to its source value
    # with a batched binary rank-select over each legal causal domain.
    pair_key = torch.sort(
        graph.edge_index[1] * graph.num_nodes + graph.edge_index[0]
    ).values
    group_start_key = targets * graph.num_nodes + domain_start
    group_begin = torch.searchsorted(pair_key, group_start_key, right=False)
    lower = torch.zeros_like(domain_size)
    upper = domain_size - 1
    max_iterations = max(int(domain_size.max()).bit_length(), 1)
    for _ in range(max_iterations):
        offset = (lower + upper) // 2
        candidate = domain_start + offset
        right = torch.searchsorted(
            pair_key, targets * graph.num_nodes + candidate, right=True
        )
        occupied_through = right - group_begin
        missing_through = offset + 1 - occupied_through
        enough = missing_through >= free_rank + 1
        upper = torch.where(enough, offset, upper)
        lower = torch.where(enough, lower, offset + 1)
    candidate = domain_start + lower
    candidate_key = targets * graph.num_nodes + candidate
    location = torch.searchsorted(pair_key, candidate_key)
    occupied = (location < pair_key.numel()) & (
        pair_key[location.clamp_max(max(pair_key.numel() - 1, 0))] == candidate_key
    )
    if bool(occupied.any()) or bool((candidate >= domain_start + domain_size).any()):
        raise RuntimeError("batched negative-source rank mapping failed")
    return torch.stack((candidate, targets)), relations


def _support_sampleable_edges(
    graph: AttentionGraph, edge_ids: torch.Tensor
) -> torch.Tensor:
    """Drop positives whose legal causal relation domain is fully occupied."""

    if edge_ids.numel() == 0:
        return edge_ids
    causal_relation = (graph.edge_index[0] >= graph.response_idx).long()
    group = graph.edge_index[1] * 2 + causal_relation
    _unique, inverse, counts = torch.unique(
        group, sorted=True, return_inverse=True, return_counts=True
    )
    target = graph.edge_index[1, edge_ids]
    relation = causal_relation[edge_ids]
    domain_size = torch.where(
        relation == RP,
        torch.full_like(target, graph.response_idx),
        target - graph.response_idx,
    )
    return edge_ids[counts[inverse[edge_ids]] < domain_size]


def _sample_cap(
    indices: torch.Tensor,
    maximum: int | None,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if maximum is None:
        return indices
    if maximum < 1:
        raise ValueError("sampling caps must be positive when provided")
    if indices.numel() <= maximum:
        return indices
    order = _randperm(
        int(indices.numel()), device=indices.device, generator=generator
    )
    return indices[order[:maximum]]


def _sample_cap_by_stratum(
    indices: torch.Tensor,
    strata: torch.Tensor,
    maximum: int | None,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample near-equally per target/relation stratum when capacity allows."""

    if strata.shape != indices.shape:
        raise ValueError("sampling strata must align with indices")
    if maximum is None or indices.numel() <= maximum:
        return indices
    if maximum < 1:
        raise ValueError("sampling caps must be positive when provided")
    unique_strata = torch.unique(strata)
    if unique_strata.numel() > maximum:
        return _sample_cap(indices, maximum, generator=generator)
    quota = max(1, maximum // int(unique_strata.numel()))
    random_order = _randperm(
        int(indices.numel()), device=indices.device, generator=generator
    )
    order = random_order[
        torch.argsort(strata[random_order], stable=True)
    ]
    ordered_strata = strata[order]
    _groups, counts = torch.unique_consecutive(
        ordered_strata, return_counts=True
    )
    position = torch.arange(indices.numel(), device=indices.device)
    starts = torch.where(
        torch.cat(
            (
                torch.ones(1, dtype=torch.bool, device=indices.device),
                ordered_strata[1:] != ordered_strata[:-1],
            )
        ),
        position,
        torch.zeros_like(position),
    )
    rank = position - torch.cummax(starts, dim=0).values
    repeated_limit = torch.repeat_interleave(
        torch.minimum(counts, torch.full_like(counts, quota)), counts
    )
    return indices[order[rank < repeated_limit]]


def _sample_group_cap_by_target(
    active_ids: torch.Tensor,
    group_keys: torch.Tensor,
    *,
    num_channels: int,
    maximum: int | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Cap row groups while retaining target-token coverage when possible."""

    if maximum is None or active_ids.numel() <= maximum:
        return active_ids
    if maximum < 1:
        raise ValueError("sampling caps must be positive when provided")
    targets = group_keys[active_ids] // num_channels
    unique_targets = torch.unique(targets)
    if unique_targets.numel() > maximum:
        return _sample_cap(active_ids, maximum, generator=generator)
    quota = max(1, maximum // int(unique_targets.numel()))
    random_order = _randperm(
        int(active_ids.numel()), device=active_ids.device, generator=generator
    )
    order = random_order[
        torch.argsort(targets[random_order], stable=True)
    ]
    ordered_targets = targets[order]
    _target, counts = torch.unique_consecutive(
        ordered_targets, return_counts=True
    )
    position = torch.arange(active_ids.numel(), device=active_ids.device)
    starts = torch.where(
        torch.cat(
            (
                torch.ones(1, dtype=torch.bool, device=active_ids.device),
                ordered_targets[1:] != ordered_targets[:-1],
            )
        ),
        position,
        torch.zeros_like(position),
    )
    rank = position - torch.cummax(starts, dim=0).values
    repeated_limit = torch.repeat_interleave(
        torch.minimum(counts, torch.full_like(counts, quota)), counts
    )
    selected = active_ids[order[rank < repeated_limit]]
    if selected.numel() > maximum:
        selected = selected[:maximum]
    return selected


def _chunked_decode(
    decoder: object,
    hidden: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    *,
    chunk_size: int,
    channel: torch.Tensor | None = None,
) -> torch.Tensor:
    if chunk_size < 1:
        raise ValueError("decoder_chunk_size must be positive")
    count = int(edge_type.numel())
    if count == 0:
        return hidden.new_empty(0)
    output: list[torch.Tensor] = []
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        if channel is None:
            output.append(decoder(hidden, edge_index[:, start:end], edge_type[start:end]))
        else:
            output.append(
                decoder(
                    hidden,
                    edge_index[:, start:end],
                    edge_type[start:end],
                    channel[start:end],
                )
            )
    return torch.cat(output)


def _distribution_group_energy(
    model: RelationAwareMaskGAE,
    hidden: torch.Tensor,
    graph: AttentionGraph,
    view: MaskedGraphView,
    *,
    generator: torch.Generator | None = None,
    max_groups: int | None = None,
    decoder_chunk_size: int = 16_384,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CE and response target for each active (target, channel) row."""

    trace_masked = ~view.visible_edge_mask[graph.trace_edge_id]
    if not bool(trace_masked.any()):
        return (
            hidden.new_empty(0),
            torch.empty(0, dtype=torch.long, device=hidden.device),
        )

    trace_target = graph.edge_index[1, graph.trace_edge_id]
    group_key = trace_target * graph.num_channels + graph.trace_channel
    all_keys, inverse = torch.unique(group_key, sorted=True, return_inverse=True)
    masked_count = torch.zeros(all_keys.numel(), dtype=torch.long, device=hidden.device)
    masked_count.index_add_(0, inverse[trace_masked], torch.ones_like(inverse[trace_masked]))
    active_group = masked_count > 0
    active_trace = active_group[inverse]
    selected_inverse = inverse[active_trace]
    active_ids = torch.nonzero(active_group, as_tuple=False).flatten()
    active_ids = _sample_group_cap_by_target(
        active_ids,
        all_keys,
        num_channels=graph.num_channels,
        maximum=max_groups,
        generator=generator,
    )
    selected_group = torch.zeros_like(active_group)
    selected_group[active_ids] = True
    active_trace = selected_group[inverse]
    selected_inverse = inverse[active_trace]
    remap = torch.full_like(active_group, -1, dtype=torch.long)
    remap[active_ids] = torch.arange(active_ids.numel(), device=hidden.device)
    local_group = remap[selected_inverse]

    edge_ids = graph.trace_edge_id[active_trace]
    edge_index = graph.edge_index[:, edge_ids]
    edge_type = graph.edge_type[edge_ids]
    channels = graph.trace_channel[active_trace]
    logits = _chunked_decode(
        model.decode_distribution_entry,
        hidden,
        edge_index,
        edge_type,
        channel=channels,
        chunk_size=decoder_chunk_size,
    )
    target_weight = graph.trace_value[active_trace]

    groups = active_ids.numel()
    retained_mass = torch.zeros(groups, dtype=target_weight.dtype, device=hidden.device)
    retained_mass.index_add_(0, local_group, target_weight)
    active_key = all_keys[active_ids]
    active_target = active_key // graph.num_channels
    active_channel = active_key.remainder(graph.num_channels)
    # The diagonal is self-attention, so only ``1 - a_ii`` is available to
    # causal history.  OTHER absorbs selected-out pairs and cache-censored
    # values; it must not accidentally absorb the diagonal itself.
    history_mass = (
        1.0 - graph.node_attr[active_target, active_channel]
    ).clamp_min(torch.finfo(target_weight.dtype).eps)
    other_mass = (history_mass - retained_mass).clamp_min(
        torch.finfo(target_weight.dtype).eps
    )
    normalizer = retained_mass + other_mass
    entry_probability = target_weight / normalizer[local_group]
    other_probability = other_mass / normalizer

    other_feature = torch.cat(
        (hidden[active_target], model.channel_embedding(active_channel)), dim=1
    )
    other_logits = model.other_decoder(other_feature).squeeze(1)

    maximum = torch.full(
        (groups,), -torch.inf, dtype=logits.dtype, device=hidden.device
    )
    maximum.scatter_reduce_(0, local_group, logits, reduce="amax", include_self=True)
    maximum = torch.maximum(maximum, other_logits)
    exponential_sum = torch.exp(other_logits - maximum).index_add(
        0, local_group, torch.exp(logits - maximum[local_group])
    )
    log_partition = maximum + exponential_sum.log()
    entry_log_probability = logits - log_partition[local_group]
    other_log_probability = other_logits - log_partition
    per_group = torch.zeros(groups, dtype=logits.dtype, device=hidden.device)
    per_group.index_add_(
        0, local_group, -entry_probability * entry_log_probability
    )
    per_group = per_group - other_probability * other_log_probability
    return per_group, active_target


def _distribution_cross_entropy(
    model: RelationAwareMaskGAE,
    hidden: torch.Tensor,
    graph: AttentionGraph,
    view: MaskedGraphView,
    *,
    generator: torch.Generator | None = None,
    max_groups: int | None = None,
    decoder_chunk_size: int = 16_384,
) -> torch.Tensor:
    """Reconstruct retained row mass plus an explicit censored OTHER bucket."""

    energy, _target = _distribution_group_energy(
        model,
        hidden,
        graph,
        view,
        generator=generator,
        max_groups=max_groups,
        decoder_chunk_size=decoder_chunk_size,
    )
    return energy.mean() if energy.numel() else hidden.sum() * 0.0


def _mean_by_target(
    values: torch.Tensor, target: torch.Tensor, *, node_count: int
) -> torch.Tensor:
    output = values.new_zeros(node_count)
    count = values.new_zeros(node_count)
    if values.numel():
        output.index_add_(0, target, values)
        count.index_add_(0, target, torch.ones_like(values))
    return output / count.clamp_min(1.0)


def reconstruction_energy_by_node(
    model: RelationAwareMaskGAE,
    graph: AttentionGraph,
    view: MaskedGraphView,
    *,
    generator: torch.Generator | None = None,
    support_weight: float = 1.0,
    attention_weight: float = 1.0,
    distribution_weight: float = 1.0,
    node_weight: float = 1.0,
    max_support_edges: int | None = 8_192,
    max_weight_traces: int | None = 65_536,
    max_distribution_groups: int | None = 512,
    decoder_chunk_size: int = 16_384,
    hidden: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return label-free reconstruction energy assigned to response targets."""

    hidden = model.encode(graph, view) if hidden is None else hidden
    if hidden.shape != (graph.num_nodes, model.embedding_dim):
        raise ValueError("provided hidden states do not align with the graph/model")
    zero = hidden.new_zeros(graph.num_nodes)
    support = zero.clone()
    support_rp = zero.clone()
    support_rr = zero.clone()
    support_masked = _support_sampleable_edges(graph, view.masked_edge_ids)
    if support_masked.numel():
        support_causal_type = (
            graph.edge_index[0, support_masked] >= graph.response_idx
        ).long()
        support_stratum = (
            graph.edge_index[1, support_masked] * 2 + support_causal_type
        )
        support_masked = _sample_cap_by_stratum(
            support_masked,
            support_stratum,
            max_support_edges,
            generator=generator,
        )
    if support_masked.numel():
        positive_index = graph.edge_index[:, support_masked]
        positive_type = graph.edge_type[support_masked]
        negative_index, negative_type = sample_support_negatives(
            graph, support_masked, generator=generator
        )
        positive = F.binary_cross_entropy_with_logits(
            _chunked_decode(
                model.decode_support,
                hidden,
                positive_index,
                positive_type,
                chunk_size=decoder_chunk_size,
            ),
            torch.ones(support_masked.numel(), device=hidden.device),
            reduction="none",
        )
        negative = F.binary_cross_entropy_with_logits(
            _chunked_decode(
                model.decode_support,
                hidden,
                negative_index,
                negative_type,
                chunk_size=decoder_chunk_size,
            ),
            torch.zeros(support_masked.numel(), device=hidden.device),
            reduction="none",
        )
        support_entry = 0.5 * (positive + negative)
        support = _mean_by_target(
            support_entry,
            positive_index[1],
            node_count=graph.num_nodes,
        )
        causal_rp_entry = _causal_edge_type(graph, support_masked) == RP
        support_rp = _mean_by_target(
            support_entry[causal_rp_entry],
            positive_index[1, causal_rp_entry],
            node_count=graph.num_nodes,
        )
        support_rr = _mean_by_target(
            support_entry[~causal_rp_entry],
            positive_index[1, ~causal_rp_entry],
            node_count=graph.num_nodes,
        )

    weight = zero.clone()
    weight_rp = zero.clone()
    weight_rr = zero.clone()
    trace_masked = ~view.visible_edge_mask[graph.trace_edge_id]
    trace_ids = torch.nonzero(trace_masked, as_tuple=False).flatten()
    if trace_ids.numel():
        candidate_edge = graph.trace_edge_id[trace_ids]
        candidate_causal_type = (
            graph.edge_index[0, candidate_edge] >= graph.response_idx
        ).long()
        trace_stratum = (
            graph.edge_index[1, candidate_edge] * 2 + candidate_causal_type
        )
        trace_ids = _sample_cap_by_stratum(
            trace_ids,
            trace_stratum,
            max_weight_traces,
            generator=generator,
        )
    if trace_ids.numel():
        trace_edge = graph.trace_edge_id[trace_ids]
        predicted = torch.sigmoid(
            _chunked_decode(
                model.decode_weight,
                hidden,
                graph.edge_index[:, trace_edge],
                graph.edge_type[trace_edge],
                channel=graph.trace_channel[trace_ids],
                chunk_size=decoder_chunk_size,
            )
        )
        trace_energy = F.smooth_l1_loss(
            predicted, graph.trace_value[trace_ids], reduction="none"
        )
        weight = _mean_by_target(
            trace_energy,
            graph.edge_index[1, trace_edge],
            node_count=graph.num_nodes,
        )
        trace_rp = _causal_edge_type(graph, trace_edge) == RP
        weight_rp = _mean_by_target(
            trace_energy[trace_rp],
            graph.edge_index[1, trace_edge[trace_rp]],
            node_count=graph.num_nodes,
        )
        weight_rr = _mean_by_target(
            trace_energy[~trace_rp],
            graph.edge_index[1, trace_edge[~trace_rp]],
            node_count=graph.num_nodes,
        )

    group_energy, group_target = _distribution_group_energy(
        model,
        hidden,
        graph,
        view,
        generator=generator,
        max_groups=max_distribution_groups,
        decoder_chunk_size=decoder_chunk_size,
    )
    distribution = _mean_by_target(
        group_energy, group_target, node_count=graph.num_nodes
    )

    node = zero.clone()
    if bool(view.node_mask.any()):
        predicted_node = model.node_decoder(hidden[view.node_mask])
        target_node = graph.node_attr[view.node_mask]
        cosine = F.cosine_similarity(predicted_node, target_node, dim=1, eps=1e-8)
        node[view.node_mask] = (1.0 - cosine).clamp_min(0.0) ** 2
    total = (
        support_weight * support
        + attention_weight * weight
        + distribution_weight * distribution
        + node_weight * node
    )
    return {
        "support": support,
        "support_rp": support_rp,
        "support_rr": support_rr,
        "weight": weight,
        "weight_rp": weight_rp,
        "weight_rr": weight_rr,
        "distribution": distribution,
        "node": node,
        "total": total,
    }


def reconstruction_losses(
    model: RelationAwareMaskGAE,
    graph: AttentionGraph,
    view: MaskedGraphView,
    *,
    generator: torch.Generator | None = None,
    support_weight: float = 1.0,
    attention_weight: float = 1.0,
    distribution_weight: float = 1.0,
    node_weight: float = 1.0,
    max_support_edges: int | None = 8_192,
    max_weight_traces: int | None = 65_536,
    max_distribution_groups: int | None = 512,
    decoder_chunk_size: int = 16_384,
    hidden: torch.Tensor | None = None,
) -> ReconstructionLosses:
    """Compute label-free typed support, trace, distribution, and node losses."""

    hidden = model.encode(graph, view) if hidden is None else hidden
    if hidden.shape != (graph.num_nodes, model.embedding_dim):
        raise ValueError("provided hidden states do not align with the graph/model")
    masked = view.masked_edge_ids
    zero = hidden.sum() * 0.0

    support_masked = _support_sampleable_edges(graph, masked)
    if support_masked.numel():
        support_causal_type = (
            graph.edge_index[0, support_masked] >= graph.response_idx
        ).long()
        support_stratum = (
            graph.edge_index[1, support_masked] * 2 + support_causal_type
        )
        support_masked = _sample_cap_by_stratum(
            support_masked,
            support_stratum,
            max_support_edges,
            generator=generator,
        )
    if support_masked.numel():
        positive_index = graph.edge_index[:, support_masked]
        positive_type = graph.edge_type[support_masked]
        negative_index, negative_type = sample_support_negatives(
            graph, support_masked, generator=generator
        )
        positive_logits = _chunked_decode(
            model.decode_support,
            hidden,
            positive_index,
            positive_type,
            chunk_size=decoder_chunk_size,
        )
        negative_logits = _chunked_decode(
            model.decode_support,
            hidden,
            negative_index,
            negative_type,
            chunk_size=decoder_chunk_size,
        )
        support = 0.5 * (
            F.binary_cross_entropy_with_logits(
                positive_logits, torch.ones_like(positive_logits)
            )
            + F.binary_cross_entropy_with_logits(
                negative_logits, torch.zeros_like(negative_logits)
            )
        )
    else:
        support = zero

    trace_masked = ~view.visible_edge_mask[graph.trace_edge_id]
    trace_ids = torch.nonzero(trace_masked, as_tuple=False).flatten()
    if trace_ids.numel():
        candidate_edge = graph.trace_edge_id[trace_ids]
        candidate_causal_type = (
            graph.edge_index[0, candidate_edge] >= graph.response_idx
        ).long()
        trace_stratum = (
            graph.edge_index[1, candidate_edge] * 2 + candidate_causal_type
        )
        trace_ids = _sample_cap_by_stratum(
            trace_ids,
            trace_stratum,
            max_weight_traces,
            generator=generator,
        )
    if trace_ids.numel():
        trace_edge = graph.trace_edge_id[trace_ids]
        predicted = torch.sigmoid(
            _chunked_decode(
                model.decode_weight,
                hidden,
                graph.edge_index[:, trace_edge],
                graph.edge_type[trace_edge],
                channel=graph.trace_channel[trace_ids],
                chunk_size=decoder_chunk_size,
            )
        )
        weight = F.smooth_l1_loss(predicted, graph.trace_value[trace_ids])
    else:
        weight = zero

    distribution = _distribution_cross_entropy(
        model,
        hidden,
        graph,
        view,
        generator=generator,
        max_groups=max_distribution_groups,
        decoder_chunk_size=decoder_chunk_size,
    )

    if bool(view.node_mask.any()):
        predicted_node = model.node_decoder(hidden[view.node_mask])
        target_node = graph.node_attr[view.node_mask]
        cosine = F.cosine_similarity(predicted_node, target_node, dim=1, eps=1e-8)
        node = ((1.0 - cosine).clamp_min(0.0) ** 2).mean()
    else:
        node = zero

    total = (
        support_weight * support
        + attention_weight * weight
        + distribution_weight * distribution
        + node_weight * node
    )
    return ReconstructionLosses(
        support=support,
        weight=weight,
        distribution=distribution,
        node=node,
        total=total,
    )


__all__ = [
    "MaskedGraphView",
    "ReconstructionLosses",
    "RelationAwareMaskGAE",
    "make_masked_view",
    "reconstruction_energy_by_node",
    "reconstruction_losses",
    "sample_support_negatives",
]
