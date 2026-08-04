"""Label-free graph transformations for structure and attribute ablations."""

from __future__ import annotations

from dataclasses import replace

import torch

from .graph import AttentionGraph


def _randperm(
    count: int, *, device: torch.device, generator: torch.Generator | None
) -> torch.Tensor:
    if generator is None:
        return torch.randperm(count, device=device)
    random_device = torch.device(generator.device)
    return torch.randperm(count, device=random_device, generator=generator).to(device)


def relation_preserving_source_shuffle(
    graph: AttentionGraph,
    *,
    generator: torch.Generator | None = None,
) -> AttentionGraph:
    """Rewire sources with legal degree-preserving two-edge swaps.

    A swap exchanges sources across two *different* targets in the same true
    causal RP/RR domain.  It preserves every target/relation in-degree, global
    source degree, edge order, and edge-bound trace payload while changing the
    actual adjacency.  Invalid causal pairs and duplicate token pairs are
    rejected.
    """

    if graph.num_edges == 0:
        return graph
    source = graph.edge_index[0].clone()
    target = graph.edge_index[1]
    causal_relation = (source >= graph.response_idx).long()
    occupied = {
        int(target_value) * graph.num_nodes + int(source_value)
        for source_value, target_value in graph.edge_index.t().cpu().tolist()
    }
    for relation in (0, 1):
        members = torch.nonzero(
            causal_relation == relation, as_tuple=False
        ).flatten()
        count = int(members.numel())
        if count <= 1:
            continue
        order = members[
            _randperm(count, device=source.device, generator=generator)
        ].cpu().tolist()
        used: set[int] = set()
        for position, first in enumerate(order):
            if first in used:
                continue
            # Full search is cheap for fixtures/small graphs.  On formal graphs
            # 128 randomized partners is ample while bounding transform time.
            partner_count = min(count - 1, 128)
            for offset in range(1, partner_count + 1):
                second = order[(position + offset) % count]
                if second == first or second in used:
                    continue
                source_a, source_b = int(source[first]), int(source[second])
                target_a, target_b = int(target[first]), int(target[second])
                if target_a == target_b or source_a == source_b:
                    continue
                if not (source_b < target_a and source_a < target_b):
                    continue
                if relation == 0 and not (
                    source_a < graph.response_idx and source_b < graph.response_idx
                ):
                    continue
                if relation == 1 and not (
                    source_a >= graph.response_idx and source_b >= graph.response_idx
                ):
                    continue
                old_a = target_a * graph.num_nodes + source_a
                old_b = target_b * graph.num_nodes + source_b
                new_a = target_a * graph.num_nodes + source_b
                new_b = target_b * graph.num_nodes + source_a
                if new_a == new_b or new_a in occupied or new_b in occupied:
                    continue
                source[first], source[second] = source_b, source_a
                occupied.remove(old_a)
                occupied.remove(old_b)
                occupied.add(new_a)
                occupied.add(new_b)
                used.add(first)
                used.add(second)
                break
    return replace(graph, edge_index=torch.stack((source, graph.edge_index[1])))


def collapse_relations(graph: AttentionGraph) -> AttentionGraph:
    """Remove RP/RR identity while preserving support and all attributes."""

    return replace(graph, edge_type=torch.zeros_like(graph.edge_type))


def mean_attention_heads(graph: AttentionGraph) -> AttentionGraph:
    """Collapse head identity by averaging heads separately in every layer.

    Node diagonal values are dense and therefore use an exact head mean.  The
    sparse cache censors entries below ``attention_floor``; for edge traces we
    average only retained observations and never reinterpret a missing channel
    as a zero.  Edge support and its selection score are deliberately frozen so
    this ablation changes channel identity rather than rebuilding topology.
    """

    if graph.num_heads == 1:
        return graph
    expected_channels = graph.num_layers * graph.num_heads
    if graph.node_attr.ndim != 2 or graph.node_attr.shape[1] != expected_channels:
        raise ValueError("node_attr does not match graph layer/head dimensions")
    node_attr = graph.node_attr.reshape(
        graph.num_nodes, graph.num_layers, graph.num_heads
    ).mean(dim=2)

    if graph.trace_value.numel():
        layer = graph.trace_channel // graph.num_heads
        key = graph.trace_edge_id * graph.num_layers + layer
        unique_key, inverse = torch.unique(key, sorted=True, return_inverse=True)
        trace_value = torch.zeros(
            unique_key.numel(),
            dtype=graph.trace_value.dtype,
            device=graph.trace_value.device,
        )
        trace_count = torch.zeros_like(trace_value)
        trace_value.index_add_(0, inverse, graph.trace_value)
        trace_count.index_add_(
            0, inverse, torch.ones_like(graph.trace_value, dtype=trace_count.dtype)
        )
        trace_value = trace_value / trace_count.clamp_min(1.0)
        trace_edge_id = unique_key // graph.num_layers
        trace_channel = unique_key.remainder(graph.num_layers)
    else:
        trace_edge_id = graph.trace_edge_id.clone()
        trace_channel = graph.trace_channel.clone()
        trace_value = graph.trace_value.clone()

    return replace(
        graph,
        num_heads=1,
        node_attr=node_attr,
        trace_edge_id=trace_edge_id,
        trace_channel=trace_channel,
        trace_value=trace_value,
    )


__all__ = [
    "collapse_relations",
    "mean_attention_heads",
    "relation_preserving_source_shuffle",
]
