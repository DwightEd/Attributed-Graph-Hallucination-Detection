from __future__ import annotations

import random

import torch

from .data import ColaGraph


def _random_walk_with_restart(
    graph: ColaGraph,
    seed: int,
    *,
    restart_prob: float,
    max_steps: int,
    rng: random.Random,
) -> list[int]:
    """RWR compatibility helper for the multi-graph data adapter.

    The upstream CoLA sampler used DGL 0.4's removed
    ``dgl.contrib.sampling.random_walk_with_restart`` API. This helper keeps
    the same data-side contract (restart walk from the target and collect
    local context nodes) without changing the upstream CoLA model or loss.
    """
    current = seed
    trace = [seed]
    for _ in range(max_steps):
        if rng.random() < restart_prob:
            current = seed
        neighbors = graph.neighbors[current]
        if not neighbors:
            current = seed
            continue
        current = neighbors[rng.randrange(len(neighbors))]
        trace.append(current)
    return trace


def sample_subgraph(
    graph: ColaGraph,
    target: int,
    subgraph_size: int,
    rng: random.Random,
) -> list[int]:
    """Match CoLA's ``subgraph_size-1 context nodes + target`` contract."""
    if subgraph_size < 2:
        raise ValueError("subgraph_size must be at least 2")
    reduced_size = subgraph_size - 1

    trace = _random_walk_with_restart(
        graph,
        target,
        restart_prob=1.0,
        max_steps=subgraph_size * 3,
        rng=rng,
    )
    context = list(dict.fromkeys(trace))
    retry_time = 0
    while len(context) < reduced_size:
        trace = _random_walk_with_restart(
            graph,
            target,
            restart_prob=0.9,
            max_steps=subgraph_size * 5,
            rng=rng,
        )
        context = list(dict.fromkeys(trace))
        retry_time += 1
        if len(context) <= 2 and retry_time > 10:
            context = context * reduced_size
            break

    if not context:
        context = [target]
    while len(context) < reduced_size:
        context.extend(context)
    nodes = context[:reduced_size]
    nodes.append(target)
    return nodes


def build_batch(
    graph: ColaGraph,
    targets: list[int],
    *,
    subgraph_size: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the exact tensor layout consumed by upstream CoLA ``Model``."""
    batch_size = len(targets)
    feature_dim = graph.feature_dim
    adjacency = torch.zeros(
        (batch_size, subgraph_size + 1, subgraph_size + 1),
        dtype=torch.float32,
        device=device,
    )
    features = torch.zeros(
        (batch_size, subgraph_size + 1, feature_dim),
        dtype=torch.float32,
        device=device,
    )

    for batch_index, target in enumerate(targets):
        nodes = sample_subgraph(graph, target, subgraph_size, rng)
        cur_adj = graph.normalized_subgraph_adjacency(nodes)
        cur_feat = graph.features[nodes]

        # Reproduce upstream run.py: keep the sampled adjacency in the top-left
        # block, append an isolated self-loop node, then shift the target
        # feature behind one zero row.
        adjacency[batch_index, :subgraph_size, :subgraph_size] = cur_adj.to(device)
        adjacency[batch_index, subgraph_size, subgraph_size] = 1.0
        features[batch_index, : subgraph_size - 1] = cur_feat[:-1].to(device)
        features[batch_index, subgraph_size] = cur_feat[-1].to(device)

    return features, adjacency
