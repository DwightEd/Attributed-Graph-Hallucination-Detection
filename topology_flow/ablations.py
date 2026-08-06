"""Topology-specific null models and ablations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import AttentionStore


def _nontrivial_roll(values: torch.Tensor, seed: int) -> torch.Tensor:
    if values.numel() <= 1:
        return values
    shift = 1 + int(seed % (values.numel() - 1))
    return torch.roll(values, shifts=shift)


def relation_preserving_source_shuffle(
    rows: torch.Tensor,
    *,
    response_idx: int,
    seed: int,
) -> torch.Tensor:
    """Permute source/weight alignment while preserving row marginals.

    For every response query, prompt-source weights and history-source weights
    are permuted independently.  This exactly preserves RP/RR mass, the full
    row weight multiset, support size and concentration.  It destroys which
    concrete source token carries each weight, so changes in recursive ancestry
    isolate source-incidence and path information from pooled attention stats.
    """

    output = torch.as_tensor(rows, dtype=torch.float32).clone()
    for local_query in range(len(output)):
        absolute_query = response_idx + local_query
        row = output[local_query]
        row[:response_idx] = _nontrivial_roll(
            row[:response_idx], seed + 1009 * (local_query + 1)
        )
        if absolute_query > response_idx:
            row[response_idx:absolute_query] = _nontrivial_roll(
                row[response_idx:absolute_query], seed + 9176 * (local_query + 1)
            )
    return output


@dataclass
class RelationPreservingSourceShuffleStore(AttentionStore):
    """Lazy source-incidence null with unchanged pooled row statistics."""

    base: AttentionStore
    seed: int = 42

    def __post_init__(self) -> None:
        self.sample_id = f"{self.base.sample_id}::source-shuffle"
        self.source_id = self.base.source_id
        self.original_idx = self.base.original_idx
        self.layers = self.base.layers
        self.heads = self.base.heads
        self.token_count = self.base.token_count
        self.response_idx = self.base.response_idx

    def response_rows(self, layer: int, head: int) -> torch.Tensor:
        return relation_preserving_source_shuffle(
            self.base.response_rows(layer, head),
            response_idx=self.response_idx,
            seed=self.seed + 104729 * layer + 1009 * head,
        )
