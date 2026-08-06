"""Controlled, label-free counterfactual topology erosions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import CorruptionConfig
from .contracts import AttentionStore


def _normalize_rows(rows: torch.Tensor, response_idx: int) -> torch.Tensor:
    values = torch.as_tensor(rows, dtype=torch.float32).clone()
    response_tokens, token_count = values.shape
    query = torch.arange(response_idx, response_idx + response_tokens)
    key = torch.arange(token_count)
    values.masked_fill_(key.unsqueeze(0) >= query.unsqueeze(1), 0.0)
    return values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _incidence_erosion(
    row: torch.Tensor,
    *,
    absolute_query: int,
    response_idx: int,
    config: CorruptionConfig,
) -> torch.Tensor:
    if absolute_query <= response_idx:
        return row
    output = row.clone()
    prompt = output[:response_idx]
    prompt_mass = prompt.sum()
    moved = config.prompt_transfer * prompt_mass
    if moved > 0:
        prompt.mul_(1.0 - config.prompt_transfer)
        local_start = max(response_idx, absolute_query - config.local_window)
        local_sources = torch.arange(local_start, absolute_query)
        existing = output[local_sources]
        allocation = existing + existing.mean().clamp_min(1e-6)
        allocation = allocation / allocation.sum()
        output[local_sources] += moved * allocation

    history = output[response_idx:absolute_query].clone()
    if history.numel() > 1 and float(history.sum()) > 0.0:
        positive = history[history > 0]
        sorted_weights = torch.sort(positive, descending=True).values
        history.zero_()
        nearest_count = min(len(sorted_weights), config.local_window, len(history))
        if nearest_count:
            history[-nearest_count:] = sorted_weights[:nearest_count]
            if len(sorted_weights) > nearest_count:
                history[-1] += sorted_weights[nearest_count:].sum()
        output[response_idx:absolute_query] = history
    return output / output.sum().clamp_min(1e-12)


def _support_collapse(
    row: torch.Tensor,
    *,
    absolute_query: int,
    config: CorruptionConfig,
) -> torch.Tensor:
    values = row[:absolute_query].clone()
    active = torch.nonzero(values > 0, as_tuple=False).flatten()
    if len(active) <= 1:
        return row
    keep = max(1, int(round(len(active) * config.support_keep_fraction)))
    order = active[torch.argsort(values[active], descending=True, stable=True)]
    retained = order[:keep]
    collapsed = torch.zeros_like(values)
    collapsed[retained] = values[retained].pow(config.concentration_power)
    collapsed /= collapsed.sum().clamp_min(1e-12)
    output = row.clone()
    output[:absolute_query] = collapsed
    return output


def corrupt_rows(
    rows: torch.Tensor,
    *,
    response_idx: int,
    config: CorruptionConfig,
) -> torch.Tensor:
    """Apply a deterministic mechanism-aligned pseudo-hallucination."""

    if config.mode == "all":
        raise ValueError("mode=all must be expanded into concrete counterfactuals")
    normalized = _normalize_rows(rows, response_idx)
    output = normalized.clone()
    for local_query in range(len(output)):
        absolute_query = response_idx + local_query
        row = output[local_query]
        if config.mode in {"incidence", "composite"}:
            row = _incidence_erosion(
                row,
                absolute_query=absolute_query,
                response_idx=response_idx,
                config=config,
            )
        if config.mode in {"collapse", "composite"}:
            row = _support_collapse(
                row, absolute_query=absolute_query, config=config
            )
        output[local_query] = row
    return output


@dataclass
class CorruptedAttentionStore(AttentionStore):
    """Lazy layer/head counterfactual view over an existing attention store."""

    base: AttentionStore
    corruption: CorruptionConfig

    def __post_init__(self) -> None:
        self.sample_id = f"{self.base.sample_id}::corrupted::{self.corruption.mode}"
        self.source_id = self.base.source_id
        self.original_idx = self.base.original_idx
        self.layers = self.base.layers
        self.heads = self.base.heads
        self.token_count = self.base.token_count
        self.response_idx = self.base.response_idx

    def response_rows(self, layer: int, head: int) -> torch.Tensor:
        return corrupt_rows(
            self.base.response_rows(layer, head),
            response_idx=self.response_idx,
            config=self.corruption,
        )
