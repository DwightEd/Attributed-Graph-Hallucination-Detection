from __future__ import annotations

import torch

from .config import CorruptionConfig


def _coalesce(source: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if source.numel() == 0:
        return source.long(), weight.float()
    unique, inverse = torch.unique(source.long(), sorted=True, return_inverse=True)
    merged = torch.zeros(len(unique), dtype=torch.float32)
    merged.index_add_(0, inverse, weight.float())
    keep = merged > 0
    return unique[keep], merged[keep]


def _normalise(weight: torch.Tensor) -> torch.Tensor:
    return weight / weight.sum().clamp_min(1e-12)


def incidence_erosion(
    source: torch.Tensor,
    weight: torch.Tensor,
    *,
    target: int,
    response_idx: int,
    config: CorruptionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target <= response_idx or source.numel() == 0:
        return source, weight

    weight = weight.clone()
    prompt = source < response_idx
    moved = config.prompt_transfer * weight[prompt].sum()
    weight[prompt] *= 1.0 - config.prompt_transfer

    local = torch.arange(max(response_idx, target - config.local_window), target)
    existing = torch.stack([
        weight[source == index].sum() for index in local
    ]) if local.numel() else torch.empty(0)
    if local.numel() and float(moved) > 0:
        allocation = existing + existing.mean().clamp_min(1e-6)
        allocation = allocation / allocation.sum()
        source = torch.cat((source, local))
        weight = torch.cat((weight, moved * allocation))

    source, weight = _coalesce(source, weight)
    history = source >= response_idx
    history_weights = weight[history]
    if history_weights.numel() > 1:
        ordered = torch.sort(history_weights, descending=True).values
        count = min(len(ordered), config.local_window, target - response_idx)
        nearest = torch.arange(target - count, target)
        relocated = ordered[:count].clone()
        if len(ordered) > count:
            relocated[-1] += ordered[count:].sum()
        source = torch.cat((source[~history], nearest))
        weight = torch.cat((weight[~history], relocated))
    return _coalesce(source, _normalise(weight))


def support_collapse(
    source: torch.Tensor,
    weight: torch.Tensor,
    config: CorruptionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.numel() <= 1:
        return source, weight
    keep = max(1, round(len(source) * config.support_keep_fraction))
    order = torch.argsort(weight, descending=True, stable=True)[:keep]
    source = source[order]
    weight = weight[order].pow(config.concentration_power)
    return _coalesce(source, _normalise(weight))


def corrupt_row(
    source: torch.Tensor,
    weight: torch.Tensor,
    *,
    target: int,
    response_idx: int,
    mode: str,
    config: CorruptionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.numel():
        weight = _normalise(weight)
    if mode in {"incidence", "composite"}:
        source, weight = incidence_erosion(
            source,
            weight,
            target=target,
            response_idx=response_idx,
            config=config,
        )
    if mode in {"collapse", "composite"}:
        source, weight = support_collapse(source, weight, config)
    return source, weight
