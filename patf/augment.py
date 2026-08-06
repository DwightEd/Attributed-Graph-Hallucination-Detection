from __future__ import annotations

import torch

from .config import CorruptionConfig


def _normalise(weight: torch.Tensor) -> torch.Tensor:
    return weight / weight.sum().clamp_min(1e-12)


def _coalesce(
    source: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.numel() == 0:
        return source.long(), weight.float()
    unique, inverse = torch.unique(source.long(), sorted=True, return_inverse=True)
    merged = torch.zeros(len(unique), dtype=torch.float32)
    merged.index_add_(0, inverse, weight.float())
    keep = merged > 0
    return unique[keep], merged[keep]


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

    start = max(response_idx, target - config.local_window)
    local = torch.arange(start, target)
    if local.numel() and float(moved) > 0:
        existing = torch.zeros(len(local))
        local_mask = (source >= start) & (source < target)
        if local_mask.any():
            existing.index_add_(0, source[local_mask].long() - start, weight[local_mask])
        allocation = existing + existing.mean().clamp_min(1e-6)
        allocation /= allocation.sum()
        source = torch.cat((source, local))
        weight = torch.cat((weight, moved * allocation))

    source, weight = _coalesce(source, weight)
    history = source >= response_idx
    history_weight = weight[history]
    if history_weight.numel() > 1:
        ordered = torch.sort(history_weight, descending=True).values
        count = min(len(ordered), config.local_window, target - response_idx)
        nearest = torch.arange(target - count, target)
        relocated = ordered[:count].clone()
        if len(ordered) > count:
            relocated[-1] += ordered[count:].sum()
        source = torch.cat((source[~history], nearest))
        weight = torch.cat((weight[~history], relocated))
    source, weight = _coalesce(source, weight)
    return source, _normalise(weight)


def support_collapse(
    source: torch.Tensor,
    weight: torch.Tensor,
    config: CorruptionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.numel() <= 1:
        return source, weight
    keep = max(1, round(len(source) * config.support_keep_fraction))
    order = torch.argsort(weight, descending=True, stable=True)[:keep]
    return source[order], _normalise(weight[order].pow(config.concentration_power))


def corrupt_variants(
    source: torch.Tensor,
    weight: torch.Tensor,
    *,
    target: int,
    response_idx: int,
    modes: tuple[str, ...],
    config: CorruptionConfig,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Build all requested variants without recomputing shared corruptions."""
    base = _normalise(weight) if weight.numel() else weight
    variants: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "original": (source, base)
    }
    incidence: tuple[torch.Tensor, torch.Tensor] | None = None
    if "incidence" in modes or "composite" in modes:
        incidence = incidence_erosion(
            source,
            base,
            target=target,
            response_idx=response_idx,
            config=config,
        )
    if "incidence" in modes and incidence is not None:
        variants["incidence"] = incidence
    if "collapse" in modes:
        variants["collapse"] = support_collapse(source, base, config)
    if "composite" in modes and incidence is not None:
        variants["composite"] = support_collapse(*incidence, config)
    return variants


def corrupt_row(
    source: torch.Tensor,
    weight: torch.Tensor,
    *,
    target: int,
    response_idx: int,
    mode: str,
    config: CorruptionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    return corrupt_variants(
        source,
        weight,
        target=target,
        response_idx=response_idx,
        modes=(mode,),
        config=config,
    )[mode]
