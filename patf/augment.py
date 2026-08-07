from __future__ import annotations

import torch

from .config import CorruptionConfig


def erode_attention(
    *,
    head: torch.Tensor,
    query: torch.Tensor,
    source: torch.Tensor,
    weight: torch.Tensor,
    diagonal: torch.Tensor,
    response_idx: int,
    response_tokens: int,
    config: CorruptionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downweight prompt support and sharpen local response dependence."""
    if weight.numel() == 0 and not bool(diagonal.any()):
        return weight, diagonal

    transformed = weight.clone()
    prompt = source < response_idx
    transformed[prompt] *= 1.0 - config.prompt_suppression

    history = ~prompt
    if history.any():
        lag = (response_idx + query[history] - source[history]).float()
        locality = torch.exp(
            -(lag - 1.0).clamp_min(0) / max(config.local_window, 1e-6)
        )
        transformed[history] *= 1.0 + config.locality_strength * locality

    diag = diagonal.clone() * (1.0 + config.locality_strength)
    if config.concentration_power != 1.0:
        transformed = transformed.clamp_min(0).pow(config.concentration_power)
        diag = diag.clamp_min(0).pow(config.concentration_power)

    rows = head * response_tokens + query
    row_mass = torch.zeros(diagonal.numel(), dtype=transformed.dtype)
    if transformed.numel():
        row_mass.index_add_(0, rows, transformed)
    row_mass += diag.flatten()
    denominator = row_mass.clamp_min(1e-12)
    if transformed.numel():
        transformed = transformed / denominator[rows]
    diag = diag / denominator.view_as(diag)
    return transformed, diag
