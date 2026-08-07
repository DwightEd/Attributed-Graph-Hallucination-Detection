from __future__ import annotations

import torch

from attention_cache.io import AttentionSample, SparseLayer

from .augment import erode_attention
from .config import CorruptionConfig, FlowConfig

BASE_FEATURES = (
    "direct_prompt_mass",
    "grounded_response_relay",
    "unsupported_response_feedback",
    "prompt_support",
    "support_delta",
    "response_locality",
    "attention_concentration",
    "effective_support_fraction",
    "retained_attention_mass",
    "observed_row_fraction",
)
FEATURE_NAMES = tuple(
    f"{stat}::{name}"
    for stat in ("head_center", "head_iqr")
    for name in BASE_FEATURES
)


def _scatter_sum(values: torch.Tensor, rows: torch.Tensor, row_count: int) -> torch.Tensor:
    result = torch.zeros(row_count, dtype=values.dtype, device=values.device)
    if values.numel():
        result.index_add_(0, rows, values)
    return result


def _normalise_layer(layer: SparseLayer, response_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = layer.head * response_tokens + layer.query
    row_count = layer.diagonal.numel()
    off_mass = _scatter_sum(layer.weight, rows, row_count)
    raw_mass = off_mass + layer.diagonal.flatten()
    observed = raw_mass > 0
    denominator = raw_mass.clamp_min(1e-12)
    weight = layer.weight / denominator[rows] if layer.weight.numel() else layer.weight
    diagonal = layer.diagonal / denominator.view_as(layer.diagonal)
    return weight, diagonal, raw_mass.view_as(layer.diagonal), observed.view_as(layer.diagonal)


def _head_features(
    layer: SparseLayer,
    weight: torch.Tensor,
    diagonal: torch.Tensor,
    raw_mass: torch.Tensor,
    observed: torch.Tensor,
    previous_support: torch.Tensor,
    *,
    response_idx: int,
    config: FlowConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads, response_tokens = diagonal.shape
    row_count = heads * response_tokens
    rows = layer.head * response_tokens + layer.query

    prompt = layer.source < response_idx
    direct = _scatter_sum(weight[prompt], rows[prompt], row_count).view(heads, response_tokens)

    response = ~prompt
    grounded = torch.zeros(row_count)
    unsupported = torch.zeros(row_count)
    locality_sum = torch.zeros(row_count)
    response_mass = torch.zeros(row_count)
    if response.any():
        response_rows = rows[response]
        response_source = layer.source[response] - response_idx
        response_weight = weight[response]
        previous = previous_support[response_source]
        grounded.index_add_(0, response_rows, response_weight * previous)
        unsupported.index_add_(0, response_rows, response_weight * (1.0 - previous))

        lag = (response_idx + layer.query[response] - layer.source[response]).float()
        denominator = (layer.query[response].float() - 1.0).clamp_min(1.0)
        locality = 1.0 - ((lag - 1.0) / denominator).clamp(0.0, 1.0)
        locality_sum.index_add_(0, response_rows, response_weight * locality)
        response_mass.index_add_(0, response_rows, response_weight)

    previous_grid = previous_support.unsqueeze(0).expand(heads, -1)
    grounded = grounded.view(heads, response_tokens) + diagonal * previous_grid
    unsupported = unsupported.view(heads, response_tokens) + diagonal * (1.0 - previous_grid)
    attention_support = direct + grounded
    head_support = (
        config.residual_weight * previous_grid
        + (1.0 - config.residual_weight) * attention_support
    ).clamp(0.0, 1.0)
    head_support = torch.where(observed, head_support, previous_grid)

    response_mass = response_mass.view(heads, response_tokens)
    locality = locality_sum.view(heads, response_tokens) / response_mass.clamp_min(1e-12)
    locality = torch.where(response_mass > 0, locality, torch.zeros_like(locality))

    concentration = _scatter_sum(weight.square(), rows, row_count).view(heads, response_tokens)
    concentration += diagonal.square()
    effective_support = concentration.clamp_min(1e-12).reciprocal()
    possible = torch.arange(response_tokens, dtype=torch.float32) + response_idx + 1.0
    effective_fraction = torch.log1p(effective_support) / torch.log1p(possible).unsqueeze(0)
    effective_fraction = effective_fraction.clamp(0.0, 1.0)

    count = observed.sum(dim=1).clamp_min(1).float()

    def mean(value: torch.Tensor) -> torch.Tensor:
        return (value * observed).sum(dim=1) / count

    features = torch.stack(
        (
            mean(direct),
            mean(grounded),
            mean(unsupported),
            mean(head_support),
            mean(head_support - previous_grid),
            mean(locality),
            mean(concentration),
            mean(effective_fraction),
            raw_mass.mean(dim=1),
            observed.float().mean(dim=1),
        ),
        dim=1,
    )
    return features, head_support


def _aggregate_heads(values: torch.Tensor, reducer: str) -> torch.Tensor:
    center = values.median(dim=0).values if reducer == "median" else values.mean(dim=0)
    if values.shape[0] == 1:
        iqr = torch.zeros_like(center)
    else:
        iqr = torch.quantile(values, 0.75, dim=0) - torch.quantile(values, 0.25, dim=0)
    return torch.cat((center, iqr))


def _reduce_support(support: torch.Tensor, reducer: str) -> torch.Tensor:
    return support.median(dim=0).values if reducer == "median" else support.mean(dim=0)


def extract_flow(
    sample: AttentionSample,
    *,
    flow: FlowConfig,
    corruption: CorruptionConfig,
    counterfactual: bool = False,
) -> dict[str, torch.Tensor]:
    """Trace prompt-rooted support across Transformer layers using sparse attention."""
    previous = {"original": torch.zeros(sample.response_tokens)}
    trajectories: dict[str, list[torch.Tensor]] = {"original": []}
    if counterfactual:
        previous["eroded"] = torch.zeros(sample.response_tokens)
        trajectories["eroded"] = []

    for layer_index in range(sample.num_layers):
        layer = sample.layer(layer_index)
        weight, diagonal, raw_mass, observed = _normalise_layer(layer, sample.response_tokens)
        features, support = _head_features(
            layer,
            weight,
            diagonal,
            raw_mass,
            observed,
            previous["original"],
            response_idx=sample.response_idx,
            config=flow,
        )
        trajectories["original"].append(_aggregate_heads(features, flow.head_reducer))
        previous["original"] = _reduce_support(support, flow.head_reducer)

        if counterfactual:
            eroded_weight, eroded_diagonal = erode_attention(
                head=layer.head,
                query=layer.query,
                source=layer.source,
                weight=weight,
                diagonal=diagonal,
                response_idx=sample.response_idx,
                response_tokens=sample.response_tokens,
                config=corruption,
            )
            eroded_features, eroded_support = _head_features(
                layer,
                eroded_weight,
                eroded_diagonal,
                raw_mass,
                observed,
                previous["eroded"],
                response_idx=sample.response_idx,
                config=flow,
            )
            trajectories["eroded"].append(_aggregate_heads(eroded_features, flow.head_reducer))
            previous["eroded"] = _reduce_support(eroded_support, flow.head_reducer)

    return {name: torch.stack(values) for name, values in trajectories.items()}
