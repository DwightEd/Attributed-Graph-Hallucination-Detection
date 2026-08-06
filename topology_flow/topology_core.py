from __future__ import annotations
import torch
from .config import TopologyConfig

def _prepare_rows(rows: torch.Tensor, response_idx: int, epsilon: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(rows, dtype=torch.float32).clone()
    response_tokens, token_count = values.shape
    query_ids = torch.arange(response_idx, response_idx + response_tokens)
    key_ids = torch.arange(token_count)
    values.masked_fill_(key_ids.unsqueeze(0) >= query_ids.unsqueeze(1), 0.0)
    retained_mass = values.sum(dim=1)
    observed = retained_mass > epsilon
    probabilities = torch.zeros_like(values)
    probabilities[observed] = values[observed] / retained_mass[observed, None]
    return (probabilities, retained_mass, observed)

def _normalized_hhi(probabilities: torch.Tensor, active_count: int) -> float:
    if active_count <= 1:
        return 1.0
    hhi = float(probabilities.square().sum())
    uniform = 1.0 / active_count
    return float(max(0.0, min(1.0, (hhi - uniform) / (1.0 - uniform))))

def _mass_cover_sources(row: torch.Tensor, limit: int, mass_cover: float) -> torch.Tensor:
    values = row[:limit]
    active = torch.nonzero(values > 0, as_tuple=False).flatten()
    if not len(active):
        return active
    order = active[torch.argsort(values[active], descending=True, stable=True)]
    cumulative = values[order].cumsum(0)
    crossing = int(torch.searchsorted(cumulative, torch.tensor(mass_cover)).item())
    return order[:min(crossing + 1, len(order))]

def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].mean())

def _channel_features(rows: torch.Tensor, *, response_idx: int, config: TopologyConfig) -> torch.Tensor:
    probabilities, retained_mass, observed = _prepare_rows(rows, response_idx, config.epsilon)
    response_tokens, token_count = probabilities.shape
    grounding = torch.zeros(response_tokens, dtype=torch.float32)
    expected_hops = torch.zeros_like(grounding)
    direct_values = torch.zeros_like(grounding)
    grounded_relay_values = torch.zeros_like(grounding)
    unsupported_values = torch.zeros_like(grounding)
    unknown_values = torch.zeros_like(grounding)
    support_size_values = torch.zeros_like(grounding)
    sparsity_values = torch.zeros_like(grounding)
    locality_values = torch.zeros_like(grounding)
    concentration_values = torch.zeros_like(grounding)
    reachability_values = torch.zeros_like(grounding)
    reachable = torch.zeros(response_tokens, dtype=torch.bool)
    prompt_sources: set[int] = set()
    source_counts = torch.zeros(token_count, dtype=torch.float32)
    for local_query in range(response_tokens):
        if not bool(observed[local_query]):
            continue
        absolute_query = response_idx + local_query
        row = probabilities[local_query, :absolute_query]
        prompt_mass = row[:response_idx].sum()
        history = row[response_idx:absolute_query]
        if local_query:
            prior_known = observed[:local_query].float()
            grounded_relay = torch.dot(history, grounding[:local_query] * prior_known)
            unsupported = torch.dot(history, (1.0 - grounding[:local_query]) * prior_known)
            unknown = torch.dot(history, 1.0 - prior_known)
            hop_numerator = prompt_mass + config.relay_discount * torch.dot(history, grounding[:local_query] * prior_known * (expected_hops[:local_query] + 1.0))
        else:
            grounded_relay = unsupported = unknown = torch.tensor(0.0)
            hop_numerator = prompt_mass
        current_grounding = prompt_mass + config.relay_discount * grounded_relay
        grounding[local_query] = current_grounding.clamp(0.0, 1.0)
        if float(current_grounding) > config.epsilon:
            expected_hops[local_query] = hop_numerator / current_grounding
        direct_values[local_query] = prompt_mass
        grounded_relay_values[local_query] = grounded_relay
        unsupported_values[local_query] = unsupported
        unknown_values[local_query] = unknown
        selected = _mass_cover_sources(row, absolute_query, config.mass_cover)
        selected_weights = row[selected]
        support_count = len(selected)
        support_size_values[local_query] = torch.log1p(torch.tensor(float(support_count)))
        sparsity_values[local_query] = 1.0 - torch.log1p(torch.tensor(float(support_count))) / torch.log1p(torch.tensor(float(absolute_query)))
        active_count = int((row > config.epsilon).sum())
        concentration_values[local_query] = _normalized_hhi(row, active_count)
        prompt_selected = selected[selected < response_idx]
        response_selected = selected[selected >= response_idx]
        prompt_sources.update((int(value) for value in prompt_selected.tolist()))
        source_counts[selected] += 1.0
        is_reachable = bool(len(prompt_selected))
        if len(response_selected):
            local_sources = response_selected - response_idx
            known_sources = observed[local_sources]
            if bool(known_sources.any()):
                is_reachable = is_reachable or bool(reachable[local_sources[known_sources]].any())
            rr_weights = selected_weights[selected >= response_idx]
            rr_weights = rr_weights / rr_weights.sum().clamp_min(config.epsilon)
            lag = (absolute_query - response_selected).float()
            max_lag = max(local_query, 1)
            denominator = max(max_lag - 1, 1)
            normalized_lag = ((lag - 1.0) / denominator).clamp(0.0, 1.0)
            locality_values[local_query] = 1.0 - torch.dot(rr_weights, normalized_lag)
        reachable[local_query] = is_reachable
        reachability_values[local_query] = float(is_reachable)
    prompt_coverage = len(prompt_sources) / float(response_idx)
    active_counts = source_counts[source_counts > 0]
    if len(active_counts) <= 1:
        hub_concentration = 1.0 if len(active_counts) else 0.0
    else:
        hub_probabilities = active_counts / active_counts.sum()
        hub_concentration = _normalized_hhi(hub_probabilities, len(active_counts))
    return torch.tensor([_masked_mean(direct_values, observed), _masked_mean(grounded_relay_values, observed), _masked_mean(unsupported_values, observed), _masked_mean(unknown_values, observed), _masked_mean(grounding, observed), _masked_mean(expected_hops, observed), float(retained_mass.mean()), float(observed.float().mean()), _masked_mean(support_size_values, observed), _masked_mean(sparsity_values, observed), _masked_mean(locality_values, observed), _masked_mean(concentration_values, observed), _masked_mean(reachability_values, observed), float(prompt_coverage), float(hub_concentration)], dtype=torch.float32)

def _head_center(values: torch.Tensor, reducer: str) -> torch.Tensor:
    if reducer == 'median':
        return values.median(dim=0).values
    return values.mean(dim=0)

def _head_iqr(values: torch.Tensor) -> torch.Tensor:
    if len(values) <= 1:
        return torch.zeros(values.shape[1], dtype=values.dtype)
    return torch.quantile(values, 0.75, dim=0) - torch.quantile(values, 0.25, dim=0)
