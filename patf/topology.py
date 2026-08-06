from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from attention_cache import AttentionSample

from .augment import corrupt_variants
from .config import CorruptionConfig, TopologyConfig

BASE_FEATURES = (
    "direct_prompt_mass",
    "grounded_response_relay",
    "unsupported_response_feedback",
    "unknown_response_feedback",
    "discounted_prompt_ancestry",
    "expected_prompt_hops",
    "retained_causal_mass",
    "observed_row_fraction",
    "mass_cover_support_size",
    "edge_sparsity",
    "response_locality",
    "weight_concentration",
    "prompt_rooted_reachability",
    "prompt_source_coverage",
    "source_hub_concentration",
)
FEATURE_NAMES = tuple(
    f"{stat}::{name}"
    for stat in ("head_center", "head_iqr")
    for name in BASE_FEATURES
)


@dataclass
class _ChannelState:
    response_tokens: int
    prompt_tokens: int
    token_count: int
    grounding: torch.Tensor = field(init=False)
    hops: torch.Tensor = field(init=False)
    observed: torch.Tensor = field(init=False)
    reachable: torch.Tensor = field(init=False)
    prompt_seen: torch.Tensor = field(init=False)
    source_count: torch.Tensor = field(init=False)
    sums: torch.Tensor = field(init=False)
    retained_mass: float = 0.0

    def __post_init__(self) -> None:
        self.grounding = torch.zeros(self.response_tokens)
        self.hops = torch.zeros(self.response_tokens)
        self.observed = torch.zeros(self.response_tokens, dtype=torch.bool)
        self.reachable = torch.zeros(self.response_tokens, dtype=torch.bool)
        self.prompt_seen = torch.zeros(self.prompt_tokens, dtype=torch.bool)
        self.source_count = torch.zeros(self.token_count)
        self.sums = torch.zeros(11)


def _normalised_hhi(weight: torch.Tensor) -> float:
    count = int(weight.numel())
    if count <= 1:
        return 1.0 if count else 0.0
    hhi = float(weight.square().sum())
    uniform = 1.0 / count
    return (hhi - uniform) / (1.0 - uniform)


def _mass_cover(
    source: torch.Tensor, weight: torch.Tensor, ratio: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.numel() == 0:
        return source, weight
    sorted_weight, order = torch.sort(weight, descending=True, stable=True)
    count = int(torch.searchsorted(sorted_weight.cumsum(0), ratio).item()) + 1
    chosen = order[:count]
    return source[chosen], weight[chosen]


def _finalise(state: _ChannelState) -> torch.Tensor:
    observed_count = max(int(state.observed.sum()), 1)
    mean = state.sums / observed_count
    counts = state.source_count[state.source_count > 0]
    hub = _normalised_hhi(counts / counts.sum()) if counts.numel() else 0.0
    return torch.cat(
        (
            mean[:6],
            torch.tensor(
                [
                    state.retained_mass / state.response_tokens,
                    float(state.observed.float().mean()),
                ]
            ),
            mean[6:],
            torch.tensor([float(state.prompt_seen.float().mean()), hub]),
        )
    )


def _update_state(
    state: _ChannelState,
    source: torch.Tensor,
    weight: torch.Tensor,
    *,
    raw_mass: float,
    local_query: int,
    response_idx: int,
    config: TopologyConfig,
) -> None:
    state.retained_mass += raw_mass
    if source.numel() == 0:
        return

    state.observed[local_query] = True
    target = response_idx + local_query
    prompt_mask = source < response_idx
    history_source = source[~prompt_mask] - response_idx
    history_weight = weight[~prompt_mask]
    prompt_mass = float(weight[prompt_mask].sum())

    grounded_relay = unsupported = unknown = hop_relay = 0.0
    if history_source.numel():
        known_mask = state.observed[history_source]
        if known_mask.any():
            known_source = history_source[known_mask]
            known_weight = history_weight[known_mask]
            grounded_relay = float(
                torch.dot(known_weight, state.grounding[known_source])
            )
            unsupported = float(
                torch.dot(known_weight, 1.0 - state.grounding[known_source])
            )
            hop_relay = float(
                torch.dot(
                    known_weight * state.grounding[known_source],
                    state.hops[known_source] + 1.0,
                )
            )
        unknown = float(history_weight[~known_mask].sum())

    grounding = min(
        1.0, prompt_mass + config.relay_discount * grounded_relay
    )
    state.grounding[local_query] = grounding
    if grounding > 1e-12:
        state.hops[local_query] = (
            prompt_mass + config.relay_discount * hop_relay
        ) / grounding

    selected_source, selected_weight = _mass_cover(
        source, weight, config.mass_cover
    )
    support_size = int(selected_source.numel())
    support_log = math.log1p(support_size)
    sparsity = 1.0 - support_log / math.log1p(target)
    concentration = _normalised_hhi(weight)

    selected_prompt = selected_source < response_idx
    prompt_source = selected_source[selected_prompt]
    response_source = selected_source[~selected_prompt]
    if prompt_source.numel():
        state.prompt_seen[prompt_source.long()] = True
    state.source_count.index_add_(
        0,
        selected_source.long(),
        torch.ones(support_size),
    )

    reachable = bool(prompt_source.numel())
    locality = 0.0
    if response_source.numel():
        local_source = response_source - response_idx
        reachable = reachable or bool(state.reachable[local_source].any())
        response_weight = selected_weight[~selected_prompt]
        response_weight /= response_weight.sum()
        lag = (target - response_source).float()
        denominator = max(local_query - 1, 1)
        normalised_lag = ((lag - 1.0) / denominator).clamp(0.0, 1.0)
        locality = float(1.0 - torch.dot(response_weight, normalised_lag))
    state.reachable[local_query] = reachable

    state.sums[0] += prompt_mass
    state.sums[1] += grounded_relay
    state.sums[2] += unsupported
    state.sums[3] += unknown
    state.sums[4] += grounding
    state.sums[5] += state.hops[local_query]
    state.sums[6] += support_log
    state.sums[7] += sparsity
    state.sums[8] += locality
    state.sums[9] += concentration
    state.sums[10] += float(reachable)


def _channel_features(
    sample: AttentionSample,
    channel: int,
    *,
    modes: tuple[str, ...],
    topology: TopologyConfig,
    corruption: CorruptionConfig,
) -> dict[str, torch.Tensor]:
    variants = ("original",) + modes
    states = {
        name: _ChannelState(
            sample.response_tokens, sample.response_idx, sample.token_count
        )
        for name in variants
    }

    for local_query, row in enumerate(sample.rows(channel)):
        target = sample.response_idx + local_query
        rows = corrupt_variants(
            row.source,
            row.weight,
            target=target,
            response_idx=sample.response_idx,
            modes=modes,
            config=corruption,
        )
        for name, (source, weight) in rows.items():
            _update_state(
                states[name],
                source,
                weight,
                raw_mass=row.retained_mass,
                local_query=local_query,
                response_idx=sample.response_idx,
                config=topology,
            )
    return {name: _finalise(state) for name, state in states.items()}


def extract_trajectories(
    sample: AttentionSample,
    *,
    topology: TopologyConfig,
    corruption: CorruptionConfig,
    modes: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Extract one [layers, 30] trajectory for each requested variant."""
    variants = ("original",) + modes
    trajectories = {name: [] for name in variants}

    for layer in range(sample.num_layers):
        head_features = {name: [] for name in variants}
        for head in range(sample.num_heads):
            channel = layer * sample.num_heads + head
            features = _channel_features(
                sample,
                channel,
                modes=modes,
                topology=topology,
                corruption=corruption,
            )
            for name in variants:
                head_features[name].append(features[name])

        for name in variants:
            values = torch.stack(head_features[name])
            center = (
                values.median(dim=0).values
                if topology.head_reducer == "median"
                else values.mean(dim=0)
            )
            iqr = (
                torch.quantile(values, 0.75, dim=0)
                - torch.quantile(values, 0.25, dim=0)
                if sample.num_heads > 1
                else torch.zeros_like(center)
            )
            trajectories[name].append(torch.cat((center, iqr)))

    return {
        name: torch.stack(layer_values)
        for name, layer_values in trajectories.items()
    }
