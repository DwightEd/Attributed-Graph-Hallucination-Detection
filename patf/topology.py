from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch

from .augment import corrupt_row
from .config import CorruptionConfig, TopologyConfig
from .data import AttentionSample

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
    grounding: torch.Tensor = field(init=False)
    hops: torch.Tensor = field(init=False)
    observed: torch.Tensor = field(init=False)
    reachable: torch.Tensor = field(init=False)
    sums: torch.Tensor = field(init=False)
    retained_mass: float = 0.0
    prompt_sources: set[int] = field(default_factory=set)
    source_counts: Counter[int] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self.grounding = torch.zeros(self.response_tokens)
        self.hops = torch.zeros(self.response_tokens)
        self.observed = torch.zeros(self.response_tokens, dtype=torch.bool)
        self.reachable = torch.zeros(self.response_tokens, dtype=torch.bool)
        self.sums = torch.zeros(11)


def _normalised_hhi(weight: torch.Tensor) -> float:
    count = len(weight)
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
    order = torch.argsort(weight, descending=True, stable=True)
    cumulative = weight[order].cumsum(0)
    count = int(torch.searchsorted(cumulative, torch.tensor(ratio)).item()) + 1
    chosen = order[:count]
    return source[chosen], weight[chosen]


def _finalise(state: _ChannelState, prompt_tokens: int) -> torch.Tensor:
    observed_count = int(state.observed.sum())
    values = state.sums / max(observed_count, 1)
    observed_fraction = float(state.observed.float().mean())
    retained = state.retained_mass / state.response_tokens
    prompt_coverage = len(state.prompt_sources) / prompt_tokens

    counts = torch.tensor(list(state.source_counts.values()), dtype=torch.float32)
    if counts.numel():
        hub_concentration = _normalised_hhi(counts / counts.sum())
    else:
        hub_concentration = 0.0

    return torch.cat(
        (
            values[:6],
            torch.tensor([retained, observed_fraction]),
            values[6:],
            torch.tensor([prompt_coverage, hub_concentration]),
        )
    )


def _update_state(
    state: _ChannelState,
    *,
    source: torch.Tensor,
    weight: torch.Tensor,
    raw_mass: float,
    local_query: int,
    response_idx: int,
    config: TopologyConfig,
) -> None:
    state.retained_mass += raw_mass
    if source.numel() == 0 or weight.numel() == 0:
        return

    state.observed[local_query] = True
    target = response_idx + local_query
    prompt = source < response_idx
    history = ~prompt
    prompt_mass = weight[prompt].sum()

    history_source = source[history] - response_idx
    history_weight = weight[history]
    if history_source.numel():
        known = state.observed[history_source]
        known_weight = history_weight[known]
        known_source = history_source[known]
        unknown = history_weight[~known].sum()
        grounded_relay = (
            known_weight * state.grounding[known_source]
        ).sum()
        unsupported = (
            known_weight * (1.0 - state.grounding[known_source])
        ).sum()
        hop_relay = (
            known_weight
            * state.grounding[known_source]
            * (state.hops[known_source] + 1.0)
        ).sum()
    else:
        grounded_relay = unsupported = unknown = hop_relay = torch.tensor(0.0)

    grounding = prompt_mass + config.relay_discount * grounded_relay
    state.grounding[local_query] = grounding.clamp(0.0, 1.0)
    if float(grounding) > 1e-12:
        state.hops[local_query] = (
            prompt_mass + config.relay_discount * hop_relay
        ) / grounding

    selected_source, selected_weight = _mass_cover(
        source, weight, config.mass_cover
    )
    support_size = len(selected_source)
    support_log = torch.log1p(torch.tensor(float(support_size)))
    possible_log = torch.log1p(torch.tensor(float(target)))
    sparsity = 1.0 - support_log / possible_log
    concentration = _normalised_hhi(weight)

    selected_prompt = selected_source < response_idx
    prompt_source = selected_source[selected_prompt]
    response_source = selected_source[~selected_prompt]
    state.prompt_sources.update(int(value) for value in prompt_source.tolist())
    state.source_counts.update(int(value) for value in selected_source.tolist())

    reachable = bool(prompt_source.numel())
    locality = torch.tensor(0.0)
    if response_source.numel():
        local_source = response_source - response_idx
        reachable = reachable or bool(state.reachable[local_source].any())
        response_weight = selected_weight[~selected_prompt]
        response_weight = response_weight / response_weight.sum()
        lag = (target - response_source).float()
        max_lag = max(local_query, 1)
        normalised_lag = ((lag - 1.0) / max(max_lag - 1, 1)).clamp(0.0, 1.0)
        locality = 1.0 - torch.dot(response_weight, normalised_lag)
    state.reachable[local_query] = reachable

    state.sums += torch.tensor(
        [
            float(prompt_mass),
            float(grounded_relay),
            float(unsupported),
            float(unknown),
            float(grounding),
            float(state.hops[local_query]),
            float(support_log),
            float(sparsity),
            float(locality),
            float(concentration),
            float(reachable),
        ]
    )


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
        name: _ChannelState(sample.response_tokens) for name in variants
    }

    for local_query, row in enumerate(sample.rows(channel)):
        if row.weight.numel():
            base_weight = row.weight / row.weight.sum()
        else:
            base_weight = row.weight
        target = sample.response_idx + local_query

        _update_state(
            states["original"],
            source=row.source,
            weight=base_weight,
            raw_mass=row.retained_mass,
            local_query=local_query,
            response_idx=sample.response_idx,
            config=topology,
        )
        for mode in modes:
            source, weight = corrupt_row(
                row.source,
                base_weight,
                target=target,
                response_idx=sample.response_idx,
                mode=mode,
                config=corruption,
            )
            _update_state(
                states[mode],
                source=source,
                weight=weight,
                raw_mass=row.retained_mass,
                local_query=local_query,
                response_idx=sample.response_idx,
                config=topology,
            )

    return {
        name: _finalise(state, sample.response_idx)
        for name, state in states.items()
    }


def extract_trajectories(
    sample: AttentionSample,
    *,
    topology: TopologyConfig,
    corruption: CorruptionConfig,
    modes: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    variants = ("original",) + modes
    trajectories = {name: [] for name in variants}

    for layer in range(sample.num_layers):
        per_variant = {name: [] for name in variants}
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
                per_variant[name].append(features[name])

        for name in variants:
            heads = torch.stack(per_variant[name])
            center = (
                heads.median(dim=0).values
                if topology.head_reducer == "median"
                else heads.mean(dim=0)
            )
            iqr = (
                torch.quantile(heads, 0.75, dim=0)
                - torch.quantile(heads, 0.25, dim=0)
                if sample.num_heads > 1
                else torch.zeros_like(center)
            )
            trajectories[name].append(torch.cat((center, iqr)))

    return {
        name: torch.stack(layers) for name, layers in trajectories.items()
    }
