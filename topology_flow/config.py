"""Configuration objects for prompt-anchored topology-flow modelling."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TopologyConfig:
    """Deterministic graph-view and trajectory extraction configuration.

    ``relay_discount`` penalises long response-to-response relay chains.  A
    value below one is essential: without it, every strictly causal attention
    walk eventually reaches the prompt and prompt ancestry degenerates to one.
    ``mass_cover`` defines an adaptive, scale-free support graph by retaining
    the fewest edges whose cumulative row mass reaches this value.
    """

    mass_cover: float = 0.80
    relay_discount: float = 0.85
    epsilon: float = 1e-8
    head_reducer: str = "median"

    def __post_init__(self) -> None:
        if not 0.0 < self.mass_cover <= 1.0:
            raise ValueError("mass_cover must lie in (0, 1]")
        if not 0.0 < self.relay_discount < 1.0:
            raise ValueError("relay_discount must lie in (0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.head_reducer not in {"median", "mean"}:
            raise ValueError("head_reducer must be 'median' or 'mean'")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CorruptionConfig:
    """Label-free counterfactual topology erosion used for pairwise training."""

    prompt_transfer: float = 0.45
    local_window: int = 4
    support_keep_fraction: float = 0.55
    concentration_power: float = 1.8
    mode: str = "composite"

    def __post_init__(self) -> None:
        if not 0.0 <= self.prompt_transfer < 1.0:
            raise ValueError("prompt_transfer must lie in [0, 1)")
        if self.local_window < 1:
            raise ValueError("local_window must be positive")
        if not 0.0 < self.support_keep_fraction <= 1.0:
            raise ValueError("support_keep_fraction must lie in (0, 1]")
        if self.concentration_power < 1.0:
            raise ValueError("concentration_power must be at least one")
        if self.mode not in {"incidence", "collapse", "composite", "all"}:
            raise ValueError("mode must be incidence, collapse, composite, or all")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
