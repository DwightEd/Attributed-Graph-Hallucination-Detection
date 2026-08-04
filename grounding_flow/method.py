"""Evidence-provenance transport and its graph-conditional null model.

The Transformer attention tensor is treated as an observed, typed transport
operator.  No GNN, hidden state, logit, token position feature, or
hallucination label enters this module.  Prompt segment ids are structural
coordinates only: HaluEval uses 0=template, 1=evidence, 2=question, and
3=response.

Sparse legacy caches censor attention at a recorded floor.  Missing mass is
therefore kept in an explicit ``unknown`` component; it is never silently
converted to zero or renormalised away.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, replace
from collections.abc import Callable, Sequence

import torch

from attention_graph.graph import AttentionGraph


COMPONENT_NAMES = (
    "direct_evidence",
    "direct_other_prompt",
    "grounded_relay",
    "ungrounded_relay",
    "grounded_self",
    "ungrounded_self",
    "unknown",
)
_COMPONENT_INDEX = {name: index for index, name in enumerate(COMPONENT_NAMES)}


@dataclass(frozen=True)
class EvidenceFlow:
    """Per-response-token evidence transport.

    ``values`` has shape ``[response_tokens, layers, heads, components]``.
    ``ancestry`` is an attention-rollout evidence proxy under head-mean
    propagation.  Cache-censored and inherited missing mass remains explicit;
    this is not a claim about value/output projections or residual streams.
    """

    response_token_indices: torch.Tensor
    values: torch.Tensor

    def component(self, name: str) -> torch.Tensor:
        try:
            index = _COMPONENT_INDEX[name]
        except KeyError as error:
            raise KeyError(f"unknown flow component: {name}") from error
        return self.values[..., index]

    @property
    def ancestry(self) -> torch.Tensor:
        return (
            self.component("direct_evidence")
            + self.component("grounded_relay")
            + self.component("grounded_self")
        )

    @property
    def debt(self) -> torch.Tensor:
        """Response-derived mass without known evidence ancestry."""

        return self.component("ungrounded_relay") + self.component(
            "ungrounded_self"
        )

    @property
    def known_ungrounded(self) -> torch.Tensor:
        return self.component("direct_other_prompt") + self.debt


@dataclass(frozen=True)
class NullModelConfig:
    """Controls legal source-identity swaps used as the conditional null.

    Two-edge swaps preserve every target, edge-bound layer/head payload,
    response-source type, causal validity, global combinatorial source degree,
    and a coarse causal lag bin.  They deliberately destroy which preceding
    response token receives each fixed edge payload.  Per-source weighted
    outgoing strength is not claimed as an invariant.
    """

    swaps_per_edge: int = 4
    lag_boundaries: tuple[int, ...] = (4, 8, 16, 32, 64, 128)
    max_null_attempt_factor: int = 12

    def validate(self) -> None:
        if self.swaps_per_edge < 1 or self.max_null_attempt_factor < 1:
            raise ValueError("null-model attempt counts must be positive")
        if any(value < 1 for value in self.lag_boundaries):
            raise ValueError("lag boundaries must be positive")
        if tuple(sorted(set(self.lag_boundaries))) != self.lag_boundaries:
            raise ValueError("lag boundaries must be strictly increasing")


@dataclass(frozen=True)
class RewireReport:
    attempts: int
    accepted_swaps: int
    changed_edges: int

    @property
    def changed(self) -> bool:
        return self.changed_edges > 0


@dataclass(frozen=True)
class _RewirePlan:
    """Static CPU indices reused by every conditional-null draw."""

    num_nodes: int
    original_sources: tuple[int, ...]
    targets: tuple[int, ...]
    lag_bins: tuple[int, ...]
    eligible_groups: tuple[tuple[int, ...], ...]
    base_occupied: frozenset[int]
    attempts: int


@dataclass(frozen=True)
class CalibratedFlow:
    """Real flow and position-matched null distribution."""

    real: EvidenceFlow
    null_mean: torch.Tensor
    null_std: torch.Tensor
    z_scores: torch.Tensor
    model_z_scores: torch.Tensor
    null_samples: int
    accepted_swap_fraction: float
    effective_null_fraction: float
    stable_model_fraction: float
    duplicate_null_draws: int
    calibration_status: str

    def component_z(self, name: str) -> torch.Tensor:
        try:
            index = _COMPONENT_INDEX[name]
        except KeyError as error:
            raise KeyError(f"unknown flow component: {name}") from error
        return self.z_scores[..., index]

    @property
    def ancestry_z(self) -> torch.Tensor:
        return self.model_z_scores[..., 0]

    @property
    def debt_z(self) -> torch.Tensor:
        return self.model_z_scores[..., 2]

    def model_tensor(self) -> torch.Tensor:
        """Return the full layer/head transition surface used by clustering.

        This is a fixed selection of mechanistic channels, not a learned or
        label-selected feature list.  Layer and head axes remain intact until
        the label-free projector is fitted on the training partition.
        """

        return self.model_z_scores


def _model_channels(flow: EvidenceFlow) -> torch.Tensor:
    """Derived mechanistic channels calibrated in their own units."""

    return torch.stack(
        (
            flow.ancestry,
            flow.component("grounded_relay"),
            flow.debt,
        ),
        dim=-1,
    )


def _validate_segments(
    graph: AttentionGraph, segment_ids: torch.Tensor | Sequence[int]
) -> torch.Tensor:
    segments = torch.as_tensor(segment_ids, dtype=torch.long, device=graph.node_attr.device)
    if segments.ndim != 1 or segments.numel() != graph.num_nodes:
        raise ValueError("segment_ids must contain one structural id per token")
    if bool(((segments < 0) | (segments > 3)).any()):
        raise ValueError("segment_ids must be in the range 0 through 3")
    expected_response = torch.arange(
        graph.num_nodes, device=segments.device
    ) >= graph.response_idx
    if not torch.equal(segments == 3, expected_response):
        raise ValueError("segment id 3 must mark exactly the response suffix")
    return segments


def _scatter_sum(
    group: torch.Tensor, values: torch.Tensor, *, groups: int
) -> torch.Tensor:
    result = torch.zeros(groups, dtype=values.dtype, device=values.device)
    if group.numel():
        result.index_add_(0, group, values)
    return result


def _build_layer_trace_indices(graph: AttentionGraph) -> tuple[torch.Tensor, ...]:
    """Partition trace rows once instead of scanning all rows for every layer."""

    layer_id = torch.div(
        graph.trace_channel, graph.num_heads, rounding_mode="floor"
    )
    if bool(((layer_id < 0) | (layer_id >= graph.num_layers)).any()):
        raise ValueError("trace channels fall outside declared layer/head dimensions")
    order = torch.argsort(layer_id, stable=True)
    counts = torch.bincount(layer_id, minlength=graph.num_layers).tolist()
    return tuple(torch.split(order, counts))


def compute_evidence_flow(
    graph: AttentionGraph,
    segment_ids: torch.Tensor | Sequence[int],
    *,
    evidence_segment_ids: Sequence[int] = (1,),
    mass_tolerance: float = 5e-3,
    validate: bool = True,
    _trace_indices_by_layer: tuple[torch.Tensor, ...] | None = None,
) -> EvidenceFlow:
    """Propagate evidence ancestry through the layer-token causal DAG.

    Heads are retained as parallel relations.  Their conservative ancestry
    estimates are averaged only when forming the shared representation passed
    to the next Transformer layer; no cross-layer head identity is invented.
    """

    if not evidence_segment_ids or any(value not in (0, 1, 2) for value in evidence_segment_ids):
        raise ValueError("evidence segments must be non-response structural ids")
    if not math.isfinite(mass_tolerance) or mass_tolerance < 0.0:
        raise ValueError("mass_tolerance must be finite and non-negative")
    segments = _validate_segments(graph, segment_ids)
    device = graph.node_attr.device
    dtype = graph.node_attr.dtype
    layers, heads = graph.num_layers, graph.num_heads
    response_tokens = graph.num_nodes - graph.response_idx
    response_indices = torch.arange(graph.response_idx, graph.num_nodes, device=device)
    evidence_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
    for segment in evidence_segment_ids:
        evidence_mask |= segments == int(segment)

    # Prompt source identities are treated as fixed anchors because the cache
    # contains response attention rows only.  Response provenance evolves one
    # Transformer layer at a time.
    previous_grounded = evidence_mask.to(dtype)
    # Before the first observed attention layer, response representations have
    # no demonstrated evidence ancestry.  They therefore start as known
    # ungrounded rather than zero-mass states; otherwise first-layer RR/self
    # attention would disappear and violate conservation.
    previous_ungrounded = (~evidence_mask).to(dtype)
    previous_unknown = torch.zeros(graph.num_nodes, dtype=dtype, device=device)
    values = torch.zeros(
        (response_tokens, layers, heads, len(COMPONENT_NAMES)),
        dtype=dtype,
        device=device,
    )
    diagonal = graph.node_attr.reshape(graph.num_nodes, layers, heads)
    trace_source = graph.edge_index[0, graph.trace_edge_id]
    trace_target = graph.edge_index[1, graph.trace_edge_id]
    trace_indices_by_layer = (
        _build_layer_trace_indices(graph)
        if _trace_indices_by_layer is None
        else _trace_indices_by_layer
    )
    if len(trace_indices_by_layer) != layers:
        raise ValueError("trace index plan disagrees with declared layer count")
    minimum_missing_mass = torch.full((), float("inf"), dtype=dtype, device=device)
    maximum_state_error = torch.zeros((), dtype=dtype, device=device)

    for layer in range(layers):
        first_channel = layer * heads
        layer_trace = trace_indices_by_layer[layer]
        source = trace_source[layer_trace]
        target = trace_target[layer_trace]
        attention = graph.trace_value[layer_trace].to(dtype)
        local_head = graph.trace_channel[layer_trace] - first_channel
        group = local_head * response_tokens + (target - graph.response_idx)
        group_count = heads * response_tokens

        source_is_prompt = source < graph.response_idx
        source_is_evidence = evidence_mask[source]
        source_is_response = ~source_is_prompt
        direct_evidence = _scatter_sum(
            group[source_is_evidence], attention[source_is_evidence], groups=group_count
        )
        other_prompt = source_is_prompt & ~source_is_evidence
        direct_other = _scatter_sum(
            group[other_prompt], attention[other_prompt], groups=group_count
        )
        response_group = group[source_is_response]
        response_attention = attention[source_is_response]
        response_source = source[source_is_response]
        grounded_relay = _scatter_sum(
            response_group,
            response_attention * previous_grounded[response_source],
            groups=group_count,
        )
        ungrounded_relay = _scatter_sum(
            response_group,
            response_attention * previous_ungrounded[response_source],
            groups=group_count,
        )
        unknown_relay = _scatter_sum(
            response_group,
            response_attention * previous_unknown[response_source],
            groups=group_count,
        )

        row_shape = (heads, response_tokens)
        direct_evidence = direct_evidence.reshape(row_shape)
        direct_other = direct_other.reshape(row_shape)
        grounded_relay = grounded_relay.reshape(row_shape)
        ungrounded_relay = ungrounded_relay.reshape(row_shape)
        unknown_relay = unknown_relay.reshape(row_shape)
        self_attention = diagonal[response_indices, layer, :].T
        grounded_self = self_attention * previous_grounded[response_indices].unsqueeze(0)
        ungrounded_self = self_attention * previous_ungrounded[response_indices].unsqueeze(0)
        unknown_self = self_attention * previous_unknown[response_indices].unsqueeze(0)

        observed_historical = _scatter_sum(
            group, attention, groups=group_count
        ).reshape(row_shape)
        observed_row_total = observed_historical + self_attention
        cache_unknown = 1.0 - observed_row_total
        if validate:
            minimum_missing_mass = torch.minimum(
                minimum_missing_mass, cache_unknown.min()
            )
        # Float16 cache storage may round a mathematically stochastic row a
        # few ulps above one.  Correct only that impossible numerical overage;
        # rows below one are never renormalised and retain explicit unknown.
        numerical_scale = torch.where(
            observed_row_total > 1.0,
            observed_row_total.reciprocal(),
            torch.ones_like(observed_row_total),
        )
        direct_evidence *= numerical_scale
        direct_other *= numerical_scale
        grounded_relay *= numerical_scale
        ungrounded_relay *= numerical_scale
        unknown_relay *= numerical_scale
        grounded_self *= numerical_scale
        ungrounded_self *= numerical_scale
        unknown_self *= numerical_scale
        cache_unknown = cache_unknown.clamp_min(0.0)
        unknown = unknown_relay + unknown_self + cache_unknown

        layer_values = torch.stack(
            (
                direct_evidence,
                direct_other,
                grounded_relay,
                ungrounded_relay,
                grounded_self,
                ungrounded_self,
                unknown,
            ),
            dim=-1,
        ).permute(1, 0, 2)
        values[:, layer] = layer_values

        grounded_head = direct_evidence + grounded_relay + grounded_self
        ungrounded_head = direct_other + ungrounded_relay + ungrounded_self
        next_grounded = previous_grounded.clone()
        next_ungrounded = previous_ungrounded.clone()
        next_unknown = previous_unknown.clone()
        next_grounded[response_indices] = grounded_head.mean(dim=0)
        next_ungrounded[response_indices] = ungrounded_head.mean(dim=0)
        next_unknown[response_indices] = unknown.mean(dim=0)
        if validate:
            state_sum = (
                next_grounded[response_indices]
                + next_ungrounded[response_indices]
                + next_unknown[response_indices]
            )
            maximum_state_error = torch.maximum(
                maximum_state_error, (state_sum - 1.0).abs().max()
            )
        previous_grounded, previous_ungrounded, previous_unknown = (
            next_grounded,
            next_ungrounded,
            next_unknown,
        )

    if validate:
        if float(minimum_missing_mass) < -mass_tolerance:
            raise ValueError(
                "attention row mass exceeds one beyond tolerance: "
                f"minimum missing mass {float(minimum_missing_mass)}"
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("evidence flow contains non-finite values")
        if bool(
            ((values < -mass_tolerance) | (values > 1.0 + mass_tolerance)).any()
        ):
            raise ValueError("evidence flow components fall outside probability bounds")
        row_sum = values.sum(dim=-1)
        if not torch.allclose(
            row_sum,
            torch.ones_like(row_sum),
            atol=max(mass_tolerance, 1e-7),
            rtol=0.0,
        ):
            raise ValueError("evidence flow does not conserve attention row mass")
        if float(maximum_state_error) > max(mass_tolerance, 1e-7):
            raise ValueError("cross-layer provenance state does not conserve mass")
    return EvidenceFlow(response_token_indices=response_indices, values=values)


def _lag_bin(lag: int, boundaries: tuple[int, ...]) -> int:
    return bisect.bisect_left(boundaries, lag)


def _build_rewire_plan(
    graph: AttentionGraph,
    segments: torch.Tensor,
    config: NullModelConfig,
) -> _RewirePlan:
    """Pre-index lag buckets and immutable occupancy once per source graph."""

    source_values = tuple(int(value) for value in graph.edge_index[0].tolist())
    target_values = tuple(int(value) for value in graph.edge_index[1].tolist())
    lag_bins = tuple(
        _lag_bin(target - source, config.lag_boundaries)
        for source, target in zip(source_values, target_values)
    )
    # Candidate edges only need to share the RR relation.  They do not need
    # to start in the same lag bin: after exchanging sources, each edge is
    # checked below against *its own* original bin.  Grouping by the original
    # bin here incorrectly discarded legal cross-bin swaps before those exact
    # checks could run.
    groups: dict[int, list[int]] = {}
    for edge_id, source in enumerate(source_values):
        source_segment = int(segments[source])
        if source_segment == 3:
            groups.setdefault(source_segment, []).append(edge_id)
    eligible = tuple(
        tuple(members) for members in groups.values() if len(members) >= 2
    )
    return _RewirePlan(
        num_nodes=graph.num_nodes,
        original_sources=source_values,
        targets=target_values,
        lag_bins=lag_bins,
        eligible_groups=eligible,
        base_occupied=frozenset(
            target * graph.num_nodes + source
            for source, target in zip(source_values, target_values)
        ),
        attempts=sum(len(members) for members in eligible) * config.swaps_per_edge,
    )


def rewire_source_identity(
    graph: AttentionGraph,
    segment_ids: torch.Tensor | Sequence[int],
    *,
    config: NullModelConfig | None = None,
    generator: torch.Generator | None = None,
    _plan: _RewirePlan | None = None,
) -> tuple[AttentionGraph, RewireReport]:
    """Destroy source identity using legal, typed, lag-conditioned swaps."""

    null_config = NullModelConfig() if config is None else config
    null_config.validate()
    segments = _validate_segments(graph, segment_ids)
    if graph.edge_index.device.type != "cpu" or segments.device.type != "cpu":
        raise ValueError("source rewiring must run on a CPU graph before flow transfer")
    edge_count = graph.num_edges
    if edge_count < 2:
        return graph, RewireReport(attempts=0, accepted_swaps=0, changed_edges=0)
    plan = (
        _build_rewire_plan(graph, segments, null_config)
        if _plan is None
        else _plan
    )
    if plan.num_nodes != graph.num_nodes or len(plan.original_sources) != edge_count:
        raise ValueError("rewire plan disagrees with graph dimensions")
    eligible = plan.eligible_groups
    if not eligible:
        return graph, RewireReport(attempts=0, accepted_swaps=0, changed_edges=0)
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(torch.seed())
    source = list(plan.original_sources)
    removed: set[int] = set()
    added: set[int] = set()
    attempts = plan.attempts
    draws = torch.rand((attempts, 3), generator=generator).tolist()

    def occupied_now(key: int) -> bool:
        return (
            key in plan.base_occupied and key not in removed
        ) or key in added

    def remove_key(key: int) -> None:
        if key in added:
            added.remove(key)
        else:
            removed.add(key)

    def add_key(key: int) -> None:
        if key in plan.base_occupied:
            removed.discard(key)
        else:
            added.add(key)

    accepted = 0
    for group_draw, first_draw, second_draw in draws:
        group_id = min(int(group_draw * len(eligible)), len(eligible) - 1)
        members = eligible[group_id]
        first_position = min(int(first_draw * len(members)), len(members) - 1)
        second_position = min(
            int(second_draw * (len(members) - 1)), len(members) - 2
        )
        if second_position >= first_position:
            second_position += 1
        first, second = members[first_position], members[second_position]
        source_a, source_b = int(source[first]), int(source[second])
        target_a, target_b = plan.targets[first], plan.targets[second]
        if source_a == source_b:
            continue
        if source_b >= target_a or source_a >= target_b:
            continue
        old_bin_a = plan.lag_bins[first]
        old_bin_b = plan.lag_bins[second]
        if (
            _lag_bin(target_a - source_b, null_config.lag_boundaries) != old_bin_a
            or _lag_bin(target_b - source_a, null_config.lag_boundaries) != old_bin_b
        ):
            continue
        old_a = target_a * graph.num_nodes + source_a
        old_b = target_b * graph.num_nodes + source_b
        new_a = target_a * graph.num_nodes + source_b
        new_b = target_b * graph.num_nodes + source_a
        # Both old pairs are removed atomically.  This deliberately permits
        # two payloads targeting the same token to exchange source identity:
        # the unweighted pair set is then unchanged, but the full layer/head
        # attention payload attached to each source is different.
        if new_a == new_b or (
            new_a not in (old_a, old_b) and occupied_now(new_a)
        ) or (
            new_b not in (old_a, old_b) and occupied_now(new_b)
        ):
            continue
        source[first], source[second] = source_b, source_a
        remove_key(old_a)
        remove_key(old_b)
        add_key(new_a)
        add_key(new_b)
        accepted += 1
    changed_edges = sum(
        current != original
        for current, original in zip(source, plan.original_sources)
    )
    shuffled_source = torch.tensor(source, dtype=graph.edge_index.dtype)
    shuffled = replace(
        graph, edge_index=torch.stack((shuffled_source, graph.edge_index[1]))
    )
    return shuffled, RewireReport(
        attempts=attempts, accepted_swaps=accepted, changed_edges=changed_edges
    )


def calibrate_against_null(
    graph: AttentionGraph,
    segment_ids: torch.Tensor | Sequence[int],
    *,
    num_nulls: int = 32,
    seed: int = 0,
    null_config: NullModelConfig | None = None,
    evidence_segment_ids: Sequence[int] = (1,),
    flow_device: str | torch.device | None = None,
    standard_deviation_floor: float = 1e-6,
    progress_callback: Callable[[int, int], None] | None = None,
) -> CalibratedFlow:
    """Standardise a graph's transport against changed conditional nulls."""

    if num_nulls < 2:
        raise ValueError("null calibration requires at least two changed graphs")
    if standard_deviation_floor <= 0.0 or not math.isfinite(standard_deviation_floor):
        raise ValueError("standard_deviation_floor must be finite and positive")
    null_config = NullModelConfig() if null_config is None else null_config
    null_config.validate()
    cpu_graph = graph.to("cpu")
    cpu_segments = torch.as_tensor(segment_ids, dtype=torch.long, device="cpu")
    requested = graph.node_attr.device if flow_device is None else torch.device(flow_device)
    device_graph = graph.to(requested)
    device_segments = cpu_segments.to(requested)
    rewire_plan = _build_rewire_plan(cpu_graph, cpu_segments, null_config)
    trace_indices_by_layer = _build_layer_trace_indices(device_graph)
    real = compute_evidence_flow(
        device_graph,
        device_segments,
        evidence_segment_ids=evidence_segment_ids,
        _trace_indices_by_layer=trace_indices_by_layer,
    )
    real_model = _model_channels(real)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    null_sum: torch.Tensor | None = None
    null_square_sum: torch.Tensor | None = None
    model_sum: torch.Tensor | None = None
    model_square_sum: torch.Tensor | None = None
    accepted_nulls = 0
    effective_nulls = 0
    total_attempts = 0
    total_accepted = 0
    duplicate_null_draws = 0
    seen_null_sources: set[bytes] = set()
    tries = 0
    maximum_tries = num_nulls * null_config.max_null_attempt_factor
    while accepted_nulls < num_nulls and tries < maximum_tries:
        tries += 1
        shuffled, report = rewire_source_identity(
            cpu_graph,
            cpu_segments,
            config=null_config,
            generator=generator,
            _plan=rewire_plan,
        )
        total_attempts += report.attempts
        total_accepted += report.accepted_swaps
        if report.attempts == 0:
            break
        if not report.changed:
            continue
        null_identity = shuffled.edge_index[0].contiguous().numpy().tobytes()
        if null_identity in seen_null_sources:
            duplicate_null_draws += 1
            continue
        seen_null_sources.add(null_identity)
        # Static node/channel tensors stay resident on the flow device.  A
        # null changes only source assignment, so transfer only edge_index.
        device_null = replace(
            device_graph, edge_index=shuffled.edge_index.to(requested)
        )
        null_flow = compute_evidence_flow(
            device_null,
            device_segments,
            evidence_segment_ids=evidence_segment_ids,
            validate=False,
            _trace_indices_by_layer=trace_indices_by_layer,
        )
        null_value = null_flow.values
        model_value = _model_channels(null_flow)
        if bool((model_value - real_model).abs().max() > 1e-8):
            effective_nulls += 1
        if null_sum is None:
            null_sum = torch.zeros_like(null_value)
            null_square_sum = torch.zeros_like(null_value)
            model_sum = torch.zeros_like(model_value)
            model_square_sum = torch.zeros_like(model_value)
        null_sum.add_(null_value)
        null_square_sum.add_(null_value.square())
        model_sum.add_(model_value)
        model_square_sum.add_(model_value.square())
        accepted_nulls += 1
        if progress_callback is not None:
            progress_callback(accepted_nulls, num_nulls)
    if accepted_nulls == 0:
        zeros = torch.zeros_like(real.values)
        return CalibratedFlow(
            real=real,
            null_mean=real.values.clone(),
            null_std=zeros,
            z_scores=zeros,
            model_z_scores=torch.zeros_like(real_model),
            null_samples=0,
            accepted_swap_fraction=0.0,
            effective_null_fraction=0.0,
            stable_model_fraction=0.0,
            duplicate_null_draws=duplicate_null_draws,
            calibration_status="unswappable",
        )
    if (
        null_sum is None
        or null_square_sum is None
        or model_sum is None
        or model_square_sum is None
    ):  # pragma: no cover - guarded by accepted_nulls above
        raise RuntimeError("null calibration did not accumulate statistics")
    null_mean = null_sum / accepted_nulls
    null_variance = (
        null_square_sum / accepted_nulls - null_mean.square()
    ).clamp_min(0.0)
    null_std = null_variance.sqrt()
    stable = null_std >= standard_deviation_floor
    z_scores = torch.where(
        stable,
        (real.values - null_mean) / null_std.clamp_min(standard_deviation_floor),
        torch.zeros_like(real.values),
    )
    model_mean = model_sum / accepted_nulls
    model_variance = (
        model_square_sum / accepted_nulls - model_mean.square()
    ).clamp_min(0.0)
    model_std = model_variance.sqrt()
    model_stable = model_std >= standard_deviation_floor
    model_z_scores = torch.where(
        model_stable,
        (real_model - model_mean) / model_std.clamp_min(standard_deviation_floor),
        torch.zeros_like(real_model),
    )
    accepted_fraction = (
        float(total_accepted / total_attempts) if total_attempts else 0.0
    )
    return CalibratedFlow(
        real=real,
        null_mean=null_mean,
        null_std=null_std,
        z_scores=z_scores,
        model_z_scores=model_z_scores,
        null_samples=accepted_nulls,
        accepted_swap_fraction=accepted_fraction,
        effective_null_fraction=float(effective_nulls / accepted_nulls),
        stable_model_fraction=float(model_stable.to(torch.float32).mean()),
        duplicate_null_draws=duplicate_null_draws,
        calibration_status=(
            "complete"
            if accepted_nulls == num_nulls
            else (
                "insufficient_unique_nulls" if accepted_nulls < 2 else "partial"
            )
        ),
    )


__all__ = [
    "COMPONENT_NAMES",
    "CalibratedFlow",
    "EvidenceFlow",
    "NullModelConfig",
    "RewireReport",
    "calibrate_against_null",
    "compute_evidence_flow",
    "rewire_source_identity",
]
