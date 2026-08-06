"""Layer-expanded operational attention transport for evidence provenance."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..data.graph import Relation, Segment
from .data import TeacherTargets

BLOCK_RANKING_MARGIN = 1e-6


@dataclass(frozen=True)
class TransportOutput:
    direct: torch.Tensor
    history_lower: torch.Tensor
    history_upper: torch.Tensor
    support_lower: torch.Tensor
    support_upper: torch.Tensor
    block: torch.Tensor
    row_available: torch.Tensor
    unsupported_history_lower: torch.Tensor
    unsupported_history_upper: torch.Tensor

    @property
    def token_risk(self) -> torch.Tensor:
        """Lower endpoint of the operational censored-attention interval."""

        return self.unsupported_history_lower.clamp(0.0, 1.0)

    @property
    def overall_evidence_deficit_lower(self) -> torch.Tensor:
        return (1.0 - self.support_upper).clamp(0.0, 1.0)


@dataclass(frozen=True)
class _LayerStatistics:
    observed_weight: torch.Tensor
    history_weight: torch.Tensor
    direct_numerator: torch.Tensor
    direct_upper_numerator: torch.Tensor
    history_lower_numerator: torch.Tensor
    support_upper_numerator: torch.Tensor
    history_upper_numerator: torch.Tensor
    grounded_history_lower_numerator: torch.Tensor
    grounded_history_upper_numerator: torch.Tensor
    block_numerator: torch.Tensor


def _tensor(
    graph: Mapping[str, object], name: str, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if name not in graph:
        raise ValueError(f"prediction-event graph is missing {name}")
    return torch.as_tensor(graph[name], device=device).to(dtype)


class CausalProvenanceTransport(nn.Module):
    """Transport operational evidence ancestry through a token-layer DAG.

    Prediction event ``t`` uses the attention row at predictor ``t-1``.  A
    response K/V source at position ``p`` is therefore the state produced by
    row ``p`` at the previous transformer layer, not prediction event ``p``.
    Expanding the state over layers makes diagonal attention a legal
    cross-layer edge instead of a spurious previous-token edge.

    Only relation/layer/head surrogate gates and one operational residual
    persistence coefficient per layer are learned. There is no token embedding,
    hidden state, generic GNN/MLP, graph pooling network, or classifier.
    """

    def __init__(self, *, num_layers: int, num_heads: int) -> None:
        super().__init__()
        if num_layers <= 0 or num_heads <= 0:
            raise ValueError("layer/head counts must be positive")
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_channels = self.num_layers * self.num_heads
        self.raw_relation_channel_gates = nn.Parameter(
            torch.zeros((len(Relation), self.num_channels), dtype=torch.float32)
        )
        # This is an operational persistence coefficient, not a physical
        # decomposition of the transformer's additive residual stream.
        self.raw_residual_persistence = nn.Parameter(
            torch.zeros(self.num_layers, dtype=torch.float32)
        )

    def gates(self) -> torch.Tensor:
        # Unit mean removes an unidentifiable global scale while retaining
        # non-negative relative channel reliability.
        positive = F.softplus(self.raw_relation_channel_gates) + 1e-8
        return positive / positive.mean()

    def residual_persistence(self) -> torch.Tensor:
        """Return one bounded operational residual coefficient per layer."""

        return torch.sigmoid(self.raw_residual_persistence)

    @torch.no_grad()
    def rewire_diagnostics(self, graph: Mapping[str, object]) -> dict[str, object]:
        """Measure how much the exact registered three-view control changes."""

        if graph.get("schema") != "cept-prediction-event-graph-v2":
            raise ValueError("unsupported prediction-event graph schema")
        if (
            int(graph["num_layers"]) != self.num_layers
            or int(graph["num_heads"]) != self.num_heads
        ):
            raise ValueError("graph layer/head shape disagrees with transport model")
        device = self.raw_relation_channel_gates.device
        segment = _tensor(
            graph, "segment_ids", device=device, dtype=torch.long
        ).flatten()
        targets = _tensor(
            graph, "target_token_positions", device=device, dtype=torch.long
        ).flatten()
        predictors = _tensor(
            graph, "predictor_positions", device=device, dtype=torch.long
        ).flatten()
        row_available = _tensor(
            graph, "row_available", device=device, dtype=torch.bool
        ).flatten()
        edge_index = _tensor(graph, "edge_index", device=device, dtype=torch.long)
        edge_relation = _tensor(
            graph, "edge_relation", device=device, dtype=torch.long
        ).flatten()
        trace_edge = _tensor(
            graph, "trace_edge_id", device=device, dtype=torch.long
        ).flatten()
        trace_channel = _tensor(
            graph, "trace_channel", device=device, dtype=torch.long
        ).flatten()
        trace_value = _tensor(
            graph, "trace_value", device=device, dtype=torch.float32
        ).flatten()
        unknown = _tensor(
            graph, "unknown_mass", device=device, dtype=torch.float32
        )
        self._validate_graph_vectors(
            segment=segment,
            targets=targets,
            predictors=predictors,
            row_available=row_available,
            edge_index=edge_index,
            edge_relation=edge_relation,
            trace_edge=trace_edge,
            trace_channel=trace_channel,
            trace_value=trace_value,
            unknown=unknown,
        )
        sources = edge_index[0].index_select(0, trace_edge)
        trace_events = edge_index[1].index_select(0, trace_edge)
        relations = edge_relation.index_select(0, trace_edge)
        ry_mask = relations == int(Relation.RY)
        trace_count = int(sources.numel())
        ry_count = int(ry_mask.sum().item())
        views: list[dict[str, int | float]] = []
        changed_total = 0
        ry_changed_total = 0
        for offset in (1, 2, 3):
            rewired = self._rewire_sources(
                sources=sources,
                relations=relations,
                trace_events=trace_events,
                trace_channel=trace_channel,
                predictors=predictors,
                segment=segment,
                offset=offset,
            )
            changed = rewired != sources
            changed_count = int(changed.sum().item())
            ry_changed_count = int((changed & ry_mask).sum().item())
            changed_total += changed_count
            ry_changed_total += ry_changed_count
            views.append(
                {
                    "offset": offset,
                    "traces": trace_count,
                    "changed_traces": changed_count,
                    "changed_trace_fraction": changed_count / max(1, trace_count),
                    "ry_traces": ry_count,
                    "ry_changed_traces": ry_changed_count,
                    "ry_changed_trace_fraction": ry_changed_count / max(1, ry_count),
                }
            )
        trace_view_pairs = 3 * trace_count
        ry_trace_view_pairs = 3 * ry_count
        return {
            "views": views,
            "trace_view_pairs": trace_view_pairs,
            "changed_trace_view_pairs": changed_total,
            "changed_trace_fraction": changed_total / max(1, trace_view_pairs),
            "ry_trace_view_pairs": ry_trace_view_pairs,
            "ry_changed_trace_view_pairs": ry_changed_total,
            "ry_changed_trace_fraction": ry_changed_total
            / max(1, ry_trace_view_pairs),
        }

    def forward(
        self,
        graph: Mapping[str, object],
        *,
        seed_positions: torch.Tensor | Sequence[int],
        block_positions: Mapping[str, Sequence[int]] | None = None,
        incidence: str = "true",
        residual_mode: str = "learned",
    ) -> TransportOutput:
        if incidence not in {"true", "rewired", "one_hop", "mass_only"}:
            raise ValueError("incidence must be true, rewired, one_hop, or mass_only")
        if residual_mode not in {"learned", "none"}:
            raise ValueError("residual_mode must be learned or none")
        if graph.get("schema") != "cept-prediction-event-graph-v2":
            raise ValueError("unsupported prediction-event graph schema")
        if (
            int(graph["num_layers"]) != self.num_layers
            or int(graph["num_heads"]) != self.num_heads
        ):
            raise ValueError("graph layer/head shape disagrees with transport model")

        device = self.raw_relation_channel_gates.device
        segment = _tensor(
            graph, "segment_ids", device=device, dtype=torch.long
        ).flatten()
        targets = _tensor(
            graph, "target_token_positions", device=device, dtype=torch.long
        ).flatten()
        predictors = _tensor(
            graph, "predictor_positions", device=device, dtype=torch.long
        ).flatten()
        row_available = _tensor(
            graph, "row_available", device=device, dtype=torch.bool
        ).flatten()
        edge_index = _tensor(graph, "edge_index", device=device, dtype=torch.long)
        edge_relation = _tensor(
            graph, "edge_relation", device=device, dtype=torch.long
        ).flatten()
        trace_edge = _tensor(
            graph, "trace_edge_id", device=device, dtype=torch.long
        ).flatten()
        trace_channel = _tensor(
            graph, "trace_channel", device=device, dtype=torch.long
        ).flatten()
        trace_value = _tensor(
            graph, "trace_value", device=device, dtype=torch.float32
        ).flatten()
        unknown = _tensor(graph, "unknown_mass", device=device, dtype=torch.float32)
        self._validate_graph_vectors(
            segment=segment,
            targets=targets,
            predictors=predictors,
            row_available=row_available,
            edge_index=edge_index,
            edge_relation=edge_relation,
            trace_edge=trace_edge,
            trace_channel=trace_channel,
            trace_value=trace_value,
            unknown=unknown,
        )

        seeds = torch.as_tensor(
            seed_positions, device=device, dtype=torch.long
        ).flatten()
        if seeds.numel() == 0 or torch.unique(seeds).numel() != seeds.numel():
            raise ValueError("seed_positions must be non-empty and unique")
        if bool(((seeds < 0) | (seeds >= segment.numel())).any()) or bool(
            (segment.index_select(0, seeds) != int(Segment.EVIDENCE)).any()
        ):
            raise ValueError("every provenance seed must be an evidence token")

        block_items = tuple(
            (block_id, tuple(int(position) for position in positions))
            for block_id, positions in (block_positions or {}).items()
        )
        if any(not block_id or not positions for block_id, positions in block_items):
            raise ValueError("transport history blocks must be non-empty")
        for _, positions in block_items:
            if any(
                position < 0
                or position >= segment.numel()
                or int(segment[position]) != int(Segment.RESPONSE)
                for position in positions
            ):
                raise ValueError("transport block positions must be response tokens")

        sources = (
            edge_index[0].index_select(0, trace_edge)
            if trace_edge.numel()
            else torch.empty(0, dtype=torch.long, device=device)
        )
        trace_events = (
            edge_index[1].index_select(0, trace_edge)
            if trace_edge.numel()
            else torch.empty(0, dtype=torch.long, device=device)
        )
        relations = (
            edge_relation.index_select(0, trace_edge)
            if trace_edge.numel()
            else torch.empty(0, dtype=torch.long, device=device)
        )
        source_views = (sources,)
        if incidence == "rewired" and sources.numel():
            source_views = tuple(
                self._rewire_sources(
                    sources=sources,
                    relations=relations,
                    trace_events=trace_events,
                    trace_channel=trace_channel,
                    predictors=predictors,
                    segment=segment,
                    offset=offset,
                )
                for offset in (1, 2, 3)
            )

        gate = self.gates()
        trace_weights = (
            trace_value * gate[relations, trace_channel]
            if trace_value.numel()
            else trace_value
        )
        # The old cache censors relation and endpoint for discarded attention.
        # The largest relation gate defines the evidence-favourable endpoint of
        # this registered attention-DAG interval; it is not a physical K/V bound.
        unknown_gate = gate.max(dim=0).values

        sequence_length = segment.numel()
        block_masks = torch.zeros((len(block_items), sequence_length), device=device)
        for block_index, (_, positions) in enumerate(block_items):
            block_masks[block_index, torch.tensor(positions, device=device)] = 1.0

        positions = torch.arange(sequence_length, device=device)
        seed_mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        seed_mask[seeds] = True
        # Prompt rows are absent from the legacy cache. A non-seed prompt token
        # can acquire seed ancestry only after one layer, and only when a seed
        # occurs causally before it. Its operational lower bound remains zero.
        reachable_prompt = (
            (segment != int(Segment.RESPONSE))
            & ~seed_mask
            & (positions > seeds.min())
        )
        learned_residual = self.residual_persistence()
        outputs = tuple(
            self._transport_view(
                sources=view,
                trace_events=trace_events,
                relations=relations,
                trace_channel=trace_channel,
                trace_weights=trace_weights,
                unknown=unknown,
                unknown_gate=unknown_gate,
                segment=segment,
                predictors=predictors,
                row_available=row_available,
                seeds=seeds,
                block_masks=block_masks,
                reachable_prompt=reachable_prompt,
                learned_residual=learned_residual,
                incidence=incidence,
                residual_mode=residual_mode,
            )
            for view in source_views
        )
        if len(outputs) == 1:
            return outputs[0]

        def mean(name: str) -> torch.Tensor:
            return torch.stack([getattr(output, name) for output in outputs]).mean(0)

        # Each permutation is a coherent graph through every layer. Averaging
        # only final outputs prevents impossible paths that splice offsets.
        return TransportOutput(
            direct=mean("direct"),
            history_lower=mean("history_lower"),
            history_upper=mean("history_upper"),
            support_lower=mean("support_lower"),
            support_upper=mean("support_upper"),
            block=mean("block"),
            row_available=row_available,
            unsupported_history_lower=mean("unsupported_history_lower"),
            unsupported_history_upper=mean("unsupported_history_upper"),
        )

    def _transport_view(
        self,
        *,
        sources: torch.Tensor,
        trace_events: torch.Tensor,
        relations: torch.Tensor,
        trace_channel: torch.Tensor,
        trace_weights: torch.Tensor,
        unknown: torch.Tensor,
        unknown_gate: torch.Tensor,
        segment: torch.Tensor,
        predictors: torch.Tensor,
        row_available: torch.Tensor,
        seeds: torch.Tensor,
        block_masks: torch.Tensor,
        reachable_prompt: torch.Tensor,
        learned_residual: torch.Tensor,
        incidence: str,
        residual_mode: str,
    ) -> TransportOutput:
        """Run one coherent operational incidence view through every layer."""

        device = segment.device
        sequence_length = segment.numel()
        event_count = predictors.numel()
        eps = torch.finfo(torch.float32).eps
        direct_state = torch.zeros(sequence_length, device=device)
        direct_state[seeds] = 1.0
        direct_upper_state = direct_state.clone()
        history_lower_state = torch.zeros(sequence_length, device=device)
        support_upper_state = direct_state.clone()
        history_upper_state = torch.zeros(sequence_length, device=device)
        block_total = torch.zeros(
            (block_masks.shape[0], event_count), device=device
        )
        unsupported_lower_total = torch.zeros(event_count, device=device)
        unsupported_upper_total = torch.zeros(event_count, device=device)
        exposure_count = torch.zeros(event_count, device=device)

        for layer in range(self.num_layers):
            layer_mask = (
                trace_channel.div(self.num_heads, rounding_mode="floor") == layer
            )
            layer_events = trace_events[layer_mask]
            layer_relations = relations[layer_mask]
            layer_weights = trace_weights[layer_mask]
            layer_sources = sources[layer_mask]
            channels = slice(layer * self.num_heads, (layer + 1) * self.num_heads)
            unknown_weight = (unknown[:, channels] * unknown_gate[channels]).sum(dim=1)
            previous_direct = direct_state
            previous_direct_upper = direct_upper_state
            previous_history_lower = history_lower_state
            previous_support_lower = previous_direct + previous_history_lower
            previous_support_upper = support_upper_state
            previous_history_upper = history_upper_state
            history_source = (
                previous_direct if incidence == "one_hop" else previous_support_lower
            )
            history_upper_source = (
                previous_direct_upper
                if incidence == "one_hop"
                else previous_support_upper
            )
            statistics = self._layer_statistics(
                sources=layer_sources,
                events=layer_events,
                relations=layer_relations,
                weights=layer_weights,
                predictors=predictors,
                segment=segment,
                incidence=incidence,
                previous_direct=previous_direct,
                previous_direct_upper=previous_direct_upper,
                previous_support_upper=previous_support_upper,
                history_source=history_source,
                history_upper_source=history_upper_source,
                block_masks=block_masks,
            )
            denominator = statistics.observed_weight + unknown_weight
            active = row_available & (denominator > eps)
            safe_denominator = denominator.clamp_min(eps)
            attention_direct = statistics.direct_numerator / safe_denominator
            attention_direct_upper = (
                statistics.direct_upper_numerator + unknown_weight
            ) / safe_denominator
            attention_history_lower = (
                statistics.history_lower_numerator / safe_denominator
            )
            if incidence == "one_hop":
                attention_support_upper = (
                    statistics.direct_upper_numerator
                    + statistics.history_upper_numerator
                    + unknown_weight
                ) / safe_denominator
                attention_support_upper = attention_support_upper.clamp(0.0, 1.0)
            else:
                attention_support_upper = (
                    statistics.support_upper_numerator + unknown_weight
                ) / safe_denominator
            attention_history_upper = (
                statistics.history_upper_numerator + unknown_weight
            ) / safe_denominator

            rho = (
                learned_residual[layer]
                if residual_mode == "learned"
                else learned_residual.new_zeros(())
            )
            attention_scale = 1.0 - rho
            query = predictors[active]
            direct_state = previous_direct.clone()
            direct_upper_state = previous_direct_upper.clone()
            history_lower_state = previous_history_lower.clone()
            support_upper_state = previous_support_upper.clone()
            history_upper_state = previous_history_upper.clone()
            direct_state[query] = (
                rho * previous_direct.index_select(0, query)
                + attention_scale * attention_direct[active]
            )
            direct_upper_state[query] = (
                rho * previous_direct_upper.index_select(0, query)
                + attention_scale * attention_direct_upper[active]
            ).clamp(0.0, 1.0)
            history_lower_state[query] = (
                rho * previous_history_lower.index_select(0, query)
                + attention_scale * attention_history_lower[active]
            )
            support_upper_state[query] = (
                rho * previous_support_upper.index_select(0, query)
                + attention_scale * attention_support_upper[active]
            ).clamp(0.0, 1.0)
            history_upper_state[query] = (
                rho * previous_history_upper.index_select(0, query)
                + attention_scale * attention_history_upper[active]
            ).clamp(0.0, 1.0)

            history_fraction = statistics.history_weight / safe_denominator
            grounded_history_lower = (
                statistics.grounded_history_lower_numerator / safe_denominator
            )
            grounded_history_upper = (
                statistics.grounded_history_upper_numerator / safe_denominator
            )
            unknown_fraction = unknown_weight / safe_denominator
            unsupported_lower_total += torch.where(
                active,
                attention_scale
                * (history_fraction - grounded_history_upper).clamp(0.0, 1.0),
                torch.zeros_like(history_fraction),
            )
            unsupported_upper_total += torch.where(
                active,
                attention_scale
                * (
                    history_fraction - grounded_history_lower + unknown_fraction
                ).clamp(0.0, 1.0),
                torch.zeros_like(history_fraction),
            )
            exposure_count += active.float()

            if block_masks.shape[0]:
                block_total += attention_scale * (
                    statistics.block_numerator / safe_denominator.unsqueeze(0)
                )

            # This uncertainty is available only to the next layer because K/V
            # at layer l reads the state produced by layer l-1.
            direct_upper_state = torch.where(
                reachable_prompt,
                torch.ones_like(direct_upper_state),
                direct_upper_state,
            )
            support_upper_state = torch.where(
                reachable_prompt,
                torch.ones_like(support_upper_state),
                support_upper_state,
            )

        direct = direct_state.index_select(0, predictors)
        history_lower = history_lower_state.index_select(0, predictors)
        history_upper = history_upper_state.index_select(0, predictors)
        support_lower = (direct + history_lower).clamp(0.0, 1.0)
        support_upper = support_upper_state.index_select(0, predictors).clamp(0.0, 1.0)
        unavailable = ~row_available
        direct = torch.where(unavailable, torch.zeros_like(direct), direct)
        history_lower = torch.where(
            unavailable, torch.zeros_like(history_lower), history_lower
        )
        history_upper = torch.where(
            unavailable, torch.ones_like(history_upper), history_upper
        )
        support_lower = torch.where(
            unavailable, torch.zeros_like(support_lower), support_lower
        )
        support_upper = torch.where(
            unavailable, torch.ones_like(support_upper), support_upper
        )
        count = exposure_count.clamp_min(1.0)
        unsupported_history_lower = unsupported_lower_total / count
        unsupported_history_upper = unsupported_upper_total / count
        unsupported_history_lower = torch.where(
            unavailable,
            torch.zeros_like(unsupported_history_lower),
            unsupported_history_lower,
        )
        unsupported_history_upper = torch.where(
            unavailable,
            torch.ones_like(unsupported_history_upper),
            unsupported_history_upper,
        )
        return TransportOutput(
            direct=direct,
            history_lower=history_lower,
            history_upper=history_upper,
            support_lower=support_lower,
            support_upper=support_upper,
            block=block_total / float(self.num_layers),
            row_available=row_available,
            unsupported_history_lower=unsupported_history_lower,
            unsupported_history_upper=unsupported_history_upper,
        )

    def _layer_statistics(
        self,
        *,
        sources: torch.Tensor,
        events: torch.Tensor,
        relations: torch.Tensor,
        weights: torch.Tensor,
        predictors: torch.Tensor,
        segment: torch.Tensor,
        incidence: str,
        previous_direct: torch.Tensor,
        previous_direct_upper: torch.Tensor,
        previous_support_upper: torch.Tensor,
        history_source: torch.Tensor,
        history_upper_source: torch.Tensor,
        block_masks: torch.Tensor,
    ) -> _LayerStatistics:
        """Aggregate one layer in O(traces) memory using GPU scatter-add."""

        event_count = predictors.numel()
        device = segment.device

        def zeros() -> torch.Tensor:
            return torch.zeros(event_count, device=device)

        def scatter(values: torch.Tensor) -> torch.Tensor:
            return zeros().index_add(0, events, values)

        empty_blocks = torch.zeros((block_masks.shape[0], event_count), device=device)
        if not weights.numel():
            return _LayerStatistics(
                *(zeros() for _ in range(9)),
                block_numerator=empty_blocks,
            )

        history_mask = relations == int(Relation.RY)
        observed_weight = scatter(weights)
        history_weight = scatter(weights * history_mask.float())
        if incidence == "mass_only":
            return self._mass_only_statistics(
                events=events,
                relations=relations,
                weights=weights,
                predictors=predictors,
                segment=segment,
                previous_direct=previous_direct,
                previous_direct_upper=previous_direct_upper,
                previous_support_upper=previous_support_upper,
                history_source=history_source,
                history_upper_source=history_upper_source,
                block_masks=block_masks,
                observed_weight=observed_weight,
                history_weight=history_weight,
            )

        direct_value = previous_direct.index_select(0, sources)
        direct_upper_value = previous_direct_upper.index_select(0, sources)
        support_upper_value = previous_support_upper.index_select(0, sources)
        history_value = history_source.index_select(0, sources)
        history_upper_value = history_upper_source.index_select(0, sources)
        non_history = (~history_mask).float()
        direct_numerator = scatter(weights * non_history * direct_value)
        direct_upper_numerator = scatter(weights * non_history * direct_upper_value)
        history_lower_numerator = scatter(
            weights * history_mask.float() * history_value
        )
        support_upper_numerator = scatter(weights * support_upper_value)
        history_upper_numerator = scatter(
            weights * history_mask.float() * history_upper_value
        )
        grounded_history_lower_numerator = scatter(
            weights * history_mask.float() * history_value
        )
        grounded_history_upper_numerator = scatter(
            weights * history_mask.float() * history_upper_value
        )
        block_numerator = empty_blocks
        if block_masks.shape[0]:
            block_rows = []
            for block_mask in block_masks:
                block_value = block_mask.index_select(
                    0, sources
                ) * history_source.index_select(0, sources)
                block_rows.append(scatter(weights * history_mask.float() * block_value))
            block_numerator = torch.stack(block_rows)
        return _LayerStatistics(
            observed_weight=observed_weight,
            history_weight=history_weight,
            direct_numerator=direct_numerator,
            direct_upper_numerator=direct_upper_numerator,
            history_lower_numerator=history_lower_numerator,
            support_upper_numerator=support_upper_numerator,
            history_upper_numerator=history_upper_numerator,
            grounded_history_lower_numerator=grounded_history_lower_numerator,
            grounded_history_upper_numerator=grounded_history_upper_numerator,
            block_numerator=block_numerator,
        )

    def _mass_only_statistics(
        self,
        *,
        events: torch.Tensor,
        relations: torch.Tensor,
        weights: torch.Tensor,
        predictors: torch.Tensor,
        segment: torch.Tensor,
        previous_direct: torch.Tensor,
        previous_direct_upper: torch.Tensor,
        previous_support_upper: torch.Tensor,
        history_source: torch.Tensor,
        history_upper_source: torch.Tensor,
        block_masks: torch.Tensor,
        observed_weight: torch.Tensor,
        history_weight: torch.Tensor,
    ) -> _LayerStatistics:
        """Remove incidence while preserving relation/layer/head mass."""

        event_count = predictors.numel()
        relation_mass = torch.zeros((len(Relation), event_count), device=segment.device)
        flat = relations * event_count + events
        relation_mass = (
            relation_mass.flatten().index_add(0, flat, weights).view_as(relation_mass)
        )

        def domain_mean(state: torch.Tensor, relation: Relation) -> torch.Tensor:
            if relation == Relation.EY:
                values = state[segment == int(Segment.EVIDENCE)]
                mean = values.mean() if values.numel() else state.new_tensor(0.0)
                return mean.expand(event_count)
            if relation == Relation.QY:
                values = state[segment == int(Segment.QUERY)]
                mean = values.mean() if values.numel() else state.new_tensor(0.0)
                return mean.expand(event_count)
            response = (segment == int(Segment.RESPONSE)).to(state.dtype)
            cumulative_value = torch.cumsum(state * response, dim=0)
            cumulative_count = torch.cumsum(response, dim=0).clamp_min(1.0)
            return cumulative_value.index_select(
                0, predictors
            ) / cumulative_count.index_select(0, predictors)

        evidence_mass = relation_mass[int(Relation.EY)]
        query_mass = relation_mass[int(Relation.QY)]
        response_mass = relation_mass[int(Relation.RY)]

        def non_history_message(state: torch.Tensor) -> torch.Tensor:
            return evidence_mass * domain_mean(
                state, Relation.EY
            ) + query_mass * domain_mean(state, Relation.QY)

        def all_relation_message(state: torch.Tensor) -> torch.Tensor:
            return non_history_message(state) + response_mass * domain_mean(
                state, Relation.RY
            )

        block_rows = [
            response_mass * domain_mean(history_source * mask, Relation.RY)
            for mask in block_masks
        ]
        block_numerator = (
            torch.stack(block_rows)
            if block_rows
            else torch.zeros((0, event_count), device=segment.device)
        )
        return _LayerStatistics(
            observed_weight=observed_weight,
            history_weight=history_weight,
            direct_numerator=non_history_message(previous_direct),
            direct_upper_numerator=non_history_message(previous_direct_upper),
            history_lower_numerator=response_mass
            * domain_mean(history_source, Relation.RY),
            support_upper_numerator=all_relation_message(previous_support_upper),
            history_upper_numerator=response_mass
            * domain_mean(history_upper_source, Relation.RY),
            grounded_history_lower_numerator=response_mass
            * domain_mean(history_source, Relation.RY),
            grounded_history_upper_numerator=response_mass
            * domain_mean(history_upper_source, Relation.RY),
            block_numerator=block_numerator,
        )

    @staticmethod
    def _rewire_sources(
        *,
        sources: torch.Tensor,
        relations: torch.Tensor,
        trace_events: torch.Tensor,
        trace_channel: torch.Tensor,
        predictors: torch.Tensor,
        segment: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        """Vectorize legal endpoint rotations on the tensor's current device."""

        if offset <= 0:
            raise ValueError("rewire offset must be positive")
        rewired = sources.clone()
        sequence_length = segment.numel()

        def rotate_fixed_domain(relation: Relation, domain_segment: Segment) -> None:
            mask = relations == int(relation)
            domain = torch.nonzero(segment == int(domain_segment)).flatten()
            if domain.numel() == 0:
                if bool(mask.any()):
                    raise ValueError("rewire relation has an empty source domain")
                return
            rank_by_position = torch.full(
                (sequence_length,), -1, dtype=torch.long, device=segment.device
            )
            rank_by_position[domain] = torch.arange(
                domain.numel(), dtype=torch.long, device=segment.device
            )
            ranks = rank_by_position.index_select(0, sources[mask])
            if bool((ranks < 0).any()):
                raise ValueError("trace source lies outside its legal relation domain")
            if domain.numel() == 1:
                return
            shift = 1 + (offset - 1) % (domain.numel() - 1)
            rewired[mask] = domain[(ranks + shift).remainder(domain.numel())]

        rotate_fixed_domain(Relation.EY, Segment.EVIDENCE)
        rotate_fixed_domain(Relation.QY, Segment.QUERY)

        response_indices = torch.nonzero(relations == int(Relation.RY)).flatten()
        if response_indices.numel() == 0:
            return rewired
        response_sources = sources.index_select(0, response_indices)
        events = trace_events.index_select(0, response_indices)
        event_predictors = predictors.index_select(0, events)
        lag = event_predictors - response_sources
        if bool(
            (
                (segment.index_select(0, response_sources) != int(Segment.RESPONSE))
                | (lag < 0)
            ).any()
        ):
            raise ValueError("trace source lies outside its legal response domain")

        # Preserve each target's lag regime. Source assignments are permuted
        # only among traces with the same target and floor(log2(lag + 1)).
        lag_bucket = torch.floor(torch.log2((lag + 1).float())).long()
        group_key = events * sequence_length + lag_bucket
        local_channel = trace_channel.index_select(0, response_indices)
        hash_key = torch.remainder(
            local_channel * 1_000_003
            + response_sources * 97_409
            + response_indices * 65_537,
            2_147_483_647,
        )
        hash_order = torch.argsort(hash_key, stable=True)
        order = hash_order[
            torch.argsort(group_key.index_select(0, hash_order), stable=True)
        ]
        sorted_indices = response_indices.index_select(0, order)
        sorted_groups = group_key.index_select(0, order)
        group_start = torch.ones(
            sorted_groups.numel(), dtype=torch.bool, device=segment.device
        )
        group_start[1:] = sorted_groups[1:] != sorted_groups[:-1]
        group_id = torch.cumsum(group_start.long(), dim=0) - 1
        group_size = torch.bincount(group_id)
        group_offset = torch.cumsum(group_size, dim=0) - group_size
        positions = torch.arange(sorted_groups.numel(), device=segment.device)
        within_group_rank = positions - group_offset.index_select(0, group_id)
        size_at_trace = group_size.index_select(0, group_id)
        span = (size_at_trace - 1).clamp_min(1)
        shift = 1 + torch.remainder(torch.full_like(span, offset - 1), span)
        donor_rank = group_offset.index_select(0, group_id) + torch.remainder(
            within_group_rank + shift, size_at_trace
        )
        sorted_sources = sources.index_select(0, sorted_indices)
        rewired[sorted_indices] = sorted_sources.index_select(0, donor_rank)
        return rewired

    def _validate_graph_vectors(
        self,
        *,
        segment: torch.Tensor,
        targets: torch.Tensor,
        predictors: torch.Tensor,
        row_available: torch.Tensor,
        edge_index: torch.Tensor,
        edge_relation: torch.Tensor,
        trace_edge: torch.Tensor,
        trace_channel: torch.Tensor,
        trace_value: torch.Tensor,
        unknown: torch.Tensor,
    ) -> None:
        if (
            targets.numel() == 0
            or predictors.shape != targets.shape
            or row_available.shape != targets.shape
            or not torch.equal(predictors, targets - 1)
        ):
            raise ValueError("graph prediction-event vectors are inconsistent")
        if bool(((predictors < 0) | (predictors >= segment.numel())).any()):
            raise ValueError("graph predictor positions are out of bounds")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.shape[1] != edge_relation.numel()
        ):
            raise ValueError("graph edge arrays are inconsistent")
        if not (trace_edge.numel() == trace_channel.numel() == trace_value.numel()):
            raise ValueError("graph trace arrays are inconsistent")
        if trace_edge.numel() and (
            int(trace_edge.min()) < 0
            or int(trace_edge.max()) >= edge_index.shape[1]
            or int(trace_channel.min()) < 0
            or int(trace_channel.max()) >= self.num_channels
        ):
            raise ValueError("graph trace indices are out of bounds")
        if edge_index.numel() and (
            int(edge_index[0].min()) < 0
            or int(edge_index[0].max()) >= segment.numel()
            or int(edge_index[1].min()) < 0
            or int(edge_index[1].max()) >= targets.numel()
        ):
            raise ValueError("graph edge endpoints are out of bounds")
        if unknown.shape != (targets.numel(), self.num_channels):
            raise ValueError("graph unknown_mass has an inconsistent shape")


def transport_loss(
    output: TransportOutput,
    targets: TeacherTargets,
    *,
    block_weight: float = 1.0,
) -> torch.Tensor:
    if not math.isfinite(block_weight) or block_weight < 0:
        raise ValueError("block_weight must be finite and non-negative")
    if (
        output.support_lower.shape != targets.support.shape
        or output.history_lower.shape != targets.history.shape
        or targets.event_weight.shape != targets.support.shape
        or targets.positive_mask.shape != targets.support.shape
        or targets.null_mask.shape != targets.support.shape
        or targets.contradictory_mask.shape != targets.support.shape
    ):
        raise ValueError("transport output and teacher event shapes disagree")
    device = output.direct.device
    available = output.row_available.to(device).bool()
    weight = targets.event_weight.to(device)
    positive = targets.positive_mask.to(device).bool() & available
    null = targets.null_mask.to(device).bool() & available
    contradictory = targets.contradictory_mask.to(device).bool() & available
    if bool(
        ((positive & null) | (positive & contradictory) | (null & contradictory)).any()
    ):
        raise ValueError("teacher event populations must be disjoint")
    support_target = targets.support.to(output.direct.device)
    history_target = targets.history.to(output.direct.device)
    event_error = F.smooth_l1_loss(
        output.support_lower, support_target, reduction="none"
    )
    event_error = event_error + F.smooth_l1_loss(
        output.history_lower, history_target, reduction="none"
    )
    population_losses: list[torch.Tensor] = []
    for population in (positive, null):
        population_weight = weight * population.float()
        denominator = population_weight.sum()
        if bool(denominator > 0):
            population_losses.append(
                (event_error * population_weight).sum() / denominator
            )
    loss = (
        torch.stack(population_losses).mean()
        if population_losses
        else event_error.sum() * 0.0
    )
    if output.block.numel() or targets.block.numel():
        if output.block.shape != targets.block.shape:
            raise ValueError("transport output and teacher block shapes disagree")
        mask = targets.block_mask.to(
            output.direct.device
        ) & output.row_available.unsqueeze(0)
        target_block = targets.block.to(output.direct.device)
        pair_losses: list[torch.Tensor] = []
        for event in range(output.block.shape[1]):
            visible = torch.nonzero(mask[:, event]).flatten()
            if visible.numel() < 2:
                continue
            pairs = torch.combinations(visible, r=2)
            target_difference = (
                target_block[pairs[:, 0], event] - target_block[pairs[:, 1], event]
            )
            informative = target_difference.abs() > BLOCK_RANKING_MARGIN
            if not bool(informative.any()):
                continue
            pairs = pairs[informative]
            direction = torch.sign(target_difference[informative])
            prediction_difference = (
                output.block[pairs[:, 0], event] - output.block[pairs[:, 1], event]
            )
            pair_losses.append(F.softplus(-direction * prediction_difference / 0.1))
        if pair_losses:
            loss = loss + block_weight * torch.cat(pair_losses).mean()
    return loss


def response_risk(token_risk: torch.Tensor, row_available: torch.Tensor) -> float:
    """Pre-registered top-10% CVaR over available response prediction events."""

    risk = torch.as_tensor(token_risk).detach().float().flatten()
    available = torch.as_tensor(row_available).detach().bool().flatten()
    if risk.shape != available.shape:
        raise ValueError("token risk and row availability must align")
    selected = risk[available]
    if selected.numel() == 0:
        return 0.0
    count = max(1, math.ceil(0.10 * selected.numel()))
    return float(torch.topk(selected, k=count).values.mean())


__all__ = [
    "BLOCK_RANKING_MARGIN",
    "CausalProvenanceTransport",
    "TransportOutput",
    "response_risk",
    "transport_loss",
]
