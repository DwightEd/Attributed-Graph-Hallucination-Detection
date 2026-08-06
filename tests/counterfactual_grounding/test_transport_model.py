from __future__ import annotations

import pytest
import torch

from counterfactual_grounding.transport.model import (
    CausalProvenanceTransport,
    response_risk,
)


def _chain_graph(*, first_unknown: bool = False) -> dict[str, object]:
    # Layer 0: evidence 0 -> response state 2.
    # Layer 1: response state 2 -> response state 3.
    # Layer 2: response state 3 -> response state 4.
    # This is impossible to represent by a layer-collapsed token recurrence.
    return {
        "schema": "cept-prediction-event-graph-v2",
        "num_layers": 3,
        "num_heads": 1,
        "segment_ids": torch.tensor([0, 1, 2, 2, 2, 2], dtype=torch.int8),
        "target_token_positions": torch.tensor([2, 3, 4, 5]),
        "predictor_positions": torch.tensor([1, 2, 3, 4]),
        "row_available": torch.tensor([False, True, True, True]),
        "edge_index": torch.tensor([[0, 2, 3], [1, 2, 3]]),
        "edge_relation": torch.tensor([0, 2, 2], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0, 1, 2]),
        "trace_channel": torch.tensor([0, 1, 2]),
        "trace_value": torch.tensor([1.0, 1.0, 1.0]),
        "unknown_mass": torch.tensor(
            [
                [1.0 if first_unknown else 0.0] * 3,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    }


def test_recursive_true_incidence_propagates_beyond_one_hop():
    model = CausalProvenanceTransport(num_layers=3, num_heads=1)
    graph = _chain_graph()

    true = model(
        graph,
        seed_positions=torch.tensor([0]),
        incidence="true",
        residual_mode="none",
    )
    one_hop = model(
        graph,
        seed_positions=torch.tensor([0]),
        incidence="one_hop",
        residual_mode="none",
    )

    assert true.support_lower.tolist() == pytest.approx([0.0, 1.0, 1.0, 1.0])
    assert one_hop.support_lower[-1].item() == pytest.approx(0.0)
    assert true.unsupported_history_lower[-1].item() == pytest.approx(0.0)


def test_censored_first_event_is_unknown_not_unsupported():
    model = CausalProvenanceTransport(num_layers=3, num_heads=1)

    output = model(
        _chain_graph(first_unknown=True),
        seed_positions=torch.tensor([0]),
        incidence="true",
    )

    assert output.support_lower[0].item() == pytest.approx(0.0)
    assert output.support_upper[0].item() == pytest.approx(1.0)
    assert output.token_risk[0].item() == pytest.approx(0.0)
    assert response_risk(
        torch.tensor([1.0, 0.0]), torch.tensor([False, True])
    ) == pytest.approx(0.0)


def test_diagonal_response_attention_uses_same_token_previous_layer_state():
    graph = {
        "schema": "cept-prediction-event-graph-v2",
        "num_layers": 2,
        "num_heads": 1,
        "segment_ids": torch.tensor([0, 1, 2, 2], dtype=torch.int8),
        "target_token_positions": torch.tensor([2, 3]),
        "predictor_positions": torch.tensor([1, 2]),
        "row_available": torch.tensor([False, True]),
        # At layer 0 evidence grounds response state 2. At layer 1 the
        # diagonal source 2 must read that exact previous-layer state.
        "edge_index": torch.tensor([[0, 2], [1, 1]]),
        "edge_relation": torch.tensor([0, 2], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0, 1]),
        "trace_channel": torch.tensor([0, 1]),
        "trace_value": torch.tensor([1.0, 1.0]),
        "unknown_mass": torch.zeros((2, 2)),
    }
    model = CausalProvenanceTransport(num_layers=2, num_heads=1)

    output = model(
        graph,
        seed_positions=torch.tensor([0]),
        incidence="true",
        residual_mode="none",
    )

    assert output.support_lower.tolist() == pytest.approx([0.0, 1.0])
    assert output.history_lower.tolist() == pytest.approx([0.0, 1.0])
    assert output.unsupported_history_lower[1].item() == pytest.approx(0.0)


def test_layer_order_rejects_an_impossible_reverse_depth_path():
    graph = {
        "schema": "cept-prediction-event-graph-v2",
        "num_layers": 2,
        "num_heads": 1,
        "segment_ids": torch.tensor([0, 1, 2, 2, 2], dtype=torch.int8),
        "target_token_positions": torch.tensor([2, 3, 4]),
        "predictor_positions": torch.tensor([1, 2, 3]),
        "row_available": torch.tensor([False, True, True]),
        # Layer 0 asks state 3 to read ungrounded state 2. Evidence reaches
        # state 2 only at layer 1, too late to traverse the earlier edge.
        "edge_index": torch.tensor([[2, 0], [2, 1]]),
        "edge_relation": torch.tensor([2, 0], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0, 1]),
        "trace_channel": torch.tensor([0, 1]),
        "trace_value": torch.tensor([1.0, 1.0]),
        "unknown_mass": torch.zeros((3, 2)),
    }
    model = CausalProvenanceTransport(num_layers=2, num_heads=1)

    output = model(
        graph,
        seed_positions=torch.tensor([0]),
        incidence="true",
        residual_mode="none",
    )

    assert output.support_lower[1].item() == pytest.approx(1.0)
    assert output.support_lower[2].item() == pytest.approx(0.0)


def test_mass_only_preserves_relation_mass_but_removes_endpoint_identity():
    def graph(source: int) -> dict[str, object]:
        return {
            "schema": "cept-prediction-event-graph-v2",
            "num_layers": 1,
            "num_heads": 1,
            "segment_ids": torch.tensor([0, 0, 1, 2], dtype=torch.int8),
            "target_token_positions": torch.tensor([3]),
            "predictor_positions": torch.tensor([2]),
            "row_available": torch.tensor([True]),
            "edge_index": torch.tensor([[source], [0]]),
            "edge_relation": torch.tensor([0], dtype=torch.int8),
            "trace_edge_id": torch.tensor([0]),
            "trace_channel": torch.tensor([0]),
            "trace_value": torch.tensor([1.0]),
            "unknown_mass": torch.zeros((1, 1)),
        }

    model = CausalProvenanceTransport(num_layers=1, num_heads=1)
    true_seed = model(
        graph(0),
        seed_positions=torch.tensor([0]),
        incidence="true",
        residual_mode="none",
    )
    true_other = model(
        graph(1),
        seed_positions=torch.tensor([0]),
        incidence="true",
        residual_mode="none",
    )
    mass_seed = model(
        graph(0),
        seed_positions=torch.tensor([0]),
        incidence="mass_only",
        residual_mode="none",
    )
    mass_other = model(
        graph(1),
        seed_positions=torch.tensor([0]),
        incidence="mass_only",
        residual_mode="none",
    )

    assert true_seed.support_lower.item() == pytest.approx(1.0)
    assert true_other.support_lower.item() == pytest.approx(0.0)
    assert mass_seed.support_lower.item() == pytest.approx(0.5)
    assert mass_other.support_lower.item() == pytest.approx(0.5)


def test_student_parameters_are_only_low_capacity_transport_gates():
    model = CausalProvenanceTransport(num_layers=2, num_heads=3)

    assert sum(parameter.numel() for parameter in model.parameters()) == 20
    assert set(dict(model.named_parameters())) == {
        "raw_relation_channel_gates",
        "raw_residual_persistence",
    }


def test_inference_uses_all_evidence_as_one_union_seed_without_max_pooling():
    graph = {
        "schema": "cept-prediction-event-graph-v2",
        "num_layers": 1,
        "num_heads": 1,
        "segment_ids": torch.tensor([0, 0, 1, 2, 2], dtype=torch.int8),
        "target_token_positions": torch.tensor([3, 4]),
        "predictor_positions": torch.tensor([2, 3]),
        "row_available": torch.tensor([True, True]),
        "edge_index": torch.tensor([[0, 1], [0, 1]]),
        "edge_relation": torch.tensor([0, 0], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0, 1]),
        "trace_channel": torch.tensor([0, 0]),
        "trace_value": torch.tensor([1.0, 1.0]),
        "unknown_mass": torch.zeros((2, 1)),
    }
    model = CausalProvenanceTransport(num_layers=1, num_heads=1)

    union = model(
        graph,
        seed_positions=torch.tensor([0, 1]),
        incidence="true",
        residual_mode="none",
    )

    assert union.direct.tolist() == pytest.approx([1.0, 1.0])


def test_learned_residual_persistence_keeps_prior_layer_support():
    graph = {
        "schema": "cept-prediction-event-graph-v2",
        "num_layers": 2,
        "num_heads": 1,
        "segment_ids": torch.tensor([1, 0, 2, 2], dtype=torch.int8),
        "target_token_positions": torch.tensor([3]),
        "predictor_positions": torch.tensor([2]),
        "row_available": torch.tensor([True]),
        # Layer 0 grounds response state 2 from evidence 1. Layer 1 reads an
        # ungrounded query before that seed, so only residual persistence can
        # retain the layer-0 support.
        "edge_index": torch.tensor([[1, 0], [0, 0]]),
        "edge_relation": torch.tensor([0, 1], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0, 1]),
        "trace_channel": torch.tensor([0, 1]),
        "trace_value": torch.tensor([1.0, 1.0]),
        "unknown_mass": torch.zeros((1, 2)),
    }
    model = CausalProvenanceTransport(num_layers=2, num_heads=1)

    learned = model(
        graph,
        seed_positions=torch.tensor([1]),
        residual_mode="learned",
    )
    none = model(
        graph,
        seed_positions=torch.tensor([1]),
        residual_mode="none",
    )

    assert learned.support_lower.item() == pytest.approx(0.25)
    assert none.support_lower.item() == pytest.approx(0.0)
    learned.support_lower.sum().backward()
    gradient = model.raw_residual_persistence.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient[1].abs().item() > 0.0


@pytest.mark.parametrize(
    ("segments", "seed", "query", "expected_upper"),
    [
        ([0, 1, 2, 2], 0, 1, 1.0),
        ([1, 0, 2, 2], 1, 0, 0.0),
    ],
)
def test_unobserved_prompt_state_upper_respects_layer_and_causal_order(
    segments: list[int], seed: int, query: int, expected_upper: float
):
    graph = {
        "schema": "cept-prediction-event-graph-v2",
        "num_layers": 2,
        "num_heads": 1,
        "segment_ids": torch.tensor(segments, dtype=torch.int8),
        "target_token_positions": torch.tensor([3]),
        "predictor_positions": torch.tensor([2]),
        "row_available": torch.tensor([True]),
        "edge_index": torch.tensor([[query], [0]]),
        "edge_relation": torch.tensor([1], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0, 0]),
        "trace_channel": torch.tensor([0, 1]),
        "trace_value": torch.tensor([1.0, 1.0]),
        "unknown_mass": torch.zeros((1, 2)),
    }
    model = CausalProvenanceTransport(num_layers=2, num_heads=1)

    output = model(
        graph,
        seed_positions=torch.tensor([seed]),
        residual_mode="none",
    )

    assert output.support_lower.item() == pytest.approx(0.0)
    assert output.support_upper.item() == pytest.approx(expected_upper)


def test_one_layer_cannot_use_unobserved_prompt_attention_from_the_same_layer():
    graph = {
        "schema": "cept-prediction-event-graph-v2",
        "num_layers": 1,
        "num_heads": 1,
        "segment_ids": torch.tensor([0, 1, 2, 2], dtype=torch.int8),
        "target_token_positions": torch.tensor([3]),
        "predictor_positions": torch.tensor([2]),
        "row_available": torch.tensor([True]),
        "edge_index": torch.tensor([[1], [0]]),
        "edge_relation": torch.tensor([1], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0]),
        "trace_channel": torch.tensor([0]),
        "trace_value": torch.tensor([1.0]),
        "unknown_mass": torch.zeros((1, 1)),
    }
    model = CausalProvenanceTransport(num_layers=1, num_heads=1)

    output = model(
        graph,
        seed_positions=torch.tensor([0]),
        residual_mode="none",
    )

    assert output.support_upper.item() == pytest.approx(0.0)


def test_rewired_views_recur_independently_without_cross_view_paths(monkeypatch):
    graph = {
        "schema": "cept-prediction-event-graph-v2",
        "num_layers": 2,
        "num_heads": 1,
        "segment_ids": torch.tensor(
            [0, 0, 0, 0, 1, 2, 2, 2, 2, 2], dtype=torch.int8
        ),
        "target_token_positions": torch.tensor([5, 6, 7, 8, 9]),
        "predictor_positions": torch.tensor([4, 5, 6, 7, 8]),
        "row_available": torch.tensor([False, True, True, True, True]),
        # For offsets 1/2/3, layer-0 source 3 rewires to evidence 0/1/2,
        # so only view 1 grounds response state 5. Layer-1 source 7 rewires
        # to response 8/5/6, so only view 2 reads state 5. No coherent view
        # contains the full seed -> 5 -> 8 path.
        "edge_index": torch.tensor([[3, 7], [1, 4]]),
        "edge_relation": torch.tensor([0, 2], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0, 1]),
        "trace_channel": torch.tensor([0, 1]),
        "trace_value": torch.tensor([1.0, 1.0]),
        "unknown_mass": torch.zeros((5, 2)),
    }
    model = CausalProvenanceTransport(num_layers=2, num_heads=1)

    def controlled_rewire(*, sources, offset, **_):
        mappings = {
            1: torch.tensor([0, 8], device=sources.device),
            2: torch.tensor([1, 5], device=sources.device),
            3: torch.tensor([2, 6], device=sources.device),
        }
        return mappings[offset]

    monkeypatch.setattr(
        CausalProvenanceTransport,
        "_rewire_sources",
        staticmethod(controlled_rewire),
    )

    output = model(
        graph,
        seed_positions=torch.tensor([0]),
        incidence="rewired",
        residual_mode="none",
    )

    assert output.support_lower[4].item() == pytest.approx(0.0)


def test_rewire_source_rotation_is_vectorized_and_preserves_legal_domains():
    segment = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2, 2], dtype=torch.long)
    sources = torch.tensor([1, 2, 6, 7], dtype=torch.long)
    relations = torch.tensor([0, 1, 2, 2], dtype=torch.long)
    trace_events = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    trace_channel = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    predictors = torch.tensor([7, 8], dtype=torch.long)

    rewired = CausalProvenanceTransport._rewire_sources(
        sources=sources,
        relations=relations,
        trace_events=trace_events,
        trace_channel=trace_channel,
        predictors=predictors,
        segment=segment,
        offset=1,
    )

    assert rewired[:2].tolist() == [0, 3]
    assert sorted(rewired[2:].tolist()) == [6, 7]
    assert rewired[2:].tolist() != sources[2:].tolist()
    original_lag_bucket = torch.floor(
        torch.log2((predictors[1] - sources[2:] + 1).float())
    )
    rewired_lag_bucket = torch.floor(
        torch.log2((predictors[1] - rewired[2:] + 1).float())
    )
    assert torch.equal(original_lag_bucket, rewired_lag_bucket)


def test_rewire_diagnostics_reuses_three_views_without_mutating_graph():
    graph = {
        "schema": "cept-prediction-event-graph-v2",
        "num_layers": 2,
        "num_heads": 2,
        "segment_ids": torch.tensor(
            [0, 0, 1, 1, 2, 2, 2, 2, 2, 2], dtype=torch.int8
        ),
        "target_token_positions": torch.tensor([8, 9]),
        "predictor_positions": torch.tensor([7, 8]),
        "row_available": torch.tensor([True, True]),
        "edge_index": torch.tensor([[1, 2, 6, 7], [0, 0, 1, 1]]),
        "edge_relation": torch.tensor([0, 1, 2, 2], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0, 1, 2, 3]),
        "trace_channel": torch.tensor([0, 1, 2, 3]),
        "trace_value": torch.ones(4),
        "unknown_mass": torch.zeros((2, 4)),
    }
    original_edges = graph["edge_index"].clone()
    model = CausalProvenanceTransport(num_layers=2, num_heads=2)

    report = model.rewire_diagnostics(graph)

    assert len(report["views"]) == 3
    assert report["changed_trace_fraction"] == pytest.approx(1.0)
    assert report["ry_changed_trace_fraction"] == pytest.approx(1.0)
    assert all(
        view["changed_trace_fraction"] == pytest.approx(1.0)
        and view["ry_changed_trace_fraction"] == pytest.approx(1.0)
        for view in report["views"]
    )
    assert torch.equal(graph["edge_index"], original_edges)


def test_learned_residual_scales_new_unsupported_attention_exposure():
    graph = {
        "schema": "cept-prediction-event-graph-v2",
        "num_layers": 1,
        "num_heads": 1,
        "segment_ids": torch.tensor([0, 1, 2, 2, 2], dtype=torch.int8),
        "target_token_positions": torch.tensor([4]),
        "predictor_positions": torch.tensor([3]),
        "row_available": torch.tensor([True]),
        "edge_index": torch.tensor([[2], [0]]),
        "edge_relation": torch.tensor([2], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0]),
        "trace_channel": torch.tensor([0]),
        "trace_value": torch.tensor([1.0]),
        "unknown_mass": torch.zeros((1, 1)),
    }
    model = CausalProvenanceTransport(num_layers=1, num_heads=1)

    learned = model(
        graph, seed_positions=torch.tensor([0]), residual_mode="learned"
    )
    none = model(graph, seed_positions=torch.tensor([0]), residual_mode="none")

    assert learned.token_risk.item() == pytest.approx(0.5)
    assert none.token_risk.item() == pytest.approx(1.0)


def test_learned_residual_scales_new_history_block_contribution():
    graph = {
        "schema": "cept-prediction-event-graph-v2",
        "num_layers": 2,
        "num_heads": 1,
        "segment_ids": torch.tensor([0, 1, 2, 2], dtype=torch.int8),
        "target_token_positions": torch.tensor([2, 3]),
        "predictor_positions": torch.tensor([1, 2]),
        "row_available": torch.tensor([False, True]),
        "edge_index": torch.tensor([[0, 2], [1, 1]]),
        "edge_relation": torch.tensor([0, 2], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0, 1]),
        "trace_channel": torch.tensor([0, 1]),
        "trace_value": torch.tensor([1.0, 1.0]),
        "unknown_mass": torch.zeros((2, 2)),
    }
    model = CausalProvenanceTransport(num_layers=2, num_heads=1)

    learned = model(
        graph,
        seed_positions=torch.tensor([0]),
        block_positions={"relay": [2]},
        residual_mode="learned",
    )
    none = model(
        graph,
        seed_positions=torch.tensor([0]),
        block_positions={"relay": [2]},
        residual_mode="none",
    )

    assert learned.block[0, 1].item() == pytest.approx(0.125)
    assert none.block[0, 1].item() == pytest.approx(0.5)


def test_one_hop_counts_censored_mass_once_in_support_upper():
    graph = {
        "schema": "cept-prediction-event-graph-v2",
        "num_layers": 1,
        "num_heads": 1,
        "segment_ids": torch.tensor([0, 1, 2, 2], dtype=torch.int8),
        "target_token_positions": torch.tensor([3]),
        "predictor_positions": torch.tensor([2]),
        "row_available": torch.tensor([True]),
        "edge_index": torch.tensor([[1], [0]]),
        "edge_relation": torch.tensor([1], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0]),
        "trace_channel": torch.tensor([0]),
        "trace_value": torch.tensor([0.5]),
        "unknown_mass": torch.tensor([[0.5]]),
    }
    model = CausalProvenanceTransport(num_layers=1, num_heads=1)

    output = model(
        graph,
        seed_positions=torch.tensor([0]),
        incidence="one_hop",
        residual_mode="none",
    )

    assert output.support_upper.item() == pytest.approx(0.5)


def test_censored_current_edges_do_not_double_reduce_observed_history_lower():
    graph = {
        "schema": "cept-prediction-event-graph-v2",
        "num_layers": 2,
        "num_heads": 1,
        "segment_ids": torch.tensor([0, 1, 2, 2, 2], dtype=torch.int8),
        "target_token_positions": torch.tensor([3, 4]),
        "predictor_positions": torch.tensor([2, 3]),
        "row_available": torch.tensor([True, True]),
        # Layer 0 gives response source 2 support_upper=0.5. Layer 1 sends
        # observed RY mass H=0.6 through it, with current unknown U=0.2.
        "edge_index": torch.tensor([[0, 1, 2], [0, 0, 1]]),
        "edge_relation": torch.tensor([0, 1, 2], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0, 1, 2]),
        "trace_channel": torch.tensor([0, 0, 1]),
        "trace_value": torch.tensor([0.5, 0.5, 0.6]),
        "unknown_mass": torch.tensor([[0.0, 0.0], [0.0, 0.2]]),
    }
    model = CausalProvenanceTransport(num_layers=2, num_heads=1)

    output = model(
        graph, seed_positions=torch.tensor([0]), residual_mode="none"
    )

    assert output.token_risk[1].item() == pytest.approx(0.375)
