from __future__ import annotations

import bisect
import unittest
from dataclasses import replace
from unittest import mock

import torch

import grounding_flow.method as flow_method

from attention_graph.graph import GraphBuildConfig, build_attention_graph
from grounding_flow.method import (
    NullModelConfig,
    calibrate_against_null,
    compute_evidence_flow,
    rewire_source_identity,
)


def _flow_graph():
    """Two-layer graph where token 3 inherits evidence through token 2."""

    layers, heads, tokens, response_idx = 2, 1, 4, 2
    # CSR rows are ordered by (layer, head, response query).  Diagonal entries
    # are stored separately, so every listed source is strictly historical.
    rows = [
        [(0, 0.8), (1, 0.1)],       # layer 0, response token 2
        [(1, 0.1), (2, 0.8)],       # layer 0, response token 3
        [(0, 0.8), (1, 0.1)],       # layer 1, response token 2
        [(1, 0.1), (2, 0.8)],       # layer 1, response token 3
    ]
    row_ptr = [0]
    columns: list[int] = []
    values: list[float] = []
    for row in rows:
        columns.extend(source for source, _ in row)
        values.extend(value for _, value in row)
        row_ptr.append(len(columns))
    diagonal = torch.zeros((layers, heads, tokens), dtype=torch.float32)
    diagonal[:, :, response_idx:] = 0.1
    sample = {
        "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
        "num_attention_layers": layers,
        "num_attention_heads": heads,
        "attention_diagonal": diagonal,
        "response_idx": response_idx,
        "response_row_ptr": torch.tensor(row_ptr),
        "response_column_indices": torch.tensor(columns),
        "response_values": torch.tensor(values),
        "attention_floor": 0.01,
        "token_ids": torch.arange(tokens),
        "source_id": "pair-1",
        "response_id": "response-1",
        "sample_id": "response-1",
    }
    graph = build_attention_graph(
        sample,
        GraphBuildConfig(
            selection="threshold", threshold=0.01, max_edges_per_target=None
        ),
    )
    return graph, torch.tensor([1, 2, 3, 3], dtype=torch.long)


def _rewirable_graph():
    layers, heads, tokens, response_idx = 2, 1, 8, 4
    base_rows = [
        [(0, 0.3), (1, 0.3), (2, 0.1), (3, 0.1)],
        [(0, 0.1), (1, 0.1), (2, 0.3), (3, 0.3), (4, 0.1)],
        [(0, 0.2), (1, 0.2), (2, 0.2), (3, 0.2), (4, 0.1), (5, 0.05)],
        [(0, 0.1), (1, 0.1), (2, 0.1), (3, 0.1), (4, 0.15), (5, 0.15), (6, 0.1)],
    ]
    rows = base_rows * layers
    row_ptr = [0]
    columns: list[int] = []
    values: list[float] = []
    for row in rows:
        columns.extend(source for source, _ in row)
        values.extend(value for _, value in row)
        row_ptr.append(len(columns))
    diagonal = torch.zeros((layers, heads, tokens), dtype=torch.float32)
    for layer in range(layers):
        for query, row in enumerate(base_rows, start=response_idx):
            diagonal[layer, 0, query] = 1.0 - sum(value for _, value in row)
    sample = {
        "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
        "num_attention_layers": layers,
        "num_attention_heads": heads,
        "attention_diagonal": diagonal,
        "response_idx": response_idx,
        "response_row_ptr": torch.tensor(row_ptr),
        "response_column_indices": torch.tensor(columns),
        "response_values": torch.tensor(values),
        "attention_floor": 0.01,
        "token_ids": torch.arange(tokens),
        "source_id": "pair-2",
        "response_id": "response-2",
        "sample_id": "response-2",
    }
    graph = build_attention_graph(
        sample,
        GraphBuildConfig(
            selection="threshold", threshold=0.01, max_edges_per_target=None
        ),
    )
    return graph, torch.tensor([1, 1, 2, 2, 3, 3, 3, 3], dtype=torch.long)


def _cross_bin_rewirable_graph():
    """RR edges in distinct lag bins that still admit legal cross-bin swaps."""

    layers, heads, tokens, response_idx = 1, 1, 17, 4
    response_edges = {
        8: (4, 0.4),  # lag 4 -> bin 0
        11: (6, 0.4),  # lag 5 -> bin 1
        16: (7, 0.4),  # lag 9 -> bin 2
    }
    rows = [
        [response_edges[query]] if query in response_edges else [(0, 0.2)]
        for query in range(response_idx, tokens)
    ]
    row_ptr = [0]
    columns: list[int] = []
    values: list[float] = []
    for row in rows:
        columns.extend(source for source, _ in row)
        values.extend(value for _, value in row)
        row_ptr.append(len(columns))
    diagonal = torch.zeros((layers, heads, tokens), dtype=torch.float32)
    for query, row in enumerate(rows, start=response_idx):
        diagonal[0, 0, query] = 1.0 - sum(value for _, value in row)
    sample = {
        "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
        "num_attention_layers": layers,
        "num_attention_heads": heads,
        "attention_diagonal": diagonal,
        "response_idx": response_idx,
        "response_row_ptr": torch.tensor(row_ptr),
        "response_column_indices": torch.tensor(columns),
        "response_values": torch.tensor(values),
        "attention_floor": 0.01,
        "token_ids": torch.arange(tokens),
        "source_id": "pair-cross-bin",
        "response_id": "response-cross-bin",
        "sample_id": "response-cross-bin",
    }
    graph = build_attention_graph(
        sample,
        GraphBuildConfig(
            selection="threshold", threshold=0.01, max_edges_per_target=None
        ),
    )
    return graph, torch.tensor([1, 2, 2, 2] + [3] * 13, dtype=torch.long)


class EvidenceFlowTests(unittest.TestCase):
    def test_flow_conserves_each_attention_row_and_detects_grounded_relay(self):
        graph, segment_ids = _flow_graph()

        result = compute_evidence_flow(graph, segment_ids)

        torch.testing.assert_close(
            result.values.sum(dim=-1),
            torch.ones_like(result.values[..., 0]),
            atol=1e-6,
            rtol=0.0,
        )
        # At layer 1 token 3 attends token 2 with weight .8.  Token 2 carried
        # .8 evidence ancestry from layer 0, so the legitimate relay is .64.
        self.assertAlmostEqual(
            float(result.component("grounded_relay")[1, 1, 0]), 0.64, places=6
        )
        self.assertAlmostEqual(
            float(result.component("ungrounded_relay")[1, 1, 0]), 0.16, places=6
        )
        self.assertAlmostEqual(float(result.ancestry[1, 1, 0]), 0.64, places=6)

    def test_small_float16_row_overage_is_corrected_without_hiding_missing_mass(self):
        graph, segment_ids = _flow_graph()
        diagonal = graph.node_attr.clone()
        diagonal[graph.response_idx :] += 1e-3
        rounded = replace(graph, node_attr=diagonal)

        result = compute_evidence_flow(rounded, segment_ids)

        torch.testing.assert_close(
            result.values.sum(dim=-1),
            torch.ones_like(result.values[..., 0]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_source_rewiring_preserves_marginals_and_changes_identity(self):
        graph, segment_ids = _rewirable_graph()

        shuffled, report = rewire_source_identity(
            graph,
            segment_ids,
            config=NullModelConfig(swaps_per_edge=8, lag_boundaries=(4, 8, 16)),
            generator=torch.Generator().manual_seed(17),
        )

        self.assertGreater(report.accepted_swaps, 0)
        self.assertFalse(torch.equal(graph.edge_index[0], shuffled.edge_index[0]))
        self.assertTrue(torch.equal(graph.edge_index[1], shuffled.edge_index[1]))
        self.assertTrue(torch.equal(graph.edge_type, shuffled.edge_type))
        self.assertTrue(torch.equal(graph.trace_edge_id, shuffled.trace_edge_id))
        self.assertTrue(torch.equal(graph.trace_channel, shuffled.trace_channel))
        self.assertTrue(torch.equal(graph.trace_value, shuffled.trace_value))
        self.assertEqual(
            sorted(graph.edge_index[0].tolist()),
            sorted(shuffled.edge_index[0].tolist()),
        )
        self.assertTrue(bool((shuffled.edge_index[0] < shuffled.edge_index[1]).all()))
        before_segments = segment_ids[graph.edge_index[0]]
        after_segments = segment_ids[shuffled.edge_index[0]]
        self.assertTrue(torch.equal(before_segments, after_segments))

    def test_source_rewiring_considers_legal_cross_bin_edge_pairs(self):
        graph, segment_ids = _cross_bin_rewirable_graph()
        boundaries = (4, 8)

        shuffled, report = rewire_source_identity(
            graph,
            segment_ids,
            config=NullModelConfig(
                swaps_per_edge=1, lag_boundaries=boundaries
            ),
            generator=torch.Generator().manual_seed(0),
        )

        self.assertGreater(report.attempts, 0)
        self.assertGreater(report.accepted_swaps, 0)
        self.assertFalse(torch.equal(graph.edge_index[0], shuffled.edge_index[0]))
        self.assertTrue(torch.equal(graph.edge_index[1], shuffled.edge_index[1]))
        self.assertEqual(
            sorted(graph.edge_index[0].tolist()),
            sorted(shuffled.edge_index[0].tolist()),
        )
        self.assertTrue(bool((shuffled.edge_index[0] < shuffled.edge_index[1]).all()))
        for edge_id in range(graph.num_edges):
            target = int(graph.edge_index[1, edge_id])
            before_lag = target - int(graph.edge_index[0, edge_id])
            after_lag = target - int(shuffled.edge_index[0, edge_id])
            self.assertEqual(
                bisect.bisect_left(boundaries, before_lag),
                bisect.bisect_left(boundaries, after_lag),
            )
        for field in (
            "node_attr",
            "node_context",
            "response_mask",
            "edge_type",
            "edge_score",
            "trace_edge_id",
            "trace_channel",
            "trace_value",
            "token_ids",
        ):
            with self.subTest(field=field):
                self.assertTrue(
                    torch.equal(getattr(graph, field), getattr(shuffled, field))
                )

    def test_null_calibration_is_finite_and_keeps_invariant_direct_mass_at_zero(self):
        graph, segment_ids = _rewirable_graph()

        calibrated = calibrate_against_null(
            graph,
            segment_ids,
            num_nulls=4,
            seed=23,
            null_config=NullModelConfig(
                swaps_per_edge=8, lag_boundaries=(4, 8, 16)
            ),
        )

        self.assertEqual(calibrated.null_samples, 4)
        self.assertTrue(bool(torch.isfinite(calibrated.z_scores).all()))
        direct = calibrated.component_z("direct_evidence")
        torch.testing.assert_close(direct, torch.zeros_like(direct))
        self.assertGreater(calibrated.accepted_swap_fraction, 0.0)
        self.assertGreater(calibrated.effective_null_fraction, 0.0)
        self.assertEqual(calibrated.calibration_status, "complete")

    def test_calibration_preindexes_static_graph_structure_once(self):
        graph, segment_ids = _rewirable_graph()

        with mock.patch.object(
            flow_method,
            "_build_rewire_plan",
            wraps=flow_method._build_rewire_plan,
        ) as rewire_plan, mock.patch.object(
            flow_method,
            "_build_layer_trace_indices",
            wraps=flow_method._build_layer_trace_indices,
        ) as trace_plan:
            calibrate_against_null(
                graph,
                segment_ids,
                num_nulls=4,
                seed=29,
                null_config=NullModelConfig(
                    swaps_per_edge=8, lag_boundaries=(4, 8, 16)
                ),
            )

        self.assertEqual(rewire_plan.call_count, 1)
        self.assertEqual(trace_plan.call_count, 1)

    def test_unswappable_response_graph_is_explicit_not_fake_null_data(self):
        graph, segment_ids = _flow_graph()

        calibrated = calibrate_against_null(
            graph,
            segment_ids,
            num_nulls=4,
            seed=5,
            null_config=NullModelConfig(swaps_per_edge=2, lag_boundaries=(4, 8)),
        )

        self.assertEqual(calibrated.null_samples, 0)
        self.assertEqual(calibrated.effective_null_fraction, 0.0)
        self.assertEqual(calibrated.calibration_status, "unswappable")
        torch.testing.assert_close(
            calibrated.model_tensor(), torch.zeros_like(calibrated.model_tensor())
        )

    def test_duplicate_null_draws_do_not_fake_sample_size(self):
        graph, segment_ids = _rewirable_graph()
        shuffled, report = rewire_source_identity(
            graph,
            segment_ids,
            config=NullModelConfig(swaps_per_edge=8, lag_boundaries=(4, 8, 16)),
            generator=torch.Generator().manual_seed(37),
        )
        self.assertTrue(report.changed)

        with mock.patch.object(
            flow_method,
            "rewire_source_identity",
            return_value=(shuffled, report),
        ):
            calibrated = calibrate_against_null(
                graph,
                segment_ids,
                num_nulls=4,
                seed=41,
                null_config=NullModelConfig(
                    swaps_per_edge=8,
                    lag_boundaries=(4, 8, 16),
                    max_null_attempt_factor=2,
                ),
            )

        self.assertEqual(calibrated.null_samples, 1)
        self.assertGreater(calibrated.duplicate_null_draws, 0)
        self.assertEqual(calibrated.calibration_status, "insufficient_unique_nulls")


if __name__ == "__main__":
    unittest.main()
