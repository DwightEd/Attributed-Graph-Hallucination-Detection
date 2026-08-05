"""Gate 0 contracts for CEPT prediction-event graph construction.

These tests deliberately target a new package.  They are the RED phase of
TDD: production code must not be added here merely to make collection pass.
"""

from __future__ import annotations

import unittest

import torch

from counterfactual_grounding.data.graph import (
    Relation,
    Segment,
    build_prediction_event_graph,
)


def _formal_sparse_fixture() -> dict[str, object]:
    """Return a one-channel cache whose response rows have observable shifts.

    Positions 0--1 are query, 2--3 are evidence, and 4--6 are response.
    The formal cache stores query rows for response positions 4, 5, and 6.
    For next-token prediction, however, those rows predict target positions
    5, 6, and 7.  Consequently row 0 belongs to event 1, row 1 belongs to
    event 2, and row 2 must not appear in this sample's graph.
    """

    return {
        "attention_cache_schema": (
            "ragtruth-all-layers-all-heads-sparse-response-csr-v1"
        ),
        "attention_cache_fingerprint": "prediction-event-fixture-v1",
        "source_id": "source-1",
        "response_id": "response-1",
        "sample_id": "response-1",
        "split": "train",
        "response_idx": 4,
        "num_attention_layers": 1,
        "num_attention_heads": 1,
        "attention_floor": 0.05,
        "token_ids": torch.tensor(
            [101, 102, 201, 202, 301, 302, 303], dtype=torch.long
        ),
        "segment_ids": torch.tensor(
            [
                Segment.QUERY,
                Segment.QUERY,
                Segment.EVIDENCE,
                Segment.EVIDENCE,
                Segment.RESPONSE,
                Segment.RESPONSE,
                Segment.RESPONSE,
            ],
            dtype=torch.int8,
        ),
        # The predictor's diagonal is visible context for its next token and
        # therefore becomes a predictor-token -> event edge.
        "attention_diagonal": torch.tensor(
            [[[0.01, 0.01, 0.01, 0.01, 0.35, 0.25, 0.25]]],
            dtype=torch.float32,
        ),
        # row(position=4): retained=.10+.20+.15+.35=.80
        # row(position=5): retained=.10+.15+.20+.25=.70
        # row(position=6): sentinel=.75+.25=1.00; it predicts position 7 and
        # must not be consumed for any event in this response.
        "response_row_ptr": torch.tensor([0, 3, 6, 7], dtype=torch.long),
        "response_column_indices": torch.tensor(
            [0, 2, 3, 1, 2, 4, 0], dtype=torch.int32
        ),
        "response_values": torch.tensor(
            [0.10, 0.20, 0.15, 0.10, 0.15, 0.20, 0.75],
            dtype=torch.float32,
        ),
    }


class PredictionEventGraphGate0Tests(unittest.TestCase):
    def test_target_tokens_use_the_previous_position_as_predictor(self):
        graph = build_prediction_event_graph(_formal_sparse_fixture())

        self.assertEqual(graph.target_token_ids.tolist(), [301, 302, 303])
        self.assertEqual(graph.target_token_positions.tolist(), [4, 5, 6])
        self.assertEqual(graph.predictor_positions.tolist(), [3, 4, 5])
        self.assertEqual(graph.row_available.tolist(), [False, True, True])

        # The final cached row belongs to a token after this response.  Its
        # sentinel value makes an off-by-one implementation observable.
        self.assertNotIn(0.75, graph.trace_value.tolist())

    def test_first_response_event_is_preserved_when_its_prompt_row_is_absent(self):
        graph = build_prediction_event_graph(_formal_sparse_fixture())

        first_event_edges = graph.edge_index[1] == 0
        self.assertFalse(bool(first_event_edges.any()))
        torch.testing.assert_close(
            graph.retained_mass_by_relation[0],
            torch.zeros((1, 3), dtype=torch.float32),
        )
        torch.testing.assert_close(
            graph.unknown_mass[0],
            torch.ones(1, dtype=torch.float32),
        )

    def test_edges_are_typed_by_source_segment_and_include_predictor_diagonal(self):
        graph = build_prediction_event_graph(_formal_sparse_fixture())

        typed_edges = {
            (int(source), int(event), Relation(int(relation)))
            for source, event, relation in zip(
                graph.edge_index[0].tolist(),
                graph.edge_index[1].tolist(),
                graph.edge_relation.tolist(),
                strict=True,
            )
        }
        self.assertEqual(
            typed_edges,
            {
                (0, 1, Relation.QY),
                (2, 1, Relation.EY),
                (3, 1, Relation.EY),
                (4, 1, Relation.RY),
                (1, 2, Relation.QY),
                (2, 2, Relation.EY),
                (4, 2, Relation.RY),
                (5, 2, Relation.RY),
            },
        )

        predictor_edge = torch.nonzero(
            (graph.edge_index[0] == 4) & (graph.edge_index[1] == 1),
            as_tuple=False,
        ).flatten()
        self.assertEqual(predictor_edge.numel(), 1)
        edge_id = int(predictor_edge.item())
        trace = graph.trace_edge_id == edge_id
        self.assertEqual(graph.trace_channel[trace].tolist(), [0])
        torch.testing.assert_close(
            graph.trace_value[trace],
            torch.tensor([0.35], dtype=torch.float32),
        )

    def test_floor_censored_attention_is_kept_as_unknown_not_assigned_to_a_relation(self):
        graph = build_prediction_event_graph(_formal_sparse_fixture())

        # Dimension order is [prediction event, channel, EY/QY/RY].
        expected_retained = torch.tensor(
            [
                [[0.00, 0.00, 0.00]],
                [[0.35, 0.10, 0.35]],
                [[0.15, 0.10, 0.45]],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            graph.retained_mass_by_relation,
            expected_retained,
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            graph.unknown_mass,
            torch.tensor([[1.00], [0.20], [0.30]], dtype=torch.float32),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            graph.retained_mass_by_relation.sum(dim=-1) + graph.unknown_mass,
            torch.ones((3, 1), dtype=torch.float32),
            atol=1e-6,
            rtol=0,
        )

    def test_graph_builder_rejects_any_hallucination_label_field(self):
        sample = _formal_sparse_fixture()
        sample["y_token"] = torch.tensor([0, 0, 0], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "label|label-blind|forbidden"):
            build_prediction_event_graph(sample)

    def test_csr_columns_reject_negative_duplicate_and_future_sources(self):
        mutations = {
            "negative": [-1, 2, 3, 1, 2, 4, 0],
            "duplicate": [0, 0, 3, 1, 2, 4, 0],
            "future": [0, 2, 4, 1, 2, 4, 0],
        }
        for name, columns in mutations.items():
            with self.subTest(name=name):
                sample = _formal_sparse_fixture()
                sample["response_column_indices"] = torch.tensor(
                    columns, dtype=torch.int32
                )
                with self.assertRaisesRegex(
                    ValueError, "negative|future|sorted|unique|causal"
                ):
                    build_prediction_event_graph(sample)

    def test_dtype_rounding_excess_is_explicit_not_misclassified_as_unknown(self):
        sample = _formal_sparse_fixture()
        sample["cache_dtype"] = "bfloat16"
        diagonal = torch.as_tensor(sample["attention_diagonal"]).to(torch.bfloat16)
        diagonal[0, 0, 4] = 0.40234375
        sample["attention_diagonal"] = diagonal
        values = torch.as_tensor(sample["response_values"]).to(torch.bfloat16)
        values[:3] = torch.tensor([0.2, 0.2, 0.2], dtype=torch.bfloat16)
        sample["response_values"] = values

        graph = build_prediction_event_graph(sample)

        self.assertGreater(float(graph.rounding_excess_mass[1, 0]), 0.0)
        torch.testing.assert_close(
            graph.retained_mass_by_relation.sum(dim=-1)
            + graph.unknown_mass
            - graph.rounding_excess_mass,
            torch.ones((3, 1)),
            atol=1e-6,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
