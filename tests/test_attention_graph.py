from __future__ import annotations

import unittest

import torch

from attention_graph.graph import GraphBuildConfig, build_attention_graph


def _formal_sample() -> dict[str, object]:
    layers, heads, token_count, response_idx = 2, 2, 6, 3
    response_tokens = token_count - response_idx
    rows: dict[tuple[int, int], list[tuple[int, float]]] = {
        (0, 0): [(0, 0.20), (1, 0.06)],
        (1, 0): [(0, 0.10), (2, 0.04)],
        (2, 0): [(0, 0.15)],
        (3, 0): [(0, 0.12)],
        (0, 1): [(0, 0.03), (3, 0.40)],
        (1, 1): [(3, 0.30)],
        (2, 1): [(3, 0.25)],
        (3, 1): [(3, 0.20)],
        (0, 2): [(1, 0.08), (4, 0.35)],
        (1, 2): [(1, 0.07), (4, 0.25)],
        (2, 2): [(2, 0.055), (4, 0.20)],
        (3, 2): [(4, 0.15)],
    }
    columns: list[int] = []
    values: list[float] = []
    row_ptr = [0]
    for channel in range(layers * heads):
        for query in range(response_tokens):
            for source, value in rows.get((channel, query), []):
                columns.append(source)
                values.append(value)
            row_ptr.append(len(columns))
    diagonal = (
        torch.arange(layers * heads * token_count, dtype=torch.float16)
        .reshape(layers, heads, token_count)
        .div(100)
    )
    return {
        "cache_format": "formal_sparse_csr",
        "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
        "attention_cache_fingerprint": "fixture-v1",
        "cache_dtype": "float16",
        "input_policy": "full_context_no_truncation",
        "was_truncated": False,
        "source_id": "source-1",
        "response_id": "response-1",
        "sample_id": "response-1",
        "response_idx": response_idx,
        "num_attention_layers": layers,
        "num_attention_heads": heads,
        "attention_floor": 0.01,
        "token_ids": torch.arange(token_count, dtype=torch.long),
        "attention_diagonal": diagonal,
        "response_row_ptr": torch.tensor(row_ptr, dtype=torch.long),
        "response_column_indices": torch.tensor(columns, dtype=torch.int32),
        "response_values": torch.tensor(values, dtype=torch.float16),
    }


class AttentionGraphConstructionTests(unittest.TestCase):
    def test_preserves_ordered_layer_head_node_and_sparse_edge_channels(self):
        sample = _formal_sample()
        graph = build_attention_graph(
            sample,
            GraphBuildConfig(selection="threshold", threshold=0.05),
            device="cpu",
        )

        expected_nodes = sample["attention_diagonal"].permute(2, 0, 1).reshape(6, 4).float()
        torch.testing.assert_close(graph.node_attr, expected_nodes)
        self.assertEqual(graph.num_layers, 2)
        self.assertEqual(graph.num_heads, 2)
        self.assertEqual(graph.edge_index.shape[0], 2)
        self.assertEqual(graph.trace_edge_id.shape, graph.trace_channel.shape)
        self.assertEqual(graph.trace_channel.shape, graph.trace_value.shape)
        trace_key = graph.trace_edge_id * graph.num_channels + graph.trace_channel
        self.assertTrue(bool((trace_key[1:] > trace_key[:-1]).all()))

        edge_id = torch.nonzero(
            (graph.edge_index[0] == 0) & (graph.edge_index[1] == 3), as_tuple=False
        ).item()
        selected = graph.trace_edge_id == edge_id
        self.assertEqual(graph.trace_channel[selected].tolist(), [0, 1, 2, 3])
        torch.testing.assert_close(
            graph.trace_value[selected],
            torch.tensor([0.20, 0.10, 0.15, 0.12]),
            atol=2e-4,
            rtol=0,
        )

    def test_threshold_retains_natural_degree_and_combined_topk_does_not_fill_relations(self):
        sample = _formal_sample()
        threshold = build_attention_graph(
            sample,
            GraphBuildConfig(selection="threshold", threshold=0.05),
            device="cpu",
        )
        global_topk = build_attention_graph(
            sample,
            GraphBuildConfig(selection="global_topk", top_k=1),
            device="cpu",
        )
        typed_topk = build_attention_graph(
            sample,
            GraphBuildConfig(selection="typed_topk", top_k=1),
            device="cpu",
        )

        threshold_degrees = torch.bincount(threshold.edge_index[1], minlength=6)[3:]
        global_degrees = torch.bincount(global_topk.edge_index[1], minlength=6)[3:]
        typed_degrees = torch.bincount(typed_topk.edge_index[1], minlength=6)[3:]
        self.assertEqual(threshold_degrees.tolist(), [2, 1, 3])
        self.assertEqual(global_degrees.tolist(), [1, 1, 1])
        self.assertEqual(typed_degrees.tolist(), [1, 2, 2])

    def test_head_identity_is_not_collapsed_to_summary_statistics(self):
        sample = _formal_sample()
        swapped = _formal_sample()
        response_tokens = 3
        values = swapped["response_values"].clone()
        row_ptr = swapped["response_row_ptr"]
        for query in range(response_tokens):
            row_a = query
            row_b = response_tokens + query
            a0, a1 = int(row_ptr[row_a]), int(row_ptr[row_a + 1])
            b0, b1 = int(row_ptr[row_b]), int(row_ptr[row_b + 1])
            if a1 - a0 == b1 - b0:
                original_a = values[a0:a1].clone()
                values[a0:a1] = values[b0:b1]
                values[b0:b1] = original_a
        swapped["response_values"] = values

        config = GraphBuildConfig(selection="threshold", threshold=0.01)
        graph_a = build_attention_graph(sample, config, device="cpu")
        graph_b = build_attention_graph(swapped, config, device="cpu")

        self.assertTrue(torch.equal(graph_a.edge_index, graph_b.edge_index))
        self.assertFalse(torch.equal(graph_a.trace_value, graph_b.trace_value))
        self.assertTrue(torch.equal(graph_a.trace_channel, graph_b.trace_channel))

    def test_graph_construction_rejects_training_labels(self):
        sample = _formal_sample()
        sample["y_token"] = torch.zeros(6)
        with self.assertRaisesRegex(ValueError, "label-blind"):
            build_attention_graph(sample, GraphBuildConfig(), device="cpu")


if __name__ == "__main__":
    unittest.main()
