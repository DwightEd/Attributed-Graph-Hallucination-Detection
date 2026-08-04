from __future__ import annotations

import unittest

import torch

from attention_graph.ablation import (
    collapse_relations,
    mean_attention_heads,
    relation_preserving_source_shuffle,
)
from attention_graph.graph import GraphBuildConfig, build_attention_graph
from attention_graph.model import (
    RelationAwareMaskGAE,
    make_masked_view,
    reconstruction_energy_by_node,
    reconstruction_losses,
    sample_support_negatives,
)
from attention_graph.train import structural_direction_anchor
from tests.test_attention_graph import _formal_sample


def _graph():
    return build_attention_graph(
        _formal_sample(),
        GraphBuildConfig(selection="threshold", threshold=0.05),
        device="cpu",
    )


class AttentionGraphAblationTests(unittest.TestCase):
    def test_source_shuffle_preserves_relation_target_and_edge_trace_payload(self):
        graph = _graph()
        shuffled = relation_preserving_source_shuffle(
            graph, generator=torch.Generator().manual_seed(101)
        )

        self.assertTrue(torch.equal(shuffled.edge_index[1], graph.edge_index[1]))
        self.assertTrue(torch.equal(shuffled.edge_type, graph.edge_type))
        self.assertTrue(torch.equal(shuffled.trace_edge_id, graph.trace_edge_id))
        self.assertTrue(torch.equal(shuffled.trace_channel, graph.trace_channel))
        self.assertTrue(torch.equal(shuffled.trace_value, graph.trace_value))
        self.assertFalse(torch.equal(shuffled.edge_index[0], graph.edge_index[0]))
        self.assertTrue(bool((shuffled.edge_index[0] < shuffled.edge_index[1]).all()))
        relation_domain = torch.where(
            shuffled.edge_type == 0,
            shuffled.edge_index[0] < shuffled.response_idx,
            shuffled.edge_index[0] >= shuffled.response_idx,
        )
        self.assertTrue(bool(relation_domain.all()))
        original_causal = (graph.edge_index[0] >= graph.response_idx).long()
        shuffled_causal = (shuffled.edge_index[0] >= graph.response_idx).long()
        original_target_degree = torch.bincount(
            graph.edge_index[1] * 2 + original_causal,
            minlength=graph.num_nodes * 2,
        )
        shuffled_target_degree = torch.bincount(
            shuffled.edge_index[1] * 2 + shuffled_causal,
            minlength=graph.num_nodes * 2,
        )
        self.assertTrue(torch.equal(original_target_degree, shuffled_target_degree))
        self.assertTrue(
            torch.equal(
                torch.bincount(graph.edge_index[0], minlength=graph.num_nodes),
                torch.bincount(shuffled.edge_index[0], minlength=graph.num_nodes),
            )
        )
        shuffled_pairs = shuffled.edge_index[1] * graph.num_nodes + shuffled.edge_index[0]
        self.assertEqual(torch.unique(shuffled_pairs).numel(), graph.num_edges)

        torch.manual_seed(107)
        model = RelationAwareMaskGAE(
            num_layers=graph.num_layers,
            num_heads=graph.num_heads,
            embedding_dim=16,
            message_passing_steps=2,
            dropout=0.0,
        ).eval()
        original_view = make_masked_view(
            graph, edge_mask_rate=0.0, node_mask_rate=0.0
        )
        shuffled_view = make_masked_view(
            shuffled, edge_mask_rate=0.0, node_mask_rate=0.0
        )
        with torch.no_grad():
            original_embedding = model.graph_embedding(
                model.encode(graph, original_view), graph
            )
            shuffled_embedding = model.graph_embedding(
                model.encode(shuffled, shuffled_view), shuffled
            )
        self.assertGreater(
            float((original_embedding - shuffled_embedding).abs().max()), 1e-4
        )

    def test_collapse_relations_changes_only_relation_identity(self):
        graph = _graph()
        collapsed = collapse_relations(graph)

        self.assertTrue(torch.equal(collapsed.edge_type, torch.zeros_like(graph.edge_type)))
        self.assertTrue(torch.equal(collapsed.edge_index, graph.edge_index))
        self.assertTrue(torch.equal(collapsed.node_attr, graph.node_attr))
        self.assertTrue(torch.equal(collapsed.trace_edge_id, graph.trace_edge_id))
        self.assertTrue(torch.equal(collapsed.trace_channel, graph.trace_channel))
        self.assertTrue(torch.equal(collapsed.trace_value, graph.trace_value))

    def test_relation_collapsed_graph_still_uses_legal_rp_rr_negative_domains(self):
        graph = _graph()
        collapsed = collapse_relations(graph)
        rr_edge = torch.nonzero(
            (collapsed.edge_index[0] >= collapsed.response_idx)
            & (collapsed.edge_index[1] == 5),
            as_tuple=False,
        ).flatten()[:1]
        negative_index, _negative_type = sample_support_negatives(
            collapsed,
            rr_edge,
            generator=torch.Generator().manual_seed(101),
        )
        self.assertGreaterEqual(
            int(negative_index[0, 0]), collapsed.response_idx
        )

        model = RelationAwareMaskGAE(
            num_layers=collapsed.num_layers,
            num_heads=collapsed.num_heads,
            embedding_dim=8,
            message_passing_steps=1,
            dropout=0.0,
        )
        generator = torch.Generator().manual_seed(103)
        view = make_masked_view(
            collapsed,
            edge_mask_rate=0.5,
            node_mask_rate=0.5,
            generator=generator,
        )

        losses = reconstruction_losses(
            model, collapsed, view, generator=generator
        )

        self.assertTrue(torch.isfinite(losses.total))

        original_view = make_masked_view(
            graph,
            edge_mask_rate=0.5,
            node_mask_rate=0.5,
            generator=torch.Generator().manual_seed(109),
        )
        collapsed_view = make_masked_view(
            collapsed,
            edge_mask_rate=0.5,
            node_mask_rate=0.5,
            generator=torch.Generator().manual_seed(109),
        )
        self.assertTrue(
            torch.equal(original_view.visible_edge_mask, collapsed_view.visible_edge_mask)
        )
        self.assertEqual(
            structural_direction_anchor(graph),
            structural_direction_anchor(collapsed),
        )

        causal_model = RelationAwareMaskGAE(
            num_layers=graph.num_layers,
            num_heads=graph.num_heads,
            embedding_dim=8,
            message_passing_steps=0,
            dropout=0.0,
        ).eval()
        with torch.no_grad():
            causal_model.relation_embedding.weight.zero_()
            original_energy = reconstruction_energy_by_node(
                causal_model,
                graph,
                original_view,
                generator=torch.Generator().manual_seed(113),
            )
            collapsed_energy = reconstruction_energy_by_node(
                causal_model,
                collapsed,
                collapsed_view,
                generator=torch.Generator().manual_seed(113),
            )
        for name in ("support_rp", "support_rr", "weight_rp", "weight_rr"):
            torch.testing.assert_close(original_energy[name], collapsed_energy[name])

    def test_mean_heads_removes_head_identity_without_changing_support(self):
        graph = _graph()
        averaged = mean_attention_heads(graph)

        self.assertEqual(averaged.num_layers, graph.num_layers)
        self.assertEqual(averaged.num_heads, 1)
        self.assertEqual(averaged.num_channels, graph.num_layers)
        expected_node = graph.node_attr.reshape(
            graph.num_nodes, graph.num_layers, graph.num_heads
        ).mean(dim=2)
        torch.testing.assert_close(averaged.node_attr, expected_node)
        self.assertTrue(torch.equal(averaged.edge_index, graph.edge_index))
        self.assertTrue(torch.equal(averaged.edge_type, graph.edge_type))
        self.assertTrue(torch.equal(averaged.edge_score, graph.edge_score))

        edge_id = torch.nonzero(
            (graph.edge_index[0] == 0) & (graph.edge_index[1] == 3),
            as_tuple=False,
        ).item()
        selected = averaged.trace_edge_id == edge_id
        self.assertEqual(averaged.trace_channel[selected].tolist(), [0, 1])
        torch.testing.assert_close(
            averaged.trace_value[selected],
            torch.tensor([0.15, 0.135]),
            atol=2e-4,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
