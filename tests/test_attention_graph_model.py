from __future__ import annotations

import unittest
from unittest.mock import patch
from dataclasses import replace

import torch

from attention_graph.graph import GraphBuildConfig, build_attention_graph
from attention_graph.model import (
    RelationAwareMaskGAE,
    make_masked_view,
    reconstruction_energy_by_node,
    reconstruction_losses,
    sample_support_negatives,
)
from tests.test_attention_graph import _formal_sample


def _graph():
    return build_attention_graph(
        _formal_sample(),
        GraphBuildConfig(selection="threshold", threshold=0.05),
        device="cpu",
    )


def _model() -> RelationAwareMaskGAE:
    torch.manual_seed(7)
    return RelationAwareMaskGAE(
        num_layers=2,
        num_heads=2,
        embedding_dim=16,
        message_passing_steps=2,
        dropout=0.0,
    )


class RelationAwareMaskedGraphTests(unittest.TestCase):
    def test_stratified_edge_mask_keeps_one_visible_edge_and_preserves_singletons(self):
        graph = _graph()
        view = make_masked_view(
            graph,
            edge_mask_rate=0.8,
            node_mask_rate=0.5,
            channel_drop_rate=0.0,
            generator=torch.Generator().manual_seed(11),
        )

        groups = graph.edge_index[1] * 2 + graph.edge_type
        for group in torch.unique(groups):
            members = groups == group
            degree = int(members.sum())
            visible = int((members & view.visible_edge_mask).sum())
            if degree == 1:
                self.assertEqual(visible, 1)
            else:
                self.assertGreaterEqual(visible, 1)
                self.assertLess(visible, degree)
        self.assertTrue(bool((view.node_mask & graph.response_mask).any()))
        self.assertFalse(bool((view.node_mask & ~graph.response_mask).any()))

    def test_masked_edge_values_are_not_visible_to_encoder(self):
        graph = _graph()
        view = make_masked_view(
            graph,
            edge_mask_rate=0.5,
            node_mask_rate=0.0,
            channel_drop_rate=0.0,
            generator=torch.Generator().manual_seed(3),
        )
        self.assertTrue(bool((~view.visible_edge_mask).any()))
        masked_trace = ~view.visible_edge_mask[graph.trace_edge_id]
        changed_values = graph.trace_value.clone()
        changed_values[masked_trace] = changed_values[masked_trace] * 0.01
        changed = replace(graph, trace_value=changed_values)
        model = _model().eval()

        with torch.no_grad():
            hidden_a = model.encode(graph, view)
            hidden_b = model.encode(changed, view)
        torch.testing.assert_close(hidden_a, hidden_b)

    def test_rewiring_same_relation_changes_response_embeddings(self):
        graph = _graph()
        source = graph.edge_index[0].clone()
        rp_edges = torch.nonzero(graph.edge_type == 0, as_tuple=False).flatten()
        edge = int(rp_edges[0])
        target = int(graph.edge_index[1, edge])
        replacement = (int(source[edge]) + 1) % graph.response_idx
        if replacement == int(source[edge]):
            replacement = (replacement + 1) % graph.response_idx
        source[edge] = replacement
        rewired = replace(graph, edge_index=torch.stack((source, graph.edge_index[1])))
        view = make_masked_view(graph, edge_mask_rate=0.0, node_mask_rate=0.0)
        rewired_view = replace(view, graph=rewired)
        model = _model().eval()

        with torch.no_grad():
            original_hidden = model.encode(graph, view)
            rewired_hidden = model.encode(rewired, rewired_view)
        self.assertFalse(torch.allclose(original_hidden[target], rewired_hidden[target]))

    def test_relation_and_head_identity_change_encoder_output(self):
        graph = _graph()
        view = make_masked_view(graph, edge_mask_rate=0.0, node_mask_rate=0.0)
        collapsed = replace(graph, edge_type=torch.zeros_like(graph.edge_type))
        collapsed_view = replace(view, graph=collapsed)
        swapped_channel = graph.trace_channel.clone()
        swapped_channel = torch.where(
            swapped_channel == 0,
            torch.ones_like(swapped_channel),
            torch.where(
                swapped_channel == 1,
                torch.zeros_like(swapped_channel),
                swapped_channel,
            ),
        )
        swapped = replace(graph, trace_channel=swapped_channel)
        swapped_view = replace(view, graph=swapped)
        model = _model().eval()

        with torch.no_grad():
            base = model.encode(graph, view)
            no_types = model.encode(collapsed, collapsed_view)
            no_head_identity = model.encode(swapped, swapped_view)
        self.assertFalse(torch.allclose(base[graph.response_mask], no_types[graph.response_mask]))
        self.assertFalse(
            torch.allclose(base[graph.response_mask], no_head_identity[graph.response_mask])
        )

    def test_zero_message_steps_is_a_true_feature_only_ablation(self):
        graph = _graph()
        source = graph.edge_index[0].roll(1)
        rewired = replace(graph, edge_index=torch.stack((source, graph.edge_index[1])))
        view = make_masked_view(graph, edge_mask_rate=0.0, node_mask_rate=0.0)
        rewired_view = replace(view, graph=rewired)
        torch.manual_seed(8)
        model = RelationAwareMaskGAE(
            num_layers=2,
            num_heads=2,
            embedding_dim=16,
            message_passing_steps=0,
            dropout=0.0,
        ).eval()

        with torch.no_grad():
            base = model.encode(graph, view)
            changed = model.encode(rewired, rewired_view)
        torch.testing.assert_close(base, changed)

    def test_negative_edges_match_target_relation_and_causal_domain(self):
        graph = _graph()
        view = make_masked_view(
            graph,
            edge_mask_rate=0.5,
            node_mask_rate=0.0,
            generator=torch.Generator().manual_seed(5),
        )
        negative_index, negative_type = sample_support_negatives(
            graph,
            view.masked_edge_ids,
            generator=torch.Generator().manual_seed(13),
        )
        positive_target = graph.edge_index[1, view.masked_edge_ids]
        positive_type = graph.edge_type[view.masked_edge_ids]

        self.assertTrue(torch.equal(negative_index[1], positive_target))
        self.assertTrue(torch.equal(negative_type, positive_type))
        self.assertTrue(bool((negative_index[0] < negative_index[1]).all()))
        self.assertTrue(
            bool(
                torch.where(
                    negative_type == 0,
                    negative_index[0] < graph.response_idx,
                    negative_index[0] >= graph.response_idx,
                ).all()
            )
        )
        original = {
            (int(source), int(target))
            for source, target in graph.edge_index.t().tolist()
        }
        self.assertTrue(
            all(tuple(edge) not in original for edge in negative_index.t().tolist())
        )

    def test_negative_rank_mapping_handles_more_than_64_occupied_sources(self):
        base = _graph()
        node_count, response_idx, target = 132, 130, 130
        source = torch.arange(80, dtype=torch.long)
        edge_index = torch.stack((source, torch.full_like(source, target)))
        node_attr = torch.zeros(node_count, base.num_channels)
        response_mask = torch.arange(node_count) >= response_idx
        position = torch.arange(node_count, dtype=torch.float32)
        node_context = torch.stack(
            (
                position / (node_count - 1),
                (~response_mask).float(),
                response_mask.float(),
            ),
            dim=1,
        )
        graph = replace(
            base,
            response_idx=response_idx,
            node_attr=node_attr,
            node_context=node_context,
            response_mask=response_mask,
            edge_index=edge_index,
            edge_type=torch.zeros(80, dtype=torch.long),
            edge_score=torch.ones(80),
            trace_edge_id=torch.empty(0, dtype=torch.long),
            trace_channel=torch.empty(0, dtype=torch.long),
            trace_value=torch.empty(0),
            token_ids=torch.arange(node_count),
        )

        negative_index, negative_type = sample_support_negatives(
            graph,
            torch.tensor([0]),
            generator=torch.Generator().manual_seed(3),
        )

        self.assertEqual(int(negative_type[0]), 0)
        self.assertGreaterEqual(int(negative_index[0, 0]), 80)
        self.assertLess(int(negative_index[0, 0]), response_idx)

    def test_all_reconstruction_terms_are_finite_and_message_branch_gets_gradient(self):
        graph = _graph()
        view = make_masked_view(
            graph,
            edge_mask_rate=0.5,
            node_mask_rate=0.5,
            channel_drop_rate=0.25,
            generator=torch.Generator().manual_seed(17),
        )
        model = _model().train()
        losses = reconstruction_losses(
            model,
            graph,
            view,
            generator=torch.Generator().manual_seed(19),
        )

        for name in ("support", "weight", "distribution", "node", "total"):
            self.assertTrue(torch.isfinite(getattr(losses, name)))
        losses.total.backward()
        message_grad = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if "message_layers" in name and parameter.grad is not None
        )
        relation_grad = float(model.relation_embedding.weight.grad.abs().sum())
        self.assertGreater(message_grad, 0.0)
        self.assertGreater(relation_grad, 0.0)

    def test_scoring_exposes_per_response_node_reconstruction_energy(self):
        graph = _graph()
        view = make_masked_view(
            graph,
            edge_mask_rate=0.5,
            node_mask_rate=0.5,
            channel_drop_rate=0.0,
            generator=torch.Generator().manual_seed(29),
        )
        model = _model().eval()

        with torch.no_grad():
            energy = reconstruction_energy_by_node(
                model,
                graph,
                view,
                generator=torch.Generator().manual_seed(31),
            )

        for name in ("support", "weight", "distribution", "node", "total"):
            self.assertEqual(energy[name].shape, (graph.num_nodes,))
            self.assertTrue(torch.isfinite(energy[name]).all())
        selected_targets = view.node_mask.clone()
        if view.masked_edge_ids.numel():
            selected_targets[graph.edge_index[1, view.masked_edge_ids]] = True
        self.assertTrue(bool((energy["total"][selected_targets] > 0).any()))
        self.assertTrue(torch.equal(energy["total"][~graph.response_mask], torch.zeros(3)))

    def test_reconstruction_sampling_caps_and_decoder_chunks_are_enforced(self):
        graph = _graph()
        view = make_masked_view(
            graph,
            edge_mask_rate=0.75,
            node_mask_rate=0.5,
            channel_drop_rate=0.0,
            generator=torch.Generator().manual_seed(53),
        )
        model = _model().eval()

        with (
            patch.object(
                model, "decode_support", wraps=model.decode_support
            ) as support,
            patch.object(
                model, "decode_weight", wraps=model.decode_weight
            ) as weight,
            patch.object(
                model,
                "decode_distribution_entry",
                wraps=model.decode_distribution_entry,
            ) as distribution,
        ):
            losses = reconstruction_losses(
                model,
                graph,
                view,
                generator=torch.Generator().manual_seed(59),
                max_support_edges=1,
                max_weight_traces=2,
                max_distribution_groups=1,
                decoder_chunk_size=1,
            )

        self.assertTrue(torch.isfinite(losses.total))
        self.assertTrue(
            all(call.args[1].shape[1] <= 1 for call in support.call_args_list)
        )
        self.assertTrue(
            all(call.args[1].shape[1] <= 1 for call in weight.call_args_list)
        )
        self.assertTrue(
            all(call.args[1].shape[1] <= 1 for call in distribution.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
