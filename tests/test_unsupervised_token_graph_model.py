import inspect
import unittest

import torch

from unsupervised_token_graph.graph import build_token_graph
from unsupervised_token_graph.model import (
    TokenGraphMaskedAutoencoder,
    make_answer_block_mask,
    masked_reconstruction_loss,
)
from unsupervised_token_graph.trace import assign_segment_ids


class SegmentTraceTests(unittest.TestCase):
    def test_assign_segment_ids_reserves_zero_for_template_and_special_tokens(self):
        segment_char_spans = {
            "passage": (10, 14),
            "question": (20, 24),
            "answer": (30, 34),
        }
        offset_mapping = [
            (0, 0),
            (10, 12),
            (12, 14),
            (14, 20),
            (20, 24),
            (30, 32),
            (32, 34),
        ]
        special_tokens_mask = [1, 0, 0, 0, 0, 0, 0]

        segment_ids = assign_segment_ids(
            offset_mapping,
            segment_char_spans,
            special_tokens_mask,
        )

        self.assertEqual(torch.as_tensor(segment_ids).tolist(), [0, 1, 1, 0, 2, 3, 3])


class TokenGraphConstructionTests(unittest.TestCase):
    def test_graph_keeps_causal_cross_segment_edges_and_encodes_both_segments(self):
        input_ids = torch.tensor([10, 11, 20, 21, 30, 31], dtype=torch.long)
        segment_ids = torch.tensor([1, 1, 2, 2, 3, 3], dtype=torch.long)
        attention = torch.zeros((1, 1, 6, 6), dtype=torch.float32)
        attention[0, 0, 2, 0] = 0.20  # passage -> question
        attention[0, 0, 4, 2] = 0.30  # question -> answer
        attention[0, 0, 5, 1] = 0.40  # passage -> answer
        attention[0, 0, 1, 4] = 0.90  # future key; must not become an edge
        attention[0, 0, 2, 2] = 0.90  # self attention; must not become an edge

        graph = build_token_graph(
            input_ids,
            attention,
            segment_ids,
            tau=0.05,
            include_prefix_edges=True,
        )

        forbidden_keys = {"label", "labels", "target", "y", "y_token"}
        self.assertTrue(set(graph).isdisjoint(forbidden_keys))
        self.assertFalse(any("label" in key.casefold() for key in graph))
        self.assertEqual(graph["x"].shape[0], len(input_ids))
        self.assertEqual(torch.as_tensor(graph["token_ids"]).tolist(), input_ids.tolist())
        self.assertEqual(
            torch.as_tensor(graph["segment_ids"]).tolist(), segment_ids.tolist()
        )

        edge_index = torch.as_tensor(graph["edge_index"])
        edges = {tuple(edge) for edge in edge_index.t().tolist()}
        self.assertEqual(edges, {(0, 2), (2, 4), (1, 5)})
        self.assertTrue(all(source < destination for source, destination in edges))

        edge_marks = torch.as_tensor(graph["edge_mark"], dtype=torch.float32)
        self.assertEqual(tuple(edge_marks.shape), (3, 8))
        for edge_position, (source, destination) in enumerate(edge_index.t().tolist()):
            expected = torch.cat(
                (
                    torch.nn.functional.one_hot(segment_ids[source], num_classes=4),
                    torch.nn.functional.one_hot(
                        segment_ids[destination], num_classes=4
                    ),
                )
            ).to(torch.float32)
            torch.testing.assert_close(edge_marks[edge_position], expected)


class MaskedAutoencoderContractTests(unittest.TestCase):
    def test_answer_block_mask_never_masks_template_passage_or_question(self):
        segment_ids = torch.tensor([0, 1, 1, 2, 2, 3, 3, 3, 3])

        mask = make_answer_block_mask(
            segment_ids,
            mask_ratio=0.5,
            generator=torch.Generator().manual_seed(7),
        )

        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(int(mask.sum()), 2)
        self.assertFalse(bool(mask[segment_ids != 3].any()))

    def test_reconstruction_loss_ignores_every_unmasked_node(self):
        predictions = torch.tensor(
            [[100.0, -100.0], [2.0, 0.0], [-100.0, 100.0]]
        )
        targets = torch.zeros_like(predictions)
        masked_nodes = torch.tensor([False, True, False])

        loss = masked_reconstruction_loss(predictions, targets, masked_nodes)
        changed_unmasked = predictions.clone()
        changed_unmasked[~masked_nodes] *= 1000.0
        changed_loss = masked_reconstruction_loss(
            changed_unmasked,
            targets,
            masked_nodes,
        )

        torch.testing.assert_close(loss, torch.tensor(2.0))
        torch.testing.assert_close(changed_loss, loss)

    def test_model_is_pure_pytorch_and_its_training_signature_has_no_labels(self):
        forward_parameters = list(
            inspect.signature(TokenGraphMaskedAutoencoder.forward).parameters
        )
        self.assertEqual(
            forward_parameters,
            [
                "self",
                "x",
                "edge_index",
                "edge_attr",
                "edge_mark",
                "masked_nodes",
            ],
        )
        for callable_object in (
            TokenGraphMaskedAutoencoder.__init__,
            TokenGraphMaskedAutoencoder.forward,
        ):
            parameters = inspect.signature(callable_object).parameters
            self.assertNotIn("labels", parameters)
            self.assertNotIn("y", parameters)

        model = TokenGraphMaskedAutoencoder(
            node_dim=3,
            edge_dim=1,
            edge_mark_dim=8,
            hidden_dim=5,
            num_layers=1,
            dropout=0.0,
        )
        self.assertIsInstance(model, torch.nn.Module)
        self.assertFalse(
            any(
                type(module).__module__.startswith("torch_geometric")
                for module in model.modules()
            )
        )

    def test_target_mean_is_invariant_to_the_target_nodes_own_out_degree(self):
        torch.manual_seed(13)
        model = TokenGraphMaskedAutoencoder(
            node_dim=3,
            edge_dim=1,
            edge_mark_dim=8,
            hidden_dim=5,
            num_layers=1,
            dropout=0.0,
        ).eval()
        x = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
            ]
        )
        masked_nodes = torch.tensor([False, False, True, False, False])

        incoming_only = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
        with_target_outgoing = torch.tensor(
            [[0, 1, 2, 2], [2, 2, 3, 4]], dtype=torch.long
        )
        incoming_edge_attr = torch.ones((2, 1))
        expanded_edge_attr = torch.ones((4, 1))
        incoming_edge_mark = torch.zeros((2, 8))
        expanded_edge_mark = torch.zeros((4, 8))

        with torch.no_grad():
            output_without_outgoing = model(
                x,
                incoming_only,
                incoming_edge_attr,
                incoming_edge_mark,
                masked_nodes,
            )
            output_with_outgoing = model(
                x,
                with_target_outgoing,
                expanded_edge_attr,
                expanded_edge_mark,
                masked_nodes,
            )

        self.assertEqual(tuple(output_without_outgoing.shape), (5, 3))
        self.assertEqual(tuple(output_with_outgoing.shape), (5, 3))
        torch.testing.assert_close(
            output_without_outgoing[2],
            output_with_outgoing[2],
        )


if __name__ == "__main__":
    unittest.main()
