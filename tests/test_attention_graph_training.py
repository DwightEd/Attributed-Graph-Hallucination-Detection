from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from attention_graph.graph import GraphBuildConfig, build_attention_graph
from attention_graph.model import RelationAwareMaskGAE
from attention_graph.ablation import collapse_relations
from attention_graph.train import (
    TrainingConfig,
    _token_direction_anchors,
    fit_two_component_mixture,
    relation_preserving_source_shuffle,
    score_graphs,
    score_tokens,
    structural_direction_anchor,
    train_relation_mae,
)
from tests.test_attention_graph import _formal_sample


def _graphs(count: int = 4):
    base = build_attention_graph(
        _formal_sample(),
        GraphBuildConfig(selection="threshold", threshold=0.05),
        device="cpu",
    )
    graphs = []
    for index in range(count):
        node_attr = (base.node_attr + index * 0.002).clamp_max(1.0)
        graphs.append(
            replace(
                base,
                source_id=f"source-{index}",
                response_id=f"response-{index}",
                sample_id=f"response-{index}",
                node_attr=node_attr,
            )
        )
    return graphs


def _model() -> RelationAwareMaskGAE:
    torch.manual_seed(23)
    return RelationAwareMaskGAE(
        num_layers=2,
        num_heads=2,
        embedding_dim=12,
        message_passing_steps=1,
        dropout=0.0,
    )


class AttentionGraphTrainingTests(unittest.TestCase):
    def test_direction_anchor_is_an_orientation_prior_not_an_input_feature(self):
        graph = _graphs(1)[0]
        anchor = structural_direction_anchor(graph)

        self.assertEqual(
            set(anchor),
            {
                "prompt_mass_fraction",
                "response_mass_fraction",
                "mean_in_degree",
                "normalized_lag",
                "retained_concentration",
                "direction_score",
            },
        )
        self.assertAlmostEqual(
            anchor["prompt_mass_fraction"] + anchor["response_mass_fraction"],
            1.0,
            places=5,
        )
        self.assertEqual(anchor, structural_direction_anchor(collapse_relations(graph)))

    def test_vectorized_token_direction_anchors_match_reference_semantics(self):
        graph = _graphs(1)[0]
        anchors = _token_direction_anchors(graph)

        for target in range(graph.response_idx, graph.num_nodes):
            target_edges = graph.edge_index[1] == target
            edge_ids = torch.nonzero(target_edges, as_tuple=False).flatten()
            if edge_ids.numel():
                traces = torch.isin(graph.trace_edge_id, edge_ids)
                trace_edges = graph.trace_edge_id[traces]
                values = graph.trace_value[traces]
                causal_type = (
                    graph.edge_index[0, trace_edges] >= graph.response_idx
                )
                rp_mass = float(values[~causal_type].sum())
                rr_mass = float(values[causal_type].sum())
                total = max(rp_mass + rr_mass, np.finfo(np.float64).eps)
                prompt_fraction = rp_mass / total
                response_fraction = rr_mass / total
                sources = graph.edge_index[0, trace_edges].float()
                lag = (float(target) - sources) / max(float(target), 1.0)
                normalized_lag = float(
                    (lag * values).sum() / values.sum().clamp_min(1e-8)
                )
                channels = graph.trace_channel[traces]
                unique_channels, inverse = torch.unique(channels, return_inverse=True)
                channel_mass = torch.zeros(unique_channels.numel())
                channel_square = torch.zeros_like(channel_mass)
                channel_mass.index_add_(0, inverse, values)
                channel_square.index_add_(0, inverse, values.square())
                concentration = float(
                    (
                        channel_square
                        / channel_mass.square().clamp_min(1e-12)
                    ).mean()
                )
            else:
                prompt_fraction = 0.0
                response_fraction = 0.0
                normalized_lag = 0.0
                concentration = 0.0
            in_degree = int(target_edges.sum())
            degree_density = in_degree / max(target, 1)
            expected = {
                "prompt_mass_fraction": float(prompt_fraction),
                "response_mass_fraction": float(response_fraction),
                "in_degree": float(in_degree),
                "normalized_lag": float(normalized_lag),
                "retained_concentration": float(concentration),
                "direction_score": float(
                    -prompt_fraction
                    + response_fraction
                    - degree_density
                    - normalized_lag
                    + concentration
                ),
            }
            for name, value in expected.items():
                self.assertAlmostEqual(anchors[target][name], value, places=6)

        collapsed = _token_direction_anchors(collapse_relations(graph))
        self.assertEqual(set(anchors), set(collapsed))
        for target in anchors:
            for name in anchors[target]:
                self.assertAlmostEqual(
                    anchors[target][name], collapsed[target][name], places=6
                )

    def test_two_component_mixture_does_not_assume_hallucinations_are_the_smaller_cluster(self):
        rng = np.random.default_rng(5)
        small = rng.normal(-2.0, 0.15, size=(20, 3))
        large = rng.normal(2.0, 0.15, size=(80, 3))
        features = np.concatenate((small, large), axis=0)
        direction = np.concatenate((np.full(20, -1.0), np.full(80, 1.0)))

        mixture = fit_two_component_mixture(features, direction, seed=31)
        probabilities = mixture.hallucination_probability(features)

        self.assertGreater(float(probabilities[20:].mean()), 0.9)
        self.assertLess(float(probabilities[:20].mean()), 0.1)
        self.assertGreater(mixture.component_weights[mixture.hallucination_component], 0.5)

    def test_relation_preserving_shuffle_keeps_degree_type_and_trace_targets(self):
        graph = _graphs(1)[0]
        shuffled = relation_preserving_source_shuffle(
            graph, generator=torch.Generator().manual_seed(37)
        )

        self.assertTrue(torch.equal(graph.edge_index[1], shuffled.edge_index[1]))
        self.assertTrue(torch.equal(graph.edge_type, shuffled.edge_type))
        self.assertTrue(torch.equal(graph.trace_edge_id, shuffled.trace_edge_id))
        self.assertTrue(torch.equal(graph.trace_value, shuffled.trace_value))
        self.assertFalse(torch.equal(graph.edge_index[0], shuffled.edge_index[0]))

    def test_training_runs_real_epochs_updates_parameters_and_writes_checkpoint(self):
        graphs = _graphs(4)
        model = _model()
        progress: list[tuple[str, int, int]] = []
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        with tempfile.TemporaryDirectory() as directory:
            result = train_relation_mae(
                model,
                train_graphs=graphs[:3],
                validation_graphs=graphs[3:],
                config=TrainingConfig(
                    epochs=3,
                    patience=3,
                    learning_rate=1e-3,
                    edge_mask_rate=0.5,
                    node_mask_rate=0.5,
                    channel_drop_rate=0.0,
                    seed=41,
                ),
                output_dir=Path(directory),
                progress_callback=lambda stage, current, total: progress.append(
                    (stage, current, total)
                ),
            )
            checkpoint = Path(directory) / "encoder.pt"
            self.assertTrue(checkpoint.is_file())

        self.assertEqual(len(result.history), 3)
        self.assertGreaterEqual(result.best_epoch, 1)
        changed = any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in model.named_parameters()
        )
        self.assertTrue(changed)
        for epoch in range(1, 4):
            self.assertEqual(
                [
                    (current, total)
                    for stage, current, total in progress
                    if stage == f"train_epoch_{epoch}"
                ],
                [(1, 3), (2, 3), (3, 3)],
            )
            self.assertEqual(
                [
                    (current, total)
                    for stage, current, total in progress
                    if stage == f"validation_epoch_{epoch}"
                ],
                [(1, 1)],
            )

    def test_scoring_returns_graph_embeddings_free_mixture_and_component_energies(self):
        graphs = _graphs(4)
        model = _model()
        progress: list[tuple[str, int, int]] = []
        scored, mixture = score_graphs(
            model,
            fit_graphs=graphs[:3],
            score_graphs=graphs[3:],
            num_views=2,
            seed=43,
            progress_callback=lambda stage, current, total: progress.append(
                (stage, current, total)
            ),
        )

        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0]["sample_id"], "response-3")
        self.assertTrue(0.0 <= scored[0]["hallucination_probability"] <= 1.0)
        self.assertIn("support_energy", scored[0])
        self.assertIn("distribution_energy", scored[0])
        self.assertEqual(len(scored[0]["graph_embedding"]), 12)
        self.assertAlmostEqual(sum(mixture.component_weights), 1.0, places=6)
        self.assertEqual(
            progress,
            [
                ("mixture_fit", 1, 3),
                ("mixture_fit", 2, 3),
                ("mixture_fit", 3, 3),
                ("test_scoring", 1, 1),
            ],
        )

    def test_token_mixture_scores_every_response_token_without_rare_class_constraint(self):
        graphs = _graphs(4)
        model = _model()
        records, mixture = score_tokens(
            model,
            fit_graphs=graphs[:3],
            score_graphs=graphs[3:],
            mask_stride=2,
            max_fit_tokens=100,
            seed=47,
        )

        self.assertEqual(len(records), 3)
        self.assertEqual([record["token_idx"] for record in records], [3, 4, 5])
        self.assertTrue(
            all(0.0 <= record["score"] <= 1.0 for record in records)
        )
        self.assertTrue(all("rp_support_energy" in record for record in records))
        self.assertAlmostEqual(sum(mixture.component_weights), 1.0, places=6)

    def test_embedding_only_scoring_is_a_true_non_reconstruction_baseline(self):
        graphs = _graphs(4)
        model = RelationAwareMaskGAE(
            num_layers=2,
            num_heads=2,
            embedding_dim=12,
            message_passing_steps=0,
            dropout=0.0,
        )

        responses, response_mixture = score_graphs(
            model,
            fit_graphs=graphs[:3],
            score_graphs=graphs[3:],
            num_views=1,
            include_reconstruction=False,
            seed=61,
        )
        tokens, token_mixture = score_tokens(
            model,
            fit_graphs=graphs[:3],
            score_graphs=graphs[3:],
            include_reconstruction=False,
            seed=67,
        )

        self.assertEqual(len(response_mixture.feature_median), 12)
        self.assertEqual(len(token_mixture.feature_median), 12)
        self.assertNotIn("support_energy", responses[0])
        self.assertNotIn("rp_support_energy", tokens[0])

    def test_scoring_forwards_nondefault_reconstruction_sampling_limits(self):
        graphs = _graphs(4)
        model = _model()
        limits = {
            "max_support_edges": 1,
            "max_weight_traces": 2,
            "max_distribution_groups": 1,
            "decoder_chunk_size": 1,
        }

        with patch(
            "attention_graph.train.reconstruction_losses",
            wraps=__import__(
                "attention_graph.train", fromlist=["reconstruction_losses"]
            ).reconstruction_losses,
        ) as response_loss:
            score_graphs(
                model,
                fit_graphs=graphs[:3],
                score_graphs=graphs[3:],
                num_views=1,
                seed=71,
                **limits,
            )
        self.assertTrue(response_loss.call_args_list)
        self.assertTrue(
            all(
                all(call.kwargs[name] == value for name, value in limits.items())
                for call in response_loss.call_args_list
            )
        )

        with patch(
            "attention_graph.train.reconstruction_energy_by_node",
            wraps=__import__(
                "attention_graph.train", fromlist=["reconstruction_energy_by_node"]
            ).reconstruction_energy_by_node,
        ) as token_energy:
            score_tokens(
                model,
                fit_graphs=graphs[:3],
                score_graphs=graphs[3:],
                mask_stride=1,
                max_fit_tokens=100,
                seed=73,
                **limits,
            )
        self.assertTrue(token_energy.call_args_list)
        self.assertTrue(
            all(
                all(call.kwargs[name] == value for name, value in limits.items())
                for call in token_energy.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
