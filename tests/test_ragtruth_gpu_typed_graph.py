"""Contracts for the GPU-resident, label-blind RAGTruth token-graph path.

These are deliberately tiny fixtures: they run on CPU-only CI, but exercise CUDA
when it is available.  They specify the boundary between the legacy ``.pt``
attention cache and the new typed neighbourhood autoencoder without requiring
RAGTruth labels during graph construction, training, or scoring.
"""

from __future__ import annotations

import inspect
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

import torch

from unsupervised_token_graph.ragtruth_graph import (
    build_compact_topk_graph,
    load_attention_sample,
)
from unsupervised_token_graph.typed_model import (
    TypedNeighborhoodAutoencoder,
    score_masked_tokens,
    typed_reconstruction_loss,
)


FORBIDDEN_LABEL_KEYS = {"label", "labels", "target", "y", "y_token"}


def _field(value, name):
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _keys(value):
    if isinstance(value, Mapping):
        return set(value)
    return set(vars(value))


def _contains_label_key(value) -> bool:
    return any("label" in str(name).casefold() for name in _keys(value))


def _test_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _legacy_attention() -> torch.Tensor:
    """Two layers/two heads with unambiguous causal top-2 predecessors."""

    attention = torch.zeros((2, 2, 5, 5), dtype=torch.float32)
    # answer token 3: both selected neighbours are source tokens 0 and 2
    attention[:, :, 3, 0] = 0.90
    attention[:, :, 3, 2] = 0.80
    attention[:, :, 3, 1] = 0.10
    # answer token 4: one source token and one earlier answer token
    attention[:, :, 4, 1] = 0.95
    attention[:, :, 4, 3] = 0.85
    attention[:, :, 4, 2] = 0.15
    return attention


def _write_legacy_cache(path: Path, *, include_hidden: bool = True) -> None:
    sample = {
            "source_id": "source-42",
            "original_idx": 19,
            "response_idx": 3,
            "token_ids": torch.tensor([10, 11, 12, 13, 14]),
            "attention": _legacy_attention(),
            # This legacy-only field must not reach graph construction by default.
            "hallucination_labels": torch.tensor([0, 0, 0, 1, 0]),
        }
    if include_hidden:
        sample["hidden"] = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    torch.save(sample, path)


class LegacyAttentionCacheContractTests(unittest.TestCase):
    def test_mmap_loader_moves_allowlisted_tensors_to_requested_device_and_drops_labels(self):
        device = _test_device()
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "sample_19.pt"
            _write_legacy_cache(cache_path)

            sample = load_attention_sample(
                cache_path,
                device=device,
                mmap=True,
                include_labels=False,
            )

        self.assertEqual(sample["source_id"], "source-42")
        self.assertEqual(sample["original_idx"], 19)
        self.assertEqual(sample["response_idx"], 3)
        self.assertFalse(_contains_label_key(sample))
        self.assertTrue(FORBIDDEN_LABEL_KEYS.isdisjoint(_keys(sample)))
        for name in ("token_ids", "attention", "hidden"):
            self.assertEqual(sample[name].device, device)
        self.assertEqual(sample["attention"].shape, (2, 2, 5, 5))

    def test_loader_exposes_no_label_parameter_to_graph_training_callers(self):
        parameters = inspect.signature(load_attention_sample).parameters

        self.assertIn("include_labels", parameters)
        self.assertNotIn("labels", parameters)
        self.assertNotIn("y_token", parameters)

    def test_attention_only_legacy_cache_remains_buildable_without_hidden_states(self):
        """Some already-extracted RAGTruth caches contain attention only."""

        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "attention_only.pt"
            _write_legacy_cache(cache_path, include_hidden=False)
            sample = load_attention_sample(
                cache_path, device=device, mmap=True, include_labels=False
            )
            graph = build_compact_topk_graph(sample, top_k=2, device=device)

        self.assertNotIn("hidden", sample)
        self.assertEqual(_field(graph, "x").shape[0], 5)


class CompactTopKGraphContractTests(unittest.TestCase):
    def test_topk_graph_is_causal_typed_label_free_and_stays_on_requested_device(self):
        device = _test_device()
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "sample_19.pt"
            _write_legacy_cache(cache_path)
            sample = load_attention_sample(
                cache_path, device=device, mmap=True, include_labels=False
            )
            graph = build_compact_topk_graph(sample, top_k=2, device=device)

        required = {
            "x",
            "edge_index",
            "edge_attr",
            "edge_type",
            "response_mask",
            "original_idx",
            "source_id",
        }
        self.assertTrue(required.issubset(_keys(graph)))
        self.assertFalse(_contains_label_key(graph))
        self.assertTrue(FORBIDDEN_LABEL_KEYS.isdisjoint(_keys(graph)))
        self.assertEqual(_field(graph, "source_id"), "source-42")
        self.assertEqual(int(_field(graph, "original_idx")), 19)

        x = _field(graph, "x")
        edge_index = _field(graph, "edge_index")
        edge_attr = _field(graph, "edge_attr")
        edge_type = _field(graph, "edge_type")
        response_mask = _field(graph, "response_mask")
        self.assertEqual(x.device, device)
        self.assertEqual(edge_index.device, device)
        self.assertEqual(edge_attr.device, device)
        self.assertEqual(edge_type.device, device)
        self.assertEqual(response_mask.device, device)
        self.assertEqual(response_mask.dtype, torch.bool)
        self.assertEqual(
            response_mask.cpu().tolist(), [False, False, False, True, True]
        )
        self.assertEqual(edge_index.shape, (2, 4))
        self.assertEqual(edge_attr.shape, (4, 8))
        self.assertEqual(edge_type.shape, (4,))

        edges = {tuple(edge) for edge in edge_index.t().cpu().tolist()}
        self.assertEqual(edges, {(0, 3), (2, 3), (1, 4), (3, 4)})
        self.assertTrue(bool((edge_index[0] < edge_index[1]).all()))
        self.assertTrue(bool(response_mask[edge_index[1]].all()))

        # Source->response and answer-history->response edges must remain distinct
        # relations even though both are selected by the same vectorised top-k pass.
        source_relation = edge_type[edge_index[1] == 3]
        history_relation = edge_type[(edge_index[0] == 3) & (edge_index[1] == 4)]
        self.assertTrue(bool((source_relation == source_relation[0]).all()))
        self.assertNotEqual(int(source_relation[0]), int(history_relation[0]))


class TypedNeighborhoodAutoencoderContractTests(unittest.TestCase):
    def _graph(self):
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "sample_19.pt"
            _write_legacy_cache(cache_path)
            sample = load_attention_sample(
                cache_path, device=device, mmap=True, include_labels=False
            )
            return build_compact_topk_graph(sample, top_k=2, device=device)

    def test_typed_reconstruction_is_masked_response_only_and_has_gradients(self):
        graph = self._graph()
        x = _field(graph, "x")
        edge_attr = _field(graph, "edge_attr")
        edge_type = _field(graph, "edge_type")
        masked_nodes = _field(graph, "response_mask").clone()
        model = TypedNeighborhoodAutoencoder(
            node_dim=x.shape[1],
            edge_dim=edge_attr.shape[1],
            num_edge_types=int(edge_type.max()) + 1,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            context_dim=_field(graph, "node_context").shape[1],
        )

        outputs = model(graph, masked_nodes)
        self.assertTrue(
            {
                "node_reconstruction",
                "neighborhood_mean",
                "neighborhood_log_variance",
                "route_stats",
            }.issubset(outputs)
        )
        self.assertEqual(outputs["node_reconstruction"].shape, x.shape)
        # Axis 1 separates source->response from answer-history->response.
        self.assertEqual(outputs["neighborhood_mean"].shape[:2], (len(x), 2))
        self.assertEqual(
            outputs["neighborhood_log_variance"].shape,
            outputs["neighborhood_mean"].shape,
        )
        self.assertEqual(outputs["route_stats"].shape[:2], (len(x), 2))
        loss = typed_reconstruction_loss(outputs, graph, masked_nodes)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_scores_are_token_aligned_and_cannot_read_legacy_hallucination_labels(self):
        graph = self._graph()
        x = _field(graph, "x")
        edge_attr = _field(graph, "edge_attr")
        edge_type = _field(graph, "edge_type")
        model = TypedNeighborhoodAutoencoder(
            node_dim=x.shape[1],
            edge_dim=edge_attr.shape[1],
            num_edge_types=int(edge_type.max()) + 1,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            context_dim=_field(graph, "node_context").shape[1],
        )
        masked_nodes = _field(graph, "response_mask")

        parameters = inspect.signature(score_masked_tokens).parameters
        self.assertNotIn("labels", parameters)
        self.assertNotIn("y", parameters)
        self.assertNotIn("y_token", parameters)
        scores = score_masked_tokens(model, graph, masked_nodes)

        self.assertTrue(
            {"original_idx", "source_id", "token_idx", "scores"}.issubset(scores)
        )
        self.assertEqual(scores["source_id"], "source-42")
        self.assertEqual(int(scores["original_idx"]), 19)
        self.assertEqual(scores["token_idx"].tolist(), [3, 4])
        self.assertEqual(scores["scores"].shape, (2,))
        self.assertTrue(torch.isfinite(scores["scores"]).all())
        self.assertFalse(_contains_label_key(scores))


if __name__ == "__main__":
    unittest.main()
