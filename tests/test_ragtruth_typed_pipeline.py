"""CPU contracts for label-gated sparse-RAGTruth graph pipeline primitives."""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from unsupervised_token_graph.ragtruth_pipeline import (
    collate_graphs,
    discover_attention_paths,
    evaluate_token_score_records,
    load_cached_token_labels,
    make_answer_mask,
    split_paths_by_source,
)
from unsupervised_token_graph.typed_experiment import (
    _require_checkpoint_graph_semantics,
)


def _graph(*, source_id: str, sample_id: str, node_count: int, edges):
    """A label-free v3 graph whose final two nodes are response tokens."""

    edge_index = torch.tensor(edges, dtype=torch.long)
    return {
        "schema_version": "ragtruth_typed_topk_v3",
        "source_id": source_id,
        "sample_id": sample_id,
        "response_idx": node_count - 2,
        "x": torch.arange(node_count * 3, dtype=torch.float32).reshape(node_count, 3),
        "node_context": torch.zeros((node_count, 3), dtype=torch.float32),
        "edge_index": edge_index,
        "edge_attr": torch.ones((edge_index.shape[1], 8), dtype=torch.float32),
        "edge_type": torch.zeros(edge_index.shape[1], dtype=torch.long),
        "response_mask": torch.tensor([False] * (node_count - 2) + [True, True]),
        "neighbor_mean_target": torch.zeros((node_count, 2, 3)),
        "neighbor_log_variance_target": torch.zeros((node_count, 2, 3)),
        "route_stats_target": torch.zeros((node_count, 2, 4)),
    }


class GraphCollationContractTests(unittest.TestCase):
    def test_collate_offsets_edges_and_preserves_v3_sample_identities(self):
        first = _graph(
            source_id="s1", sample_id="response-11", node_count=3,
            edges=[[0, 1], [1, 2]],
        )
        second = _graph(
            source_id="s2", sample_id="response-12", node_count=4,
            edges=[[0, 2], [2, 3]],
        )

        batch = collate_graphs([first, second])

        self.assertEqual(batch["x"].shape, (7, 3))
        self.assertEqual(batch["graph_ptr"].tolist(), [0, 3, 7])
        self.assertEqual(batch["edge_index"].tolist(), [[0, 1, 3, 5], [1, 2, 5, 6]])
        self.assertEqual(batch["source_id"], ["s1", "s2"])
        self.assertEqual(batch["sample_id"], ["response-11", "response-12"])
        self.assertNotIn("original_idx", batch)

    def test_collate_recursively_rejects_y_token_before_training(self):
        graph = _graph(
            source_id="s1", sample_id="response-11", node_count=3,
            edges=[[0, 1], [1, 2]],
        )
        graph["metadata"] = {"upstream": {"y_token": torch.tensor([0, 1])}}

        with self.assertRaisesRegex(ValueError, "label|y_token"):
            collate_graphs([graph])
        parameters = inspect.signature(collate_graphs).parameters
        self.assertNotIn("labels", parameters)
        self.assertNotIn("y_token", parameters)

    def test_legacy_original_idx_is_optional_metadata_not_the_batch_identity(self):
        graph = _graph(
            source_id="legacy-source", sample_id="legacy-response-7", node_count=3,
            edges=[[0, 1], [1, 2]],
        )
        graph["original_idx"] = 7

        batch = collate_graphs([graph])

        self.assertEqual(batch["source_id"], ["legacy-source"])
        self.assertEqual(batch["sample_id"], ["legacy-response-7"])
        self.assertEqual(batch["original_idx"], [7])

    def test_collate_rejects_mixed_legacy_original_idx_metadata(self):
        legacy = _graph(
            source_id="legacy-source", sample_id="legacy-response-7", node_count=3,
            edges=[[0, 1], [1, 2]],
        )
        legacy["original_idx"] = 7
        formal = _graph(
            source_id="formal-source", sample_id="formal-response-8", node_count=3,
            edges=[[0, 1], [1, 2]],
        )

        with self.assertRaisesRegex(ValueError, "original_idx.*every graph"):
            collate_graphs([legacy, formal])


class SourceGroupedSplitContractTests(unittest.TestCase):
    def test_split_keeps_all_response_samples_for_one_source_in_one_partition(self):
        records = [
            {"path": Path("a_0.pt"), "source_id": "source-a", "sample_id": "ra0"},
            {"path": Path("a_1.pt"), "source_id": "source-a", "sample_id": "ra1"},
            {"path": Path("b_0.pt"), "source_id": "source-b", "sample_id": "rb0"},
            {"path": Path("c_0.pt"), "source_id": "source-c", "sample_id": "rc0"},
        ]

        splits = split_paths_by_source(
            records, train_fraction=0.5, validation_fraction=0.25, seed=3
        )

        self.assertEqual(set(splits), {"train", "validation", "test"})
        source_by_path = {row["path"]: row["source_id"] for row in records}
        assigned = [path for paths in splits.values() for path in paths]
        self.assertEqual(set(assigned), set(source_by_path))
        partition_by_source = {}
        for partition, paths in splits.items():
            for path in paths:
                source = source_by_path[path]
                self.assertEqual(partition_by_source.setdefault(source, partition), partition)


class AttentionDiscoveryContractTests(unittest.TestCase):
    def test_discovery_accepts_one_homogeneous_flat_cache_family(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "attention_000.pt").touch()
            (root / "attention_001.pt").touch()

            paths = discover_attention_paths(root)

        self.assertEqual([path.name for path in paths], ["attention_000.pt", "attention_001.pt"])


class AnswerMaskContractTests(unittest.TestCase):
    def test_mask_has_at_least_one_response_token_per_graph_and_never_masks_context(self):
        response_mask = torch.tensor(
            [False, False, True, True, False, True, True, True], dtype=torch.bool
        )
        graph_ptr = torch.tensor([0, 4, 8], dtype=torch.long)

        masked = make_answer_mask(
            response_mask, graph_ptr, mask_ratio=0.01,
            generator=torch.Generator().manual_seed(12),
        )

        self.assertEqual(masked.dtype, torch.bool)
        self.assertFalse(bool(masked[~response_mask].any()))
        self.assertGreaterEqual(int(masked[:4].sum()), 1)
        self.assertGreaterEqual(int(masked[4:].sum()), 1)


class LabelGatedTokenEvaluationContractTests(unittest.TestCase):
    def test_evaluation_joins_external_labels_by_source_sample_and_token(self):
        scores = [
            {"source_id": "s1", "sample_id": "r1", "token_idx": 3, "score": 0.05},
            {"source_id": "s1", "sample_id": "r1", "token_idx": 4, "score": 0.95},
            {"source_id": "s2", "sample_id": "r2", "token_idx": 2, "score": 0.20},
            {"source_id": "s2", "sample_id": "r2", "token_idx": 3, "score": 0.80},
        ]
        labels = {
            ("s1", "r1", 3): 0,
            ("s1", "r1", 4): 1,
            ("s2", "r2", 2): 0,
            ("s2", "r2", 3): 1,
        }

        report = evaluate_token_score_records(scores, labels)
        expected_labels, expected_scores = [0, 1, 0, 1], [0.05, 0.95, 0.20, 0.80]
        self.assertEqual(report["token_count"], 4)
        self.assertEqual(report["positive_count"], 2)
        self.assertAlmostEqual(report["auroc"], roc_auc_score(expected_labels, expected_scores))
        self.assertAlmostEqual(report["auprc"], average_precision_score(expected_labels, expected_scores))
        self.assertTrue(all("y_token" not in row and "label" not in row for row in scores))

    def test_evaluation_rejects_missing_or_extra_v3_token_identity(self):
        scores = [
            {"source_id": "s1", "sample_id": "r1", "token_idx": 3, "score": 0.1},
            {"source_id": "s1", "sample_id": "r1", "token_idx": 4, "score": 0.9},
        ]
        with self.assertRaisesRegex(ValueError, "align|missing|label"):
            evaluate_token_score_records(scores, {("s1", "r1", 3): 0})
        with self.assertRaisesRegex(ValueError, "align|extra|label"):
            evaluate_token_score_records(
                scores,
                {("s1", "r1", 3): 0, ("s1", "r1", 4): 1, ("s1", "r1", 5): 0},
            )

    def test_evaluation_rejects_fractional_external_labels_before_casting(self):
        scores = [
            {"source_id": "s1", "sample_id": "r1", "token_idx": 3, "score": 0.1},
            {"source_id": "s1", "sample_id": "r1", "token_idx": 4, "score": 0.9},
        ]
        with self.assertRaisesRegex(ValueError, "binary|0.*1|label"):
            evaluate_token_score_records(
                scores, {("s1", "r1", 3): 0.5, ("s1", "r1", 4): 1}
            )

    def test_legacy_label_join_falls_back_to_attention_length_without_token_ids(self):
        scores = [
            {"source_id": "legacy", "sample_id": "7", "token_idx": 3, "score": 0.1},
            {"source_id": "legacy", "sample_id": "7", "token_idx": 4, "score": 0.9},
        ]
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample_7.pt"
            torch.save(
                {
                    "source_id": "legacy",
                    "original_idx": 7,
                    "response_idx": 3,
                    "attention": torch.zeros((1, 1, 5, 5)),
                    "hallucination_labels": torch.tensor([0, 0, 0, 0, 1]),
                },
                path,
            )
            labels, alignment = load_cached_token_labels(
                temporary_directory, scores, attention_paths=[path]
            )

        self.assertEqual(labels[("legacy", "7", 3)], 0)
        self.assertEqual(labels[("legacy", "7", 4)], 1)
        self.assertEqual(alignment["matched_tokens"], 2)

    def test_cached_fractional_labels_are_rejected_before_long_conversion(self):
        scores = [
            {"source_id": "legacy", "sample_id": "7", "token_idx": 3, "score": 0.1},
            {"source_id": "legacy", "sample_id": "7", "token_idx": 4, "score": 0.9},
        ]
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample_7.pt"
            torch.save(
                {
                    "source_id": "legacy",
                    "original_idx": 7,
                    "response_idx": 3,
                    "attention": torch.zeros((1, 1, 5, 5)),
                    "hallucination_labels": torch.tensor([0, 0, 0, 0.5, 1.0]),
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "binary|0.*1|label"):
                load_cached_token_labels(
                    temporary_directory, scores, attention_paths=[path]
                )

    def test_legacy_label_shift_rejects_positive_moved_beyond_token_range(self):
        scores = [
            {"source_id": "legacy", "sample_id": "7", "token_idx": 4, "score": 0.9},
        ]
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample_7.pt"
            torch.save(
                {
                    "source_id": "legacy",
                    "original_idx": 7,
                    "response_idx": 3,
                    "attention": torch.zeros((1, 1, 5, 5)),
                    "hallucination_labels": torch.tensor([0, 0, 0, 0, 1]),
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "outside the response token range"):
                load_cached_token_labels(
                    temporary_directory,
                    scores,
                    label_shift=1,
                    attention_paths=[path],
                )

    def test_label_join_rejects_raw_cache_provenance_drift(self):
        scores = [
            {"source_id": "legacy", "sample_id": "7", "token_idx": 3, "score": 0.1},
        ]
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample_7.pt"
            torch.save(
                {
                    "source_id": "legacy",
                    "original_idx": 7,
                    "response_idx": 3,
                    "attention": torch.zeros((1, 1, 5, 5)),
                    "hallucination_labels": torch.tensor([0, 0, 0, 0, 1]),
                },
                path,
            )
            provenance = {
                ("legacy", "7"): {
                    "cache_format": "legacy_dense",
                    "attention_cache_fingerprint": None,
                    "cache_dtype": "float32",
                    "input_policy": None,
                    "was_truncated": None,
                    "attention_floor": None,
                    "response_idx": 4,
                    "token_count": 5,
                    "layers": 1,
                    "heads": 1,
                }
            }
            with self.assertRaisesRegex(RuntimeError, "provenance diverges|response_idx"):
                load_cached_token_labels(
                    temporary_directory,
                    scores,
                    attention_paths=[path],
                    expected_provenance=provenance,
                )


class GraphSemanticBindingContractTests(unittest.TestCase):
    def test_checkpoint_cannot_score_a_different_graph_semantic_signature(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ragtruth_typed_topk_corpus_v3",
                        "state": "complete",
                        "graph_semantic_signature": "formal-signature",
                    }
                ),
                encoding="utf-8",
            )
            _require_checkpoint_graph_semantics(
                {"graph_semantic_signature": "formal-signature"}, root
            )
            with self.assertRaisesRegex(RuntimeError, "semantic signature|checkpoint|graph"):
                _require_checkpoint_graph_semantics(
                    {"graph_semantic_signature": "legacy-signature"}, root
                )


if __name__ == "__main__":
    unittest.main()
