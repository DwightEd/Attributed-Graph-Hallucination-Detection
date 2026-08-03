"""Small CPU contracts for the label-gated typed RAGTruth training pipeline."""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from unsupervised_token_graph.ragtruth_pipeline import (
    collate_graphs,
    evaluate_token_score_records,
    make_answer_mask,
    split_paths_by_source,
)


def _graph(*, source_id: str, original_idx: int, node_count: int, edges):
    """Create a tiny label-free graph whose final nodes are answer tokens."""

    edge_index = torch.tensor(edges, dtype=torch.long)
    return {
        "x": torch.arange(node_count * 3, dtype=torch.float32).reshape(node_count, 3),
        "edge_index": edge_index,
        "edge_attr": torch.ones((edge_index.shape[1], 2), dtype=torch.float32),
        "edge_type": torch.zeros(edge_index.shape[1], dtype=torch.long),
        "response_mask": torch.tensor(
            [False] * (node_count - 2) + [True, True], dtype=torch.bool
        ),
        "source_id": source_id,
        "original_idx": original_idx,
    }


class GraphCollationContractTests(unittest.TestCase):
    def test_collate_offsets_edges_and_preserves_graph_boundaries(self):
        first = _graph(
            source_id="s1",
            original_idx=11,
            node_count=3,
            edges=[[0, 1], [1, 2]],
        )
        second = _graph(
            source_id="s2",
            original_idx=12,
            node_count=4,
            edges=[[0, 2], [2, 3]],
        )

        batch = collate_graphs([first, second])

        self.assertEqual(batch["x"].shape, (7, 3))
        self.assertEqual(batch["graph_ptr"].tolist(), [0, 3, 7])
        self.assertEqual(
            batch["edge_index"].tolist(), [[0, 1, 3, 5], [1, 2, 5, 6]]
        )
        self.assertEqual(batch["response_mask"].tolist(), [False, True, True, False, False, True, True])
        self.assertEqual(batch["source_id"], ["s1", "s2"])
        self.assertEqual(batch["original_idx"], [11, 12])

    def test_collate_recursively_rejects_labels_before_training(self):
        graph = _graph(
            source_id="s1",
            original_idx=11,
            node_count=3,
            edges=[[0, 1], [1, 2]],
        )
        graph["metadata"] = {"precomputed": {"token_labels": torch.tensor([0, 1])}}

        with self.assertRaisesRegex(ValueError, "label"):
            collate_graphs([graph])

        parameters = inspect.signature(collate_graphs).parameters
        self.assertNotIn("labels", parameters)
        self.assertNotIn("y_token", parameters)


class SourceGroupedSplitContractTests(unittest.TestCase):
    def test_split_assigns_each_source_to_exactly_one_partition(self):
        records = [
            {"path": Path("a_0.pt"), "source_id": "source-a"},
            {"path": Path("a_1.pt"), "source_id": "source-a"},
            {"path": Path("b_0.pt"), "source_id": "source-b"},
            {"path": Path("c_0.pt"), "source_id": "source-c"},
            {"path": Path("c_1.pt"), "source_id": "source-c"},
        ]

        splits = split_paths_by_source(
            records, train_fraction=0.5, validation_fraction=0.25, seed=3
        )

        self.assertEqual(set(splits), {"train", "validation", "test"})
        assigned = [path for paths in splits.values() for path in paths]
        self.assertEqual(set(assigned), {row["path"] for row in records})
        self.assertEqual(len(assigned), len(set(assigned)))
        source_by_path = {row["path"]: row["source_id"] for row in records}
        partition_by_source = {}
        for partition, paths in splits.items():
            for path in paths:
                source = source_by_path[path]
                self.assertEqual(partition_by_source.setdefault(source, partition), partition)


class AnswerMaskContractTests(unittest.TestCase):
    def test_mask_has_at_least_one_response_token_per_graph_and_never_masks_context(self):
        response_mask = torch.tensor(
            [False, False, True, True, False, True, True, True], dtype=torch.bool
        )
        graph_ptr = torch.tensor([0, 4, 8], dtype=torch.long)

        masked = make_answer_mask(
            response_mask,
            graph_ptr,
            mask_ratio=0.01,
            generator=torch.Generator().manual_seed(12),
        )

        self.assertEqual(masked.dtype, torch.bool)
        self.assertFalse(bool(masked[~response_mask].any()))
        self.assertGreaterEqual(int(masked[:4].sum()), 1)
        self.assertGreaterEqual(int(masked[4:].sum()), 1)


class LabelGatedTokenEvaluationContractTests(unittest.TestCase):
    def test_evaluation_joins_labels_only_at_the_end_and_matches_sklearn_metrics(self):
        score_records = [
            {"source_id": "s1", "original_idx": 1, "token_idx": 3, "score": 0.05},
            {"source_id": "s1", "original_idx": 1, "token_idx": 4, "score": 0.95},
            {"source_id": "s2", "original_idx": 2, "token_idx": 2, "score": 0.20},
            {"source_id": "s2", "original_idx": 2, "token_idx": 3, "score": 0.80},
        ]
        token_labels = {
            ("s1", 1, 3): 0,
            ("s1", 1, 4): 1,
            ("s2", 2, 2): 0,
            ("s2", 2, 3): 1,
        }

        report = evaluate_token_score_records(score_records, token_labels)
        expected_labels = [0, 1, 0, 1]
        expected_scores = [0.05, 0.95, 0.20, 0.80]

        self.assertEqual(report["token_count"], 4)
        self.assertEqual(report["positive_count"], 2)
        self.assertAlmostEqual(report["auroc"], roc_auc_score(expected_labels, expected_scores))
        self.assertAlmostEqual(
            report["auprc"], average_precision_score(expected_labels, expected_scores)
        )
        self.assertNotIn("label", score_records[0])

    def test_evaluation_rejects_any_missing_or_extra_token_label(self):
        score_records = [
            {"source_id": "s1", "original_idx": 1, "token_idx": 3, "score": 0.1},
            {"source_id": "s1", "original_idx": 1, "token_idx": 4, "score": 0.9},
        ]
        missing = {("s1", 1, 3): 0}
        extra = {
            ("s1", 1, 3): 0,
            ("s1", 1, 4): 1,
            ("s1", 1, 5): 0,
        }

        with self.assertRaisesRegex(ValueError, "align|missing|label"):
            evaluate_token_score_records(score_records, missing)
        with self.assertRaisesRegex(ValueError, "align|extra|label"):
            evaluate_token_score_records(score_records, extra)


if __name__ == "__main__":
    unittest.main()
