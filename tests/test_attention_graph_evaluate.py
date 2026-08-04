from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from attention_graph.evaluate import (
    aggregate_sentence_probabilities,
    evaluate_binary_scores,
    evaluate_predictions,
    evaluate_sentence_predictions,
    load_evaluation_labels,
    map_token_offsets_to_sentences,
    sentence_char_spans,
)
from tests.test_attention_graph_data import _write_sample


class AttentionGraphEvaluationTests(unittest.TestCase):
    def test_binary_metrics_report_baseline_and_both_orientations(self):
        report = evaluate_binary_scores([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])

        self.assertAlmostEqual(report["auroc"], 0.0)
        self.assertAlmostEqual(report["orientation_free_auroc"], 1.0)
        self.assertAlmostEqual(report["average_precision_random_baseline"], 0.5)
        self.assertGreater(
            report["orientation_free_average_precision"],
            report["average_precision"],
        )

    def test_empty_and_single_class_metrics_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_binary_scores([], [])
        with self.assertRaisesRegex(ValueError, "both classes"):
            evaluate_binary_scores([0, 0], [0.1, 0.2])

    def test_labels_are_requested_only_by_explicit_evaluation_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention_0001.pt"
            _write_sample(path, source_id="s1", response_id="r1", split="test")
            from attention_graph.data import load_attention_record as real_loader

            with patch(
                "attention_graph.evaluate.load_attention_record",
                wraps=real_loader,
            ) as loader:
                labels = load_evaluation_labels([path], mmap=True)

        self.assertTrue(loader.call_args.kwargs["include_labels"])
        self.assertEqual(labels.response_labels[("s1", "r1")], 1)
        self.assertEqual(labels.token_labels[("s1", "r1", 3)], 1)

    def test_response_and_token_evaluation_require_exact_identity_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, (source, response, positive) in enumerate(
                (("s0", "r0", False), ("s1", "r1", True))
            ):
                path = root / f"attention_{index:04d}.pt"
                _write_sample(path, source_id=source, response_id=response, split="test")
                payload = torch.load(path, weights_only=True)
                payload["y_token"] = torch.tensor(
                    [0, 0, 0, int(positive), 0, 0], dtype=torch.float32
                )
                torch.save(payload, path)
                paths.append(path)
            labels = load_evaluation_labels(paths)
            responses = [
                {
                    "source_id": "s0",
                    "sample_id": "r0",
                    "hallucination_probability": 0.1,
                },
                {
                    "source_id": "s1",
                    "sample_id": "r1",
                    "hallucination_probability": 0.9,
                },
            ]
            tokens = [
                {
                    "source_id": source,
                    "sample_id": response,
                    "token_idx": token_idx,
                    "score": 0.9 if positive and token_idx == 3 else 0.1,
                }
                for source, response, positive in (
                    ("s0", "r0", False),
                    ("s1", "r1", True),
                )
                for token_idx in (3, 4, 5)
            ]

            report = evaluate_predictions(responses, tokens, labels)

        self.assertAlmostEqual(report["response"]["auroc"], 1.0)
        self.assertAlmostEqual(report["token"]["auroc"], 1.0)
        with self.assertRaisesRegex(ValueError, "alignment"):
            evaluate_predictions(responses[:-1], tokens, labels)

    def test_sentence_offsets_and_default_mean_probability_are_generic(self):
        text = "Dr. Ada answered. Next claim!"
        spans = sentence_char_spans(text)
        offsets = {3: (0, 7), 4: (8, 17), 5: (18, 22), 6: (23, 29)}
        membership = map_token_offsets_to_sentences(text, offsets)
        records = aggregate_sentence_probabilities(
            text,
            {3: 0.2, 4: 0.6, 5: 0.9, 6: 0.3},
            offsets,
        )

        self.assertEqual(len(spans), 2)
        self.assertEqual(membership, [[3, 4], [5, 6]])
        self.assertEqual([record["pooling"] for record in records], ["mean", "mean"])
        self.assertAlmostEqual(records[0]["score"], 0.4)
        self.assertAlmostEqual(records[1]["score"], 0.6)

    def test_sentence_evaluation_uses_any_positive_member_and_exact_token_coverage(self):
        labels = type("Labels", (), {})()
        labels.token_labels = {
            ("s0", "r0", 3): 0,
            ("s0", "r0", 4): 0,
            ("s1", "r1", 3): 1,
            ("s1", "r1", 4): 0,
        }
        sentence_records = [
            {
                "source_id": "s0",
                "sample_id": "r0",
                "sentence_idx": 0,
                "token_indices": [3, 4],
                "pooling": "mean",
                "score": 0.1,
            },
            {
                "source_id": "s1",
                "sample_id": "r1",
                "sentence_idx": 0,
                "token_indices": [3, 4],
                "pooling": "mean",
                "score": 0.9,
            },
        ]

        report = evaluate_sentence_predictions(sentence_records, labels)

        self.assertEqual(report["pooling"], "mean")
        self.assertAlmostEqual(report["metrics"]["auroc"], 1.0)
        with self.assertRaisesRegex(ValueError, "coverage"):
            evaluate_sentence_predictions(sentence_records[:-1], labels)


if __name__ == "__main__":
    unittest.main()
