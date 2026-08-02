import inspect
import json
import tempfile
import unittest
from pathlib import Path

from unsupervised_token_graph.audit import UnsupervisedPatternProfiler
from unsupervised_token_graph.data import (
    compose_example,
    deterministic_split,
    write_prepared_dataset,
)
from unsupervised_token_graph.evaluation import summarize_feature_separation


def _example_ids(examples):
    return [example.example_id for example in examples]


def _nested_keys(value):
    if isinstance(value, dict):
        keys = set(value)
        for nested in value.values():
            keys.update(_nested_keys(nested))
        return keys
    if isinstance(value, list):
        keys = set()
        for nested in value:
            keys.update(_nested_keys(nested))
        return keys
    return set()


class UnsupervisedPatternProfilerTests(unittest.TestCase):
    def test_fit_is_label_blind_and_transform_names_attention_failure_modes(self):
        reference_records = [
            {
                "example_id": f"reference-{index}",
                "answer_to_passage_ratio": passage_ratio,
                "answer_self_reliance": self_reliance,
                "answer_attention_entropy": entropy,
            }
            for index, (passage_ratio, self_reliance, entropy) in enumerate(
                [
                    (0.55, 0.10, 0.20),
                    (0.60, 0.15, 0.25),
                    (0.65, 0.20, 0.30),
                    (0.70, 0.25, 0.35),
                    (0.75, 0.30, 0.40),
                ]
            )
        ]
        fit_parameters = inspect.signature(UnsupervisedPatternProfiler.fit).parameters
        self.assertNotIn("labels", fit_parameters)
        self.assertNotIn("y", fit_parameters)

        profiler = UnsupervisedPatternProfiler().fit(reference_records)
        with self.assertRaises(TypeError):
            UnsupervisedPatternProfiler().fit(reference_records, labels={})

        transformed = profiler.transform(
            [
                {
                    "example_id": "low-passage",
                    "answer_to_passage_ratio": 0.01,
                    "answer_self_reliance": 0.20,
                    "answer_attention_entropy": 0.30,
                },
                {
                    "example_id": "high-self",
                    "answer_to_passage_ratio": 0.65,
                    "answer_self_reliance": 0.99,
                    "answer_attention_entropy": 0.30,
                },
                {
                    "example_id": "high-entropy",
                    "answer_to_passage_ratio": 0.65,
                    "answer_self_reliance": 0.20,
                    "answer_attention_entropy": 0.99,
                },
            ]
        )
        patterns_by_id = {
            record["example_id"]: set(record["patterns"]) for record in transformed
        }

        self.assertIn("evidence_neglect", patterns_by_id["low-passage"])
        self.assertIn(
            "answer_self_reinforcement",
            patterns_by_id["high-self"],
        )
        self.assertIn("diffuse_attention", patterns_by_id["high-entropy"])


class LabelGatedEvaluationTests(unittest.TestCase):
    def test_feature_separation_reports_group_medians_auc_and_direction_free_score(self):
        records = [
            {
                "example_id": "normal-a",
                "answer_to_passage_ratio": 0.80,
                "answer_self_reliance": 0.10,
            },
            {
                "example_id": "normal-b",
                "answer_to_passage_ratio": 0.70,
                "answer_self_reliance": 0.20,
            },
            {
                "example_id": "error-a",
                "answer_to_passage_ratio": 0.20,
                "answer_self_reliance": 0.80,
            },
            {
                "example_id": "error-b",
                "answer_to_passage_ratio": 0.10,
                "answer_self_reliance": 0.90,
            },
        ]
        evaluation_labels = {
            "normal-a": 0,
            "normal-b": 0,
            "error-a": 1,
            "error-b": 1,
        }

        summary = summarize_feature_separation(records, evaluation_labels)

        self.assertAlmostEqual(
            summary["answer_self_reliance"]["median_label_0"], 0.15
        )
        self.assertAlmostEqual(
            summary["answer_self_reliance"]["median_label_1"], 0.85
        )
        self.assertAlmostEqual(summary["answer_self_reliance"]["auc"], 1.0)
        self.assertAlmostEqual(
            summary["answer_self_reliance"]["separability"], 1.0
        )
        self.assertAlmostEqual(
            summary["answer_to_passage_ratio"]["median_label_0"], 0.75
        )
        self.assertAlmostEqual(
            summary["answer_to_passage_ratio"]["median_label_1"], 0.15
        )
        self.assertAlmostEqual(summary["answer_to_passage_ratio"]["auc"], 0.0)
        self.assertAlmostEqual(
            summary["answer_to_passage_ratio"]["separability"], 1.0
        )


class PreparedDatasetTests(unittest.TestCase):
    def test_writer_keeps_examples_label_free_and_exports_labels_separately(self):
        examples = [
            compose_example(
                "A passage.",
                "A question?",
                "An answer.",
                example_id="example-a",
                pair_id="pair-a",
                dataset="halueval_qa",
                metadata={
                    "title": "safe metadata",
                    "candidate_role": "correct",
                    "is_hallucinated": False,
                    "y_token": [0, 0],
                },
            ),
            compose_example(
                "Another passage.",
                "Another question?",
                "Another answer.",
                example_id="example-b",
                pair_id="pair-a",
                dataset="halueval_qa",
            ),
        ]
        labels = {"example-a": 0, "example-b": 1}

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "prepared"
            write_prepared_dataset(examples, labels, output_dir)
            examples_text = (output_dir / "examples.jsonl").read_text(
                encoding="utf-8"
            )
            label_text = (output_dir / "evaluation_labels.jsonl").read_text(
                encoding="utf-8"
            )

        example_rows = [json.loads(line) for line in examples_text.splitlines()]
        label_rows = [json.loads(line) for line in label_text.splitlines()]
        forbidden_example_keys = {
            "candidate",
            "candidate_role",
            "candidate_type",
            "correct_candidate",
            "is_correct",
            "is_hallucinated",
            "label",
            "labels",
            "y",
            "y_token",
        }
        for row in example_rows:
            self.assertTrue(_nested_keys(row).isdisjoint(forbidden_example_keys))
        self.assertNotIn('"label"', examples_text)
        self.assertNotIn("y_token", examples_text)
        self.assertNotIn("is_hallucinated", examples_text)
        self.assertEqual(
            {row["example_id"]: row["label"] for row in label_rows},
            labels,
        )

    def test_deterministic_split_never_separates_candidates_from_one_pair(self):
        examples = []
        for pair_index in range(6):
            pair_id = f"pair-{pair_index}"
            for candidate_index in range(2):
                examples.append(
                    compose_example(
                        f"Passage {pair_index}",
                        f"Question {pair_index}?",
                        f"Candidate {candidate_index}",
                        example_id=f"{pair_id}-candidate-{candidate_index}",
                        pair_id=pair_id,
                        dataset="halueval_qa",
                    )
                )

        train_a, test_a = deterministic_split(
            examples,
            train_fraction=0.5,
            seed=29,
        )
        train_b, test_b = deterministic_split(
            examples,
            train_fraction=0.5,
            seed=29,
        )

        self.assertEqual(_example_ids(train_a), _example_ids(train_b))
        self.assertEqual(_example_ids(test_a), _example_ids(test_b))
        self.assertTrue(train_a)
        self.assertTrue(test_a)
        train_pairs = {example.pair_id for example in train_a}
        test_pairs = {example.pair_id for example in test_a}
        self.assertTrue(train_pairs.isdisjoint(test_pairs))
        self.assertEqual(
            set(_example_ids(train_a)) | set(_example_ids(test_a)),
            set(_example_ids(examples)),
        )


if __name__ == "__main__":
    unittest.main()
