import inspect
import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from unsupervised_token_graph.data import (
    compose_example,
    load_boolq_predictions,
    load_halueval_qa,
)
from unsupervised_token_graph.features import summarize_attention_trace
from unsupervised_token_graph.generate_boolq import (
    build_boolq_generation_prompt,
    generate_boolq_predictions,
)
from unsupervised_token_graph.scoring import RobustMahalanobisScorer


def _field(record, name):
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _record_fields(record):
    if isinstance(record, Mapping):
        return set(record)
    return set(vars(record))


def _mean_scalar(value):
    return float(np.asarray(value, dtype=float).mean())


class DatasetAdapterContractTests(unittest.TestCase):
    def test_halueval_expands_each_pair_without_putting_labels_in_examples(self):
        row = {
            "knowledge": "Mercury is the closest planet to the Sun.",
            "question": "Which planet is closest to the Sun?",
            "right_answer": "Mercury is closest to the Sun.",
            "hallucinated_answer": "Venus is closest to the Sun.",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "qa_data.json"
            dataset_path.write_text(json.dumps([row]), encoding="utf-8")

            examples, evaluation_labels = load_halueval_qa(dataset_path)

        self.assertEqual(len(examples), 2)
        self.assertEqual(len({_field(example, "pair_id") for example in examples}), 1)
        self.assertEqual(
            {_field(example, "answer") for example in examples},
            {row["right_answer"], row["hallucinated_answer"]},
        )

        example_ids = {_field(example, "example_id") for example in examples}
        self.assertEqual(set(evaluation_labels), example_ids)
        self.assertEqual(set(evaluation_labels.values()), {0, 1})

        forbidden_label_fields = {
            "candidate_role",
            "candidate_type",
            "gold_answer",
            "is_correct",
            "is_hallucinated",
            "label",
            "labels",
            "target",
            "y",
        }
        for example in examples:
            self.assertTrue(
                _record_fields(example).isdisjoint(forbidden_label_fields),
                "Ground-truth fields must be returned separately from graph inputs.",
            )

    def test_boolq_false_means_gold_no_and_not_hallucination_by_itself(self):
        boolq_rows = [
            {
                "id": "false-answered-no",
                "passage": "The Pacific Ocean is larger than the Atlantic Ocean.",
                "question": "Is the Atlantic Ocean larger than the Pacific Ocean?",
                "answer": False,
            },
            {
                "id": "false-answered-yes",
                "passage": "Mercury is the closest planet to the Sun.",
                "question": "Is Venus the closest planet to the Sun?",
                "answer": False,
            },
        ]
        prediction_rows = [
            {"id": "false-answered-no", "model_answer": "No."},
            {"id": "false-answered-yes", "model_answer": "Yes."},
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "boolq.jsonl"
            predictions_path = root / "predictions.jsonl"
            dataset_path.write_text(
                "\n".join(json.dumps(row) for row in boolq_rows) + "\n",
                encoding="utf-8",
            )
            predictions_path.write_text(
                "\n".join(json.dumps(row) for row in prediction_rows) + "\n",
                encoding="utf-8",
            )

            examples, evaluation_labels = load_boolq_predictions(
                dataset_path,
                predictions_path,
            )

        examples_by_id = {
            _field(example, "example_id"): example for example in examples
        }
        self.assertEqual(set(examples_by_id), set(evaluation_labels))
        self.assertEqual(evaluation_labels["false-answered-no"], 0)
        self.assertEqual(evaluation_labels["false-answered-yes"], 1)
        self.assertEqual(_field(examples_by_id["false-answered-no"], "answer"), "No.")
        self.assertEqual(_field(examples_by_id["false-answered-yes"], "answer"), "Yes.")
        self.assertEqual(
            _field(examples_by_id["false-answered-no"], "text"),
            build_boolq_generation_prompt(
                boolq_rows[0]["passage"], boolq_rows[0]["question"]
            )
            + "No.",
        )

        for example in examples:
            self.assertNotIn("gold_answer", _record_fields(example))
            self.assertNotIn("is_hallucinated", _record_fields(example))

    def test_boolq_generation_prompt_has_no_gold_answer_input(self):
        parameters = inspect.signature(build_boolq_generation_prompt).parameters
        self.assertEqual(list(parameters), ["passage", "question"])

        prompt = build_boolq_generation_prompt(
            "The passage states the evidence.",
            "Does the passage state the evidence?",
        )

        self.assertIn("The passage states the evidence.", prompt)
        self.assertIn("Does the passage state the evidence?", prompt)
        self.assertIn("Yes or No", prompt)

    def test_boolq_resume_rejects_predictions_from_another_model(self):
        class FakeTokenizer:
            eos_token_id = 0

            def __call__(self, prompt, **options):
                return {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.ones((1, 2), dtype=torch.long),
                    "offset_mapping": torch.tensor([[[0, 0], [0, 1]]]),
                    "special_tokens_mask": torch.tensor([[1, 0]]),
                }

            def decode(self, tokens, **options):
                return "Yes"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def generate(self, input_ids, **options):
                suffix = torch.tensor([[3]], device=input_ids.device)
                return torch.cat((input_ids, suffix), dim=1)

        records = [{"id": "example", "passage": "P", "question": "Q?"}]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "predictions.jsonl"
            generate_boolq_predictions(
                FakeModel(),
                FakeTokenizer(),
                records,
                output_path=output,
                model_id="model-a",
                max_input_tokens=16,
                max_new_tokens=2,
            )
            generated_row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(generated_row["replay_input_ids"], [1, 2, 3])
            self.assertEqual(generated_row["replay_segment_ids"][-1], 3)
            self.assertEqual(len(generated_row["replay_segment_ids"]), 3)

            with self.assertRaisesRegex(RuntimeError, "stale BoolQ prediction"):
                generate_boolq_predictions(
                    FakeModel(),
                    FakeTokenizer(),
                    records,
                    output_path=output,
                    model_id="model-b",
                    max_input_tokens=16,
                    max_new_tokens=2,
                )

    def test_boolq_generation_quarantines_non_yes_no_outputs(self):
        class FakeTokenizer:
            eos_token_id = 0

            def __init__(self):
                self.decode_count = 0

            def __call__(self, prompt, **options):
                return {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.ones((1, 2), dtype=torch.long),
                    "offset_mapping": torch.tensor([[[0, 0], [0, 1]]]),
                    "special_tokens_mask": torch.tensor([[1, 0]]),
                }

            def decode(self, tokens, **options):
                self.decode_count += 1
                return "Maybe" if self.decode_count == 1 else "No"

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def generate(self, input_ids, **options):
                suffix = torch.tensor([[3]], device=input_ids.device)
                return torch.cat((input_ids, suffix), dim=1)

        records = [
            {"id": "invalid", "passage": "P1", "question": "Q1?"},
            {"id": "valid", "passage": "P2", "question": "Q2?"},
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "predictions.jsonl"
            count = generate_boolq_predictions(
                FakeModel(),
                FakeTokenizer(),
                records,
                output_path=output,
                model_id="model-a",
                max_input_tokens=16,
                max_new_tokens=2,
            )
            valid_rows = [json.loads(line) for line in output.read_text().splitlines()]
            invalid_rows = [
                json.loads(line)
                for line in (output.parent / f"{output.name}.invalid.jsonl")
                .read_text()
                .splitlines()
            ]

        self.assertEqual(count, 1)
        self.assertEqual([row["id"] for row in valid_rows], ["valid"])
        self.assertEqual([row["id"] for row in invalid_rows], ["invalid"])

    def test_dataset_adapters_reject_ambiguous_ids_and_non_boolean_boolq_gold(self):
        duplicate_halueval = {
            "knowledge": "K",
            "question": "Q?",
            "right_answer": "same",
            "hallucinated_answer": "same",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            halueval_path = root / "halueval.json"
            halueval_path.write_text(
                json.dumps([duplicate_halueval]), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "identical candidates"):
                load_halueval_qa(halueval_path)

            boolq_path = root / "boolq.jsonl"
            predictions_path = root / "predictions.jsonl"
            boolq_path.write_text(
                json.dumps(
                    {"id": "x", "passage": "P", "question": "Q?", "answer": "false"}
                )
                + "\n",
                encoding="utf-8",
            )
            predictions_path.write_text(
                json.dumps({"id": "x", "model_answer": "No"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "actual boolean"):
                load_boolq_predictions(boolq_path, predictions_path)


class ExampleCompositionTests(unittest.TestCase):
    def test_compose_example_concatenates_all_segments_and_records_char_spans(self):
        passage = "Evidence sentence unique to the passage."
        question = "Question sentence with a distinct phrase?"
        answer = "Answer sentence with its own wording."

        example = compose_example(
            passage=passage,
            question=question,
            answer=answer,
            example_id="example-1",
        )

        text = _field(example, "text")
        spans = _field(example, "segment_char_spans")
        self.assertEqual(set(spans), {"passage", "question", "answer"})

        expected_segments = {
            "passage": passage,
            "question": question,
            "answer": answer,
        }
        for segment_name, expected_text in expected_segments.items():
            start, end = spans[segment_name]
            self.assertEqual(text[start:end], expected_text)

        passage_start, passage_end = spans["passage"]
        question_start, question_end = spans["question"]
        answer_start, answer_end = spans["answer"]
        self.assertLessEqual(passage_end, question_start)
        self.assertLessEqual(question_end, answer_start)
        self.assertEqual(passage_start, text.index(passage))
        self.assertEqual(answer_end, text.index(answer) + len(answer))


class AttentionFeatureTests(unittest.TestCase):
    def test_summary_keeps_answer_attention_mass_to_each_input_segment_separate(self):
        attention = np.zeros((1, 1, 6, 6), dtype=np.float64)
        attention[:, :, :4, :] = 1.0 / 6.0
        attention[:, :, 4, :] = np.array(
            [0.35, 0.35, 0.10, 0.10, 0.10, 0.00], dtype=np.float64
        )
        attention[:, :, 5, :] = np.array(
            [0.35, 0.35, 0.10, 0.10, 0.05, 0.05], dtype=np.float64
        )
        segment_token_spans = {
            "passage": (0, 2),
            "question": (2, 4),
            "answer": (4, 6),
        }

        summary = summarize_attention_trace(attention, segment_token_spans)

        passage_mass = _mean_scalar(summary["answer_to_passage_mass"])
        question_mass = _mean_scalar(summary["answer_to_question_mass"])
        answer_mass = _mean_scalar(summary["answer_to_answer_mass"])
        self.assertAlmostEqual(passage_mass, 0.70)
        self.assertAlmostEqual(question_mass, 0.20)
        self.assertAlmostEqual(answer_mass, 0.025)
        self.assertAlmostEqual(
            _mean_scalar(summary["answer_to_prior_answer_mass"]), 0.025
        )
        self.assertGreater(passage_mass, question_mass)
        self.assertGreater(question_mass, answer_mass)
        self.assertGreater(
            _mean_scalar(summary["answer_to_passage_token_normalized"]),
            _mean_scalar(summary["answer_to_question_token_normalized"]),
        )


class RobustMahalanobisScorerTests(unittest.TestCase):
    def test_fit_contract_does_not_accept_ground_truth_labels(self):
        fit_parameters = inspect.signature(RobustMahalanobisScorer.fit).parameters

        self.assertNotIn("labels", fit_parameters)
        self.assertNotIn("y", fit_parameters)

        scorer = RobustMahalanobisScorer()
        normal_features = np.array(
            [[-0.2, 0.0], [0.0, -0.1], [0.0, 0.1], [0.2, 0.0]],
            dtype=np.float64,
        )
        with self.assertRaises(TypeError):
            scorer.fit(normal_features, labels=np.zeros(len(normal_features)))

    def test_score_samples_assigns_a_higher_score_to_a_distant_pattern(self):
        normal_features = np.array(
            [
                [-0.20, -0.10],
                [-0.10, 0.15],
                [0.00, -0.05],
                [0.10, 0.10],
                [0.20, -0.10],
                [0.05, 0.20],
            ],
            dtype=np.float64,
        )
        scorer = RobustMahalanobisScorer().fit(normal_features)

        scores = scorer.score_samples(
            np.array([[0.0, 0.0], [8.0, 8.0]], dtype=np.float64)
        )

        self.assertEqual(np.asarray(scores).shape, (2,))
        self.assertGreater(float(scores[1]), float(scores[0]))


if __name__ == "__main__":
    unittest.main()
