import unittest
from types import SimpleNamespace

import torch

from unsupervised_token_graph.data import compose_example
from unsupervised_token_graph.extract import (
    compute_teacher_forced_statistics,
    extract_example_trace,
    summarize_trace_record,
    tokenize_example_once,
)


class _FakeTokenizer:
    def __init__(self, example):
        self.example = example
        self.call_count = 0
        self.last_options = None

    def __call__(self, text, **options):
        self.call_count += 1
        self.last_options = options
        offsets = [(0, 0)] + [
            self.example.segment_char_spans[name]
            for name in ("passage", "question", "answer")
        ]
        return {
            "input_ids": torch.tensor([[1, 11, 22, 33]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "offset_mapping": torch.tensor([offsets]),
            "special_tokens_mask": torch.tensor([[1, 0, 0, 0]]),
        }


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask, **options):
        token_count = input_ids.shape[1]
        logits = torch.zeros((1, token_count, 40), device=input_ids.device)
        attention = torch.zeros((1, 1, token_count, token_count), device=input_ids.device)
        attention[:, :, 3, 1:4] = torch.tensor([0.6, 0.2, 0.2], device=input_ids.device)
        hidden = torch.arange(token_count * 3, device=input_ids.device).reshape(1, token_count, 3).float()
        return SimpleNamespace(
            logits=logits,
            attentions=(attention,),
            hidden_states=(hidden * 0.0, hidden),
        )


class SingleTokenizationTests(unittest.TestCase):
    def test_final_composed_text_is_tokenized_once_and_segments_are_aligned(self):
        example = compose_example(
            "Passage evidence.",
            "Question?",
            "Answer.",
            example_id="sample",
        )
        tokenizer = _FakeTokenizer(example)

        encoded = tokenize_example_once(tokenizer, example, max_tokens=8)

        self.assertEqual(tokenizer.call_count, 1)
        self.assertFalse(tokenizer.last_options["truncation"])
        self.assertEqual(encoded["segment_ids"].tolist(), [0, 1, 2, 3])
        self.assertEqual(encoded["answer_mask"].tolist(), [False, False, False, True])

    def test_full_context_limit_raises_instead_of_silently_truncating(self):
        example = compose_example("P", "Q", "A", example_id="sample")
        tokenizer = _FakeTokenizer(example)

        with self.assertRaisesRegex(ValueError, "exceeds max_tokens"):
            tokenize_example_once(tokenizer, example, max_tokens=3)


class TeacherForcedStatisticsTests(unittest.TestCase):
    def test_log_probability_and_entropy_are_aligned_to_the_predicted_token(self):
        logits = torch.tensor(
            [
                [
                    [2.0, 0.0, -1.0],
                    [0.0, 2.0, -1.0],
                    [-1.0, 0.0, 2.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        )
        input_ids = torch.tensor([[0, 0, 1, 2]])

        token_log_prob, entropy, valid = compute_teacher_forced_statistics(
            logits, input_ids
        )

        self.assertEqual(token_log_prob.shape, (4,))
        self.assertEqual(entropy.shape, (4,))
        self.assertEqual(valid.tolist(), [False, True, True, True])
        expected = torch.log_softmax(logits[0, :-1], dim=-1)
        self.assertAlmostEqual(float(token_log_prob[1]), float(expected[0, 0]))
        self.assertAlmostEqual(float(token_log_prob[2]), float(expected[1, 1]))
        self.assertAlmostEqual(float(token_log_prob[3]), float(expected[2, 2]))

    def test_complete_trace_contains_model_observables_but_no_evaluation_fields(self):
        example = compose_example("P", "Q", "A", example_id="sample", pair_id="pair")

        trace = extract_example_trace(
            _FakeModel(),
            _FakeTokenizer(example),
            example,
            selected_hidden_layers=(1,),
            max_tokens=8,
        )

        self.assertEqual(tuple(trace["attention"].shape), (1, 1, 4, 4))
        self.assertEqual(tuple(trace["hidden_states"].shape), (1, 4, 3))
        self.assertEqual(trace["segment_ids"].tolist(), [0, 1, 2, 3])
        self.assertTrue(set(trace).isdisjoint({"label", "labels", "y", "y_token"}))


class TraceSummaryTests(unittest.TestCase):
    def test_trace_summary_exports_scalar_evidence_and_self_reliance_features(self):
        attention = torch.zeros((1, 1, 4, 4))
        attention[0, 0, 3] = torch.tensor([0.0, 0.6, 0.2, 0.2])
        trace = {
            "example_id": "sample",
            "pair_id": "pair",
            "dataset": "halueval_qa",
            "attention": attention,
            "segment_ids": torch.tensor([0, 1, 2, 3]),
            "token_log_prob": torch.tensor([0.0, -1.0, -1.0, -0.1]),
            "token_stat_valid": torch.tensor([False, True, True, True]),
        }

        record = summarize_trace_record(trace)

        self.assertEqual(record["example_id"], "sample")
        self.assertAlmostEqual(record["answer_to_passage_ratio"], 0.6)
        self.assertAlmostEqual(record["answer_to_question_ratio"], 0.2)
        self.assertAlmostEqual(record["answer_self_reliance"], 0.2)
        self.assertAlmostEqual(record["mean_answer_log_prob"], -0.1)


if __name__ == "__main__":
    unittest.main()
