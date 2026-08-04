"""Contracts for label-free token-to-sentence score aggregation."""

from __future__ import annotations

import unittest

from unsupervised_token_graph.ragtruth_sentences import (
    aggregate_sentence_token_scores,
    evaluate_sentence_score_records,
    sentence_char_spans,
)


class SentenceBoundaryContractTests(unittest.TestCase):
    def test_sentence_spans_cover_punctuation_quotes_and_newlines(self):
        text = 'First claim. "Second claim?"\nThird claim!'

        spans = sentence_char_spans(text)

        self.assertEqual(
            [text[start:end].strip() for start, end in spans],
            ["First claim.", '"Second claim?"', "Third claim!"],
        )
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(text))


class SentenceScoreAggregationContractTests(unittest.TestCase):
    def test_top_twenty_percent_pooling_is_label_free_and_less_noisy_than_max(self):
        identity = ("source-1", "response-1")
        text = "Clean words here. Hallucinated words appear now."
        token_offsets = {
            identity: {
                10: (0, 5),
                11: (6, 11),
                12: (12, 17),
                13: (18, 30),
                14: (31, 36),
                15: (37, 43),
                16: (44, 48),
            }
        }
        scores = [
            {"source_id": identity[0], "sample_id": identity[1], "token_idx": index, "score": score}
            for index, score in zip(range(10, 17), [0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 9.0])
        ]

        records = aggregate_sentence_token_scores(
            scores, token_offsets, {identity: text}, top_fraction=0.20
        )

        self.assertEqual(len(records), 2)
        self.assertAlmostEqual(records[0]["score"], 0.3)
        self.assertAlmostEqual(records[1]["score"], 9.0)
        self.assertEqual(records[1]["token_indices"], [13, 14, 15, 16])
        self.assertTrue(all("label" not in record for record in records))

    def test_sentence_evaluation_joins_labels_only_after_scores_are_frozen(self):
        sentence_scores = [
            {
                "source_id": "s1", "sample_id": "r1", "sentence_idx": 0,
                "token_indices": [3, 4], "score": 0.1,
            },
            {
                "source_id": "s1", "sample_id": "r1", "sentence_idx": 1,
                "token_indices": [5, 6], "score": 0.9,
            },
        ]
        token_labels = {
            ("s1", "r1", 3): 0,
            ("s1", "r1", 4): 0,
            ("s1", "r1", 5): 0,
            ("s1", "r1", 6): 1,
        }

        report = evaluate_sentence_score_records(sentence_scores, token_labels)

        self.assertEqual(report["sentence_count"], 2)
        self.assertEqual(report["positive_count"], 1)
        self.assertAlmostEqual(report["auroc"], 1.0)
        self.assertAlmostEqual(report["auprc"], 1.0)

    def test_aggregation_rejects_missing_response_token_scores(self):
        identity = ("s1", "r1")
        with self.assertRaisesRegex(ValueError, "coverage|missing|token"):
            aggregate_sentence_token_scores(
                [{"source_id": "s1", "sample_id": "r1", "token_idx": 2, "score": 0.1}],
                {identity: {2: (0, 4), 3: (5, 9)}},
                {identity: "one two."},
            )


if __name__ == "__main__":
    unittest.main()
