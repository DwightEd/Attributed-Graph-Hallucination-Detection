import json
import tempfile
import unittest
from pathlib import Path

from audit_negative_patterns import (
    audit_dataset,
    classify_error_pattern,
    merge_intervals,
)


class ErrorPatternClassificationTests(unittest.TestCase):
    def test_baseless_numeric_detail_keeps_support_and_mechanism_axes(self):
        label = {
            "label_type": "Evident Baseless Info",
            "text": "he was adrift for 20 days",
            "meta": (
                "HIGH INTRO OF NEW INFO It was not mentioned in the original "
                "source that he was adrift for 20 days."
            ),
        }

        result = classify_error_pattern(label)

        self.assertEqual(result["support_relation"], "baseless")
        self.assertEqual(result["severity"], "evident")
        self.assertEqual(result["primary_pattern"], "numeric_temporal")
        self.assertIn("unsupported_addition", result["pattern_tags"])

    def test_polarity_conflict_is_distinguished_from_generic_relation_error(self):
        label = {
            "label_type": "Evident Conflict",
            "text": "free WiFi",
            "meta": 'Original: "WiFi": "no", Generated: "free WiFi"',
        }

        result = classify_error_pattern(label)

        self.assertEqual(result["support_relation"], "conflict")
        self.assertEqual(result["primary_pattern"], "polarity_negation")
        self.assertIn("relation_predicate", result["pattern_tags"])

    def test_source_attribution_conflict_is_detected(self):
        label = {
            "label_type": "Evident Conflict",
            "text": "Passage 2",
            "meta": "The information is mentioned in Passage 1.",
        }

        result = classify_error_pattern(label)

        self.assertEqual(result["primary_pattern"], "source_attribution")


class IntervalTests(unittest.TestCase):
    def test_merge_intervals_counts_overlapping_spans_once(self):
        merged = merge_intervals([(2, 7), (5, 10), (12, 14)])

        self.assertEqual(merged, [(2, 10), (12, 14)])

    def test_merge_intervals_clamps_to_response_boundaries(self):
        merged = merge_intervals([(-3, 4), (8, 20)], upper_bound=12)

        self.assertEqual(merged, [(0, 4), (8, 12)])


class DatasetAuditTests(unittest.TestCase):
    def test_audit_separates_clean_and_hallucinated_samples_and_checks_spans(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources_path = root / "source_info.jsonl"
            responses_path = root / "response.jsonl"
            sources_path.write_text(
                json.dumps({"source_id": "s1", "task_type": "Summary"}) + "\n",
                encoding="utf-8",
            )
            responses = [
                {
                    "id": "clean",
                    "source_id": "s1",
                    "model": "model-a",
                    "split": "train",
                    "labels": [],
                    "response": "Supported answer.",
                },
                {
                    "id": "hallucinated",
                    "source_id": "s1",
                    "model": "model-a",
                    "split": "train",
                    "response": "X 2022 and free WiFi.",
                    "labels": [
                        {
                            "start": 2,
                            "end": 6,
                            "text": "2022",
                            "meta": "Original: 1945; Generated: 2022",
                            "label_type": "Evident Conflict",
                        },
                        {
                            "start": 11,
                            "end": 20,
                            "text": "free WiFi",
                            "meta": 'Original: "WiFi": "no"',
                            "label_type": "Evident Conflict",
                        },
                    ],
                },
                {
                    "id": "bad-spans",
                    "source_id": "missing-source",
                    "model": "model-b",
                    "split": "test",
                    "response": "abcdef",
                    "labels": [
                        {
                            "start": 1,
                            "end": 4,
                            "text": "wrong text",
                            "meta": "No source support",
                            "label_type": "Evident Baseless Info",
                        },
                        {
                            "start": 3,
                            "end": 6,
                            "text": "def",
                            "meta": "No source support",
                            "label_type": "Evident Baseless Info",
                        },
                        {
                            "start": 10,
                            "end": 12,
                            "text": "outside",
                            "meta": "No source support",
                            "label_type": "Unknown Label",
                        },
                    ],
                },
            ]
            response_lines = [json.dumps(response) for response in responses]
            response_lines.append("{not valid json")
            responses_path.write_text("\n".join(response_lines) + "\n", encoding="utf-8")

            result = audit_dataset(responses_path, sources_path, max_examples=2)

        summary = result.report["summary"]
        quality = result.report["data_quality"]
        self.assertEqual(summary["responses"], 3)
        self.assertEqual(summary["clean_responses"], 1)
        self.assertEqual(summary["hallucinated_responses"], 2)
        self.assertEqual(summary["annotated_spans"], 5)
        self.assertEqual(quality["malformed_response_rows"], 1)
        self.assertEqual(quality["missing_source_responses"], 1)
        self.assertEqual(quality["invalid_span_bounds"], 1)
        self.assertEqual(quality["span_text_mismatches"], 2)
        self.assertEqual(quality["overlapping_span_responses"], 1)
        self.assertEqual(quality["unknown_label_types"], 1)
        self.assertEqual(
            result.report["breakdowns"]["primary_pattern"]["numeric_temporal"],
            1,
        )
        self.assertEqual(len(result.sample_rows), 3)
        self.assertEqual(len(result.span_rows), 5)


if __name__ == "__main__":
    unittest.main()
