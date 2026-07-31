import unittest

from audit_negative_patterns import classify_error_pattern, merge_intervals


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


if __name__ == "__main__":
    unittest.main()
