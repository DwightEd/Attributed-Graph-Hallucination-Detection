from __future__ import annotations

import unittest

from grounding_flow.cli import build_parser, config_from_args


class GroundingFlowCliTests(unittest.TestCase):
    def test_run_defaults_are_the_formal_attention_flow_protocol(self):
        args = build_parser().parse_args(
            [
                "run",
                "--extraction-dir",
                "/tmp/extraction",
                "--examples",
                "/tmp/examples.jsonl",
                "--evaluation-labels",
                "/tmp/labels.jsonl",
                "--output-dir",
                "/tmp/output",
            ]
        )
        config = config_from_args(args)

        self.assertEqual(config.num_nulls, 32)
        self.assertEqual(config.evidence_segment_ids, (1,))
        self.assertEqual(config.null_lag_boundaries, (4, 8, 16, 32, 64, 128))
        self.assertEqual(config.pca_components, 32)
        self.assertEqual(config.hmm_iterations, 50)
        self.assertEqual(config.expected_candidates, 2000)
        self.assertEqual(config.min_test_pair_coverage, 0.9)
        self.assertTrue(config.require_complete_cache)
        self.assertTrue(config.group_by_prompt)
        self.assertTrue(config.resume)

    def test_comma_separated_method_parameters_and_skip_evaluation(self):
        args = build_parser().parse_args(
            [
                "run",
                "--extraction-dir",
                "/tmp/extraction",
                "--examples",
                "/tmp/examples.jsonl",
                "--output-dir",
                "/tmp/output",
                "--evidence-segments",
                "0,1",
                "--lag-boundaries",
                "2,4,16",
                "--skip-evaluation",
                "--allow-partial-cache",
                "--expected-candidates",
                "none",
                "--no-resume",
                "--no-group-by-prompt",
            ]
        )
        config = config_from_args(args)

        self.assertEqual(config.evidence_segment_ids, (0, 1))
        self.assertEqual(config.null_lag_boundaries, (2, 4, 16))
        self.assertIsNone(config.evaluation_labels_path)
        self.assertTrue(config.skip_evaluation)
        self.assertIsNone(config.expected_candidates)
        self.assertFalse(config.require_complete_cache)
        self.assertFalse(config.resume)
        self.assertFalse(config.group_by_prompt)


if __name__ == "__main__":
    unittest.main()
