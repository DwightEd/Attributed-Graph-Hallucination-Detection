from __future__ import annotations

import unittest
from pathlib import Path


class GroundingFlowShellContractTests(unittest.TestCase):
    def test_single_entrypoint_owns_all_runtime_parameters(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "grounding_flow" / "run_halueval.sh"
        text = script.read_text(encoding="utf-8")

        self.assertIn("-m grounding_flow.cli run", text)
        self.assertIn("LATEST_HALUEVAL_GRAPH_MAE_RUN.txt", text)
        self.assertIn('OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/feature_extraction}"', text)
        self.assertIn('${OUTPUT_ROOT}/halueval_grounding_flow_', text)
        self.assertIn('EXPECTED_CANDIDATES="${EXPECTED_CANDIDATES:-2000}"', text)
        self.assertIn('REQUIRE_COMPLETE_CACHE="${REQUIRE_COMPLETE_CACHE:-1}"', text)
        self.assertIn('MIN_TEST_PAIR_COVERAGE="${MIN_TEST_PAIR_COVERAGE:-0.90}"', text)
        self.assertIn("NULL_SAMPLES=", text)
        self.assertIn("PCA_COMPONENTS=", text)
        self.assertEqual(text.count("-m grounding_flow.cli run"), 1)
        self.assertNotIn("ragtruth_cli", text)
        self.assertNotIn("attention_graph.halueval_cli", text)
        self.assertNotIn("unsupervised_token_graph.train", text)


if __name__ == "__main__":
    unittest.main()
