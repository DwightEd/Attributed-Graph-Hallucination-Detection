"""Static contracts for the one-command official RAGTruth runner."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OfficialRunnerContractTests(unittest.TestCase):
    def test_runner_uses_cache_root_official_split_and_sentence_scoring(self):
        text = (ROOT / "run_ragtruth_typed_token_graph.sh").read_text(encoding="utf-8")

        self.assertIn("fresh_attention_c8847872bedf_20260731T074520Z_p876}", text)
        self.assertIn("for split in train test", text)
        self.assertIn('split_dir="${ATTENTION_DIR}/${split}"', text)
        self.assertIn("--split-policy official", text)
        self.assertIn("ragtruth_cli sentences", text)
        self.assertIn("--sentence-scores", text)
        self.assertIn('FEATURE_ROOT="${FEATURE_ROOT:-${DATA_ROOT}/feature_extraction}"', text)

    def test_history_registration_targets_the_full_v2_result_without_copying_it(self):
        text = (ROOT / "register_ragtruth_hypergraph_result.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("hypergraph_ssl_full_hierarchical_ssl_v2_full", text)
        self.assertIn("/lys/data/feature_extraction", text)
        self.assertIn("ln -s", text)
        self.assertNotIn("cp -r", text)
        self.assertNotIn("mv ", text)


if __name__ == "__main__":
    unittest.main()
