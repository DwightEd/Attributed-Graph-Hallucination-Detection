"""Contract tests for the legacy HaluEval attention-graph launcher."""

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class HaluEvalAttentionGraphShellTests(unittest.TestCase):
    def test_launcher_is_foreground_label_free_legacy_compatibility_entrypoint(self):
        script = (
            REPOSITORY_ROOT / "run_halueval_attention_graph.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("set -euo pipefail", script)
        self.assertIn(
            'SOURCE_RUN_FILE="${SOURCE_RUN_FILE:-/share/home/tm902089733300000/'
            'a903202310/lys/data/feature_extraction/LATEST_HALUEVAL_GRAPH_MAE_RUN.txt}"',
            script,
        )
        self.assertIn('SOURCE_RUN="${SOURCE_RUN:-}"', script)
        self.assertIn('if [[ -z "${SOURCE_RUN}" ]]; then', script)
        self.assertIn('if [[ ! -s "${SOURCE_RUN_FILE}" ]]; then', script)
        self.assertIn('SOURCE_RUN="$(<"${SOURCE_RUN_FILE}")"', script)
        self.assertIn('EXTRACTION_DIR="${EXTRACTION_DIR:-${SOURCE_RUN}/extraction}"', script)
        self.assertIn('EXAMPLES="${EXAMPLES:-${SOURCE_RUN}/prepared/examples.jsonl}"', script)
        self.assertIn(
            'EVALUATION_LABELS="${EVALUATION_LABELS:-${SOURCE_RUN}/prepared/evaluation_labels.jsonl}"',
            script,
        )
        self.assertIn('RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_seed${SEED}}"', script)
        self.assertIn('OUTPUT_DIR="${OUTPUT_DIR:-${SOURCE_RUN}/attention_graph_${RUN_TAG}}"', script)
        self.assertIn('TEST_FRACTION="${TEST_FRACTION:-0.20}"', script)
        self.assertIn(
            'PYTHON_BIN="${PYTHON_BIN:-/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python}"',
            script,
        )
        self.assertIn(
            'Warning: legacy tau-censored compatibility; this is not the dense/floor-0.01 protocol.',
            script,
        )

        self.assertIn('GROUP_BY_PROMPT="${GROUP_BY_PROMPT:-0}"', script)
        self.assertIn("--source-run", script)
        self.assertIn("--group-by-prompt", script)
        self.assertIn('"${PYTHON_BIN}" -m attention_graph.halueval_cli run', script)
        self.assertNotIn("nohup", script)
        self.assertNotIn("&\n", script)

    def test_launcher_forwards_graph_training_controls_and_validates_boolean_switches(self):
        script = (
            REPOSITORY_ROOT / "run_halueval_attention_graph.sh"
        ).read_text(encoding="utf-8")

        for variable, option in (
            ("CUDA_VISIBLE_DEVICES", ""),
            ("DEVICE", "--device"),
            ("SELECTION", "--selection"),
            ("THRESHOLD", "--threshold"),
            ("TOP_K", "--top-k"),
            ("MAX_EDGES_PER_TARGET", "--max-edges-per-target"),
            ("EPOCHS", "--epochs"),
            ("PATIENCE", "--patience"),
            ("SEED", "--seed"),
        ):
            self.assertIn(variable, script)
            if option:
                self.assertIn(option, script)

        self.assertIn('LIMIT_PAIRS="${LIMIT_PAIRS:-}"', script)
        self.assertIn("--limit-pairs", script)
        self.assertIn('CONVERSION_CHUNK_EDGES="${CONVERSION_CHUNK_EDGES:-8192}"', script)
        self.assertIn("--conversion-chunk-edges", script)
        for variable, option in (
            ("REQUIRE_COMPLETE_CACHE", "--require-complete-cache"),
            ("SKIP_EVALUATION", "--skip-evaluation"),
        ):
            self.assertIn(f'[[ "${{{variable}}}" != "0" && "${{{variable}}}" != "1" ]]', script)
            self.assertIn(option, script)


if __name__ == "__main__":
    unittest.main()
