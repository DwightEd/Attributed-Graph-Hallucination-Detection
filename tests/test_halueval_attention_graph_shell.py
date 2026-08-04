"""Contract tests for the legacy HaluEval attention-graph launcher."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class HaluEvalAttentionGraphShellTests(unittest.TestCase):
    def test_completed_extraction_can_start_while_source_training_is_running(self):
        bash = shutil.which("bash")
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if bash is None and git_bash.is_file():
            bash = str(git_bash)
        if bash is None:
            self.skipTest("bash is required for the launcher behavior test")

        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary_directory:
            source_run = Path(temporary_directory)
            (source_run / "extraction").mkdir()
            (source_run / "prepared").mkdir()
            (source_run / "training").mkdir()
            (source_run / "extraction" / "extraction_manifest.json").write_text(
                '{"state": "complete"}\n', encoding="utf-8"
            )
            (source_run / "prepared" / "examples.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            (source_run / "prepared" / "evaluation_labels.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
            captured_arguments = source_run / "captured-arguments.txt"
            fake_python = source_run / "fake-python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"$#\" -eq 0 ]]; then\n"
                "  printf '%s\\n' \"$PWD/attention_graph/halueval_cli.py\"\n"
                "  exit 0\n"
                "fi\n"
                "printf '%s\\n' \"$@\" > \"${CAPTURE_ARGUMENTS}\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            relative_source = source_run.relative_to(REPOSITORY_ROOT).as_posix()
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHON_BIN": f"{relative_source}/fake-python",
                    "SOURCE_RUN": relative_source,
                    "RUN_TAG": "concurrent-source-test",
                    "CAPTURE_ARGUMENTS": (
                        f"{relative_source}/captured-arguments.txt"
                    ),
                }
            )

            completed = subprocess.run(
                [bash, "./run_halueval_attention_graph.sh"],
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = captured_arguments.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments[:3], ["-m", "attention_graph.halueval_cli", "run"])
            self.assertNotIn("ragtruth_cli", "\n".join(arguments))

    def test_launcher_is_a_zero_argument_entrypoint_pinned_to_local_halueval_cli(self):
        script = (
            REPOSITORY_ROOT / "run_halueval_attention_graph.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('if [[ "$#" -ne 0 ]]; then', script)
        self.assertIn(
            'Usage: bash ./run_halueval_attention_graph.sh', script
        )
        self.assertIn(
            'export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"',
            script,
        )
        self.assertIn("import attention_graph.halueval_cli as module", script)
        self.assertIn('attention_graph" / "halueval_cli.py"', script)
        self.assertIn("Unexpected HaluEval CLI module", script)
        self.assertIn(
            "entrypoint=attention_graph.halueval_cli", script
        )
        self.assertNotIn("ragtruth_cli", script)
        self.assertEqual(
            script.count(
                '"${PYTHON_BIN}" -m attention_graph.halueval_cli run'
            ),
            1,
        )
        self.assertIn(
            'EXTRACTION_MANIFEST="${EXTRACTION_DIR}/extraction_manifest.json"',
            script,
        )
        self.assertIn("source attention extraction is not complete", script)
        self.assertIn("source Graph-MAE training is still running", script)
        self.assertNotIn("Wait for its evaluation file", script)

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
