import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.parse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = REPOSITORY_ROOT / "run_halueval_boolq_feature_pilots.sh"


def _shell_function(name: str, following_marker: str) -> str:
    script = LAUNCHER_PATH.read_text(encoding="utf-8")
    start = script.index(f"{name}() {{")
    end = script.index(f"\n}}\n\n{following_marker}", start) + 2
    return script[start:end]


def _boolq_row(index: int) -> dict[str, object]:
    return {
        "row_idx": index,
        "row": {
            "question": f"question {index}",
            "passage": f"passage {index}",
            "answer": bool(index % 2),
        },
        "truncated_cells": [],
    }


@contextmanager
def _rows_server(*, fail_at_offset: int | None = None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed.query)
            if (
                parsed.path != "/rows"
                or query.get("dataset") != ["google/boolq"]
                or query.get("config") != ["default"]
                or query.get("split") != ["validation"]
            ):
                self.send_error(400)
                return
            offset = int(query["offset"][0])
            length = int(query["length"][0])
            if fail_at_offset is not None and offset >= fail_at_offset:
                self.send_error(503)
                return
            if offset < 0 or length < 1 or length > 100 or offset + length > 3270:
                self.send_error(416)
                return
            payload = {
                "rows": [_boolq_row(index) for index in range(offset, offset + length)],
                "num_rows_total": 3270,
                "num_rows_per_page": 100,
                "partial": False,
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/rows"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class BoolQDownloadBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = shutil.which("bash")
        if cls.bash is None:
            raise unittest.SkipTest("bash is required for launcher behavior tests")
        cls.validate_function = _shell_function(
            "validate_boolq_file", "download_boolq_from_huggingface() {"
        )
        cls.download_function = _shell_function(
            "download_boolq_from_huggingface", 'if [[ "${DOWNLOAD_DATA}" == "1" ]]'
        )

    def _run_shell(self, body: str, *, endpoint: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["BOOLQ_DOWNLOAD_RETRY_DELAY_SECONDS"] = "0"
        environment["NO_PROXY"] = "127.0.0.1,localhost"
        environment["no_proxy"] = "127.0.0.1,localhost"
        script = "\n".join(
            (
                "set -euo pipefail",
                f"PYTHON_BIN={shlex.quote(sys.executable)}",
                f"BOOLQ_HF_ROWS_URL={shlex.quote(endpoint)}",
                self.validate_function,
                self.download_function,
                body,
            )
        )
        return subprocess.run(
            [self.bash, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )

    def test_downloads_all_pages_and_publishes_exactly_3270_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory, _rows_server() as endpoint:
            destination = Path(temporary_directory) / "BoolQ" / "dev.jsonl"
            result = self._run_shell(
                f"download_boolq_from_huggingface {shlex.quote(str(destination))}\n"
                f"validate_boolq_file {shlex.quote(str(destination))}",
                endpoint=endpoint,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in destination.read_text().splitlines()]
            self.assertEqual(len(rows), 3270)
            self.assertEqual(rows[0]["question"], "question 0")
            self.assertEqual(rows[-1]["question"], "question 3269")

    def test_mid_download_failure_preserves_destination_and_removes_part_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory, _rows_server(
            fail_at_offset=100
        ) as endpoint:
            destination = Path(temporary_directory) / "BoolQ" / "dev.jsonl"
            destination.parent.mkdir(parents=True)
            destination.write_text("sentinel\n", encoding="utf-8")
            result = self._run_shell(
                f"download_boolq_from_huggingface {shlex.quote(str(destination))}",
                endpoint=endpoint,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(destination.read_text(encoding="utf-8"), "sentinel\n")
            self.assertEqual(list(destination.parent.glob("dev.jsonl.*.part")), [])

    def test_invalid_existing_file_is_replaced_instead_of_reused(self):
        with tempfile.TemporaryDirectory() as temporary_directory, _rows_server() as endpoint:
            destination = Path(temporary_directory) / "BoolQ" / "dev.jsonl"
            destination.parent.mkdir(parents=True)
            destination.write_text('{"question": "broken"}\n', encoding="utf-8")
            quoted_destination = shlex.quote(str(destination))
            result = self._run_shell(
                "if [[ -s {0} ]] && validate_boolq_file {0}; then\n"
                "  echo reused\n"
                "else\n"
                "  download_boolq_from_huggingface {0}\n"
                "  validate_boolq_file {0}\n"
                "fi".format(quoted_destination),
                endpoint=endpoint,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("reused", result.stdout)
            rows = destination.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 3270)


if __name__ == "__main__":
    unittest.main()
