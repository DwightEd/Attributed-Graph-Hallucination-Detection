from __future__ import annotations

import json
import multiprocessing as mp
from multiprocessing.queues import Queue
from pathlib import Path

import pytest

from counterfactual_grounding.run_control import ExclusiveRunLock


def _attempt_lock(path: str, queue: Queue) -> None:
    try:
        with ExclusiveRunLock(Path(path), resume=True):
            queue.put("acquired")
    except RuntimeError:
        queue.put("blocked")


def test_exclusive_lock_rejects_second_owner(tmp_path: Path) -> None:
    with (
        ExclusiveRunLock(tmp_path),
        pytest.raises(RuntimeError, match="another CEPT process"),
        ExclusiveRunLock(tmp_path),
    ):
        raise AssertionError("unreachable")
    assert (tmp_path / ".cept-run.lock").exists()


def test_stale_metadata_file_does_not_hold_kernel_lock(tmp_path: Path) -> None:
    path = tmp_path / ".cept-run.lock"
    path.write_text(
        json.dumps(
            {
                "hostname": "old-host",
                "pid": 999_999_999,
                "token": "dead",
            }
        ),
        encoding="utf-8",
    )
    with ExclusiveRunLock(tmp_path, resume=True):
        assert path.exists()
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["token"] != "dead"


def test_kernel_lock_blocks_a_second_process(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    queue = context.Queue()
    with ExclusiveRunLock(tmp_path, resume=True):
        contender = context.Process(target=_attempt_lock, args=(str(tmp_path), queue))
        contender.start()
        contender.join(timeout=15)
        assert contender.exitcode == 0
        assert queue.get(timeout=2) == "blocked"
