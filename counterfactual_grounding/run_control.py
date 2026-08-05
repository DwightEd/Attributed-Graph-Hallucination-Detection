"""Kernel-backed exclusive output ownership for recoverable CEPT runs."""

from __future__ import annotations

import json
import os
import socket
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def _try_lock(descriptor: int) -> None:
    if os.name == "nt":
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(descriptor: int) -> None:
    if os.name == "nt":
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


@dataclass
class ExclusiveRunLock:
    """Hold an OS lock for the process lifetime; stale files are harmless."""

    output_dir: Path
    resume: bool = False
    _descriptor: int | None = None

    def __enter__(self) -> Self:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / ".cept-run.lock"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            _try_lock(descriptor)
        except OSError:
            os.close(descriptor)
            raise RuntimeError(
                f"another CEPT process owns the output directory: {path}"
            ) from None
        payload = {
            "schema": "cept-kernel-run-lock-v2",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "token": uuid.uuid4().hex,
            "resume": self.resume,
            "created_utc": datetime.now(UTC).isoformat(),
        }
        encoded = (json.dumps(payload) + "\n").encode("utf-8")
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, encoded)
        os.ftruncate(descriptor, len(encoded))
        os.fsync(descriptor)
        self._descriptor = descriptor
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if self._descriptor is None:
            return
        try:
            _unlock(self._descriptor)
        finally:
            os.close(self._descriptor)
            self._descriptor = None


__all__ = ["ExclusiveRunLock"]
