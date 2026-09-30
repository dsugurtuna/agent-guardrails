"""Tamper-evident, append-only audit log (JSON Lines with a SHA-256 hash chain).

Each record carries ``seq`` (0, 1, 2, ...), ``prev_hash`` (the previous record's
hash; 64 zeros for the first) and ``hash`` (SHA-256 of the record's canonical JSON
without the ``hash`` key). :func:`verify_log` recomputes the chain, so editing,
deleting, inserting or reordering records is detected.

What it cannot do on its own: detect that the *last* records were cut off, or that
someone rewrote the whole file and recomputed every hash. For that, keep the head
hash somewhere the writer cannot change (see ``expected_head``) and compare. That
is why this is called tamper-*evident*, not tamper-proof.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from ._canonical import canonical_json, sha256_hex
from .errors import AuditIntegrityError

try:  # POSIX advisory locks serialise writers across processes.
    import fcntl
except ImportError:  # pragma: no cover - Windows: in-process lock only (documented limit)
    fcntl = None  # type: ignore[assignment]

GENESIS_HASH = "0" * 64
RESERVED_KEYS = frozenset({"seq", "ts", "event", "prev_hash", "hash"})

_process_lock = threading.Lock()


def record_hash(record: dict[str, Any]) -> str:
    """Hash of a record, computed over every key except ``hash`` itself."""
    body = {k: v for k, v in record.items() if k != "hash"}
    return sha256_hex(canonical_json(body))


def _iso(ts: float) -> str:
    return (
        datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def _read_last_line(f: IO[bytes]) -> bytes | None:
    f.seek(0, os.SEEK_END)
    size = f.tell()
    if size == 0:
        return None
    pos = size
    buf = b""
    while pos > 0:
        step = min(8192, pos)
        pos -= step
        f.seek(pos)
        buf = f.read(step) + buf
        body = buf[:-1] if buf.endswith(b"\n") else buf
        idx = body.rfind(b"\n")
        if idx != -1:
            break
    if not buf.endswith(b"\n"):
        raise AuditIntegrityError("audit log does not end with a newline (partial write?)")
    body = buf[:-1]
    return body[body.rfind(b"\n") + 1 :]


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    records: int
    head: str | None
    error: str | None = None
    line: int | None = None

    def __bool__(self) -> bool:
        return self.ok


class AuditLog:
    """Append-only writer. Safe to share between threads and (on POSIX) processes."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] | None = None,
        fsync: bool = True,
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self._fsync = fsync

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        return datetime.now(tz=UTC).timestamp()

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one record and return it (including its hash)."""
        clash = RESERVED_KEYS.intersection(fields)
        if clash:
            raise ValueError(f"reserved audit keys cannot be set: {sorted(clash)}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _process_lock, self.path.open("a+b") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                seq, prev = self._tail(f)
                record: dict[str, Any] = {
                    "seq": seq,
                    "ts": _iso(self._now()),
                    "event": event,
                    **fields,
                    "prev_hash": prev,
                }
                record["hash"] = record_hash(record)
                # Written in insertion order for readability; the hash is over the
                # canonical (sorted-key) form, so key order in the file does not matter.
                # ASCII-only on disk: characters such as U+2028 or U+0085 count as line
                # breaks for some readers and would otherwise split a record in two.
                line = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
                f.write((line + "\n").encode("utf-8"))
                f.flush()
                if self._fsync:
                    os.fsync(f.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return record

    @staticmethod
    def _tail(f: IO[bytes]) -> tuple[int, str]:
        last = _read_last_line(f)
        if last is None:
            return 0, GENESIS_HASH
        try:
            record = json.loads(last)
        except json.JSONDecodeError as exc:
            raise AuditIntegrityError("last audit record is not valid JSON") from exc
        if not isinstance(record, dict) or record_hash(record) != record.get("hash"):
            # Refuse to chain onto a record that has been altered: fail closed.
            raise AuditIntegrityError("last audit record's hash does not match its content")
        seq = record.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise AuditIntegrityError("last audit record has no integer 'seq'")
        return seq + 1, str(record["hash"])

    def head(self) -> tuple[int, str]:
        """``(record_count, head_hash)``. Store the head elsewhere to detect truncation."""
        if not self.path.exists():
            return 0, GENESIS_HASH
        with self.path.open("rb") as f:
            seq, head = self._tail(f)
        return seq, head

    def verify(
        self, *, expected_head: str | None = None, expected_count: int | None = None
    ) -> VerificationResult:
        return verify_log(self.path, expected_head=expected_head, expected_count=expected_count)


def verify_log(
    path: str | Path,
    *,
    expected_head: str | None = None,
    expected_count: int | None = None,
) -> VerificationResult:
    """Recompute the hash chain of the log at ``path``.

    ``expected_head`` / ``expected_count`` are anchors kept outside the log (for
    example, printed into a ticket or shipped to another system). With them,
    truncation and whole-file rewrites are detected too.
    """
    p = Path(path)
    if not p.exists():
        return VerificationResult(False, 0, None, "audit log not found")
    prev = GENESIS_HASH
    count = 0
    with p.open("rb") as f:
        for lineno, raw in enumerate(f, start=1):
            if not raw.endswith(b"\n"):
                return VerificationResult(False, count, prev, "last line is incomplete", lineno)
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return VerificationResult(False, count, prev, "line is not valid JSON", lineno)
            if not isinstance(record, dict):
                return VerificationResult(False, count, prev, "line is not a JSON object", lineno)
            if record.get("seq") != count:
                return VerificationResult(
                    False,
                    count,
                    prev,
                    f"sequence break: expected seq {count}, found {record.get('seq')!r}",
                    lineno,
                )
            if record.get("prev_hash") != prev:
                return VerificationResult(
                    False, count, prev, "prev_hash does not match the previous record", lineno
                )
            if record_hash(record) != record.get("hash"):
                return VerificationResult(
                    False, count, prev, "record hash does not match its content", lineno
                )
            prev = str(record["hash"])
            count += 1
    if expected_count is not None and count != expected_count:
        return VerificationResult(
            False, count, prev, f"expected {expected_count} records, found {count}"
        )
    if expected_head is not None and prev != expected_head:
        return VerificationResult(False, count, prev, "head hash does not match the anchor")
    return VerificationResult(True, count, prev)
