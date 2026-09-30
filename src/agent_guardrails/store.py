"""Durable action store and approval queue, on stdlib ``sqlite3``.

One table holds every action that reached the point of execution or approval.
The same rows answer four questions: what is waiting for approval, how many
calls ran in the rate-limit window, how much each agent has spent, and whether
an identical action already happened.

Rows for queued (approve-mode) actions hold the full arguments, because the action
runs later from its row; rows for actions run at once hold only the redacted
arguments. Protect the database file like a credential store.

Why SQLite? It is in the standard library, it survives restarts, and
``BEGIN IMMEDIATE`` transactions give an atomic check-then-reserve step, so two
concurrent callers cannot both squeeze under the same rate limit or budget.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ._files import create_private_file, make_private_dir
from .errors import ActionNotFoundError, ApprovalExpiredError, InvalidTransitionError


class ActionState(StrEnum):
    PENDING = "pending"  # waiting for a human decision
    APPROVED = "approved"  # approved, not yet executed
    REJECTED = "rejected"
    EXPIRED = "expired"  # not decided or not executed within the approval TTL
    EXECUTING = "executing"  # reserved; the tool function is running
    EXECUTED = "executed"
    FAILED = "failed"  # the tool function raised
    BLOCKED = "blocked"  # stopped by a check at execution time


AWAITING_STATES = (ActionState.PENDING, ActionState.APPROVED)
RAN_STATES = (ActionState.EXECUTING, ActionState.EXECUTED, ActionState.FAILED)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS actions (
    id TEXT PRIMARY KEY,
    tool TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    args_json TEXT NOT NULL,
    args_digest TEXT NOT NULL,
    dedupe_key TEXT NOT NULL,
    cost REAL NOT NULL DEFAULT 0,
    policy_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL,
    decided_at REAL,
    decided_by TEXT,
    decision_note TEXT,
    started_at REAL,
    finished_at REAL,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS ix_actions_status ON actions(status);
CREATE INDEX IF NOT EXISTS ix_actions_dedupe ON actions(dedupe_key);
CREATE INDEX IF NOT EXISTS ix_actions_tool_started ON actions(tool, started_at);
CREATE INDEX IF NOT EXISTS ix_actions_agent_started ON actions(agent_id, started_at);
"""


@dataclass(frozen=True)
class ActionRecord:
    id: str
    tool: str
    agent_id: str
    mode: str
    status: ActionState
    args: dict[str, Any]
    args_json: str
    args_digest: str
    dedupe_key: str
    cost: float
    policy_hash: str
    created_at: float
    expires_at: float | None
    decided_at: float | None
    decided_by: str | None
    decision_note: str | None
    started_at: float | None
    finished_at: float | None
    reason: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ActionRecord:
        return cls(
            id=row["id"],
            tool=row["tool"],
            agent_id=row["agent_id"],
            mode=row["mode"],
            status=ActionState(row["status"]),
            args=json.loads(row["args_json"]),
            args_json=row["args_json"],
            args_digest=row["args_digest"],
            dedupe_key=row["dedupe_key"],
            cost=float(row["cost"]),
            policy_hash=row["policy_hash"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            decided_at=row["decided_at"],
            decided_by=row["decided_by"],
            decision_note=row["decision_note"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            reason=row["reason"],
        )


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


class Tx:
    """Queries that must run inside one ``BEGIN IMMEDIATE`` transaction."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def get(self, action_id: str) -> ActionRecord:
        row = self.conn.execute("SELECT * FROM actions WHERE id = ?", (action_id,)).fetchone()
        if row is None:
            raise ActionNotFoundError(f"no action with id {action_id!r}")
        return ActionRecord.from_row(row)

    def count_runs(self, tool: str, since: float, agent_id: str | None = None) -> int:
        sql = (
            f"SELECT COUNT(*) FROM actions WHERE tool = ? AND started_at >= ? "  # noqa: S608
            f"AND status IN ({_placeholders(len(RAN_STATES))})"
        )
        params: list[Any] = [tool, since, *RAN_STATES]
        if agent_id is not None:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        return int(self.conn.execute(sql, params).fetchone()[0])

    def spent(self, agent_id: str, since: float | None) -> float:
        sql = (
            f"SELECT COALESCE(SUM(cost), 0) FROM actions WHERE agent_id = ? "  # noqa: S608
            f"AND status IN ({_placeholders(len(RAN_STATES))})"
        )
        params: list[Any] = [agent_id, *RAN_STATES]
        if since is not None:
            sql += " AND started_at >= ?"
            params.append(since)
        return float(self.conn.execute(sql, params).fetchone()[0])

    def find_duplicate(
        self, dedupe_key: str, since: float, exclude_id: str | None = None
    ) -> ActionRecord | None:
        """An identical action that is awaiting approval, or that ran since ``since``.

        Pending and approved actions count whatever their age (their TTL bounds them).
        An action still marked ``executing`` counts only within the window, so a
        process that crashed mid-call does not block that action for ever.
        """
        sql = (
            f"SELECT * FROM actions WHERE dedupe_key = ? AND ("  # noqa: S608
            f"status IN ({_placeholders(len(AWAITING_STATES))}) "
            "OR (status IN (?, ?) AND COALESCE(finished_at, started_at) >= ?))"
        )
        params: list[Any] = [
            dedupe_key,
            *AWAITING_STATES,
            ActionState.EXECUTING,
            ActionState.EXECUTED,
            since,
        ]
        if exclude_id is not None:
            sql += " AND id != ?"
            params.append(exclude_id)
        sql += " ORDER BY created_at LIMIT 1"
        row = self.conn.execute(sql, params).fetchone()
        return ActionRecord.from_row(row) if row is not None else None

    def pending_count(self, tool: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM actions WHERE tool = ? AND status = ?",
            (tool, ActionState.PENDING),
        ).fetchone()
        return int(row[0])

    def insert(self, **values: Any) -> None:
        cols = ", ".join(values)
        self.conn.execute(
            f"INSERT INTO actions ({cols}) VALUES ({_placeholders(len(values))})",  # noqa: S608
            list(values.values()),
        )

    def update(
        self, action_id: str, *, expect: Iterable[ActionState] | None = None, **values: Any
    ) -> bool:
        """Update one row; with ``expect``, only if its status is one of those (CAS)."""
        sets = ", ".join(f"{k} = ?" for k in values)
        sql = f"UPDATE actions SET {sets} WHERE id = ?"  # noqa: S608
        params: list[Any] = [*values.values(), action_id]
        if expect is not None:
            states = list(expect)
            sql += f" AND status IN ({_placeholders(len(states))})"
            params += states
        return self.conn.execute(sql, params).rowcount == 1


class ActionStore:
    def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
        self.path = Path(path)
        self.timeout = timeout
        make_private_dir(self.path.parent)
        create_private_file(self.path)  # SQLite gives its -wal and -shm files the same mode
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # One short-lived connection per operation: sqlite3 connections must not be
        # shared across threads, and this keeps the store safe to use from many.
        conn = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[Tx]:
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield Tx(conn)
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    # -- reads ----------------------------------------------------------------------

    def get(self, action_id: str) -> ActionRecord:
        with closing(self._connect()) as conn:
            return Tx(conn).get(action_id)

    def list_actions(
        self, statuses: Iterable[ActionState] | None = None, limit: int | None = None
    ) -> list[ActionRecord]:
        sql = "SELECT * FROM actions"
        params: list[Any] = []
        if statuses is not None:
            states = list(statuses)
            sql += f" WHERE status IN ({_placeholders(len(states))})"
            params += states
        sql += " ORDER BY created_at, id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with closing(self._connect()) as conn:
            return [ActionRecord.from_row(r) for r in conn.execute(sql, params).fetchall()]

    def resolve_id(self, prefix: str) -> str:
        """Accept a full id or a unique prefix of one (handy on the command line)."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id FROM actions WHERE id LIKE ? ESCAPE '\\' LIMIT 2",
                (prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%",),
            ).fetchall()
        if len(rows) != 1:
            if not rows:
                raise ActionNotFoundError(f"no action with id {prefix!r}")
            raise ActionNotFoundError(f"id prefix {prefix!r} is ambiguous")
        return str(rows[0]["id"])

    # -- lifecycle ------------------------------------------------------------------

    def expire_stale(self, now: float) -> list[ActionRecord]:
        """Mark pending/approved actions past their TTL as expired; return them."""
        with self.transaction() as tx:
            rows = tx.conn.execute(
                "SELECT * FROM actions WHERE status IN (?, ?) AND expires_at IS NOT NULL "
                "AND expires_at <= ?",
                (ActionState.PENDING, ActionState.APPROVED, now),
            ).fetchall()
            expired = [ActionRecord.from_row(r) for r in rows]
            for rec in expired:
                tx.update(
                    rec.id,
                    expect=(ActionState.PENDING, ActionState.APPROVED),
                    status=ActionState.EXPIRED,
                    reason="approval_expired",
                    finished_at=now,
                )
        return expired

    def approve(
        self, action_id: str, *, by: str, now: float, note: str | None = None
    ) -> ActionRecord:
        """pending -> approved. Raises if the action is not pending or has expired."""
        expired = False
        with self.transaction() as tx:
            rec = tx.get(action_id)
            if rec.status is not ActionState.PENDING:
                raise InvalidTransitionError(
                    f"action {action_id} is {rec.status}; only pending actions can be approved"
                )
            if rec.expires_at is not None and rec.expires_at <= now:
                tx.update(
                    action_id,
                    status=ActionState.EXPIRED,
                    reason="approval_expired",
                    finished_at=now,
                )
                expired = True
            else:
                tx.update(
                    action_id,
                    expect=(ActionState.PENDING,),
                    status=ActionState.APPROVED,
                    decided_at=now,
                    decided_by=by,
                    decision_note=note,
                )
        if expired:
            raise ApprovalExpiredError(f"action {action_id} expired before it was approved")
        return self.get(action_id)

    def reject(
        self, action_id: str, *, by: str, now: float, note: str | None = None
    ) -> ActionRecord:
        """pending or approved -> rejected.

        Rejecting an *approved* action is how a reviewer withdraws an approval before
        it runs, for example while the kill switch is engaged during an incident.
        """
        return self.reject_from(action_id, by=by, now=now, note=note)[0]

    def reject_from(
        self, action_id: str, *, by: str, now: float, note: str | None = None
    ) -> tuple[ActionRecord, ActionState]:
        """Like :meth:`reject`, also returning the state the rejection replaced.

        The previous state is read in the same transaction as the change, so the
        audit can say reliably whether a pending request was refused or an approval
        withdrawn, even while another reviewer is deciding.
        """
        with self.transaction() as tx:
            rec = tx.get(action_id)
            allowed = (ActionState.PENDING, ActionState.APPROVED)
            if rec.status not in allowed:
                raise InvalidTransitionError(
                    f"action {action_id} is {rec.status}; only pending or approved actions "
                    "can be rejected"
                )
            tx.update(
                action_id,
                expect=allowed,
                status=ActionState.REJECTED,
                decided_at=now,
                decided_by=by,
                decision_note=note,
                finished_at=now,
            )
        return self.get(action_id), rec.status
