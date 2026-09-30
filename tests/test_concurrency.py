"""Concurrency: limits and the approval queue hold under parallel callers.

Each thread gets its own SQLite connection (as it would in a web server or a pool
of workers), so these tests exercise SQLite's locking, not a Python lock.
"""

from __future__ import annotations

import json
import multiprocessing
import sqlite3
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from agent_guardrails import (
    ActionRecord,
    ActionState,
    ActionStore,
    AuditLog,
    Guard,
    InvalidTransitionError,
    Policy,
    Reason,
    Status,
    verify_log,
)
from agent_guardrails.guard import reject_action
from agent_guardrails.store import Tx

N = 24


def _run_in_parallel(fn: Callable[[int], Status], n: int = N) -> list[Status]:
    barrier = threading.Barrier(n)

    def task(i: int) -> Status:
        barrier.wait()  # release every thread at the same moment
        return fn(i)

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(task, range(n)))


def test_rate_limit_is_exact_under_contention(tmp_path: Path) -> None:
    policy = Policy.from_yaml(
        "tools: {ping: {mode: allow, rate_limit: {max_calls: 5, window_seconds: 3600}}}\n"
    )
    guard = Guard(policy, home=tmp_path)
    ran: list[int] = []
    lock = threading.Lock()

    def ping(n: int) -> None:
        with lock:
            ran.append(n)

    guard.register("ping", ping)
    statuses = _run_in_parallel(lambda i: guard.call("ping", {"n": i}).status)
    assert statuses.count(Status.EXECUTED) == 5
    assert statuses.count(Status.BLOCKED) == N - 5
    assert len(ran) == 5


def test_budget_is_exact_under_contention(tmp_path: Path) -> None:
    policy = Policy.from_yaml(
        """
budgets: {"*": {limit: 10}}
tools: {pay: {mode: allow, cost: {fixed: 1}, dedupe_window_seconds: 0}}
"""
    )
    guard = Guard(policy, home=tmp_path)
    guard.register("pay", lambda ref: None)
    statuses = _run_in_parallel(lambda i: guard.call("pay", {"ref": str(i)}).status)
    assert statuses.count(Status.EXECUTED) == 10


def test_identical_retries_run_once(tmp_path: Path) -> None:
    policy = Policy.from_yaml("tools: {remind: {mode: allow}}\n")
    guard = Guard(policy, home=tmp_path)
    sent: list[str] = []
    guard.register("remind", lambda to, text: sent.append(to))
    statuses = _run_in_parallel(
        lambda i: guard.call("remind", {"to": "a@example.com", "text": "10am"}).status
    )
    assert statuses.count(Status.EXECUTED) == 1
    assert statuses.count(Status.DUPLICATE) == N - 1
    assert sent == ["a@example.com"]


def test_approved_action_runs_once_across_workers(tmp_path: Path) -> None:
    policy = Policy.from_yaml("tools: {send: {mode: approve}}\n")
    app = Guard(policy, home=tmp_path)
    app.register("send", lambda to: None)
    queued = app.call("send", {"to": "a@example.com"})
    assert queued.action_id is not None
    app.approve(queued.action_id, by="alice")

    sent: list[int] = []
    lock = threading.Lock()
    workers = []

    def make_sender(w: int) -> Callable[[str], None]:
        def send(to: str) -> None:
            with lock:
                sent.append(w)

        return send

    for w in range(N):
        worker = Guard(policy, home=tmp_path)  # separate guard, as in separate processes
        worker.register("send", make_sender(w))
        workers.append(worker)

    action_id = queued.action_id
    statuses = _run_in_parallel(lambda i: workers[i].execute_approved(action_id).status)
    assert statuses.count(Status.EXECUTED) == 1
    assert len(sent) == 1
    assert app.store.get(action_id).status is ActionState.EXECUTED


def test_audit_log_stays_consistent_with_parallel_writers(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"

    def write(i: int) -> Status:
        log = AuditLog(path, fsync=False)
        for j in range(10):
            log.append("executed", worker=i, n=j)
        return Status.EXECUTED

    _run_in_parallel(write, n=8)
    result = verify_log(path)
    assert result.ok and result.records == 80


def _append_from_process(path: str, worker: int) -> None:
    log = AuditLog(path, fsync=False)
    for j in range(25):
        log.append("executed", worker=worker, n=j)


def test_audit_log_with_parallel_processes(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_append_from_process, args=(str(path), w)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    result = verify_log(path)
    assert result.ok and result.records == 100


RATE_POLICY = "tools: {ping: {mode: allow, rate_limit: {max_calls: 7, window_seconds: 3600}}}\n"


def _ping_from_process(home: str, worker: int, results: Any) -> None:
    # Exactly one message per process, so an error is reported rather than leaving
    # the parent waiting for results that will never come.
    try:
        guard = Guard(Policy.from_yaml(RATE_POLICY), home=home)
        guard.register("ping", lambda n: n)
        statuses = [str(guard.call("ping", {"n": worker * 100 + j}).status) for j in range(5)]
        results.put(json.dumps(statuses))
    except Exception:
        results.put("ERROR " + traceback.format_exc())


def test_rate_limit_holds_across_processes(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    procs = [
        ctx.Process(target=_ping_from_process, args=(str(tmp_path), w, results)) for w in range(4)
    ]
    for p in procs:
        p.start()
    messages = [results.get(timeout=120) for _ in procs]
    errors = [m for m in messages if m.startswith("ERROR")]
    assert not errors, errors[0]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    statuses = [s for m in messages for s in json.loads(m)]
    assert statuses.count("executed") == 7
    assert statuses.count("blocked") == 13


def _busy() -> sqlite3.OperationalError:
    exc = sqlite3.OperationalError("database is locked")
    exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
    return exc


class _FlakyConnection:
    """Answers ``PRAGMA journal_mode=WAL`` with "database is locked" ``fails`` times.

    SQLite does this, without waiting on the busy timeout, while another process is
    switching a new database to WAL. That window is too short to hit on demand, so
    these tests simulate it; the multi-process test below exercises the real thing.
    """

    def __init__(self, fails: int, error: Callable[[], Exception] = _busy) -> None:
        self.fails, self.error, self.calls = fails, error, 0

    def execute(self, sql: str) -> None:
        self.calls += 1
        if self.calls <= self.fails:
            raise self.error()


def _enable_wal(store_timeout: float, conn: _FlakyConnection) -> None:
    store = ActionStore.__new__(ActionStore)
    store.timeout = store_timeout
    store._enable_wal(cast(sqlite3.Connection, conn))


def test_switching_to_wal_retries_while_another_process_holds_the_file() -> None:
    conn = _FlakyConnection(fails=3)
    _enable_wal(5.0, conn)
    assert conn.calls == 4


def test_switching_to_wal_gives_up_after_the_store_timeout() -> None:
    conn = _FlakyConnection(fails=10_000)
    started = time.monotonic()
    try:
        _enable_wal(0.2, conn)
    except sqlite3.OperationalError as exc:
        assert "locked" in str(exc)
    else:
        raise AssertionError("expected the busy error once the timeout ran out")
    assert time.monotonic() - started < 1.0
    assert conn.calls > 1


def test_switching_to_wal_does_not_retry_other_errors() -> None:
    conn = _FlakyConnection(fails=1, error=lambda: sqlite3.OperationalError("disk I/O error"))
    try:
        _enable_wal(5.0, conn)
    except sqlite3.OperationalError as exc:
        assert "disk I/O" in str(exc)
    else:
        raise AssertionError("expected the error to propagate")
    assert conn.calls == 1


def _open_store(path: str, barrier: Any, results: Any) -> None:
    try:
        barrier.wait(timeout=60)
        ActionStore(Path(path))
        results.put("ok")
    except Exception:
        results.put("ERROR " + traceback.format_exc())


def test_many_processes_can_create_the_same_store_at_once(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    for round_ in range(8):
        path = str(tmp_path / f"actions-{round_}.db")
        barrier = ctx.Barrier(8)
        results = ctx.Queue()
        procs = [ctx.Process(target=_open_store, args=(path, barrier, results)) for _ in range(8)]
        for p in procs:
            p.start()
        messages = [results.get(timeout=120) for _ in procs]
        for p in procs:
            p.join(timeout=60)
        errors = [m for m in messages if m != "ok"]
        assert not errors, errors[0]


class InterleavingStore(ActionStore):
    """Runs a callback once at a chosen point, as another worker would act there."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.after_commit: Callable[[], None] | None = None
        self.after_get: Callable[[], None] | None = None

    @contextmanager
    def transaction(self) -> Iterator[Tx]:
        with super().transaction() as tx:
            yield tx
        hook, self.after_commit = self.after_commit, None
        if hook is not None:
            hook()

    def get(self, action_id: str) -> ActionRecord:
        record = super().get(action_id)
        hook, self.after_get = self.after_get, None
        if hook is not None:
            hook()
        return record


def _approved_send(tmp_path: Path, policy: Policy) -> tuple[Guard, InterleavingStore, str]:
    store = InterleavingStore(tmp_path / "queue.db")
    guard = Guard(policy, home=tmp_path, store=store)
    guard.register("send", lambda to: None)
    queued = guard.call("send", {"to": "a@example.com"})
    assert queued.action_id is not None
    guard.approve(queued.action_id, by="alice")
    return guard, store, queued.action_id


def _other_worker_claims(store: ActionStore, action_id: str, results: list[bool]) -> None:
    with store.transaction() as tx:
        results.append(
            tx.update(
                action_id,
                expect=(ActionState.APPROVED,),
                status=ActionState.EXECUTING,
                started_at=time.time(),
            )
        )


def test_closing_a_duplicate_never_overwrites_another_workers_claim(tmp_path: Path) -> None:
    guard, store, action_id = _approved_send(
        tmp_path, Policy.from_yaml("tools: {send: {mode: approve}}\n")
    )
    record = store.get(action_id)
    with store.transaction() as tx:  # an identical action is in flight elsewhere
        tx.insert(
            id="act_inflight",
            tool="send",
            agent_id="default",
            mode="allow",
            status=ActionState.EXECUTING,
            args_json=record.args_json,
            args_digest=record.args_digest,
            dedupe_key=record.dedupe_key,
            policy_hash=record.policy_hash,
            created_at=time.time(),
            started_at=time.time(),
        )
    claimed: list[bool] = []

    def meanwhile() -> None:
        # Right after this worker saw the duplicate: the in-flight call fails (so it no
        # longer counts), and a second worker claims the approved action.
        with store.transaction() as tx:
            tx.update("act_inflight", status=ActionState.FAILED, finished_at=time.time())
        _other_worker_claims(store, action_id, claimed)

    store.after_commit = meanwhile
    outcome = guard.execute_approved(action_id)
    assert outcome.status is Status.DUPLICATE
    status = store.get(action_id).status
    if claimed == [True]:  # the other worker is running it: its claim must stand
        assert status is ActionState.EXECUTING
    else:
        assert status is ActionState.BLOCKED


SEND_TO = "tools: {send: {mode: approve, recipients: {fields: [to], allowed_domains: [%s]}}}\n"


def test_closing_reports_the_truth_when_another_worker_ran_it(tmp_path: Path) -> None:
    guard, store, action_id = _approved_send(tmp_path, Policy.from_yaml(SEND_TO % "example.com"))
    # This worker has a tighter policy, so it will close the action...
    guard.set_policy(Policy.from_yaml(SEND_TO % "example.org"))
    claimed: list[bool] = []
    # ...but another worker, still on the old policy, claims it first.
    store.after_get = lambda: _other_worker_claims(store, action_id, claimed)
    outcome = guard.execute_approved(action_id)
    assert claimed == [True]
    assert store.get(action_id).status is ActionState.EXECUTING
    assert outcome.reason is Reason.NOT_APPROVED  # not "recipient_not_allowed": it was not closed
    assert "not run" in outcome.message


def test_each_decision_records_the_policy_it_was_made_under(tmp_path: Path) -> None:
    # set_policy from another thread can land at any point during a call. The redact
    # hook runs mid-call, so it can stand in for that thread deterministically.
    old = Policy.from_yaml("tools: {ping: {mode: allow}}\n")
    new = Policy.from_yaml("tools: {ping: {mode: block}}\n")
    swapped: list[bool] = []

    def swap_policy_meanwhile(key: str, value: Any) -> Any:
        if not swapped:
            swapped.append(True)
            guard.set_policy(new, by="security-team")
        return value

    guard = Guard(old, home=tmp_path, redact_hook=swap_policy_meanwhile)
    guard.register("ping", lambda n: n)
    outcome = guard.call("ping", {"n": 1})
    assert swapped == [True]
    records = [json.loads(line) for line in guard.audit.path.read_text().splitlines()]
    decision = next(r for r in records if r.get("tool") == "ping")
    # The call was decided under the old policy (it ran), so it must say so.
    assert outcome.status is Status.EXECUTED
    assert decision["policy_hash"] == old.fingerprint()


def test_rejection_records_the_state_it_actually_overrode(tmp_path: Path) -> None:
    # Bob rejects while Alice approves. The audit must say whether Bob rejected a
    # pending request or withdrew Alice's approval: that is who overrode whom.
    store = InterleavingStore(tmp_path / "queue.db")
    guard = Guard(Policy.from_yaml("tools: {send: {mode: approve}}\n"), home=tmp_path, store=store)
    queued = guard.call("send", {"to": "a@example.com"})
    action_id = queued.action_id
    assert action_id is not None
    approved_first: list[bool] = []

    def alice_approves() -> None:
        try:
            store.approve(action_id, by="alice", now=time.time())
            approved_first.append(True)
        except InvalidTransitionError:
            approved_first.append(False)

    store.after_get = alice_approves
    reject_action(store, guard.audit, action_id, by="bob", now=time.time())
    records = [json.loads(line) for line in guard.audit.path.read_text().splitlines()]
    rejected = next(r for r in records if r["event"] == "rejected")
    assert rejected["was"] == ("approved" if approved_first == [True] else "pending")
