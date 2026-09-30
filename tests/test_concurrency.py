"""Concurrency: limits and the approval queue hold under parallel callers.

Each thread gets its own SQLite connection (as it would in a web server or a pool
of workers), so these tests exercise SQLite's locking, not a Python lock.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent_guardrails import (
    ActionRecord,
    ActionState,
    ActionStore,
    AuditLog,
    Guard,
    Policy,
    Reason,
    Status,
    verify_log,
)
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
    try:
        guard = Guard(Policy.from_yaml(RATE_POLICY), home=home)
        guard.register("ping", lambda n: n)
        for j in range(5):
            results.put(str(guard.call("ping", {"n": worker * 100 + j}).status))
    except Exception:  # report instead of leaving the parent waiting
        results.put("ERROR " + traceback.format_exc())


def test_rate_limit_holds_across_processes(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    procs = [
        ctx.Process(target=_ping_from_process, args=(str(tmp_path), w, results)) for w in range(4)
    ]
    for p in procs:
        p.start()
    statuses = [results.get(timeout=180) for _ in range(20)]
    assert not [s for s in statuses if s.startswith("ERROR")], statuses
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    assert statuses.count("executed") == 7
    assert statuses.count("blocked") == 13


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
