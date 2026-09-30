"""Concurrency: limits and the approval queue hold under parallel callers.

Each thread gets its own SQLite connection (as it would in a web server or a pool
of workers), so these tests exercise SQLite's locking, not a Python lock.
"""

from __future__ import annotations

import multiprocessing
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agent_guardrails import ActionState, AuditLog, Guard, Policy, Status, verify_log

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
    guard = Guard(Policy.from_yaml(RATE_POLICY), home=home)
    guard.register("ping", lambda n: n)
    for j in range(5):
        results.put(str(guard.call("ping", {"n": worker * 100 + j}).status))


def test_rate_limit_holds_across_processes(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    procs = [
        ctx.Process(target=_ping_from_process, args=(str(tmp_path), w, results)) for w in range(4)
    ]
    for p in procs:
        p.start()
    statuses = [results.get(timeout=60) for _ in range(20)]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    assert statuses.count("executed") == 7
    assert statuses.count("blocked") == 13
