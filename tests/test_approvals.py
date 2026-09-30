"""Approval lifecycle and execution-time re-validation (time-of-check vs time-of-use)."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_guardrails import (
    ActionNotFoundError,
    ActionState,
    ApprovalExpiredError,
    Guard,
    InvalidTransitionError,
    Policy,
    Reason,
    Status,
)

from .conftest import EMAIL_POLICY, Backend, FakeClock

MakeGuard = Callable[..., Guard]
ARGS = {"to": ["a@example.com"], "subject": "Budget", "body": "Figures attached."}


def _queue(guard: Guard, args: dict[str, Any] | None = None) -> str:
    outcome = guard.call("send_email", args or ARGS)
    assert outcome.status is Status.QUEUED
    assert outcome.action_id is not None
    return outcome.action_id


def _policy_with(**send_email_overrides: Any) -> Policy:
    data = json.loads(json.dumps(EMAIL_POLICY))
    data["tools"]["send_email"].update(send_email_overrides)
    return Policy.from_dict(data)


def test_approve_then_run(make_guard: MakeGuard, email_policy: Policy, backend: Backend) -> None:
    guard = make_guard(email_policy)
    action_id = _queue(guard)
    record = guard.approve(action_id, by="alice", note="checked")
    assert record.status is ActionState.APPROVED
    assert record.decided_by == "alice"
    assert backend.calls == []  # approval alone runs nothing
    [outcome] = guard.run_approved()
    assert outcome.status is Status.EXECUTED
    assert backend.names() == ["send_email"]
    assert guard.store.get(action_id).status is ActionState.EXECUTED
    assert guard.run_approved() == []  # nothing left; never runs twice


def test_reject(make_guard: MakeGuard, email_policy: Policy, backend: Backend) -> None:
    guard = make_guard(email_policy)
    action_id = _queue(guard)
    record = guard.reject(action_id, by="alice", reason="wrong recipient")
    assert record.status is ActionState.REJECTED
    assert record.decision_note == "wrong recipient"
    assert guard.run_approved() == []
    outcome = guard.execute_approved(action_id)
    assert outcome.reason is Reason.NOT_APPROVED
    assert backend.calls == []


def test_cannot_run_pending_or_decide_twice(make_guard: MakeGuard, email_policy: Policy) -> None:
    guard = make_guard(email_policy)
    action_id = _queue(guard)
    assert guard.execute_approved(action_id).reason is Reason.NOT_APPROVED
    guard.approve(action_id, by="alice")
    with pytest.raises(InvalidTransitionError):
        guard.approve(action_id, by="bob")
    with pytest.raises(ActionNotFoundError):
        guard.approve("act_missing", by="alice")


def test_approval_can_be_withdrawn_before_it_runs(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    action_id = _queue(guard)
    guard.approve(action_id, by="alice")
    record = guard.reject(action_id, by="bob", reason="changed my mind")
    assert record.status is ActionState.REJECTED
    assert guard.run_approved() == []
    assert backend.calls == []
    with pytest.raises(InvalidTransitionError):
        guard.reject(action_id, by="bob")  # already closed


def test_executed_action_cannot_be_rejected(make_guard: MakeGuard, email_policy: Policy) -> None:
    guard = make_guard(email_policy)
    action_id = _queue(guard)
    guard.approve(action_id, by="alice")
    guard.run_approved()
    with pytest.raises(InvalidTransitionError):
        guard.reject(action_id, by="bob")


def test_pending_request_expires(
    make_guard: MakeGuard, email_policy: Policy, clock: FakeClock
) -> None:
    guard = make_guard(email_policy)
    action_id = _queue(guard)
    clock.advance(601)  # TTL is 600s in the fixture policy
    with pytest.raises(ApprovalExpiredError):
        guard.approve(action_id, by="alice")
    assert guard.store.get(action_id).status is ActionState.EXPIRED
    assert guard.pending() == []


def test_approved_but_stale_action_does_not_run(
    make_guard: MakeGuard, email_policy: Policy, clock: FakeClock, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    action_id = _queue(guard)
    guard.approve(action_id, by="alice")
    clock.advance(601)
    outcome = guard.execute_approved(action_id)
    assert outcome.reason is Reason.APPROVAL_EXPIRED
    assert guard.store.get(action_id).status is ActionState.EXPIRED
    assert backend.calls == []


def test_policy_tightened_after_approval(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    action_id = _queue(guard)
    guard.approve(action_id, by="alice")
    guard.set_policy(_policy_with(mode="block"), by="security-team")
    outcome = guard.execute_approved(action_id)
    assert outcome.reason is Reason.POLICY_CHANGED
    assert guard.store.get(action_id).status is ActionState.BLOCKED
    assert backend.calls == []


def test_allow_list_changed_after_approval(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    action_id = _queue(guard)
    guard.approve(action_id, by="alice")
    guard.set_policy(
        _policy_with(recipients={"fields": ["to"], "allowed_domains": ["example.org"]})
    )
    outcome = guard.execute_approved(action_id)
    assert outcome.reason is Reason.RECIPIENT_NOT_ALLOWED
    assert "current policy" in outcome.message
    assert backend.calls == []


def test_schema_change_that_would_alter_arguments_blocks(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    action_id = _queue(guard)
    guard.approve(action_id, by="alice")
    args = dict(EMAIL_POLICY["tools"]["send_email"]["args"])
    args["priority"] = {"type": "str", "required": False, "default": "high"}
    guard.set_policy(_policy_with(args=args))
    outcome = guard.execute_approved(action_id)
    assert outcome.reason is Reason.POLICY_CHANGED
    assert backend.calls == []


def test_relaxing_to_allow_still_runs(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    action_id = _queue(guard)
    guard.approve(action_id, by="alice")
    guard.set_policy(_policy_with(mode="allow"))
    assert guard.execute_approved(action_id).status is Status.EXECUTED
    assert backend.names() == ["send_email"]


def test_budget_spent_meanwhile_keeps_it_approved(tmp_path: Path, clock: FakeClock) -> None:
    policy = Policy.from_yaml(
        """
budgets: {"*": {limit: 100, window_seconds: 86400}}
tools:
  pay: {mode: approve, cost: {field: amount}, approval_ttl_seconds: 7200}
  pay_now: {mode: allow, cost: {field: amount}}
"""
    )
    guard = Guard(policy, home=tmp_path, clock=clock)
    paid: list[float] = []
    guard.register("pay", lambda amount: paid.append(amount))
    guard.register("pay_now", lambda amount: paid.append(amount))
    action_id = _queue_pay(guard, 60)
    guard.approve(action_id, by="alice")
    assert guard.call("pay_now", {"amount": 50}).status is Status.EXECUTED
    outcome = guard.execute_approved(action_id)
    assert outcome.reason is Reason.BUDGET_EXCEEDED
    assert "stays approved" in outcome.message
    assert guard.store.get(action_id).status is ActionState.APPROVED
    assert paid == [50]


def _queue_pay(guard: Guard, amount: float) -> str:
    outcome = guard.call("pay", {"amount": amount})
    assert outcome.action_id is not None
    return outcome.action_id


def test_tampered_arguments_are_refused(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    action_id = _queue(guard)
    guard.approve(action_id, by="alice")
    tampered = dict(ARGS, to=["attacker@example.com"])
    with sqlite3.connect(guard.store.path) as conn:
        conn.execute(
            "UPDATE actions SET args_json = ? WHERE id = ?", (json.dumps(tampered), action_id)
        )
    outcome = guard.execute_approved(action_id)
    assert outcome.reason is Reason.ARGUMENTS_CHANGED
    assert backend.calls == []


def test_worker_without_the_tool_leaves_it_approved(
    tmp_path: Path, email_policy: Policy, clock: FakeClock
) -> None:
    web = Guard(email_policy, home=tmp_path, clock=clock)
    web.register("send_email", lambda to, subject, body: None)
    action_id = _queue(web)
    web.approve(action_id, by="alice")
    worker = Guard(email_policy, home=tmp_path, clock=clock)  # nothing registered
    outcome = worker.execute_approved(action_id)
    assert outcome.reason is Reason.NOT_REGISTERED
    assert worker.store.get(action_id).status is ActionState.APPROVED


def test_audit_records_approval_chain(make_guard: MakeGuard, email_policy: Policy) -> None:
    guard = make_guard(email_policy)
    action_id = _queue(guard)
    guard.approve(action_id, by="alice")
    guard.run_approved()
    records = [json.loads(line) for line in guard.audit.path.read_text().splitlines()]
    mine = [r for r in records if r.get("action_id") == action_id]
    assert [r["event"] for r in mine] == ["queued", "approved", "allowed", "executed"]
    digests = {r["args_digest"] for r in mine}
    assert len(digests) == 1  # the same arguments from request to execution
    assert mine[1]["by"] == "alice"
    assert mine[2]["approved_by"] == "alice"


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_state_files_are_private_to_their_owner(tmp_path: Path, email_policy: Policy) -> None:
    # queue.db holds the full arguments of queued actions; neither it nor the audit log
    # should be readable by other users of the host, whatever the umask.
    home = tmp_path / "state"
    old_umask = os.umask(0o022)
    try:
        guard = Guard(email_policy, home=home)
        _queue(guard)
    finally:
        os.umask(old_umask)
    for path in (home, home / "queue.db", home / "audit.jsonl", *home.glob("queue.db-*")):
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode & 0o077 == 0, f"{path.name} is {oct(mode)}"
