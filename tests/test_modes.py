"""Each mode does what it says, and nothing more."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_guardrails import (
    ActionBlockedError,
    ActionFailedError,
    ConfigurationError,
    Guard,
    Policy,
    Reason,
    Status,
    get_default_guard,
    guarded,
    set_default_guard,
    verify_log,
)

from .conftest import Backend

MakeGuard = Callable[..., Guard]


def test_allow_executes_and_returns_result(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    outcome = guard.call("search_calendar", {"day": "monday"})
    assert outcome.status is Status.EXECUTED
    assert outcome.result == ["09:00 stand-up"]
    assert backend.calls == [("search_calendar", {"day": "monday"})]
    assert "OK" in outcome.as_tool_result()
    assert not outcome.is_error


def test_draft_never_executes(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    outcome = guard.call("draft_reply", {"to": "a@example.com", "body": "Thanks!"})
    assert outcome.status is Status.DRAFTED
    assert backend.calls == []
    assert outcome.preview is not None and "Thanks!" in outcome.preview
    assert "nothing was sent" in outcome.as_tool_result()


def test_custom_preview(tmp_path: Path, email_policy: Policy) -> None:
    guard = Guard(email_policy, home=tmp_path)
    guard.register("draft_reply", lambda to, body: None, preview=lambda t, a: f"To: {a['to']}")
    assert guard.call("draft_reply", {"to": "a@example.com", "body": "x"}).preview == (
        "To: a@example.com"
    )


def test_approve_queues_without_executing(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    outcome = guard.call(
        "send_email", {"to": ["a@example.com"], "subject": "Hi", "body": "See you at 10."}
    )
    assert outcome.status is Status.QUEUED
    assert outcome.action_id is not None
    assert backend.calls == []
    assert [r.id for r in guard.pending()] == [outcome.action_id]
    text = outcome.as_tool_result()
    assert "NOT been performed" in text and outcome.action_id in text


def test_block_mode(make_guard: MakeGuard, email_policy: Policy, backend: Backend) -> None:
    outcome = make_guard(email_policy).call("delete_mailbox", {})
    assert outcome.status is Status.BLOCKED
    assert outcome.reason is Reason.TOOL_BLOCKED
    assert outcome.is_error
    assert backend.calls == []


def test_unknown_tool_blocked_by_default(make_guard: MakeGuard, email_policy: Policy) -> None:
    outcome = make_guard(email_policy).call("transfer_funds", {"amount": 10})
    assert outcome.status is Status.BLOCKED
    assert outcome.reason is Reason.UNKNOWN_TOOL


def test_unknown_tool_with_draft_default(tmp_path: Path) -> None:
    guard = Guard(Policy.from_yaml("default_mode: draft\n"), home=tmp_path)
    outcome = guard.call("anything", {"x": 1})
    assert outcome.status is Status.DRAFTED


def test_allowed_but_not_registered(tmp_path: Path, email_policy: Policy) -> None:
    outcome = Guard(email_policy, home=tmp_path).call("search_calendar", {})
    assert outcome.status is Status.BLOCKED
    assert outcome.reason is Reason.NOT_REGISTERED


def test_invalid_arguments_blocked(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    outcome = guard.call("create_event", {"title": "x" * 101, "when": "10:00"})
    assert outcome.reason is Reason.INVALID_ARGUMENTS
    missing = guard.call("create_event", {"title": "x"})
    assert missing.reason is Reason.INVALID_ARGUMENTS
    extra = guard.call("search_calendar", {"day": "today", "unexpected": True})
    assert extra.reason is Reason.INVALID_ARGUMENTS  # caught by the function signature
    assert backend.calls == []


def test_recipient_checks_apply_before_queueing(
    make_guard: MakeGuard, email_policy: Policy
) -> None:
    guard = make_guard(email_policy)
    outcome = guard.call("send_email", {"to": ["x@evil.test"], "subject": "s", "body": "b"})
    assert outcome.reason is Reason.RECIPIENT_NOT_ALLOWED
    assert guard.pending() == []


def test_tool_exception_is_reported_without_its_message(
    tmp_path: Path, email_policy: Policy
) -> None:
    guard = Guard(email_policy, home=tmp_path)

    def search_calendar(day: str = "today") -> None:
        raise RuntimeError("password=hunter2 leaked in error text")

    guard.register("search_calendar", search_calendar)
    outcome = guard.call("search_calendar", {})
    assert outcome.status is Status.FAILED
    assert outcome.reason is Reason.TOOL_ERROR
    assert "RuntimeError" in outcome.message
    assert "hunter2" not in outcome.as_tool_result()
    assert "hunter2" not in (tmp_path / "audit.jsonl").read_text()
    with pytest.raises(ActionFailedError):
        outcome.raise_for_status()


def test_raise_for_status(make_guard: MakeGuard, email_policy: Policy) -> None:
    guard = make_guard(email_policy)
    ok = guard.call("search_calendar", {})
    assert ok.raise_for_status() is ok
    with pytest.raises(ActionBlockedError) as info:
        guard.call("delete_mailbox", {}).raise_for_status()
    assert info.value.outcome.reason is Reason.TOOL_BLOCKED


def test_wrap_and_decorator_bind_positional_arguments(tmp_path: Path, email_policy: Policy) -> None:
    guard = Guard(email_policy, home=tmp_path)
    seen: list[tuple[str, str]] = []

    @guard.tool()
    def create_event(title: str, when: str) -> str:
        seen.append((title, when))
        return "ok"

    outcome = create_event("Planning", when="Tue 10:00")
    assert outcome.status is Status.EXECUTED
    assert seen == [("Planning", "Tue 10:00")]
    assert create_event.__wrapped__.__name__ == "create_event"


def test_defaults_are_applied_before_hashing(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    guard.call("search_calendar", {})
    guard.call("search_calendar", {"day": "today"})  # same action once defaults apply
    assert backend.names() == ["search_calendar"]


def test_async_and_varargs_tools_are_rejected(tmp_path: Path, email_policy: Policy) -> None:
    guard = Guard(email_policy, home=tmp_path)

    async def send_email(to: list[str]) -> None:  # pragma: no cover - never awaited
        return None

    def create_event(*titles: str) -> None:  # pragma: no cover - never called
        return None

    with pytest.raises(ConfigurationError, match="async"):
        guard.register("send_email", send_email)
    with pytest.raises(ConfigurationError, match=r"\*args"):
        guard.register("create_event", create_event)


def test_module_level_guarded_decorator(tmp_path: Path, email_policy: Policy) -> None:
    calls: list[str] = []

    @guarded("search_calendar")
    def lookup(day: str = "today") -> str:
        calls.append(day)
        return "free"

    set_default_guard(None)
    with pytest.raises(ConfigurationError):
        lookup()
    guard = Guard(email_policy, home=tmp_path)
    set_default_guard(guard)
    try:
        assert get_default_guard() is guard
        assert lookup("friday").status is Status.EXECUTED
        assert calls == ["friday"]
    finally:
        set_default_guard(None)


def test_every_decision_is_audited_with_redaction(
    make_guard: MakeGuard, email_policy: Policy, tmp_path: Path
) -> None:
    guard = make_guard(email_policy)
    guard.call("search_calendar", {})
    guard.call("draft_reply", {"to": "a@example.com", "body": "draft"})
    guard.call("send_email", {"to": ["a@example.com"], "subject": "s", "body": "very private"})
    guard.call("delete_mailbox", {})
    guard.call("x", {"api_key": "sk-123"})
    log = guard.audit.path.read_text()
    events = [line.split('"event":"')[1].split('"')[0] for line in log.splitlines()]
    assert events == ["allowed", "executed", "drafted", "queued", "blocked", "blocked"]
    assert "very private" not in log
    assert "sk-123" not in log
    assert verify_log(guard.audit.path).ok


def test_redact_hook(tmp_path: Path, email_policy: Policy) -> None:
    def hook(key: str, value: Any) -> Any:
        return "<subject hidden>" if key == "subject" else value

    guard = Guard(email_policy, home=tmp_path, redact_hook=hook)
    guard.call("send_email", {"to": ["a@example.com"], "subject": "Salary review", "body": "b"})
    log = guard.audit.path.read_text()
    assert "Salary review" not in log
    assert "<subject hidden>" in log


def test_malformed_arguments_are_blocked_and_still_audited(tmp_path: Path) -> None:
    policy = Policy.from_yaml(
        "tools: {pay: {mode: allow, args: {amount: {type: float}}}, ping: {mode: allow}}\n"
    )
    guard = Guard(policy, home=tmp_path)
    guard.register("pay", lambda amount: None)
    guard.register("ping", lambda payload: None)
    assert guard.call("pay", {"amount": float("nan")}).reason is Reason.INVALID_ARGUMENTS
    assert guard.call("ping", {"payload": float("inf")}).reason is Reason.INVALID_ARGUMENTS
    assert guard.call("ping", {"payload": object()}).reason is Reason.INVALID_ARGUMENTS
    assert guard.call("unknown", {"x": object()}).reason is Reason.UNKNOWN_TOOL
    log = guard.audit.path.read_text()
    assert "<object>" in log and '"amount":"nan"' in log
    assert verify_log(guard.audit.path).ok
