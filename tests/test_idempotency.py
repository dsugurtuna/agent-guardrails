"""An agent that retries must not send the same thing twice."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from agent_guardrails import Guard, Policy, Status
from agent_guardrails._canonical import dedupe_key

from .conftest import Backend, FakeClock

MakeGuard = Callable[..., Guard]


def test_dedupe_key_normalisation() -> None:
    a = dedupe_key("send", {"to": ["A@Example.com", "b@example.com"], "body": "Hi "}, ["to"])
    b = dedupe_key("send", {"body": "Hi", "to": ["b@example.com", "a@example.com"]}, ["to"])
    c = dedupe_key("send", {"to": ["a@example.com"], "body": "Hi"}, ["to"])
    d = dedupe_key("other", {"to": ["A@Example.com", "b@example.com"], "body": "Hi"}, ["to"])
    assert a == b
    assert a != c
    assert a != d
    # Unicode composition: precomposed vs combining accent are the same text.
    assert dedupe_key("t", {"x": "caf\u00e9"}) == dedupe_key("t", {"x": "cafe\u0301"})


def test_executed_action_is_not_repeated(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend, clock: FakeClock
) -> None:
    guard = make_guard(email_policy)
    first = guard.call("search_calendar", {"day": "mon"})
    retry = guard.call("search_calendar", {"day": " mon "})
    assert first.status is Status.EXECUTED
    assert retry.status is Status.DUPLICATE
    assert retry.action_id == first.action_id
    assert not retry.is_error
    assert "already executed" in retry.as_tool_result()
    assert backend.names() == ["search_calendar"]
    clock.advance(3601)  # default window is one hour
    assert guard.call("search_calendar", {"day": "mon"}).status is Status.EXECUTED


def test_retry_while_awaiting_approval_returns_same_request(
    make_guard: MakeGuard, email_policy: Policy
) -> None:
    guard = make_guard(email_policy)
    args = {"to": ["a@example.com", "b@example.com"], "subject": "Reminder", "body": "10am"}
    first = guard.call("send_email", args)
    again = guard.call(
        "send_email",
        {"to": ["B@example.com", "a@example.com"], "subject": "Reminder ", "body": "10am"},
    )
    assert first.status is Status.QUEUED
    assert again.status is Status.DUPLICATE
    assert again.action_id == first.action_id
    assert "awaiting approval" in again.message
    assert len(guard.pending()) == 1


def test_duplicate_of_approved_and_executed_reminder(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    args = {"to": ["a@example.com"], "subject": "Reminder", "body": "10am"}
    queued = guard.call("send_email", args)
    assert queued.action_id is not None
    guard.approve(queued.action_id, by="reviewer")
    assert guard.call("send_email", args).status is Status.DUPLICATE  # approved, due to run
    guard.run_approved()
    assert guard.call("send_email", args).status is Status.DUPLICATE  # already sent
    assert backend.names() == ["send_email"]


def test_different_arguments_are_not_duplicates(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    guard.call("create_event", {"title": "A", "when": "10:00"})
    guard.call("create_event", {"title": "A", "when": "11:00"})
    assert backend.names() == ["create_event", "create_event"]


def test_failed_action_may_be_retried(tmp_path: Path, clock: FakeClock) -> None:
    policy = Policy.from_yaml("tools: {flaky: {mode: allow}}\n")
    guard = Guard(policy, home=tmp_path, clock=clock)
    attempts: list[int] = []

    def flaky(n: int) -> str:
        attempts.append(n)
        if len(attempts) == 1:
            raise ConnectionError
        return "ok"

    guard.register("flaky", flaky)
    assert guard.call("flaky", {"n": 1}).status is Status.FAILED
    assert guard.call("flaky", {"n": 1}).status is Status.EXECUTED
    assert attempts == [1, 1]


def test_dedupe_can_be_disabled(tmp_path: Path, clock: FakeClock) -> None:
    policy = Policy.from_yaml("tools: {ping: {mode: allow, dedupe_window_seconds: 0}}\n")
    guard = Guard(policy, home=tmp_path, clock=clock)
    guard.register("ping", lambda: "pong")
    assert [guard.call("ping").status for _ in range(3)] == [Status.EXECUTED] * 3


def test_crash_mid_execution_blocks_retries_only_within_the_window(
    tmp_path: Path, clock: FakeClock
) -> None:
    policy = Policy.from_yaml("tools: {remind: {mode: allow, dedupe_window_seconds: 600}}\n")
    guard = Guard(policy, home=tmp_path, clock=clock)
    guard.register("remind", lambda to: "ok")
    # Simulate a process that reserved the action and died before recording the result.
    first = guard.call("remind", {"to": "a@example.com"})
    assert first.action_id is not None
    with guard.store.transaction() as tx:
        tx.update(first.action_id, status="executing", finished_at=None)
    assert guard.call("remind", {"to": "a@example.com"}).status is Status.DUPLICATE
    clock.advance(601)
    assert guard.call("remind", {"to": "a@example.com"}).status is Status.EXECUTED


def test_numbers_that_are_equal_in_json_are_the_same_action(
    tmp_path: Path, clock: FakeClock
) -> None:
    # A model may retry {"amount": 10} as {"amount": 10.0}; JSON does not tell them apart.
    policy = Policy.from_yaml("tools: {pay: {mode: allow}}\n")
    guard = Guard(policy, home=tmp_path, clock=clock)
    paid: list[float] = []
    guard.register("pay", lambda amount, to: paid.append(amount))
    assert guard.call("pay", {"amount": 10, "to": "shop"}).status is Status.EXECUTED
    assert guard.call("pay", {"amount": 10.0, "to": "shop"}).status is Status.DUPLICATE
    assert guard.call("pay", {"amount": 10.5, "to": "shop"}).status is Status.EXECUTED
    assert paid == [10, 10.5]
    assert dedupe_key("t", {"x": -0.0}) == dedupe_key("t", {"x": 0})
    assert dedupe_key("t", {"x": True}) != dedupe_key("t", {"x": 1})


def test_recipients_are_compared_as_addresses(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    base = {"subject": "Reminder", "body": "10am"}
    first = guard.call("send_email", {**base, "to": ["sam@example.com"]})
    assert first.status is Status.QUEUED
    for retry in (
        ["Sam <sam@example.com>"],  # display name added
        ["sam@example.com", "SAM@example.com"],  # same person listed twice
    ):
        again = guard.call("send_email", {**base, "to": retry})
        assert again.status is Status.DUPLICATE, retry
        assert again.action_id == first.action_id
    # Scalar, comma-separated and list forms name the same people.
    assert dedupe_key("s", {"to": "a@x.test, b@x.test"}, ["to"]) == dedupe_key(
        "s", {"to": ["b@x.test", "a@x.test"]}, ["to"]
    )
    assert dedupe_key("s", {"to": "a@x.test"}, ["to"]) == dedupe_key(
        "s", {"to": ["a@x.test"]}, ["to"]
    )
    assert len(guard.pending()) == 1
