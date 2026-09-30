"""Rate limits, budgets and queue size (OWASP LLM10:2025 Unbounded Consumption)."""

from __future__ import annotations

from pathlib import Path

from agent_guardrails import Guard, Policy, Reason, Status

from .conftest import FakeClock


def _guard(tmp_path: Path, clock: FakeClock, yaml_text: str) -> tuple[Guard, list[float]]:
    guard = Guard(Policy.from_yaml(yaml_text), home=tmp_path, clock=clock)
    paid: list[float] = []

    def pay(amount: float, ref: str = "") -> str:
        paid.append(amount)
        return "paid"

    def ping(n: int) -> int:
        return n

    guard.register("pay", pay)
    guard.register("ping", ping)
    return guard, paid


RATE = """
tools:
  ping:
    mode: allow
    rate_limit: {max_calls: 3, window_seconds: 60}
"""


def test_rate_limit_blocks_then_recovers(tmp_path: Path, clock: FakeClock) -> None:
    guard, _ = _guard(tmp_path, clock, RATE)
    statuses = [guard.call("ping", {"n": i}).status for i in range(5)]
    assert statuses == [Status.EXECUTED] * 3 + [Status.BLOCKED] * 2
    blocked = guard.call("ping", {"n": 99})
    assert blocked.reason is Reason.RATE_LIMITED
    assert "limit is 3" in blocked.message
    clock.advance(61)
    assert guard.call("ping", {"n": 100}).status is Status.EXECUTED


def test_rate_limit_is_global_unless_per_agent(tmp_path: Path, clock: FakeClock) -> None:
    guard, _ = _guard(tmp_path, clock, RATE)
    for i in range(3):
        guard.call("ping", {"n": i}, agent_id="a")
    assert guard.call("ping", {"n": 9}, agent_id="b").reason is Reason.RATE_LIMITED

    per_agent, _ = _guard(tmp_path / "other", clock, RATE.replace("60}", "60, per_agent: true}"))
    for i in range(3):
        per_agent.call("ping", {"n": i}, agent_id="a")
    assert per_agent.call("ping", {"n": 9}, agent_id="b").status is Status.EXECUTED
    assert per_agent.call("ping", {"n": 10}, agent_id="a").reason is Reason.RATE_LIMITED


def test_failed_calls_count_towards_the_rate_limit(tmp_path: Path, clock: FakeClock) -> None:
    guard = Guard(Policy.from_yaml(RATE), home=tmp_path, clock=clock)

    def ping(n: int) -> int:
        raise TimeoutError

    guard.register("ping", ping)
    for i in range(3):
        assert guard.call("ping", {"n": i}).status is Status.FAILED
    # We cannot know whether a failed call had its side effect, so it still counts.
    assert guard.call("ping", {"n": 4}).reason is Reason.RATE_LIMITED


BUDGET = """
budgets:
  "*": {limit: 10, window_seconds: 3600}
  intern: {limit: 1}
tools:
  pay:
    mode: allow
    args: {amount: {type: float}, ref: {type: str, required: false, default: ""}}
    cost: {field: amount}
    dedupe_window_seconds: 0
"""


def test_budget_by_argument(tmp_path: Path, clock: FakeClock) -> None:
    guard, paid = _guard(tmp_path, clock, BUDGET)
    assert guard.call("pay", {"amount": 6}).status is Status.EXECUTED
    assert guard.call("pay", {"amount": 4}).status is Status.EXECUTED
    over = guard.call("pay", {"amount": 0.01})
    assert over.reason is Reason.BUDGET_EXCEEDED
    assert "spent 10 of its 10" in over.message
    assert paid == [6, 4]
    clock.advance(3601)
    assert guard.call("pay", {"amount": 5}).status is Status.EXECUTED


def test_budget_is_per_agent(tmp_path: Path, clock: FakeClock) -> None:
    guard, _ = _guard(tmp_path, clock, BUDGET)
    assert guard.call("pay", {"amount": 2}, agent_id="intern").reason is Reason.BUDGET_EXCEEDED
    assert guard.call("pay", {"amount": 1}, agent_id="intern").status is Status.EXECUTED
    assert guard.call("pay", {"amount": 9}, agent_id="lead").status is Status.EXECUTED


def test_negative_amount_cannot_refill_budget(tmp_path: Path, clock: FakeClock) -> None:
    guard, paid = _guard(tmp_path, clock, BUDGET)
    assert guard.call("pay", {"amount": -100}).reason is Reason.INVALID_ARGUMENTS
    assert paid == []


def test_fixed_cost(tmp_path: Path, clock: FakeClock) -> None:
    yaml_text = """
budgets: {"*": {limit: 1}}
tools:
  ping: {mode: allow, cost: {fixed: 0.4}}
"""
    guard, _ = _guard(tmp_path, clock, yaml_text)
    results = [guard.call("ping", {"n": i}).status for i in range(3)]
    assert results == [Status.EXECUTED, Status.EXECUTED, Status.BLOCKED]


def test_queue_size_limit(tmp_path: Path, clock: FakeClock) -> None:
    yaml_text = """
tools:
  pay: {mode: approve, max_pending: 2}
"""
    guard, _ = _guard(tmp_path, clock, yaml_text)
    assert guard.call("pay", {"amount": 1}).status is Status.QUEUED
    assert guard.call("pay", {"amount": 2}).status is Status.QUEUED
    full = guard.call("pay", {"amount": 3})
    assert full.reason is Reason.QUEUE_FULL


def test_request_costing_more_than_whole_budget_is_not_queued(
    tmp_path: Path, clock: FakeClock
) -> None:
    yaml_text = """
budgets: {"*": {limit: 50}}
tools:
  pay: {mode: approve, cost: {field: amount}}
"""
    guard, _ = _guard(tmp_path, clock, yaml_text)
    outcome = guard.call("pay", {"amount": 500})
    assert outcome.reason is Reason.BUDGET_EXCEEDED
    assert guard.pending() == []
