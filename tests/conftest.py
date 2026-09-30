from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_guardrails import Guard, Policy


class FakeClock:
    """A controllable clock so TTLs and windows can be tested without sleeping."""

    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def _no_env_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_GUARDRAILS_KILL", raising=False)
    monkeypatch.delenv("AGENT_GUARDRAILS_HOME", raising=False)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


EMAIL_POLICY: dict[str, Any] = {
    "tools": {
        "search_calendar": {"mode": "allow"},
        "create_event": {
            "mode": "allow",
            "args": {"title": {"type": "str", "max_length": 100}, "when": {"type": "str"}},
            "rate_limit": {"max_calls": 3, "window_seconds": 60},
        },
        "draft_reply": {"mode": "draft"},
        "send_email": {
            "mode": "approve",
            "args": {
                "to": {"type": "list[str]", "min_items": 1},
                "subject": {"type": "str", "max_length": 200},
                "body": {"type": "str"},
            },
            "recipients": {
                "fields": ["to"],
                "allowed_domains": ["example.com", "*.example.org"],
                "max_recipients": 3,
            },
            "approval_ttl_seconds": 600,
            "dedupe_window_seconds": 3600,
            "redact_fields": ["body"],
        },
        "delete_mailbox": {"mode": "block"},
    }
}


@pytest.fixture
def email_policy() -> Policy:
    return Policy.from_dict(EMAIL_POLICY)


class Backend:
    """Records calls so tests can assert what actually ran."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


@pytest.fixture
def backend() -> Backend:
    return Backend()


@pytest.fixture
def make_guard(tmp_path: Path, clock: FakeClock, backend: Backend) -> Callable[..., Guard]:
    """Build a guard with the standard fake tools registered."""

    def _make(policy: Policy, **kwargs: Any) -> Guard:
        kwargs.setdefault("home", tmp_path / "state")
        kwargs.setdefault("clock", clock)
        guard = Guard(policy, **kwargs)

        def search_calendar(day: str = "today") -> list[str]:
            backend.calls.append(("search_calendar", {"day": day}))
            return ["09:00 stand-up"]

        def create_event(title: str, when: str) -> str:
            backend.calls.append(("create_event", {"title": title, "when": when}))
            return f"created {title}"

        def draft_reply(to: str, body: str) -> str:
            backend.calls.append(("draft_reply", {"to": to, "body": body}))
            return "should never run"

        def send_email(to: list[str], subject: str, body: str) -> str:
            backend.calls.append(("send_email", {"to": to, "subject": subject, "body": body}))
            return "sent"

        def delete_mailbox() -> None:
            backend.calls.append(("delete_mailbox", {}))

        for fn in (search_calendar, create_event, draft_reply, send_email, delete_mailbox):
            guard.register(fn.__name__, fn)
        return guard

    return _make
