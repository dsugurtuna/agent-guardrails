from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agent_guardrails import ActionState, Guard, KillSwitch, Policy, Reason, Status

from .conftest import Backend

MakeGuard = Callable[..., Guard]
ARGS = {"to": ["a@example.com"], "subject": "s", "body": "b"}


def test_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ks = KillSwitch(flag_file=tmp_path / "KILL")
    assert not ks.engaged
    ks.engage("maintenance", by="ops")
    status = ks.status()
    assert (status.engaged, status.source, status.reason) == (True, "file", "maintenance")
    assert ks.release().engaged is False

    api_only = KillSwitch(flag_file=None)
    api_only.engage("api stop")
    assert api_only.status().source == "api"
    api_only.release()
    assert not api_only.engaged

    monkeypatch.setenv("AGENT_GUARDRAILS_KILL", "1")
    status = ks.release()
    assert status.engaged and status.source == "env"  # release cannot clear the env var
    monkeypatch.setenv("AGENT_GUARDRAILS_KILL", "0")
    assert not ks.engaged


def test_hand_made_flag_file_still_engages(tmp_path: Path) -> None:
    (tmp_path / "KILL").write_text("")
    status = KillSwitch(flag_file=tmp_path / "KILL").status()
    assert status.engaged and status.reason == "flag file present"


def test_blocks_every_mode(make_guard: MakeGuard, email_policy: Policy, backend: Backend) -> None:
    guard = make_guard(email_policy)
    guard.engage_kill_switch("incident 42", by="ops")
    for tool, args in [
        ("search_calendar", {}),
        ("draft_reply", {"to": "a@example.com", "body": "x"}),
        ("send_email", ARGS),
    ]:
        outcome = guard.call(tool, args)
        assert outcome.status is Status.BLOCKED
        assert outcome.reason is Reason.KILL_SWITCH
        assert "incident 42" in outcome.message
    assert backend.calls == []
    assert guard.pending() == []


def test_blocks_approved_actions_and_keeps_them_for_review(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    queued = guard.call("send_email", ARGS)
    assert queued.action_id is not None
    guard.approve(queued.action_id, by="alice")

    guard.engage_kill_switch("stop everything")
    [outcome] = guard.run_approved()
    assert outcome.reason is Reason.KILL_SWITCH
    assert guard.store.get(queued.action_id).status is ActionState.APPROVED
    assert backend.calls == []

    guard.release_kill_switch(by="ops")
    [outcome] = guard.run_approved()
    assert outcome.status is Status.EXECUTED
    assert backend.names() == ["send_email"]


def test_flag_file_shared_between_processes(tmp_path: Path, email_policy: Policy) -> None:
    app = Guard(email_policy, home=tmp_path)
    app.register("search_calendar", lambda day="today": [])
    operator = KillSwitch(flag_file=tmp_path / "KILL")  # e.g. the CLI on another host
    operator.engage("from the operator")
    assert app.call("search_calendar", {}).reason is Reason.KILL_SWITCH
    operator.release()
    assert app.call("search_calendar", {}).status is Status.EXECUTED


class FlipsOnSecondCheck(KillSwitch):
    """Engages between the first check and the last-moment check."""

    def __init__(self) -> None:
        super().__init__(flag_file=None, env_var=None)
        self.checks = 0

    def status(self):  # type: ignore[no-untyped-def]
        self.checks += 1
        if self.checks >= 2:
            self.engage("pulled mid-flight")
        return super().status()


def test_last_moment_check_before_running_the_tool(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy, kill_switch=FlipsOnSecondCheck())
    outcome = guard.call("search_calendar", {})
    assert outcome.reason is Reason.KILL_SWITCH
    assert backend.calls == []
    assert outcome.action_id is not None
    assert guard.store.get(outcome.action_id).status is ActionState.BLOCKED


@pytest.mark.parametrize("value", ["1", "true", "TRUE ", "yes", "stop", "engaged", "y", "enabled"])
def test_env_var_engages_unless_explicitly_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    # An operator in an incident types whatever comes to mind. Anything that is not
    # clearly "off" must stop the agent: a kill switch fails closed.
    monkeypatch.setenv("AGENT_GUARDRAILS_KILL", value)
    status = KillSwitch(flag_file=tmp_path / "KILL").status()
    assert status.engaged and status.source == "env"


@pytest.mark.parametrize("value", ["", "  ", "0", "false", "False", "no", "off", " OFF "])
def test_env_var_explicitly_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("AGENT_GUARDRAILS_KILL", value)
    assert not KillSwitch(flag_file=tmp_path / "KILL").engaged
