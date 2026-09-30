from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_guardrails import ActionState, Guard, Policy, Status
from agent_guardrails.cli import main

POLICY = Policy.from_yaml("tools: {send: {mode: approve}}\n")


def _queued(home: Path) -> tuple[Guard, str, list[str]]:
    guard = Guard(POLICY, home=home)
    sent: list[str] = []
    guard.register("send", lambda to: sent.append(to))
    outcome = guard.call("send", {"to": "a@example.com"})
    assert outcome.action_id is not None
    return guard, outcome.action_id, sent


def test_list_show_approve_then_app_runs_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    guard, action_id, sent = _queued(tmp_path)
    home = ["--home", str(tmp_path)]

    assert main([*home, "queue", "list"]) == 0
    out = capsys.readouterr().out
    assert action_id in out and "send" in out

    assert main([*home, "queue", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in listed] == [action_id]

    assert main([*home, "queue", "show", action_id[:10]]) == 0
    assert json.loads(capsys.readouterr().out)["args"] == {"to": "a@example.com"}

    assert main([*home, "queue", "approve", action_id, "--by", "alice"]) == 0
    assert "Approved" in capsys.readouterr().out
    assert sent == []  # the CLI never runs tools
    [outcome] = guard.run_approved()
    assert outcome.status is Status.EXECUTED
    assert sent == ["a@example.com"]

    assert main([*home, "queue", "list"]) == 0
    assert "No pending actions" in capsys.readouterr().out


def test_reject_and_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    guard, action_id, _ = _queued(tmp_path)
    home = ["--home", str(tmp_path)]
    assert main([*home, "queue", "reject", action_id, "--by", "bob", "--reason", "no"]) == 0
    assert guard.store.get(action_id).status is ActionState.REJECTED
    assert main([*home, "queue", "approve", action_id, "--by", "bob"]) == 1
    assert "only pending" in capsys.readouterr().err
    assert main([*home, "queue", "approve", "act_does_not_exist"]) == 1


def test_audit_verify_and_head(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _queued(tmp_path)
    home = ["--home", str(tmp_path)]
    assert main([*home, "audit", "head"]) == 0
    head = json.loads(capsys.readouterr().out)["head"]
    assert main([*home, "audit", "verify", "--expected-head", head]) == 0
    assert "chain intact" in capsys.readouterr().out

    log = tmp_path / "audit.jsonl"
    log.write_text(log.read_text().replace('"send"', '"sned"'))
    assert main(["audit", "verify", str(log)]) == 1
    assert "FAILED at line 1" in capsys.readouterr().out


def test_kill_switch_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    guard, _, _ = _queued(tmp_path)
    home = ["--home", str(tmp_path)]
    assert main([*home, "kill", "status"]) == 0
    assert "not engaged" in capsys.readouterr().out
    assert main([*home, "kill", "on", "--reason", "drill", "--by", "ops"]) == 0
    assert main([*home, "kill", "status"]) == 0
    assert "ENGAGED via file: drill" in capsys.readouterr().out
    assert guard.call("send", {"to": "b@example.com"}).status is Status.BLOCKED
    assert main([*home, "kill", "off", "--by", "ops"]) == 0
    events = [json.loads(line)["event"] for line in (tmp_path / "audit.jsonl").open()]
    assert "kill_switch_engaged" in events and "kill_switch_released" in events


def test_kill_off_reports_env_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_GUARDRAILS_KILL", "true")
    assert main(["--home", str(tmp_path), "kill", "off"]) == 1
    assert "still engaged via env" in capsys.readouterr().out


def test_home_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _queued(tmp_path)
    monkeypatch.setenv("AGENT_GUARDRAILS_HOME", str(tmp_path))
    assert main(["queue", "list", "--json"]) == 0
