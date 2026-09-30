"""End to end: the offline email and calendar example behaves as documented."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from agent_guardrails import Status, verify_log

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "email_calendar_assistant.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("email_calendar_assistant", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_example_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary = _load().run(tmp_path)
    statuses = [status for _, status in summary["steps"]]
    assert statuses == [
        Status.EXECUTED,  # 1 read calendar
        Status.EXECUTED,  # 2 create event
        Status.DRAFTED,  # 3 draft reply
        Status.QUEUED,  # 4 email queued
        Status.DUPLICATE,  # 5 retry while pending
        Status.BLOCKED,  # 6 outside domain
        Status.BLOCKED,  # 7 delete blocked
        Status.DUPLICATE,  # 9 retry after sending
        Status.EXECUTED,  # 10.1 sms
        Status.EXECUTED,  # 10.2 sms
        Status.BLOCKED,  # 10.3 over budget
        Status.QUEUED,  # 11 follow-up queued
        Status.BLOCKED,  # 13 kill switch
    ]
    # The one approved email was sent exactly once; the follow-up never was.
    assert summary["emails_sent"] == ["Reminder: project sync at 10:00"]
    assert summary["events"] == ["Stand-up", "Project sync"]
    assert summary["sms_sent"] == 2
    assert summary["pending"] == 0
    assert summary["audit_ok"] is True
    assert summary["tamper_detected"] is True
    assert verify_log(tmp_path / "audit.jsonl").ok
    out = capsys.readouterr().out
    assert "BLOCKED [kill_switch]" in out
    # Message bodies are redacted from the audit log.
    assert "Agenda: status" not in (tmp_path / "audit.jsonl").read_text()


def test_example_main_runs_in_a_temp_dir(capsys: pytest.CaptureFixture[str]) -> None:
    assert _load().main([]) == 0
    assert "emails sent" in capsys.readouterr().out
