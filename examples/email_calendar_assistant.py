"""Offline demo: a generic email and calendar assistant behind agent-guardrails.

No model and no network. A scripted "agent" makes the tool calls an LLM agent
might make, and a fake backend records what really happened. Every number
printed is produced by this script.

    python examples/email_calendar_assistant.py
    python examples/email_calendar_assistant.py --home ./demo-state   # keep the state
    agent-guardrails --home ./demo-state queue list -s all             # then inspect it
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_guardrails import Guard, Outcome, Policy, Status, verify_log

POLICY_PATH = Path(__file__).with_name("policy.yaml")


@dataclass
class FakeBackend:
    """Stands in for a mail server, a calendar and an SMS gateway."""

    events: list[dict[str, Any]] = field(
        default_factory=lambda: [{"title": "Stand-up", "start": "2026-10-01T09:00"}]
    )
    outbox: list[dict[str, Any]] = field(default_factory=list)
    sms: list[dict[str, Any]] = field(default_factory=list)

    def list_events(self, day: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["start"].startswith(day)]

    def create_event(self, title: str, start: str, attendees: list[str]) -> str:
        self.events.append({"title": title, "start": start, "attendees": attendees})
        return f"event '{title}' created"

    def draft_reply(self, to: str, body: str) -> str:  # pragma: no cover - draft mode
        raise AssertionError("draft mode must never call the real function")

    def send_email(self, to: list[str], subject: str, body: str) -> str:
        self.outbox.append({"to": to, "subject": subject, "body": body})
        return f"email '{subject}' sent to {len(to)} recipient(s)"

    def send_sms(self, to: str, text: str) -> str:
        self.sms.append({"to": to, "text": text})
        return "sms sent"

    def delete_event(self, title: str) -> str:  # pragma: no cover - blocked by policy
        raise AssertionError("blocked by policy; must never run")


def build_guard(home: Path, backend: FakeBackend, agent_id: str = "assistant") -> Guard:
    guard = Guard(Policy.from_yaml(POLICY_PATH), home=home, agent_id=agent_id)
    for name in ("list_events", "create_event", "send_email", "send_sms", "delete_event"):
        guard.register(name, getattr(backend, name))
    guard.register(
        "draft_reply",
        backend.draft_reply,
        preview=lambda tool, a: f"To: {a['to']}\n\n{a['body']}",
    )
    return guard


def _show(step: str, outcome: Outcome) -> None:
    label = str(outcome.status).upper()
    extra = f" [{outcome.reason}]" if outcome.reason else ""
    print(f"\n{step}\n  -> {label}{extra}")
    for line in outcome.as_tool_result().splitlines():
        print(f"     | {line}")


def run(home: Path) -> dict[str, Any]:
    """Run the scenario. Returns a summary (used by the end-to-end test)."""
    backend = FakeBackend()
    guard = build_guard(home, backend)
    seen: list[tuple[str, Status]] = []

    def step(label: str, tool: str, /, **args: Any) -> Outcome:
        outcome = guard.call(tool, args)
        seen.append((label, outcome.status))
        _show(label, outcome)
        return outcome

    print("== Agent turn: the user asks to set up tomorrow's project sync ==")
    step("1. Read the calendar (allow)", "list_events", day="2026-10-01")
    step(
        "2. Create an event (allow, within limits)",
        "create_event",
        title="Project sync",
        start="2026-10-01T10:00",
        attendees=["sam@example.com"],
    )
    step(
        "3. Draft a reply (draft: nothing is sent)",
        "draft_reply",
        to="sam@example.com",
        body="Hi Sam, I have booked 10:00 tomorrow for the project sync.",
    )
    reminder = {
        "to": ["team@example.com"],
        "subject": "Reminder: project sync at 10:00",
        "body": "Agenda: status, risks, next steps.",
    }
    queued = step("4. Email the team (approve: queued for a human)", "send_email", **reminder)
    step("5. The agent retries the same email (duplicate)", "send_email", **reminder)
    step(
        "6. Email an outside address (blocked: domain not allow-listed)",
        "send_email",
        to=["someone@partner.test"],
        subject="Minutes",
        body="Attached.",
    )
    step("7. Delete an event (blocked by policy)", "delete_event", title="Stand-up")

    print("\n== A reviewer approves the queued email (same call the CLI makes) ==")
    assert queued.action_id is not None
    guard.approve(queued.action_id, by="reviewer@example.com")
    print(f"  approved {queued.action_id}")
    print("\n== The application's worker runs approved actions, re-validating each ==")
    for outcome in guard.run_approved():
        _show("8. Execute approved email", outcome)
    step("9. The agent retries again after it was sent (duplicate)", "send_email", **reminder)

    print("\n== Spend limits: SMS costs 1 credit, budget is 2 per day ==")
    for n in (1, 2, 3):
        step(f"10.{n} Send SMS #{n}", "send_sms", to="+447700900123", text=f"Sync at 10 ({n})")

    print("\n== Kill switch ==")
    second = step(
        "11. Queue a follow-up email",
        "send_email",
        to=["team@example.com"],
        subject="Notes from the project sync",
        body="Notes to follow.",
    )
    assert second.action_id is not None
    guard.approve(second.action_id, by="reviewer@example.com")
    guard.engage_kill_switch("demo: suspicious behaviour reported", by="operator")
    print("  operator engaged the kill switch")
    for outcome in guard.run_approved():
        _show("12. Worker tries the approved follow-up", outcome)
    step("13. Agent tries to read the calendar", "list_events", day="2026-10-01")
    guard.release_kill_switch(by="operator")
    guard.reject(second.action_id, by="operator", reason="reviewed during incident")
    print("  operator released the switch and rejected the follow-up after review")

    print("\n== Audit log ==")
    log_path = guard.audit.path
    count, head = guard.audit.head()
    result = verify_log(log_path, expected_head=head)
    print(f"  {count} records; chain intact: {result.ok}; head {head[:16]}...")
    tampered = home / "audit-tampered.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    # Pretend someone rewrites history: the draft (never sent) now claims "executed".
    i = next(n for n, line in enumerate(lines) if '"event":"drafted"' in line)
    lines[i] = lines[i].replace('"event":"drafted"', '"event":"executed"')
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad = verify_log(tampered)
    print(f"  edited copy verifies: {bad.ok} (line {bad.line}: {bad.error})")

    summary = {
        "steps": seen,
        "emails_sent": [m["subject"] for m in backend.outbox],
        "events": [e["title"] for e in backend.events],
        "sms_sent": len(backend.sms),
        "audit_records": count,
        "audit_ok": result.ok,
        "tamper_detected": not bad.ok,
        "pending": len(guard.pending()),
    }
    print("\n== What actually happened in the backend ==")
    print(f"  emails sent: {summary['emails_sent']}")
    print(f"  calendar:    {summary['events']}")
    print(f"  SMS sent:    {summary['sms_sent']}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--home", type=Path, help="keep state here instead of a temp dir")
    args = parser.parse_args(argv)
    if args.home is not None:
        if args.home.exists():
            shutil.rmtree(args.home)
        run(args.home)
        print(f"\nState kept in {args.home}. Try:")
        print(f"  agent-guardrails --home {args.home} queue list -s all")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        run(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
