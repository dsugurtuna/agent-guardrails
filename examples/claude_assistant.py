"""The same assistant driven by Claude through the tool-use API.

Needs ``pip install "agent-guardrails[anthropic]"`` and Anthropic credentials
(``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile). The tools still act on
the fake backend, so nothing real is sent. This file is not run by the test
suite; ``tests/test_claude_adapter.py`` covers the loop with a fake client.

    python examples/claude_assistant.py --home ./claude-demo "Remind the team about the sync"
    agent-guardrails --home ./claude-demo queue list
    agent-guardrails --home ./claude-demo queue approve <id> --by you
    python examples/claude_assistant.py --home ./claude-demo --run-approved
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import anthropic

from agent_guardrails.adapters.claude import DEFAULT_MODEL, run_tool_loop, tool_definition
from email_calendar_assistant import FakeBackend, build_guard

SYSTEM = (
    "You are an email and calendar assistant for staff at example.com. Today is 2026-09-30. "
    "Some tools only draft or queue actions for human approval; when a tool result says so, "
    "tell the user plainly what happened and do not retry the same call."
)


def tools_for(policy: Any) -> list[dict[str, Any]]:
    return [
        tool_definition(
            policy,
            "list_events",
            "List calendar events for one day (YYYY-MM-DD). Call this before proposing times.",
        ),
        tool_definition(
            policy,
            "create_event",
            "Create a calendar event. Attendees must be example.com addresses.",
        ),
        tool_definition(
            policy,
            "draft_reply",
            "Draft a reply for the user to review. Nothing is sent.",
            input_schema={
                "type": "object",
                "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                "required": ["to", "body"],
                "additionalProperties": False,
            },
        ),
        tool_definition(
            policy,
            "send_email",
            "Send an email. A human must approve it first; the result tells you it is queued.",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Claude-driven demo of agent-guardrails")
    parser.add_argument("request", nargs="?", default="Remind the team about tomorrow's sync.")
    parser.add_argument("--home", type=Path, default=Path("./claude-demo"))
    parser.add_argument("--run-approved", action="store_true", help="execute approved actions")
    args = parser.parse_args()

    backend = FakeBackend()
    guard = build_guard(args.home, backend)

    if args.run_approved:
        for outcome in guard.run_approved():
            print(outcome.status, outcome.as_tool_result())
        print("Fake outbox:", backend.outbox)
        return 0

    client = anthropic.Anthropic()
    try:
        result = run_tool_loop(
            client,
            guard,
            model=DEFAULT_MODEL,
            system=SYSTEM,
            messages=[{"role": "user", "content": args.request}],
            tools=tools_for(guard.policy),
            output_config={"effort": "medium"},  # this model's default, stated explicitly
        )
    except anthropic.AuthenticationError:
        print("No valid Anthropic credentials found. Run the offline example instead.")
        return 1
    except anthropic.APIStatusError as exc:
        print(f"API error {exc.status_code}: {exc.message}")
        return 1
    except anthropic.APIConnectionError:
        print("Could not reach the API.")
        return 1

    print(f"Stopped: {result.stop_reason} after {result.turns} request(s)\n")
    for outcome in result.outcomes:
        print(f"- {outcome.tool}: {outcome.status} {outcome.reason or ''}")
    print(f"\nClaude: {result.text}")
    for record in guard.pending():
        print(f"\nAwaiting approval: {record.id} {record.tool} {record.args}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
