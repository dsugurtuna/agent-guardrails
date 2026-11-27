"""The same assistant driven via OpenAI or Azure OpenAI Chat Completions.

Runs in offline demo mode by default (no ``openai`` package, no network, no API keys)
using a scripted client that emits OpenAI-shaped tool calls against ``FakeBackend``:

    python examples/openai_assistant.py
    python examples/openai_assistant.py --home ./openai-demo
    agent-guardrails --home ./openai-demo queue list
    agent-guardrails --home ./openai-demo queue approve <id> --by you
    python examples/openai_assistant.py --home ./openai-demo --run-approved

To run against a real endpoint (needs ``pip install "agent-guardrails[openai]"``):

    OPENAI_API_KEY=... python examples/openai_assistant.py --provider openai
    AZURE_OPENAI_ENDPOINT=https://example.openai.azure.com/ \\
      AZURE_OPENAI_API_KEY=... \\
      python examples/openai_assistant.py --provider azure --model my-deployment
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_guardrails import verify_log
from agent_guardrails.adapters.openai import (
    DEFAULT_MODEL,
    OpenAIClientLike,
    ToolLoopResult,
    run_tool_loop,
    tool_definition,
)

_EXAMPLES_DIR = str(Path(__file__).resolve().parent)
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from email_calendar_assistant import FakeBackend, build_guard  # noqa: E402

SYSTEM = (
    "You are an email and calendar assistant for staff at example.com. Today is 2026-10-01. "
    "Some tools only draft or queue actions for human approval; when a tool result says so, "
    "tell the user plainly what happened and do not retry the same call."
)


def tools_for(policy: Any) -> list[dict[str, Any]]:
    """Build OpenAI function tool definitions backed by ``policy`` schemas."""
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
            parameters={
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


@dataclass
class _DemoFunction:
    name: str
    arguments: str


@dataclass
class _DemoToolCall:
    id: str
    function: _DemoFunction
    type: str = "function"


@dataclass
class _DemoMessage:
    content: str | None = None
    tool_calls: list[_DemoToolCall] | None = None
    role: str = "assistant"
    refusal: str | None = None


@dataclass
class _DemoChoice:
    finish_reason: str
    message: _DemoMessage


@dataclass
class _DemoCompletion:
    choices: list[_DemoChoice]


@dataclass
class _DemoCompletions:
    script: list[_DemoCompletion]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> _DemoCompletion:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.script.pop(0)


@dataclass
class _DemoChat:
    completions: _DemoCompletions


class _DemoOpenAIClient:
    """Offline fake client producing standard OpenAI ChatCompletion responses."""

    def __init__(self, script: list[_DemoCompletion]) -> None:
        self.chat = _DemoChat(completions=_DemoCompletions(list(script)))


def _demo_client() -> _DemoOpenAIClient:
    return _DemoOpenAIClient(
        [
            _DemoCompletion(
                choices=[
                    _DemoChoice(
                        finish_reason="tool_calls",
                        message=_DemoMessage(
                            content=None,
                            tool_calls=[
                                _DemoToolCall(
                                    id="call_1",
                                    function=_DemoFunction(
                                        name="list_events",
                                        arguments=json.dumps({"day": "2026-10-01"}),
                                    ),
                                ),
                                _DemoToolCall(
                                    id="call_2",
                                    function=_DemoFunction(
                                        name="draft_reply",
                                        arguments=json.dumps(
                                            {
                                                "to": "sam@example.com",
                                                "body": "Stand-up is at 09:00 today.",
                                            }
                                        ),
                                    ),
                                ),
                                _DemoToolCall(
                                    id="call_3",
                                    function=_DemoFunction(
                                        name="send_email",
                                        arguments=json.dumps(
                                            {
                                                "to": ["team@example.com"],
                                                "subject": "Reminder: project sync",
                                                "body": "Please review the agenda before 10:00.",
                                            }
                                        ),
                                    ),
                                ),
                                _DemoToolCall(
                                    id="call_4",
                                    function=_DemoFunction(
                                        name="delete_event",
                                        arguments=json.dumps({"title": "Stand-up"}),
                                    ),
                                ),
                            ],
                        ),
                    )
                ]
            ),
            _DemoCompletion(
                choices=[
                    _DemoChoice(
                        finish_reason="stop",
                        message=_DemoMessage(
                            content=(
                                "Checked today's calendar (09:00 Stand-up), prepared a draft "
                                "reply to Sam, and queued the team reminder email for human "
                                "approval. Deleting Stand-up was blocked by policy."
                            )
                        ),
                    )
                ]
            ),
        ]
    )


def _live_client(provider: str) -> tuple[Any, OpenAIClientLike]:
    try:
        openai_mod = importlib.import_module("openai")
    except ModuleNotFoundError:
        print('Missing optional dependency. Install with: pip install "agent-guardrails[openai]"')
        raise SystemExit(1) from None

    if provider == "azure":
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
        if not endpoint:
            print("Set AZURE_OPENAI_ENDPOINT when using --provider azure.")
            raise SystemExit(1)
        client: OpenAIClientLike = openai_mod.AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        )
        return openai_mod, client

    client = openai_mod.OpenAI()
    return openai_mod, client


def run_demo(home: Path, request: str = "Remind the team about today's sync.") -> ToolLoopResult:
    """Run the offline OpenAI Chat Completions demo against ``home``."""
    backend = FakeBackend()
    guard = build_guard(home, backend)
    client = _demo_client()
    result = run_tool_loop(
        client,
        guard,
        system=SYSTEM,
        messages=[{"role": "user", "content": request}],
        tools=tools_for(guard.policy),
        model=DEFAULT_MODEL,
    )
    print(f"Stopped: {result.stop_reason} after {result.turns} request(s)\n")
    for outcome in result.outcomes:
        extra = f" ({outcome.reason})" if outcome.reason else ""
        print(f"- {outcome.tool}: {outcome.status}{extra}")
    print(f"\nAssistant: {result.text}")
    for record in guard.pending():
        print(f"\nAwaiting approval: {record.id} {record.tool} {record.args}")
    audit = verify_log(guard.audit.path)
    print(f"\nAudit log: {audit.records} records; chain intact: {audit.ok}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenAI / Azure OpenAI demo of agent-guardrails")
    parser.add_argument("request", nargs="?", default="Remind the team about today's sync.")
    parser.add_argument("--home", type=Path, help="keep state here instead of a temp dir")
    parser.add_argument(
        "--provider",
        choices=("demo", "openai", "azure"),
        default="demo",
        help="use the offline demo client (default), OpenAI, or Azure OpenAI",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", DEFAULT_MODEL),
        help="OpenAI model or Azure OpenAI deployment name",
    )
    parser.add_argument("--run-approved", action="store_true", help="execute approved actions")
    args = parser.parse_args(argv)

    if args.run_approved:
        home = args.home or Path("./openai-demo")
        backend = FakeBackend()
        guard = build_guard(home, backend)
        for outcome in guard.run_approved():
            print(outcome.status, outcome.as_tool_result())
        print("Fake outbox:", backend.outbox)
        return 0

    if args.provider == "demo":
        if args.home is not None:
            if args.home.exists():
                shutil.rmtree(args.home)
            run_demo(args.home, args.request)
            print(f"\nState kept in {args.home}. Try:")
            print(f"  agent-guardrails --home {args.home} queue list -s all")
            return 0
        with tempfile.TemporaryDirectory() as tmp:
            run_demo(Path(tmp), args.request)
        return 0

    home = args.home or Path("./openai-demo")
    backend = FakeBackend()
    guard = build_guard(home, backend)
    openai_mod, client = _live_client(args.provider)
    try:
        result = run_tool_loop(
            client,
            guard,
            model=args.model,
            system=SYSTEM,
            messages=[{"role": "user", "content": args.request}],
            tools=tools_for(guard.policy),
        )
    except openai_mod.AuthenticationError:
        print("No valid OpenAI/Azure OpenAI credentials found. Run the offline demo instead.")
        return 1
    except openai_mod.APIStatusError as exc:
        print(f"API error {exc.status_code}: {exc.message}")
        return 1
    except openai_mod.APIConnectionError:
        print("Could not reach the API.")
        return 1

    print(f"Stopped: {result.stop_reason} after {result.turns} request(s)\n")
    for outcome in result.outcomes:
        extra = f" ({outcome.reason})" if outcome.reason else ""
        print(f"- {outcome.tool}: {outcome.status}{extra}")
    print(f"\nAssistant: {result.text}")
    for record in guard.pending():
        print(f"\nAwaiting approval: {record.id} {record.tool} {record.args}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
