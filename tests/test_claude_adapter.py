"""The Claude adapter, with a scripted fake client (no network, no API key)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_guardrails import ConfigurationError, Guard, Policy, Status
from agent_guardrails.adapters.claude import (
    DEFAULT_MODEL,
    FALLBACK_BETA,
    handle_tool_uses,
    run_tool_loop,
    tool_definition,
    tool_result_block,
)

from .conftest import Backend

MakeGuard = Callable[..., Guard]


@dataclass
class Block:
    type: str
    id: str = ""
    name: str = ""
    input: Any = None
    text: str = ""


@dataclass
class Message:
    content: list[Block]
    stop_reason: str


@dataclass
class FakeMessages:
    script: list[Message]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> Message:
        # Snapshot the history as sent, since the caller keeps appending to it.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script: list[Message]) -> None:
        self.messages = FakeMessages(list(script))
        self.beta = type("Beta", (), {"messages": FakeMessages(self.messages.script)})()


def tool_use(id_: str, name: str, **inp: Any) -> Block:
    return Block(type="tool_use", id=id_, name=name, input=inp)


EMAIL_ARGS = {"to": ["a@example.com"], "subject": "Hi", "body": "See you"}


def test_full_loop_with_mixed_outcomes(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    client = FakeClient(
        [
            Message(
                [
                    Block(type="text", text="Checking and emailing."),
                    tool_use("t1", "search_calendar", day="monday"),
                    tool_use("t2", "send_email", **EMAIL_ARGS),
                    tool_use("t3", "delete_mailbox"),
                ],
                "tool_use",
            ),
            Message([Block(type="text", text="Done: the email awaits approval.")], "end_turn"),
        ]
    )
    result = run_tool_loop(
        client,
        guard,
        messages=[{"role": "user", "content": "Email Alice about Monday"}],
        tools=[],
        system="You are an assistant.",
    )
    assert result.stop_reason == "end_turn"
    assert result.turns == 2
    assert result.text == "Done: the email awaits approval."
    assert [o.status for o in result.outcomes] == [
        Status.EXECUTED,
        Status.QUEUED,
        Status.BLOCKED,
    ]
    assert backend.names() == ["search_calendar"]

    # Requests used the beta endpoint with server-side fallbacks and the default model.
    sent = client.beta.messages.requests
    assert all(r["model"] == DEFAULT_MODEL == "claude-opus-5-5" for r in sent)
    assert all(r["betas"] == [FALLBACK_BETA] and r["fallbacks"] == "default" for r in sent)
    assert sent[0]["system"] == "You are an assistant."

    # All tool results went back together, in order, in one user message.
    results_msg = sent[1]["messages"][-1]
    assert results_msg["role"] == "user"
    ids = [b["tool_use_id"] for b in results_msg["content"]]
    assert ids == ["t1", "t2", "t3"]
    by_id = {b["tool_use_id"]: b for b in results_msg["content"]}
    assert "is_error" not in by_id["t1"]
    assert "QUEUED FOR HUMAN APPROVAL" in by_id["t2"]["content"]
    assert "is_error" not in by_id["t2"]
    assert by_id["t3"]["is_error"] is True
    assert "BLOCKED BY POLICY" in by_id["t3"]["content"]


def test_no_tools_run_on_refusal(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    client = FakeClient([Message([tool_use("t1", "search_calendar")], "refusal")])
    result = run_tool_loop(client, guard, messages=[], tools=[])
    assert result.stop_reason == "refusal"
    assert result.outcomes == []
    assert backend.calls == []


def test_no_tools_run_when_tool_input_may_be_truncated(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    client = FakeClient([Message([tool_use("t1", "search_calendar")], "max_tokens")])
    result = run_tool_loop(client, guard, messages=[], tools=[])
    assert result.stop_reason == "max_tokens"
    assert backend.calls == []


def test_pause_turn_is_resumed_and_max_turns_caps_the_loop(
    make_guard: MakeGuard, email_policy: Policy
) -> None:
    guard = make_guard(email_policy)
    paused = Message([Block(type="text", text="...")], "pause_turn")
    client = FakeClient([paused, paused, paused])
    result = run_tool_loop(client, guard, messages=[], tools=[], max_turns=3)
    assert result.stop_reason == "max_turns"
    assert result.turns == 3


def test_without_fallback_uses_plain_messages_endpoint(
    make_guard: MakeGuard, email_policy: Policy
) -> None:
    guard = make_guard(email_policy)
    client = FakeClient([Message([Block(type="text", text="hi")], "end_turn")])
    client.messages.script = [Message([Block(type="text", text="hi")], "end_turn")]
    result = run_tool_loop(
        client, guard, messages=[], tools=[], server_side_fallback=False, model="claude-x"
    )
    assert result.text == "hi"
    assert client.messages.requests[0]["model"] == "claude-x"
    assert "fallbacks" not in client.messages.requests[0]


def test_extra_betas_are_kept(make_guard: MakeGuard, email_policy: Policy) -> None:
    guard = make_guard(email_policy)
    client = FakeClient([Message([], "end_turn")])
    run_tool_loop(client, guard, messages=[], tools=[], betas=["some-other-beta"])
    assert client.beta.messages.requests[0]["betas"] == ["some-other-beta", FALLBACK_BETA]


def test_non_object_input_is_rejected(make_guard: MakeGuard, email_policy: Policy) -> None:
    guard = make_guard(email_policy)
    results, outcomes = handle_tool_uses(
        guard, [Block(type="tool_use", id="t", name="x", input="oops")]
    )
    assert results[0]["is_error"] is True
    assert outcomes[0].status is Status.BLOCKED


def test_tool_result_block_shapes(make_guard: MakeGuard, email_policy: Policy) -> None:
    guard = make_guard(email_policy)
    drafted = guard.call("draft_reply", {"to": "a@example.com", "body": "Thanks"})
    block = tool_result_block("toolu_1", drafted)
    assert block["type"] == "tool_result" and block["tool_use_id"] == "toolu_1"
    assert "DRAFT ONLY" in block["content"] and "is_error" not in block


def test_tool_definition_from_policy(email_policy: Policy) -> None:
    definition = tool_definition(email_policy, "send_email", "Send an email. Needs approval.")
    schema = definition["input_schema"]
    assert definition["name"] == "send_email"
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"to", "subject", "body"}
    assert schema["additionalProperties"] is False
    with pytest.raises(ConfigurationError):
        tool_definition(email_policy, "search_calendar", "no schema in policy")
    custom = tool_definition(
        email_policy, "search_calendar", "Search", input_schema={"type": "object"}
    )
    assert custom["input_schema"] == {"type": "object"}


def test_real_sdk_types_are_handled(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    beta = pytest.importorskip("anthropic.types.beta")
    guard = make_guard(email_policy)
    message = beta.BetaMessage.model_construct(
        id="msg_1",
        type="message",
        role="assistant",
        model=DEFAULT_MODEL,
        stop_reason="tool_use",
        content=[
            beta.BetaTextBlock(type="text", text="Looking.", citations=None),
            beta.BetaToolUseBlock(
                type="tool_use", id="toolu_1", name="search_calendar", input={"day": "fri"}
            ),
        ],
    )
    results, outcomes = handle_tool_uses(guard, message.content)
    assert outcomes[0].status is Status.EXECUTED
    assert results[0]["tool_use_id"] == "toolu_1"
    assert backend.calls == [("search_calendar", {"day": "fri"})]
