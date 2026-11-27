"""The OpenAI and Azure OpenAI adapter, tested offline with fake clients (0 skips)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agent_guardrails import ConfigurationError, Guard, Policy, Reason, Status, verify_log
from agent_guardrails.adapters import claude, openai
from agent_guardrails.adapters.openai import (
    DEFAULT_MODEL,
    handle_tool_calls,
    run_tool_loop,
    tool_definition,
    tool_result_message,
    tools_from_policy,
)

from .conftest import Backend

MakeGuard = Callable[..., Guard]


@dataclass
class FunctionCall:
    name: str
    arguments: Any


@dataclass
class ToolCall:
    id: str
    function: FunctionCall
    type: str = "function"


@dataclass
class CustomToolCall:
    id: str
    type: str = "custom"
    custom: dict[str, Any] = field(default_factory=lambda: {"name": "search_calendar"})


@dataclass
class ChatMessage:
    content: Any = None
    tool_calls: list[Any] | None = None
    role: str = "assistant"
    refusal: str | None = None


@dataclass
class Choice:
    finish_reason: str | None
    message: ChatMessage
    content_filter_results: dict[str, Any] | None = None


@dataclass
class ChatCompletion:
    choices: list[Choice]
    prompt_filter_results: list[dict[str, Any]] | None = None


@dataclass
class FakeCompletions:
    script: list[ChatCompletion]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> ChatCompletion:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.script.pop(0)


@dataclass
class FakeChat:
    completions: FakeCompletions


class FakeOpenAIClient:
    def __init__(self, script: list[ChatCompletion]) -> None:
        self.chat = FakeChat(completions=FakeCompletions(list(script)))


class FakeAzureOpenAIClient:
    """Mimics ``openai.AzureOpenAI`` configuration and response metadata."""

    def __init__(
        self,
        script: list[ChatCompletion],
        *,
        azure_endpoint: str = "https://example-resource.openai.azure.com/",
        api_version: str = "2025-03-01-preview",
    ) -> None:
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.chat = FakeChat(completions=FakeCompletions(list(script)))


def tc(id_: str, name: str, **args: Any) -> ToolCall:
    return ToolCall(id=id_, function=FunctionCall(name=name, arguments=json.dumps(args)))


EMAIL_ARGS = {"to": ["a@example.com"], "subject": "Hi", "body": "See you"}


def test_adapters_namespace_exposes_claude_and_openai_without_sdks() -> None:
    assert openai.run_tool_loop is run_tool_loop
    assert openai.tool_definition is tool_definition
    assert callable(claude.run_tool_loop)


def test_multi_turn_loop_with_parallel_mixed_outcomes(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    initial_messages = [{"role": "user", "content": "Check Monday, email Alice, delete mail."}]
    client = FakeOpenAIClient(
        [
            ChatCompletion(
                choices=[
                    Choice(
                        finish_reason="tool_calls",
                        message=ChatMessage(
                            content=None,
                            tool_calls=[
                                tc("call_1", "search_calendar", day="monday"),
                                tc("call_2", "send_email", **EMAIL_ARGS),
                                tc("call_3", "delete_mailbox"),
                            ],
                        ),
                    )
                ]
            ),
            ChatCompletion(
                choices=[
                    Choice(
                        finish_reason="stop",
                        message=ChatMessage(content="Done: the email awaits approval."),
                    )
                ]
            ),
        ]
    )

    result = run_tool_loop(
        client,
        guard,
        messages=initial_messages,
        tools=[tool_definition(email_policy, "send_email", "Send email")],
        system="You are an assistant.",
        temperature=0.0,
    )

    # Caller's list was not mutated.
    assert len(initial_messages) == 1
    assert result.stop_reason == "stop"
    assert result.turns == 2
    assert result.text == "Done: the email awaits approval."
    assert result.refusal is None
    assert [o.status for o in result.outcomes] == [
        Status.EXECUTED,
        Status.QUEUED,
        Status.BLOCKED,
    ]
    assert backend.names() == ["search_calendar"]

    sent = client.chat.completions.requests
    assert all(r["model"] == DEFAULT_MODEL == "gpt-4.1" for r in sent)
    assert all(r["temperature"] == 0.0 for r in sent)
    assert all("betas" not in r and "fallbacks" not in r for r in sent)
    assert sent[0]["messages"][0] == {"role": "system", "content": "You are an assistant."}

    # Turn 2 received assistant tool_calls followed by one role="tool" message per tool_call_id.
    turn2_msgs = sent[1]["messages"]
    assert turn2_msgs[2]["role"] == "assistant"
    assert [c["id"] for c in turn2_msgs[2]["tool_calls"]] == ["call_1", "call_2", "call_3"]
    tool_msgs = turn2_msgs[3:]
    assert [m["role"] for m in tool_msgs] == ["tool", "tool", "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_1", "call_2", "call_3"]
    assert all("is_error" not in m for m in tool_msgs)
    assert "OK: 'search_calendar' was executed" in tool_msgs[0]["content"]
    assert "QUEUED FOR HUMAN APPROVAL" in tool_msgs[1]["content"]
    assert "BLOCKED BY POLICY" in tool_msgs[2]["content"]


def test_parallel_duplicate_tool_calls_in_same_turn(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    client = FakeOpenAIClient(
        [
            ChatCompletion(
                choices=[
                    Choice(
                        finish_reason="tool_calls",
                        message=ChatMessage(
                            tool_calls=[
                                tc("c1", "send_email", **EMAIL_ARGS),
                                tc("c2", "send_email", **EMAIL_ARGS),
                            ]
                        ),
                    )
                ]
            ),
            ChatCompletion(choices=[Choice(finish_reason="stop", message=ChatMessage("Queued."))]),
        ]
    )

    result = run_tool_loop(client, guard, messages=[{"role": "user", "content": "Send"}])
    assert [o.status for o in result.outcomes] == [Status.QUEUED, Status.DUPLICATE]
    assert len(guard.pending()) == 1
    assert backend.calls == []
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    assert "NOT REPEATED" in tool_msgs[1]["content"]


def test_malformed_json_arguments_fail_closed_via_guard(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    bad_payloads: list[Any] = [
        '{"to": ["a@example.com"',  # truncated JSON
        "",  # empty string
        "   ",  # whitespace only
        "[1, 2, 3]",  # JSON array, not object
        "42",  # JSON number
        '"oops"',  # JSON string
        "null",  # JSON null must not coerce to {}
        '{"day": "mon", "day": "tue"}',  # duplicate keys
        '{"outer": {"k": 1, "k": 2}}',  # nested duplicate keys
        '{"day": NaN}',  # non-standard JSON constant
        '{"day": Infinity}',  # non-standard JSON constant
        '{"day": 1e500}',  # float overflow to inf
        '{"api_key": "sk-secret-do-not-log", "broken',  # secret inside broken JSON
        '[{"password": "hunter2-array-secret"}]',  # secret inside non-object JSON array
        '{"day": "mon\\ud800"}',  # lone surrogate inside JSON object
        None,  # non-string arguments
        123,  # non-string arguments
    ]
    calls = [
        ToolCall(id=f"bad_{i}", function=FunctionCall(name="search_calendar", arguments=payload))
        for i, payload in enumerate(bad_payloads)
    ]

    messages, outcomes = handle_tool_calls(guard, calls)
    assert len(outcomes) == len(bad_payloads)
    assert all(o.status is Status.BLOCKED for o in outcomes)
    assert all(o.reason is Reason.INVALID_ARGUMENTS for o in outcomes)
    assert all(
        m["role"] == "tool" and m["tool_call_id"] == f"bad_{i}" for i, m in enumerate(messages)
    )
    assert backend.calls == []

    audit_text = guard.audit.path.read_text(encoding="utf-8")
    assert "sk-secret-do-not-log" not in audit_text
    assert "hunter2-array-secret" not in audit_text
    assert "sk-secret-do-not-log" not in json.dumps(messages)
    assert "hunter2-array-secret" not in json.dumps(messages)
    verified = verify_log(guard.audit.path)
    assert verified.ok and verified.records == len(bad_payloads)


def test_malformed_json_respects_guard_check_order_for_blocked_tool_and_kill_switch(
    make_guard: MakeGuard, email_policy: Policy
) -> None:
    guard = make_guard(email_policy)
    blocked_msgs, blocked_outcomes = handle_tool_calls(
        guard,
        [ToolCall(id="c1", function=FunctionCall(name="delete_mailbox", arguments="{broken"))],
    )
    assert blocked_outcomes[0].status is Status.BLOCKED
    assert blocked_outcomes[0].reason is Reason.TOOL_BLOCKED
    assert blocked_msgs[0]["tool_call_id"] == "c1"

    guard.engage_kill_switch("incident", by="ops")
    ks_msgs, ks_outcomes = handle_tool_calls(
        guard,
        [ToolCall(id="c2", function=FunctionCall(name="search_calendar", arguments="{broken"))],
    )
    assert ks_outcomes[0].status is Status.BLOCKED
    assert ks_outcomes[0].reason is Reason.KILL_SWITCH
    assert ks_msgs[0]["tool_call_id"] == "c2"


def test_redacted_tool_result_messages(tmp_path: Path) -> None:
    policy = Policy.from_yaml(
        """
tools:
  fetch_account:
    mode: allow
    redact_fields: [iban]
  fail_tool:
    mode: allow
  draft_note:
    mode: draft
"""
    )
    guard = Guard(policy, home=tmp_path)
    guard.register(
        "fetch_account",
        lambda user: {
            "user": user,
            "api_key": "sk-live-secret-99",
            "nested": [{"password": "pw-secret", "iban": "GB29NWBK60161331926819"}],
        },
    )

    def fail_tool() -> None:
        raise RuntimeError("secret-token=abc123 must stay hidden")

    guard.register("fail_tool", fail_tool)

    msgs, outcomes = handle_tool_calls(
        guard,
        [
            tc("call_ok", "fetch_account", user="alice"),
            tc("call_err", "fail_tool"),
            tc("call_draft", "draft_note", text="hello"),
        ],
    )
    assert [o.status for o in outcomes] == [Status.EXECUTED, Status.FAILED, Status.DRAFTED]
    assert msgs[0]["role"] == "tool" and msgs[0]["tool_call_id"] == "call_ok"
    assert "sk-live-secret-99" not in msgs[0]["content"]
    assert "pw-secret" not in msgs[0]["content"]
    assert "GB29NWBK60161331926819" not in msgs[0]["content"]
    assert "[REDACTED]" in msgs[0]["content"]
    assert "alice" in msgs[0]["content"]

    assert msgs[1]["role"] == "tool" and msgs[1]["tool_call_id"] == "call_err"
    assert "RuntimeError" in msgs[1]["content"]
    assert "abc123" not in msgs[1]["content"]

    assert msgs[2]["role"] == "tool" and msgs[2]["tool_call_id"] == "call_draft"
    assert "DRAFT ONLY" in msgs[2]["content"]

    # Standalone tool_result_message also redacts DEFAULT_REDACT_FIELDS and escapes surrogates.
    standalone = tool_result_message("id\ud800", outcomes[0])
    assert standalone["role"] == "tool"
    assert standalone["tool_call_id"] == "id\\ud800"
    assert "sk-live-secret-99" not in standalone["content"]
    assert "pw-secret" not in standalone["content"]


def test_no_tools_run_on_refusal_content_filter_length_or_empty_choices(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)

    # 1. Model refusal (even if tool_calls were unexpectedly attached)
    refusal_client = FakeOpenAIClient(
        [
            ChatCompletion(
                choices=[
                    Choice(
                        finish_reason="stop",
                        message=ChatMessage(
                            content=None,
                            refusal="I cannot assist with that request.",
                            tool_calls=[tc("t1", "search_calendar", day="monday")],
                        ),
                    )
                ]
            )
        ]
    )
    res_refusal = run_tool_loop(refusal_client, guard, messages=[])
    assert res_refusal.stop_reason == "refusal"
    assert res_refusal.refusal == "I cannot assist with that request."
    assert res_refusal.text == ""
    assert res_refusal.outcomes == []
    assert backend.calls == []

    # 2. Content filter
    filtered_client = FakeOpenAIClient(
        [
            ChatCompletion(
                choices=[
                    Choice(
                        finish_reason="content_filter",
                        message=ChatMessage(tool_calls=[tc("t1", "search_calendar", day="monday")]),
                    )
                ]
            )
        ]
    )
    res_filter = run_tool_loop(filtered_client, guard, messages=[])
    assert res_filter.stop_reason == "content_filter"
    assert res_filter.outcomes == []
    assert backend.calls == []

    # 3. Truncated tool call (finish_reason == "length" with tool_calls)
    truncated_client = FakeOpenAIClient(
        [
            ChatCompletion(
                choices=[
                    Choice(
                        finish_reason="length",
                        message=ChatMessage(
                            tool_calls=[
                                ToolCall(
                                    id="t1",
                                    function=FunctionCall("search_calendar", '{"day": "mon'),
                                )
                            ]
                        ),
                    )
                ]
            )
        ]
    )
    res_len = run_tool_loop(truncated_client, guard, messages=[])
    assert res_len.stop_reason == "length"
    assert res_len.outcomes == []
    assert backend.calls == []

    # 4. Empty choices (e.g. Azure prompt filter response)
    empty_client = FakeOpenAIClient([ChatCompletion(choices=[])])
    res_empty = run_tool_loop(empty_client, guard, messages=[])
    assert res_empty.stop_reason == "empty_choices"
    assert res_empty.text == ""
    assert res_empty.outcomes == []
    assert backend.calls == []


def test_max_turns_caps_loop_and_rejects_invalid_max_turns(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    with pytest.raises(ConfigurationError, match="max_turns"):
        run_tool_loop(FakeOpenAIClient([]), guard, messages=[], max_turns=0)

    loop_turn = ChatCompletion(
        choices=[
            Choice(
                finish_reason="tool_calls",
                message=ChatMessage(tool_calls=[tc("t1", "search_calendar", day="mon")]),
            )
        ]
    )
    loop_turn_2 = ChatCompletion(
        choices=[
            Choice(
                finish_reason="tool_calls",
                message=ChatMessage(tool_calls=[tc("t2", "search_calendar", day="tue")]),
            )
        ]
    )
    loop_turn_3 = ChatCompletion(
        choices=[
            Choice(
                finish_reason="tool_calls",
                message=ChatMessage(tool_calls=[tc("t3", "search_calendar", day="wed")]),
            )
        ]
    )
    client = FakeOpenAIClient([loop_turn, loop_turn_2, loop_turn_3])
    result = run_tool_loop(client, guard, messages=[], max_turns=3)
    assert result.stop_reason == "max_turns"
    assert result.turns == 3
    assert len(result.outcomes) == 3
    assert len(backend.calls) == 3


def test_server_side_fallback_safe_behavior_and_stream_rejection(
    make_guard: MakeGuard, email_policy: Policy
) -> None:
    guard = make_guard(email_policy)

    # server_side_fallback=False is accepted and never forwarded to chat.completions.create.
    client = FakeOpenAIClient(
        [ChatCompletion(choices=[Choice(finish_reason="stop", message=ChatMessage(content="ok"))])]
    )
    res = run_tool_loop(
        client,
        guard,
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        server_side_fallback=False,
    )
    assert res.text == "ok"
    sent_req = client.chat.completions.requests[0]
    assert "server_side_fallback" not in sent_req
    assert "tools" not in sent_req  # empty tools omitted so OpenAI/Azure does not reject tools=[]

    # server_side_fallback=True fails closed with ConfigurationError instead of silent ignore.
    with pytest.raises(ConfigurationError, match="server_side_fallback"):
        run_tool_loop(client, guard, messages=[], server_side_fallback=True)

    # stream=True is rejected explicitly before calling the API.
    with pytest.raises(ConfigurationError, match="stream"):
        run_tool_loop(client, guard, messages=[], stream=True)


def test_azure_openai_client_shape_and_forced_tool_choice(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)
    azure_client = FakeAzureOpenAIClient(
        [
            ChatCompletion(
                prompt_filter_results=[{"prompt_index": 0, "content_filter_results": {}}],
                choices=[
                    Choice(
                        finish_reason="stop",  # happens with forced tool_choice on Azure/OpenAI
                        content_filter_results={"hate": {"filtered": False}},
                        message=ChatMessage(
                            content="",
                            tool_calls=[
                                tc("call_az_1", "create_event", title="Sync", when="10:00")
                            ],
                        ),
                    )
                ],
            ),
            ChatCompletion(
                choices=[
                    Choice(
                        finish_reason="stop",
                        message=ChatMessage(
                            content="Booked Sync.",
                            tool_calls=[],  # Azure SDK/proxies sometimes return [] instead of None
                        ),
                    )
                ]
            ),
        ]
    )

    result = run_tool_loop(
        azure_client,
        guard,
        model="gpt-41-prod-deployment",
        messages=[{"role": "user", "content": "Book Sync"}],
        tools=[tool_definition(email_policy, "create_event", "Create a calendar event")],
        tool_choice={"type": "function", "function": {"name": "create_event"}},
        extra_headers={"x-ms-region": "uksouth"},
    )

    assert result.stop_reason == "stop"
    assert result.turns == 2
    assert result.text == "Booked Sync."
    assert [o.status for o in result.outcomes] == [Status.EXECUTED]
    assert backend.calls == [("create_event", {"title": "Sync", "when": "10:00"})]

    reqs = azure_client.chat.completions.requests
    assert reqs[0]["model"] == "gpt-41-prod-deployment"
    assert reqs[0]["extra_headers"] == {"x-ms-region": "uksouth"}
    # Assistant history messages stay clean plain dicts without Azure response extras
    # or empty tool_calls lists.
    turn2_assistant = reqs[1]["messages"][1]
    assert set(turn2_assistant) == {"role", "content", "tool_calls"}
    assert result.messages[-1] == {"role": "assistant", "content": "Booked Sync."}


def test_tool_schema_export_from_policy() -> None:
    policy = Policy.from_yaml(
        """
tools:
  list_events:
    mode: allow
    description: List calendar events.
    args:
      day: {type: str, pattern: '^\\d{4}-\\d{2}-\\d{2}$'}
      kind: {type: str, choices: [work, personal], required: false, default: work}
  draft_reply:
    mode: draft
  delete_mailbox:
    mode: block
    args:
      confirm: {type: bool}
"""
    )

    defn = tool_definition(policy, "list_events")
    assert defn["type"] == "function"
    fn = defn["function"]
    assert fn["name"] == "list_events"
    assert fn["description"] == "List calendar events."
    assert "strict" not in fn
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["additionalProperties"] is False
    assert "title" not in params
    assert params["required"] == ["day"]
    assert params["properties"]["kind"]["anyOf"][0]["enum"] == ["work", "personal"]

    strict_defn = tool_definition(policy, "list_events", "List events", strict=True)
    assert strict_defn["function"]["strict"] is True

    with pytest.raises(ConfigurationError, match="no argument schema"):
        tool_definition(policy, "draft_reply", "Draft")

    with pytest.raises(ConfigurationError, match="no description"):
        tool_definition(policy, "delete_mailbox")

    exported = tools_from_policy(
        policy,
        descriptions={"draft_reply": "Draft reply"},
        parameters={
            "draft_reply": {"type": "object", "properties": {}, "additionalProperties": False}
        },
    )
    assert [t["function"]["name"] for t in exported] == ["list_events", "draft_reply"]

    exported_with_blocked = tools_from_policy(
        policy,
        descriptions={"draft_reply": "Draft reply", "delete_mailbox": "Delete mailbox"},
        parameters={"draft_reply": {"type": "object"}},
        include_blocked=True,
    )
    assert [t["function"]["name"] for t in exported_with_blocked] == [
        "list_events",
        "draft_reply",
        "delete_mailbox",
    ]


def test_dict_shaped_tool_calls_unsupported_types_and_multimodal_text(
    make_guard: MakeGuard, email_policy: Policy, backend: Backend
) -> None:
    guard = make_guard(email_policy)

    msgs, outcomes = handle_tool_calls(
        guard,
        [
            {
                "id": "dict_1",
                "type": "function",
                "function": {"name": "search_calendar", "arguments": '{"day": "friday"}'},
            },
            CustomToolCall(id="custom_1"),
        ],
    )
    assert [o.status for o in outcomes] == [Status.EXECUTED, Status.BLOCKED]
    assert outcomes[1].reason is Reason.INVALID_ARGUMENTS
    assert [m["tool_call_id"] for m in msgs] == ["dict_1", "custom_1"]
    assert backend.calls == [("search_calendar", {"day": "friday"})]

    multimodal_client = FakeOpenAIClient(
        [
            ChatCompletion(
                choices=[
                    Choice(
                        finish_reason="stop",
                        message=ChatMessage(
                            content=[
                                {"type": "text", "text": "Line 1"},
                                {"type": "text", "text": "Line 2"},
                            ]
                        ),
                    )
                ]
            )
        ]
    )
    res = run_tool_loop(multimodal_client, guard, messages=[])
    assert res.text == "Line 1\nLine 2"
