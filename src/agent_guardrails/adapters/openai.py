"""Run guarded tools inside an OpenAI or Azure OpenAI Chat Completions loop.

Install with ``pip install "agent-guardrails[openai]"``. This module does not import
``openai`` itself: it talks to any client matching :class:`OpenAIClientLike`
(``openai.OpenAI``, ``openai.AzureOpenAI``, or an offline test double) via
``client.chat.completions.create``.

Every ``tool_calls`` entry emitted on an assistant message is strictly parsed as JSON
and routed through :meth:`Guard.call`. Malformed JSON, duplicate JSON keys, non-standard
constants (``NaN``/``Infinity``), and non-object JSON arguments fail closed through
:class:`~agent_guardrails.guard.Guard` so they are blocked and audited cleanly without
leaking raw malformed payloads. Each :class:`~agent_guardrails.outcomes.Outcome` is
returned as a redacted ``{"role": "tool", "tool_call_id": ..., "content": ...}``
message.

Why a small hand-written loop? Because the guard—not the model or SDK—decides whether
each tool function runs; because a turn that ends in ``content_filter``, a ``refusal``,
or ``length`` with truncated tool-call arguments must not run any tools at all; and
because the loop is short enough to read in full.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .._canonical import utf8_safe
from ..errors import ConfigurationError
from ..guard import Guard
from ..outcomes import Outcome, Status
from ..policy import DEFAULT_REDACT_FIELDS, Mode, Policy
from ..redaction import Redactor

DEFAULT_MODEL = "gpt-4.1"
"""Default OpenAI model name. For Azure OpenAI, pass your deployment name as ``model``."""

_MALFORMED_JSON_MARKER = "<malformed JSON arguments>"
_NULL_JSON_MARKER = "<JSON null arguments>"
_UNSUPPORTED_TOOL_CALL_MARKER = "<unsupported tool call>"


class FunctionCallLike(Protocol):
    """Function name and JSON-encoded arguments on an OpenAI tool call."""

    @property
    def name(self) -> str: ...

    @property
    def arguments(self) -> str: ...


class ToolCallLike(Protocol):
    """One tool call emitted on a Chat Completions assistant message."""

    @property
    def id(self) -> str: ...

    @property
    def type(self) -> str: ...

    @property
    def function(self) -> FunctionCallLike: ...


class ChatMessageLike(Protocol):
    """Assistant message returned inside a Chat Completions choice."""

    @property
    def content(self) -> Any: ...

    @property
    def tool_calls(self) -> Sequence[Any] | None: ...


class ChoiceLike(Protocol):
    """One choice in a ChatCompletion response."""

    @property
    def finish_reason(self) -> str | None: ...

    @property
    def message(self) -> ChatMessageLike: ...


class ChatCompletionLike(Protocol):
    """Minimal ChatCompletion response shape."""

    @property
    def choices(self) -> Sequence[ChoiceLike]: ...


class CompletionsClientLike(Protocol):
    """Structural interface for ``client.chat.completions``."""

    def create(self, **kwargs: Any) -> Any: ...


class ChatNamespaceLike(Protocol):
    """Structural interface for ``client.chat``."""

    @property
    def completions(self) -> CompletionsClientLike: ...


class OpenAIClientLike(Protocol):
    """Structural interface satisfied by ``openai.OpenAI`` and ``openai.AzureOpenAI``."""

    @property
    def chat(self) -> ChatNamespaceLike: ...


class _StrictJSONError(ValueError):
    """Raised on duplicate JSON keys or non-standard constants (NaN/Infinity)."""


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise _StrictJSONError(f"duplicate JSON key {key!r}")
        record[key] = value
    return record


def _no_constants(name: str) -> Any:
    raise _StrictJSONError(f"non-standard JSON constant {name}")


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_arguments(raw_arguments: Any) -> Any:
    """Parse OpenAI tool ``arguments`` strictly; return a non-Mapping marker on failure.

    Returning a non-Mapping marker (never ``None``, which ``Guard.call`` treats as ``{}``)
    ensures malformed JSON still flows through ``Guard.call`` for policy-order checks
    (kill switch, mode, audit logging) and fails closed with ``Reason.INVALID_ARGUMENTS``.
    """
    if not isinstance(raw_arguments, str):
        return _MALFORMED_JSON_MARKER
    try:
        parsed = json.loads(
            raw_arguments,
            object_pairs_hook=_unique_keys,
            parse_constant=_no_constants,
        )
    except (ValueError, RecursionError):
        return _MALFORMED_JSON_MARKER
    if parsed is None:
        return _NULL_JSON_MARKER
    return parsed


def tool_result_message(
    tool_call_id: str,
    outcome: Outcome,
    *,
    redactor: Redactor | None = None,
) -> dict[str, Any]:
    """Build an OpenAI ``role: "tool"`` message for one guarded :class:`Outcome`.

    Sensitive fields in structured tool outputs are redacted via ``redactor`` (defaulting
    to the policy's configured ``redact_fields`` when called via :func:`handle_tool_calls`,
    or :class:`~agent_guardrails.redaction.Redactor` with ``DEFAULT_REDACT_FIELDS`` when
    called standalone), and lone surrogates are escaped so serialisation never fails.
    """
    effective = outcome
    if outcome.status is Status.EXECUTED and isinstance(outcome.result, dict | list | tuple):
        active_redactor = redactor if redactor is not None else Redactor(DEFAULT_REDACT_FIELDS)
        effective = replace(outcome, result=active_redactor.redact(outcome.result))
    return {
        "role": "tool",
        "tool_call_id": utf8_safe(tool_call_id),
        "content": utf8_safe(effective.as_tool_result()),
    }


def tool_definition(
    policy: Policy,
    name: str,
    description: str | None = None,
    *,
    parameters: Mapping[str, Any] | None = None,
    strict: bool | None = None,
) -> dict[str, Any]:
    """An OpenAI / Azure OpenAI function tool definition backed by the policy's schema.

    One schema then serves both purposes: telling the model what to send, and
    validating what it actually sent.
    """
    tool = policy.tool(name)
    if parameters is None:
        model = tool.schema_model if tool is not None else None
        if model is None:
            raise ConfigurationError(
                f"tool '{name}' has no argument schema in the policy; pass parameters"
            )
        schema = model.model_json_schema()
        schema.pop("title", None)
    else:
        schema = dict(parameters)

    resolved_description = description
    if resolved_description is None and tool is not None:
        resolved_description = tool.description
    if resolved_description is None:
        raise ConfigurationError(
            f"tool '{name}' has no description in the policy; pass description"
        )

    function: dict[str, Any] = {
        "name": name,
        "description": resolved_description,
        "parameters": schema,
    }
    if strict is not None:
        function["strict"] = strict
    return {"type": "function", "function": function}


def tools_from_policy(
    policy: Policy,
    *,
    descriptions: Mapping[str, str] | None = None,
    parameters: Mapping[str, Mapping[str, Any]] | None = None,
    include_blocked: bool = False,
    strict: bool | None = None,
) -> list[dict[str, Any]]:
    """Export OpenAI function tool definitions for all active tools in ``policy``.

    Tools in ``block`` mode are excluded by default so they are not advertised to the
    model.
    """
    desc_map = descriptions or {}
    param_map = parameters or {}
    definitions: list[dict[str, Any]] = []
    for name, tool in policy.tools.items():
        if not include_blocked and tool.mode is Mode.BLOCK:
            continue
        definitions.append(
            tool_definition(
                policy,
                name,
                desc_map.get(name),
                parameters=param_map.get(name),
                strict=strict,
            )
        )
    return definitions


def handle_tool_calls(
    guard: Guard,
    tool_calls: Iterable[Any],
    *,
    agent_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[Outcome]]:
    """Run every tool call in ``tool_calls`` through ``guard``, in order.

    Returns the ``role: "tool"`` messages (append each to the conversation history) and
    the corresponding :class:`~agent_guardrails.outcomes.Outcome` objects.
    """
    messages: list[dict[str, Any]] = []
    outcomes: list[Outcome] = []
    for tool_call in tool_calls:
        raw_id = _field(tool_call, "id", "")
        call_id = raw_id if isinstance(raw_id, str) else str(raw_id or "")
        call_type = _field(tool_call, "type", "function")
        fn = _field(tool_call, "function")
        tool_name: Any
        tool_args: Any
        if call_type != "function" or fn is None:
            raw_name = (
                _field(fn, "name", "")
                if fn is not None
                else _field(_field(tool_call, "custom"), "name", "")
            )
            tool_name = raw_name if isinstance(raw_name, str) else ""
            tool_args = _UNSUPPORTED_TOOL_CALL_MARKER
        else:
            tool_name = _field(fn, "name")
            tool_args = _parse_arguments(_field(fn, "arguments"))
        outcome = guard.call(tool_name, tool_args, agent_id=agent_id)

        policy = guard.policy
        tool_policy = policy.tool(outcome.tool) if outcome.tool else None
        redactor = Redactor(policy.redact_fields)
        if tool_policy is not None and tool_policy.redact_fields:
            redactor = redactor.with_fields(tool_policy.redact_fields)

        outcomes.append(outcome)
        messages.append(tool_result_message(call_id, outcome, redactor=redactor))
    return messages, outcomes


def _first_message(completion: Any) -> Any:
    if completion is None:
        return None
    choices = list(_field(completion, "choices") or ())
    if not choices:
        return None
    return _field(choices[0], "message")


@dataclass
class ToolLoopResult:
    """Where the loop stopped, the full message history and every guarded outcome."""

    final_message: Any
    messages: list[dict[str, Any]]
    outcomes: list[Outcome] = field(default_factory=list)
    stop_reason: str | None = None
    turns: int = 0

    @property
    def text(self) -> str:
        """The text content of the final choice message, or ``""`` if absent."""
        message = _first_message(self.final_message)
        if message is None:
            return ""
        content = _field(message, "content")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list | tuple):
            parts = [
                str(_field(part, "text", "")) for part in content if _field(part, "type") == "text"
            ]
            return "\n".join(parts)
        return str(content)

    @property
    def refusal(self) -> str | None:
        """The refusal string on the final choice message, if present."""
        message = _first_message(self.final_message)
        if message is None:
            return None
        value = _field(message, "refusal")
        return value if isinstance(value, str) and value else None


def _serialize_assistant_message(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
    role = _field(message, "role", "assistant") or "assistant"
    content = _field(message, "content")
    if isinstance(content, str):
        content = utf8_safe(content)
    msg: dict[str, Any] = {
        "role": str(role),
        "content": content,
    }
    refusal = _field(message, "refusal")
    if isinstance(refusal, str) and refusal:
        msg["refusal"] = utf8_safe(refusal)
    if tool_calls:
        serialized: list[dict[str, Any]] = []
        for tc in tool_calls:
            raw_id = _field(tc, "id", "")
            raw_type = _field(tc, "type", "function") or "function"
            entry: dict[str, Any] = {
                "id": utf8_safe(raw_id) if isinstance(raw_id, str) else str(raw_id or ""),
                "type": str(raw_type),
            }
            fn = _field(tc, "function")
            if fn is not None:
                raw_name = _field(fn, "name", "")
                raw_args = _field(fn, "arguments", "")
                if isinstance(raw_args, str):
                    args_str = utf8_safe(raw_args)
                elif isinstance(raw_args, Mapping):
                    args_str = json.dumps(dict(raw_args))
                else:
                    args_str = str(raw_args or "")
                entry["function"] = {
                    "name": (
                        utf8_safe(raw_name) if isinstance(raw_name, str) else str(raw_name or "")
                    ),
                    "arguments": args_str,
                }
            serialized.append(entry)
        msg["tool_calls"] = serialized
    return msg


def run_tool_loop(
    client: OpenAIClientLike,
    guard: Guard,
    *,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] = (),
    model: str = DEFAULT_MODEL,
    max_turns: int = 10,
    system: str | None = None,
    server_side_fallback: bool = False,
    agent_id: str | None = None,
    **create_kwargs: Any,
) -> ToolLoopResult:
    """Call OpenAI or Azure OpenAI, route tool calls through ``guard``, and loop until done.

    ``client`` is an ``openai.OpenAI`` or ``openai.AzureOpenAI`` instance (or compatible
    test double exposing ``client.chat.completions.create``). For Azure OpenAI, pass your
    deployment name as ``model``. Extra keyword arguments (``temperature``,
    ``tool_choice``, ``parallel_tool_calls``, ``max_completion_tokens``, ...) are passed
    through unchanged. ``server_side_fallback`` defaults to ``False``; passing ``True``
    raises :class:`~agent_guardrails.errors.ConfigurationError` because OpenAI Chat
    Completions has no Anthropic-style server-side refusal fallback endpoint.

    The loop stops *without running that turn's tools* when ``choices`` is empty, when
    ``finish_reason`` is ``"content_filter"``, when the message carries a ``refusal``, or
    when ``finish_reason`` is ``"length"`` while ``tool_calls`` are present (arguments may
    be truncated). It also stops after ``max_turns`` requests with ``stop_reason`` set to
    ``"max_turns"``.
    """
    if max_turns < 1:
        raise ConfigurationError("max_turns must be at least 1")
    if server_side_fallback:
        raise ConfigurationError(
            "server_side_fallback is not supported by the OpenAI adapter; "
            "omit it or pass server_side_fallback=False"
        )
    if create_kwargs.get("stream"):
        raise ConfigurationError(
            "streaming responses are not supported by run_tool_loop; pass stream=False"
        )

    history: list[dict[str, Any]] = [dict(m) for m in messages]
    if system is not None:
        history.insert(0, {"role": "system", "content": system})

    outcomes: list[Outcome] = []
    response: Any = None
    params: dict[str, Any] = {"model": model, **create_kwargs}
    if tools:
        params["tools"] = [dict(t) for t in tools]

    for turn in range(1, max_turns + 1):
        response = client.chat.completions.create(**params, messages=list(history))
        choices = list(_field(response, "choices") or ())
        if not choices:
            return ToolLoopResult(response, history, outcomes, "empty_choices", turn)

        choice = choices[0]
        finish_reason = _field(choice, "finish_reason")
        message = _field(choice, "message")
        if message is None:
            return ToolLoopResult(response, history, outcomes, finish_reason, turn)

        tool_calls = list(_field(message, "tool_calls") or ())
        refusal = _field(message, "refusal")

        if finish_reason == "content_filter":
            return ToolLoopResult(response, history, outcomes, "content_filter", turn)
        if isinstance(refusal, str) and refusal:
            return ToolLoopResult(response, history, outcomes, "refusal", turn)
        if finish_reason == "length" and tool_calls:
            return ToolLoopResult(response, history, outcomes, "length", turn)
        if finish_reason not in ("tool_calls", "stop", None) and tool_calls:
            return ToolLoopResult(response, history, outcomes, str(finish_reason), turn)

        history.append(_serialize_assistant_message(message, tool_calls))
        if not tool_calls:
            return ToolLoopResult(response, history, outcomes, finish_reason, turn)

        tool_messages, turn_outcomes = handle_tool_calls(guard, tool_calls, agent_id=agent_id)
        outcomes.extend(turn_outcomes)
        history.extend(tool_messages)

    return ToolLoopResult(response, history, outcomes, "max_turns", max_turns)


__all__ = [
    "DEFAULT_MODEL",
    "ChatCompletionLike",
    "ChatMessageLike",
    "ChatNamespaceLike",
    "ChoiceLike",
    "CompletionsClientLike",
    "FunctionCallLike",
    "OpenAIClientLike",
    "ToolCallLike",
    "ToolLoopResult",
    "handle_tool_calls",
    "run_tool_loop",
    "tool_definition",
    "tool_result_message",
    "tools_from_policy",
]
