"""Run guarded tools inside a Claude tool-use loop (Anthropic Python SDK).

Install with ``pip install "agent-guardrails[anthropic]"``. This module does not
import ``anthropic`` itself: it calls the client you pass in, so it also works with
test doubles.

Every ``tool_use`` block Claude emits goes through :meth:`Guard.call`. The
:class:`~agent_guardrails.outcomes.Outcome` becomes the ``tool_result``: executed
results come back as normal content; drafts, queued approvals and duplicates come
back as plain explanations; blocked and failed calls are flagged ``is_error`` so
the model knows the action did not happen.

Why a small hand-written loop rather than the SDK's beta tool runner? Because the
guard, not the SDK, must decide whether each tool function runs; because a turn
that ended in ``refusal``, or hit ``max_tokens`` part-way through a tool call, must
not run any tools at all; and because the loop is short enough to read in full.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..errors import ConfigurationError
from ..guard import Guard
from ..outcomes import Outcome, Reason, Status
from ..policy import Policy

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
"""Beta header for server-side refusal fallbacks with ``fallbacks="default"``."""


def tool_result_block(tool_use_id: str, outcome: Outcome) -> dict[str, Any]:
    """A ``tool_result`` content block for one guarded call."""
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": outcome.as_tool_result(),
    }
    if outcome.is_error:
        block["is_error"] = True
    return block


def tool_definition(
    policy: Policy,
    name: str,
    description: str,
    *,
    input_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A Claude tool definition whose ``input_schema`` comes from the policy's schema.

    One schema then serves both purposes: telling the model what to send, and
    validating what it actually sent.
    """
    if input_schema is None:
        tool = policy.tool(name)
        model = tool.schema_model if tool is not None else None
        if model is None:
            raise ConfigurationError(
                f"tool '{name}' has no argument schema in the policy; pass input_schema"
            )
        schema = model.model_json_schema()
        schema.pop("title", None)
    else:
        schema = dict(input_schema)
    return {"name": name, "description": description, "input_schema": schema}


def _blocks_of_type(content: Iterable[Any], kind: str) -> list[Any]:
    return [b for b in content if getattr(b, "type", None) == kind]


def handle_tool_uses(
    guard: Guard, content: Iterable[Any], *, agent_id: str | None = None
) -> tuple[list[dict[str, Any]], list[Outcome]]:
    """Run every ``tool_use`` block in ``content`` through the guard, in order.

    Returns the ``tool_result`` blocks (send them back together, in one user
    message) and the outcomes.
    """
    results: list[dict[str, Any]] = []
    outcomes: list[Outcome] = []
    for block in _blocks_of_type(content, "tool_use"):
        tool_input = getattr(block, "input", None)
        if isinstance(tool_input, Mapping):
            outcome = guard.call(block.name, tool_input, agent_id=agent_id)
        else:
            outcome = Outcome(
                Status.BLOCKED,
                str(block.name),
                "tool input was not a JSON object.",
                reason=Reason.INVALID_ARGUMENTS,
            )
        outcomes.append(outcome)
        results.append(tool_result_block(block.id, outcome))
    return results, outcomes


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
        """The text blocks of the final message, joined."""
        if self.final_message is None:
            return ""
        texts = _blocks_of_type(getattr(self.final_message, "content", []), "text")
        return "\n".join(str(b.text) for b in texts)


def run_tool_loop(
    client: Any,
    guard: Guard,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 16000,
    max_turns: int = 10,
    server_side_fallback: bool = True,
    agent_id: str | None = None,
    **create_kwargs: Any,
) -> ToolLoopResult:
    """Call Claude, route its tool calls through ``guard``, and repeat until it is done.

    ``client`` is an ``anthropic.Anthropic`` instance. With ``server_side_fallback``
    (the default) requests go through ``client.beta.messages.create`` with
    ``fallbacks="default"``, so a request the model declines is retried server-side
    on Anthropic's recommended fallback model; set it to ``False`` to use
    ``client.messages.create`` instead. Extra keyword arguments (``system``,
    ``output_config``, ...) are passed through unchanged.

    The loop stops, without running tools, when:

    - ``stop_reason`` is ``"refusal"``;
    - ``stop_reason`` is ``"max_tokens"`` and the response contains a tool call
      (its input may be truncated; retry with a larger ``max_tokens``);
    - ``max_turns`` requests have been made (``stop_reason`` is then ``"max_turns"``).
    """
    history = list(messages)
    outcomes: list[Outcome] = []
    response: Any = None
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "tools": tools,
        **create_kwargs,
    }
    for turn in range(1, max_turns + 1):
        response = _create(client, params, history, server_side_fallback)
        stop = response.stop_reason
        tool_uses = _blocks_of_type(response.content, "tool_use")
        if stop == "refusal" or (stop == "max_tokens" and tool_uses):
            return ToolLoopResult(response, history, outcomes, stop, turn)
        history.append({"role": "assistant", "content": response.content})
        if stop == "pause_turn":
            continue  # a server-side tool paused; re-send so the server resumes
        if stop != "tool_use" or not tool_uses:
            return ToolLoopResult(response, history, outcomes, stop, turn)
        results, turn_outcomes = handle_tool_uses(guard, response.content, agent_id=agent_id)
        outcomes += turn_outcomes
        history.append({"role": "user", "content": results})
    return ToolLoopResult(response, history, outcomes, "max_turns", max_turns)


def _create(
    client: Any, params: dict[str, Any], history: list[dict[str, Any]], fallback: bool
) -> Any:
    kwargs = {**params, "messages": history}
    if fallback:
        betas = list(kwargs.pop("betas", []))
        if FALLBACK_BETA not in betas:
            betas.append(FALLBACK_BETA)
        return client.beta.messages.create(**kwargs, betas=betas, fallbacks="default")
    return client.messages.create(**kwargs)
