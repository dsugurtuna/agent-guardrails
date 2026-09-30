"""Result objects returned for every guarded call.

An :class:`Outcome` is designed to be handed back to the model as a tool result.
Its text says plainly what happened and what the model should do next, because a
model that is told "queued for approval, do not retry" behaves better than one
that sees a bare exception.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .errors import ActionBlockedError, ActionFailedError


class Status(StrEnum):
    EXECUTED = "executed"
    DRAFTED = "drafted"
    QUEUED = "queued"
    DUPLICATE = "duplicate"
    BLOCKED = "blocked"
    FAILED = "failed"


class Reason(StrEnum):
    """Machine-readable reason codes for blocked or failed outcomes."""

    KILL_SWITCH = "kill_switch"
    UNKNOWN_TOOL = "unknown_tool"
    TOOL_BLOCKED = "tool_blocked"
    NOT_REGISTERED = "not_registered"
    INVALID_ARGUMENTS = "invalid_arguments"
    RECIPIENT_NOT_ALLOWED = "recipient_not_allowed"
    TOO_MANY_RECIPIENTS = "too_many_recipients"
    RATE_LIMITED = "rate_limited"
    BUDGET_EXCEEDED = "budget_exceeded"
    QUEUE_FULL = "queue_full"
    NOT_APPROVED = "not_approved"
    APPROVAL_EXPIRED = "approval_expired"
    POLICY_CHANGED = "policy_changed"
    ARGUMENTS_CHANGED = "arguments_changed"
    TOOL_ERROR = "tool_error"


# Reasons that may clear on their own. An approved action blocked for one of these
# stays approved (until it expires) instead of being closed.
TRANSIENT_REASONS = frozenset({Reason.KILL_SWITCH, Reason.RATE_LIMITED, Reason.BUDGET_EXCEEDED})


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)


@dataclass(frozen=True)
class Outcome:
    """What the guard decided, and what happened."""

    status: Status
    tool: str
    message: str
    action_id: str | None = None
    reason: Reason | None = None
    result: Any = None
    preview: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def executed(self) -> bool:
        return self.status is Status.EXECUTED

    @property
    def is_error(self) -> bool:
        """True when the tool result should be flagged as an error to the model."""
        return self.status in (Status.BLOCKED, Status.FAILED)

    def as_tool_result(self) -> str:
        """Plain-text content suitable for a tool result sent back to a model."""
        ref = f" (action_id={self.action_id})" if self.action_id else ""
        if self.status is Status.EXECUTED:
            body = "" if self.result is None else f"\nResult: {_render(self.result)}"
            return f"OK: '{self.tool}' was executed{ref}.{body}"
        if self.status is Status.DRAFTED:
            return (
                f"DRAFT ONLY: '{self.tool}' is in draft mode, so nothing was sent or changed. "
                f"Show the draft to the user.\n{self.preview or ''}"
            ).rstrip()
        if self.status is Status.QUEUED:
            return (
                f"QUEUED FOR HUMAN APPROVAL: '{self.tool}' has NOT been performed yet{ref}. "
                "Tell the user it is awaiting approval. Do not retry it."
            )
        if self.status is Status.DUPLICATE:
            return f"NOT REPEATED: {self.message}{ref} Do not retry it."
        if self.status is Status.BLOCKED:
            return (
                f"BLOCKED BY POLICY ({self.reason}): {self.message} "
                "Do not retry the same call; explain the limit to the user instead."
            )
        return f"FAILED ({self.reason}): {self.message}"

    def raise_for_status(self) -> Outcome:
        """Return ``self`` unless blocked or failed, in which case raise a typed error."""
        if self.status is Status.BLOCKED:
            raise ActionBlockedError(self)
        if self.status is Status.FAILED:
            raise ActionFailedError(self)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "tool": self.tool,
            "message": self.message,
            "action_id": self.action_id,
            "reason": None if self.reason is None else str(self.reason),
            "preview": self.preview,
            "details": self.details,
        }
