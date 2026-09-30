"""Typed exceptions.

Policy decisions (blocked, queued, drafted...) are *not* exceptions: they are
returned as :class:`~agent_guardrails.outcomes.Outcome` objects so an agent loop
can hand them straight back to the model. Exceptions are reserved for
programming and operator errors, plus :meth:`Outcome.raise_for_status` for
callers who prefer exceptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .outcomes import Outcome


class GuardrailError(Exception):
    """Base class for every error raised by agent-guardrails."""


class PolicyError(GuardrailError, ValueError):
    """The policy (Python or YAML) is invalid or inconsistent."""


class ConfigurationError(GuardrailError):
    """The guard is wired up incorrectly (for example, no default guard is set)."""


class ActionNotFoundError(GuardrailError, LookupError):
    """No queued action has this id."""


class InvalidTransitionError(GuardrailError):
    """The requested state change is not allowed (for example, approving a rejected action)."""


class ApprovalExpiredError(InvalidTransitionError):
    """The action's approval window has passed; it can no longer be approved or executed."""


class AuditIntegrityError(GuardrailError):
    """The audit log is corrupt, so a new record cannot be chained onto it safely."""


class ActionBlockedError(GuardrailError):
    """Raised by :meth:`Outcome.raise_for_status` for a blocked action."""

    def __init__(self, outcome: Outcome) -> None:
        super().__init__(outcome.message)
        self.outcome = outcome


class ActionFailedError(GuardrailError):
    """Raised by :meth:`Outcome.raise_for_status` when the tool itself raised."""

    def __init__(self, outcome: Outcome) -> None:
        super().__init__(outcome.message)
        self.outcome = outcome
