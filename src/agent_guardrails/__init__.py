"""agent-guardrails: controls for AI agents that take actions with side effects.

Read freely, write carefully.
"""

from .audit import AuditLog, VerificationResult, verify_log
from .errors import (
    ActionBlockedError,
    ActionFailedError,
    ActionNotFoundError,
    ApprovalExpiredError,
    AuditIntegrityError,
    ConfigurationError,
    GuardrailError,
    InvalidTransitionError,
    PolicyError,
)
from .guard import (
    Guard,
    GuardedTool,
    default_home,
    get_default_guard,
    guarded,
    set_default_guard,
)
from .killswitch import KillSwitch, KillSwitchStatus
from .outcomes import Outcome, Reason, Status
from .policy import ArgSpec, Budget, Cost, Mode, Policy, RateLimit, RecipientRule, ToolPolicy
from .redaction import REDACTED, Redactor
from .store import ActionRecord, ActionState, ActionStore

__version__ = "0.1.0"

__all__ = [
    "REDACTED",
    "ActionBlockedError",
    "ActionFailedError",
    "ActionNotFoundError",
    "ActionRecord",
    "ActionState",
    "ActionStore",
    "ApprovalExpiredError",
    "ArgSpec",
    "AuditIntegrityError",
    "AuditLog",
    "Budget",
    "ConfigurationError",
    "Cost",
    "Guard",
    "GuardedTool",
    "GuardrailError",
    "InvalidTransitionError",
    "KillSwitch",
    "KillSwitchStatus",
    "Mode",
    "Outcome",
    "Policy",
    "PolicyError",
    "RateLimit",
    "Reason",
    "RecipientRule",
    "Redactor",
    "Status",
    "ToolPolicy",
    "VerificationResult",
    "__version__",
    "default_home",
    "get_default_guard",
    "guarded",
    "set_default_guard",
    "verify_log",
]
