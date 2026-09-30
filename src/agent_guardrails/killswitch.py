"""Kill switch: one global stop for every guarded action.

Three independent ways to engage it, because the person who needs to stop an agent
may not have access to the same place as the person who deployed it:

- an environment variable (set at deploy time; cannot be cleared by the process).
  Any value engages it except an empty one or an explicit "off" (``0``, ``false``,
  ``no``, ``off``): an operator who types ``stop`` in an incident must not be ignored,
- a flag file (shared by the app, workers and the CLI),
- an in-process API call.

It is checked on every call, again immediately before a tool function runs, and
before any approved action is executed.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_ENV_VAR = "AGENT_GUARDRAILS_KILL"
_OFF = {"", "0", "false", "no", "off"}


@dataclass(frozen=True)
class KillSwitchStatus:
    engaged: bool
    source: str | None = None  # "env", "file" or "api"
    reason: str | None = None


class KillSwitch:
    def __init__(
        self, *, flag_file: str | Path | None = None, env_var: str | None = DEFAULT_ENV_VAR
    ) -> None:
        self.flag_file = Path(flag_file) if flag_file is not None else None
        self.env_var = env_var
        self._lock = threading.Lock()
        self._engaged_reason: str | None = None

    def status(self) -> KillSwitchStatus:
        if self.env_var:
            value = os.environ.get(self.env_var, "")
            if value.strip().lower() not in _OFF:  # fail closed on unrecognised values
                return KillSwitchStatus(True, "env", f"environment variable {self.env_var} is set")
        if self.flag_file is not None and self.flag_file.exists():
            return KillSwitchStatus(True, "file", self._read_reason(self.flag_file))
        with self._lock:
            if self._engaged_reason is not None:
                return KillSwitchStatus(True, "api", self._engaged_reason)
        return KillSwitchStatus(False)

    @property
    def engaged(self) -> bool:
        return self.status().engaged

    def engage(self, reason: str = "engaged", *, by: str | None = None) -> None:
        """Engage in this process and, if configured, write the flag file for everyone."""
        with self._lock:
            self._engaged_reason = reason
        if self.flag_file is not None:
            self.flag_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {"reason": reason, "by": by, "at": datetime.now(tz=UTC).isoformat()}
            self.flag_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    def release(self) -> KillSwitchStatus:
        """Clear the API flag and the flag file. An env var stays set; the status says so."""
        with self._lock:
            self._engaged_reason = None
        if self.flag_file is not None:
            self.flag_file.unlink(missing_ok=True)
        return self.status()

    @staticmethod
    def _read_reason(path: Path) -> str:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data.get("reason") or "flag file present")
        except (OSError, ValueError, AttributeError):
            # An empty or hand-made flag file still engages the switch: fail closed.
            return "flag file present"
