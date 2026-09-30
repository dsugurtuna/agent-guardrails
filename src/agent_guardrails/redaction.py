"""Redaction of argument values before they are written to the audit log.

Why a plain ``[REDACTED]`` marker rather than a hash of the value? Because hashes
of short or guessable secrets (a PIN, a password) can be brute-forced, which would
turn the audit log into a leak.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

REDACTED = "[REDACTED]"

RedactHook = Callable[[str, Any], Any]
"""Called as ``hook(key, value)`` for every key; return the value to log instead."""


class Redactor:
    def __init__(self, fields: Iterable[str] = (), hook: RedactHook | None = None) -> None:
        self._fields = {f.casefold() for f in fields}
        self._hook = hook

    def with_fields(self, extra: Iterable[str]) -> Redactor:
        return Redactor(self._fields | {f.casefold() for f in extra}, self._hook)

    def redact(self, value: Any) -> Any:
        """Return a redacted deep copy of ``value`` (dicts and lists are walked)."""
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                skey = str(key)
                if skey.casefold() in self._fields:
                    out[skey] = REDACTED
                    continue
                redacted = self.redact(item)
                if self._hook is not None:
                    redacted = self._hook(skey, redacted)
                out[skey] = redacted
            return out
        if isinstance(value, list | tuple):
            return [self.redact(v) for v in value]
        return value
