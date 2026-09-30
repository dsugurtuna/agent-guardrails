"""Redaction of argument values before they are written to the audit log.

Why a plain ``[REDACTED]`` marker rather than a hash of the value? Because hashes
of short or guessable secrets (a PIN, a password) can be brute-forced, which would
turn the audit log into a leak.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

REDACTED = "[REDACTED]"

RedactHook = Callable[[str, Any], Any]
"""Called as ``hook(key, value)`` for every key; return the value to log instead."""


def _texts(value: Any, out: set[str]) -> None:
    """Add the text of every leaf of ``value`` to ``out`` (as a message might quote it)."""
    if isinstance(value, dict):
        for item in value.values():
            _texts(item, out)
    elif isinstance(value, list | tuple):
        for item in value:
            _texts(item, out)
    elif isinstance(value, str):
        if value:
            out.update({value, repr(value)[1:-1]})  # as written by f"{value!r}" too
    elif isinstance(value, int | float) and not isinstance(value, bool):
        out.add(str(value))


class Redactor:
    def __init__(self, fields: Iterable[str] = (), hook: RedactHook | None = None) -> None:
        self._fields = {f.casefold() for f in fields}
        self._hook = hook

    def with_fields(self, extra: Iterable[str]) -> Redactor:
        return Redactor(self._fields | {f.casefold() for f in extra}, self._hook)

    def redact(self, value: Any, hidden: set[str] | None = None) -> Any:
        """Return a redacted deep copy of ``value`` (dicts and lists are walked).

        With ``hidden``, the text of every value that was hidden (by field name or by
        the hook) is added to it, so it can also be removed from free text such as a
        blocked call's detail message (see :func:`scrub`).
        """
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                skey = str(key)
                if skey.casefold() in self._fields:
                    out[skey] = REDACTED
                    if hidden is not None:
                        _texts(item, hidden)
                    continue
                redacted = self.redact(item, hidden)
                if self._hook is not None:
                    hooked = self._hook(skey, redacted)
                    if hidden is not None and hooked != redacted:
                        _texts(item, hidden)
                    redacted = hooked
                out[skey] = redacted
            return out
        if isinstance(value, list | tuple):
            return [self.redact(v, hidden) for v in value]
        return value


def scrub(text: str, hidden: Iterable[str]) -> str:
    """Replace every occurrence of a hidden value in ``text`` with ``[REDACTED]``.

    Why? Messages can quote arguments ("recipient domain not on the allow-list:
    a@b.test"), and a redacted argument must not reach the log that way either.
    One pass, longest value first, so a replacement is never matched again.
    """
    values = sorted({h for h in hidden if h}, key=len, reverse=True)
    if not values:
        return text
    return re.sub("|".join(re.escape(v) for v in values), REDACTED, text)
