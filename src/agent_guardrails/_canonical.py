"""Canonical JSON and hashing helpers.

Every hash in this package is computed over *canonical* JSON: sorted keys, no
insignificant whitespace, UTF-8. Why? Because two dicts that mean the same thing
must hash the same, whatever order the model happened to emit the keys in.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

_LDH_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` deterministically. Raises ``TypeError`` for non-JSON values."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(obj: Any) -> str:
    """SHA-256 of the canonical JSON form of ``obj``."""
    return sha256_hex(canonical_json(obj))


def canonical_hostname(value: str) -> str | None:
    """Lower-case ASCII hostname without its trailing dot, or ``None`` if not a hostname.

    Why so strict? A domain is compared as text here but used by software that may
    read it differently: a NUL byte ends a C string; ``#``, ``/`` or ``?`` end the host
    part of a URL; and IDNA maps Unicode in ways ``str.lower()`` does not (the Kelvin
    sign lower-cases to ASCII ``k``). Only letters, digits and hyphens in non-empty
    labels are accepted, so the name that is checked is the name that gets resolved.
    Internationalised domains must be written in their ASCII (``xn--``) form.
    """
    if not value.isascii():  # before lower(): some non-ASCII letters lower-case to ASCII
        return None
    host = value.lower()
    if host.endswith("."):
        host = host[:-1]  # one trailing dot is the fully qualified form of the same name
    if not host or len(host) > 253:
        return None
    if not all(_LDH_LABEL.fullmatch(label) for label in host.split(".")):
        return None
    return host


def _normalise_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def normalise(obj: Any) -> Any:
    """Normalise a JSON value for duplicate detection.

    Strings are NFC-normalised and stripped of surrounding whitespace, so a retry
    that differs only by a trailing space or a different Unicode composition of
    the same character is still recognised as the same action. Paraphrases are
    *not* recognised; that is a documented limit.
    """
    if isinstance(obj, str):
        return _normalise_text(obj)
    if isinstance(obj, dict):
        return {str(k): normalise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalise(v) for v in obj]
    return obj


def dedupe_key(tool: str, args: dict[str, Any], recipient_fields: Iterable[str] = ()) -> str:
    """Key used to recognise "the same action" within the de-duplication window.

    Recipient fields are case-folded and sorted, because ``[A@x.com, b@x.com]`` and
    ``[b@x.com, a@x.com]`` deliver the same message to the same people.
    """
    norm: dict[str, Any] = normalise(args)
    for field in recipient_fields:
        value = norm.get(field)
        if isinstance(value, str):
            norm[field] = value.casefold()
        elif isinstance(value, list):
            norm[field] = sorted(str(v).casefold() for v in value)
    return digest({"tool": tool, "args": norm})
