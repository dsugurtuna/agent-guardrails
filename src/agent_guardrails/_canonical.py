"""Canonical JSON and hashing helpers.

Every hash in this package is computed over *canonical* JSON: sorted keys, no
insignificant whitespace, UTF-8. Why? Because two dicts that mean the same thing
must hash the same, whatever order the model happened to emit the keys in.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable
from typing import Any


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
