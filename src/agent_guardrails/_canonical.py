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
from email.utils import getaddresses
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
    the same character is still recognised as the same action. Whole-number floats
    become integers, because JSON does not distinguish ``10`` from ``10.0`` and a
    model may emit either. Paraphrases are *not* recognised; that is a documented
    limit.
    """
    if isinstance(obj, str):
        return _normalise_text(obj)
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)  # bool is not a float, so True stays distinct from 1
    if isinstance(obj, dict):
        return {str(k): normalise(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [normalise(v) for v in obj]
    return obj


def _address_set(value: Any) -> Any:
    """The set of bare, case-folded addresses a recipient field names, sorted."""
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list):
        return value
    found: set[str] = set()
    for item in items:
        text = str(item)
        parsed = [addr for _name, addr in getaddresses([text]) if addr]
        found.update(addr.casefold() for addr in (parsed or [text]))
    return sorted(found)


def dedupe_key(tool: str, args: dict[str, Any], recipient_fields: Iterable[str] = ()) -> str:
    """Key used to recognise "the same action" within the de-duplication window.

    Recipient fields are reduced to the sorted set of case-folded bare addresses,
    because ``[A@x.com, b@x.com]``, ``[b@x.com, a@x.com]``, ``"a@x.com, b@x.com"``
    and ``["Ann <a@x.com>", "b@x.com", "b@x.com"]`` deliver the same message to the
    same people. (The recipient check has already refused anything ambiguous.)
    """
    norm: dict[str, Any] = normalise(args)
    for field in recipient_fields:
        if norm.get(field) is not None:
            norm[field] = _address_set(norm[field])
    return digest({"tool": tool, "args": norm})
