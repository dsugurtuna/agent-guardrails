"""Stateless checks: argument schema, recipients and cost.

These run on every call, in every mode, and again at execution time for
approved actions. They need no database, so they are easy to test exhaustively.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from email.utils import getaddresses
from typing import Any

from pydantic import BaseModel, ValidationError

from ._canonical import canonical_hostname, canonical_json
from .outcomes import Reason
from .policy import Cost, RecipientRule, ToolPolicy


@dataclass(frozen=True)
class Violation:
    reason: Reason
    detail: str


@dataclass(frozen=True)
class ValidatedArgs:
    """Arguments after validation.

    ``stored`` is the JSON form that is checked, hashed, queued and audited. ``call``
    is what the tool function receives, rebuilt from ``stored`` (and required to
    match the original validation), so what was checked (and approved) is exactly
    what runs, whether it runs now or later from the queue.
    """

    stored: dict[str, Any]
    call: dict[str, Any]


def _summarise(exc: ValidationError, limit: int = 5) -> str:
    # Only locations and messages: pydantic's error dicts also carry the raw input,
    # which may be sensitive, so it is deliberately left out.
    parts = []
    for err in exc.errors()[:limit]:
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        parts.append(f"{loc}: {err['msg']}")
    more = len(exc.errors()) - limit
    if more > 0:
        parts.append(f"... and {more} more")
    return "; ".join(parts)


_NOT_JSON = "arguments must be JSON-serialisable, with finite numbers and valid Unicode text"
_NOT_ROUND_TRIP = (
    "the argument model does not read back its own JSON form (a field is excluded or "
    "serialised differently), so what is checked would not be what runs"
)


def _is_json_text(value: Any) -> bool:
    """Whether ``value`` serialises to canonical JSON that can be encoded as UTF-8."""
    try:
        canonical_json(value).encode("utf-8")
    except (TypeError, ValueError):  # UnicodeEncodeError is a ValueError
        return False
    return True


def validate_arguments(
    tool: ToolPolicy | None, args: Mapping[str, Any]
) -> tuple[ValidatedArgs | None, Violation | None]:
    model: type[BaseModel] | None = tool.schema_model if tool is not None else None
    if model is None:
        plain = dict(args)
        if not _is_json_text(plain):
            return None, Violation(Reason.INVALID_ARGUMENTS, _NOT_JSON)
        return ValidatedArgs(stored=plain, call=dict(plain)), None
    try:
        instance = model.model_validate(dict(args))
    except ValidationError as exc:
        return None, Violation(Reason.INVALID_ARGUMENTS, _summarise(exc))
    stored = instance.model_dump(mode="json")
    if not _is_json_text(stored):  # pydantic accepts NaN, infinity and lone surrogates
        return None, Violation(Reason.INVALID_ARGUMENTS, _NOT_JSON)
    # Why a round trip? A model can leave a field out of its dump (Field(exclude=True))
    # or serialise it differently (a field_serializer, SecretStr). The checks see the
    # dump; the tool must not receive anything the dump does not say.
    try:
        again = model.model_validate(stored)
    except ValidationError:
        return None, Violation(Reason.INVALID_ARGUMENTS, _NOT_ROUND_TRIP)
    if again != instance:
        return None, Violation(Reason.INVALID_ARGUMENTS, _NOT_ROUND_TRIP)
    call = {name: getattr(again, name) for name in type(again).model_fields}
    return ValidatedArgs(stored=stored, call=call), None


def domain_allowed(domain: str, allowed: list[str]) -> bool:
    """Whether ``domain`` matches the allow-list. Anything that is not a plain
    hostname (see :func:`canonical_hostname`) is refused, even under ``*.`` entries."""
    if "*" in allowed:
        return True
    host = canonical_hostname(domain)
    if host is None:
        return False
    for pattern in allowed:
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]):
                return True
        elif host == pattern:
            return True
    return False


def extract_addresses(value: Any) -> list[str]:
    """Parse one field's value into bare addresses. Raises ``ValueError`` if ambiguous.

    Why so strict? Address parsing is a classic source of bypasses (quoted local
    parts, display names containing another address, stray commas, control
    characters, domains that are not hostnames). Anything that does not parse to
    exactly one ``local@hostname`` per ``@`` is rejected: fail closed.
    """
    if value is None:
        return []
    raw_items = [value] if isinstance(value, str) else value
    if not isinstance(raw_items, list):
        raise ValueError("recipient field must be a string or a list of strings")
    addresses: list[str] = []
    for raw in raw_items:
        if not isinstance(raw, str):
            raise ValueError("recipient entries must be strings")
        if not raw.isprintable():  # NUL, CR/LF, zero-width and other invisible characters
            raise ValueError(f"recipient {raw!r} contains control or invisible characters")
        parsed = [addr for _name, addr in getaddresses([raw]) if addr]
        if raw.count("@") != len(parsed) or not parsed:
            raise ValueError(f"could not parse recipient {raw!r} unambiguously")
        for addr in parsed:
            local, sep, domain = addr.rpartition("@")
            if not sep or not local or any(c.isspace() for c in addr):
                raise ValueError(f"malformed address {addr!r}")
            if canonical_hostname(domain) is None:
                raise ValueError(f"malformed address {addr!r}: the domain is not a hostname")
            addresses.append(addr)
    return addresses


def check_recipients(rule: RecipientRule | None, args: Mapping[str, Any]) -> Violation | None:
    if rule is None:
        return None
    addresses: list[str] = []
    for field in rule.fields:
        try:
            addresses += extract_addresses(args.get(field))
        except ValueError as exc:
            return Violation(Reason.RECIPIENT_NOT_ALLOWED, str(exc))
    if rule.max_recipients is not None and len(addresses) > rule.max_recipients:
        return Violation(
            Reason.TOO_MANY_RECIPIENTS,
            f"{len(addresses)} recipients exceeds the limit of {rule.max_recipients}",
        )
    refused = sorted(
        {a for a in addresses if not domain_allowed(a.rpartition("@")[2], rule.allowed_domains)}
    )
    if refused:
        return Violation(
            Reason.RECIPIENT_NOT_ALLOWED,
            f"recipient domain not on the allow-list: {', '.join(refused)}",
        )
    return None


def compute_cost(cost: Cost | None, args: Mapping[str, Any]) -> tuple[float, Violation | None]:
    if cost is None:
        return 0.0, None
    if cost.fixed is not None:
        return cost.fixed, None
    assert cost.field is not None  # guaranteed by the Cost validator  # noqa: S101
    value = args.get(cost.field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0, Violation(
            Reason.INVALID_ARGUMENTS, f"cost field '{cost.field}' must be a number"
        )
    amount = float(value)
    # A negative or non-finite amount would *increase* the remaining budget.
    if not math.isfinite(amount) or amount < 0:
        return 0.0, Violation(
            Reason.INVALID_ARGUMENTS, f"cost field '{cost.field}' must be a finite number >= 0"
        )
    return amount, None
