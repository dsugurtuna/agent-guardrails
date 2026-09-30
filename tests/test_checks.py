from __future__ import annotations

import math

import pytest
from pydantic import BaseModel, ConfigDict, Field, field_serializer

from agent_guardrails import Cost, Mode, Policy, Reason, RecipientRule, ToolPolicy
from agent_guardrails.checks import (
    check_recipients,
    compute_cost,
    domain_allowed,
    extract_addresses,
    validate_arguments,
)

RULE = RecipientRule(
    fields=["to", "cc"], allowed_domains=["example.com", "*.example.org"], max_recipients=3
)


@pytest.mark.parametrize(
    ("domain", "allowed"),
    [
        ("example.com", True),
        ("EXAMPLE.com", True),
        ("example.com.", True),
        ("mail.example.com", False),  # exact entry does not cover subdomains
        ("evil-example.com", False),
        ("example.com.evil.net", False),
        ("team.example.org", True),
        ("a.b.example.org", True),
        ("example.org", False),  # "*." means subdomains only
        ("notexample.org", False),
    ],
)
def test_domain_matching(domain: str, allowed: bool) -> None:
    assert domain_allowed(domain, ["example.com", "*.example.org"]) is allowed


def test_star_allows_any_domain() -> None:
    assert domain_allowed("anything.test", ["*"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a@example.com", ["a@example.com"]),
        ("Alice <a@example.com>", ["a@example.com"]),
        (["a@example.com", "Bob <b@example.com>"], ["a@example.com", "b@example.com"]),
        ("a@example.com, b@example.com", ["a@example.com", "b@example.com"]),
        (None, []),
    ],
)
def test_extract_addresses(value: object, expected: list[str]) -> None:
    assert extract_addresses(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        '"a@evil.com" <b@example.com>',  # display name hides a second address
        "a@example.com@evil.com",
        "no-at-sign",
        "",
        ["a@example.com", 7],
        {"to": "a@example.com"},
    ],
)
def test_ambiguous_addresses_are_rejected(value: object) -> None:
    with pytest.raises(ValueError, match=r"recipient|parse|malformed"):
        extract_addresses(value)


def test_recipients_allowed() -> None:
    assert check_recipients(RULE, {"to": ["a@example.com"], "cc": "b@team.example.org"}) is None


def test_recipient_outside_allow_list_is_named() -> None:
    v = check_recipients(RULE, {"to": ["a@example.com", "x@evil.test"]})
    assert v is not None
    assert v.reason is Reason.RECIPIENT_NOT_ALLOWED
    assert "x@evil.test" in v.detail


def test_too_many_recipients_counts_all_fields() -> None:
    v = check_recipients(
        RULE, {"to": ["a@example.com", "b@example.com"], "cc": "c@example.com, d@example.com"}
    )
    assert v is not None
    assert v.reason is Reason.TOO_MANY_RECIPIENTS


def test_hidden_second_address_fails_closed() -> None:
    v = check_recipients(RULE, {"to": ['"a@evil.test" <b@example.com>']})
    assert v is not None
    assert v.reason is Reason.RECIPIENT_NOT_ALLOWED


def test_cost_fixed_and_field() -> None:
    assert compute_cost(None, {}) == (0.0, None)
    assert compute_cost(Cost(fixed=0.5), {}) == (0.5, None)
    assert compute_cost(Cost(field="amount"), {"amount": 3}) == (3.0, None)


@pytest.mark.parametrize("amount", [-1, math.nan, math.inf, "10", True, None])
def test_cost_field_must_be_finite_non_negative_number(amount: object) -> None:
    cost, v = compute_cost(Cost(field="amount"), {"amount": amount})
    assert v is not None
    assert v.reason is Reason.INVALID_ARGUMENTS
    assert cost == 0.0


def test_validation_messages_do_not_echo_input() -> None:
    policy = Policy.from_yaml("tools: {t: {mode: allow, args: {pin: {type: int}}}}\n")
    validated, v = validate_arguments(policy.tools["t"], {"pin": "secret-value-123"})
    assert validated is None
    assert v is not None
    assert "secret-value-123" not in v.detail
    assert "pin" in v.detail


def test_schemaless_arguments_must_be_json() -> None:
    validated, v = validate_arguments(ToolPolicy(mode=Mode.ALLOW), {"when": object()})
    assert validated is None
    assert v is not None
    assert v.reason is Reason.INVALID_ARGUMENTS


@pytest.mark.parametrize(
    "address",
    [
        "a@evil.test\x00.example.org",  # NUL: C-string truncation leaves evil.test
        "a@evil.test#.example.org",  # URL delimiters: a URL built from this goes to evil.test
        "a@evil.test/.example.org",
        "a@evil.test?.example.org",
        "a@evil.test\\.example.org",
        "a@evil.test\u200b.example.org",  # invisible characters
        "a@.example.org",  # empty labels
        "a@team..example.org",
        "a@example.com..",
        "a@-team.example.org",  # labels cannot start or end with a hyphen
        "a\x00@example.com",  # control characters anywhere in the address
    ],
)
def test_domains_that_are_not_hostnames_are_refused(address: str) -> None:
    v = check_recipients(RULE, {"to": address})
    assert v is not None
    assert v.reason is Reason.RECIPIENT_NOT_ALLOWED


@pytest.mark.parametrize(
    "domain",
    ["evil.test\x00.example.org", "ex\u0430mple.com", "\u212aexample.com", "example.com.."],
)
def test_domain_allowed_refuses_non_hostnames(domain: str) -> None:
    # Cyrillic "a" and the Kelvin sign (which lower-cases to ASCII "k") are refused:
    # internationalised domains must be written in their ASCII (xn--) form.
    assert not domain_allowed(domain, ["example.com", "*.example.org", "kexample.com"])


def test_punycode_domains_can_be_allow_listed() -> None:
    rule = RecipientRule(fields=["to"], allowed_domains=["xn--bcher-kva.example"])
    assert check_recipients(rule, {"to": "a@XN--BCHER-KVA.example."}) is None
    assert check_recipients(rule, {"to": "a@bücher.example"}) is not None


class _BccHidden(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to: list[str]
    bcc: list[str] = Field(default_factory=list, exclude=True)  # left out of the dump


class _ToRewritten(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to: list[str]

    @field_serializer("to")
    def _display(self, value: list[str]) -> list[str]:
        return [a.split("@")[0] + "@example.com" for a in value]


@pytest.mark.parametrize(
    ("model", "args"),
    [
        (_BccHidden, {"to": ["a@example.com"], "bcc": ["x@evil.test"]}),
        (_ToRewritten, {"to": ["x@evil.test"]}),
    ],
)
def test_what_is_checked_is_what_the_tool_receives(
    model: type[BaseModel], args: dict[str, object]
) -> None:
    # The checks, the digest and the queue use the JSON form; the tool must not
    # receive anything that form does not say.
    tool = ToolPolicy(mode=Mode.ALLOW, args_model=model)
    rule = RecipientRule(fields=list(model.model_fields), allowed_domains=["example.com"])
    validated, v = validate_arguments(tool, args)
    if validated is not None:
        assert check_recipients(rule, validated.stored) is None  # the policy passes...
        sent = validated.call.get("to", []) + validated.call.get("bcc", [])
        assert all(a.endswith("@example.com") for a in sent)  # ...so only these may go
    else:
        assert v is not None and v.reason is Reason.INVALID_ARGUMENTS


def test_yaml_schemas_with_defaults_still_validate() -> None:
    policy = Policy.from_yaml(
        "tools: {t: {mode: allow, args: {a: {type: str}, b: {type: int, required: false, "
        "default: 3}, c: {type: 'list[float]', required: false}}}}\n"
    )
    validated, v = validate_arguments(policy.tools["t"], {"a": "x"})
    assert v is None and validated is not None
    assert validated.call == {"a": "x", "b": 3, "c": None}
