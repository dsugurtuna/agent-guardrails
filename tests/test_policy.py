from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from agent_guardrails import Mode, Policy, PolicyError, ToolPolicy

YAML = """
version: 1
default_mode: block
budgets:
  "*": {limit: 10}
  planner: {limit: 2.5, window_seconds: 3600}
tools:
  search: {mode: allow}
  send_email:
    mode: approve
    args:
      to: {type: "list[str]", min_items: 1, max_items: 5}
      subject: {type: str, max_length: 10}
      priority: {type: str, choices: [low, high], required: false, default: low}
    recipients: {fields: [to], allowed_domains: [Example.COM.]}
  purchase:
    mode: approve
    args:
      amount: {type: float, ge: 0}
    cost: {field: amount}
"""


def test_yaml_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(YAML, encoding="utf-8")
    from_file = Policy.from_yaml(path)
    from_text = Policy.from_yaml(YAML)
    assert from_file.fingerprint() == from_text.fingerprint()
    assert from_file.mode_for("search") is Mode.ALLOW
    assert from_file.mode_for("not_listed") is Mode.BLOCK
    assert from_file.budget_for("planner") is not None
    assert from_file.budget_for("planner").limit == 2.5  # type: ignore[union-attr]
    assert from_file.budget_for("someone_else").limit == 10  # type: ignore[union-attr]
    rule = from_file.tools["send_email"].recipients
    assert rule is not None and rule.allowed_domains == ["example.com"]


def test_yaml_schema_is_enforced() -> None:
    policy = Policy.from_yaml(YAML)
    model = policy.tools["send_email"].schema_model
    assert model is not None
    ok = model.model_validate({"to": ["a@example.com"], "subject": "hi"})
    assert ok.model_dump()["priority"] == "low"
    for bad in (
        {"to": [], "subject": "hi"},  # min_items
        {"to": ["a@example.com"], "subject": "x" * 11},  # max_length
        {"to": ["a@example.com"], "subject": "hi", "priority": "urgent"},  # choices
        {"to": ["a@example.com"], "subject": "hi", "cc": ["b@example.com"]},  # extra field
        {"subject": "hi"},  # required
    ):
        with pytest.raises(ValidationError):
            model.model_validate(bad)


@pytest.mark.parametrize(
    ("snippet", "message"),
    [
        ("tools: {x: {mode: allow, typo_key: 1}}", "typo_key"),
        ("default_mode: allow", "default_mode 'allow' is not permitted"),
        ("tools: {x: {mode: sometimes}}", "mode"),
        ("tools: {x: {mode: allow, cost: {fixed: 1, field: amount}}}", "exactly one"),
        ("tools: {x: {mode: allow, cost: {}}}", "exactly one"),
        (
            "tools: {x: {mode: allow, args: {a: {type: str}}, recipients: "
            "{fields: [to], allowed_domains: [a.com]}}}",
            "unknown argument",
        ),
        ("tools: {x: {mode: allow, rate_limit: {max_calls: 0, window_seconds: 1}}}", "max_calls"),
    ],
)
def test_invalid_policies_fail_loudly(snippet: str, message: str) -> None:
    with pytest.raises(PolicyError, match=message):
        Policy.from_yaml(snippet + "\n")


def test_top_level_must_be_mapping() -> None:
    with pytest.raises(PolicyError):
        Policy.from_yaml("- just\n- a list\n")


class SendArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to: list[str]
    body: str


def test_python_policy_with_args_model() -> None:
    policy = Policy(tools={"send": ToolPolicy(mode=Mode.APPROVE, args_model=SendArgs)})
    assert policy.tools["send"].schema_model is SendArgs


def test_args_and_args_model_are_exclusive() -> None:
    with pytest.raises(ValidationError, match="not both"):
        ToolPolicy(mode=Mode.ALLOW, args={}, args_model=SendArgs)


def test_fingerprint_changes_with_policy() -> None:
    a = Policy.from_yaml(YAML)
    b = Policy.from_yaml(YAML.replace("max_length: 10", "max_length: 11"))
    c = Policy(tools={"send": ToolPolicy(mode=Mode.APPROVE, args_model=SendArgs)})
    assert a.fingerprint() == Policy.from_yaml(YAML).fingerprint()
    assert a.fingerprint() != b.fingerprint()
    assert len(c.fingerprint()) == 64


def test_policy_is_immutable() -> None:
    policy = Policy.from_yaml(YAML)
    with pytest.raises(ValidationError):
        policy.default_mode = Mode.DRAFT  # type: ignore[misc]


def test_choices_on_lists_apply_to_items() -> None:
    policy = Policy.from_yaml(
        'tools: {t: {mode: allow, args: {tags: {type: "list[str]", choices: [a, b]}}}}\n'
    )
    model = policy.tools["t"].schema_model
    assert model is not None
    assert model.model_validate({"tags": ["a", "b"]}).model_dump() == {"tags": ["a", "b"]}
    with pytest.raises(ValidationError):
        model.model_validate({"tags": ["c"]})


def test_from_yaml_accepts_path_string_and_single_line_text(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text("tools: {x: {mode: allow}}\n", encoding="utf-8")
    assert Policy.from_yaml(str(path)).mode_for("x") is Mode.ALLOW
    assert Policy.from_yaml("tools: {x: {mode: draft}}").mode_for("x") is Mode.DRAFT
    with pytest.raises(PolicyError, match="file path"):
        Policy.from_yaml("missing-policy.yaml")
    with pytest.raises(PolicyError, match="not valid YAML"):
        Policy.from_yaml("tools: {x: [unclosed\n")


@pytest.mark.parametrize(
    "entry", ["*example.com", "", "exa mple.com", "*.*.example.com", "bücher.example", "a..b"]
)
def test_malformed_allow_list_entries_fail_loudly(entry: str) -> None:
    with pytest.raises(PolicyError, match="allowed_domains"):
        Policy.from_dict(
            {
                "tools": {
                    "t": {
                        "mode": "allow",
                        "recipients": {"fields": ["to"], "allowed_domains": [entry]},
                    }
                }
            }
        )


def _arg_model(spec: dict[str, object]) -> type[BaseModel]:
    policy = Policy.from_dict({"tools": {"t": {"mode": "allow", "args": {"x": spec}}}})
    model = policy.tools["t"].schema_model
    assert model is not None
    return model


@pytest.mark.parametrize(
    ("spec", "good", "bad"),
    [
        # String constraints on a list[str] apply to every item, as `choices` does.
        (
            {"type": "list[str]", "pattern": r"^[a-z]+@example\.com$"},
            ["a@example.com"],
            ["a@example.com", "x@evil.test"],
        ),
        ({"type": "list[str]", "max_length": 3}, ["abc"], ["abcd"]),
        ({"type": "list[str]", "min_length": 2}, ["ab"], ["a"]),
        # Numeric bounds on a list of numbers apply to every item.
        ({"type": "list[float]", "ge": 0}, [0.0, 5.5], [5.5, -100.0]),
        ({"type": "list[int]", "le": 5}, [1, 5], [999]),
        ({"type": "list[str]", "min_items": 1, "max_items": 2}, ["a"], ["a", "b", "c"]),
    ],
)
def test_constraints_apply_to_list_items(
    spec: dict[str, object], good: object, bad: object
) -> None:
    model = _arg_model(spec)
    model.model_validate({"x": good})
    with pytest.raises(ValidationError):
        model.model_validate({"x": bad})


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "int", "pattern": "^1$"},
        {"type": "float", "max_length": 3},
        {"type": "str", "ge": 0},
        {"type": "str", "min_items": 1},
        {"type": "bool", "min_length": 1},
        {"type": "bool", "le": 1},
        {"type": "list[str]", "ge": 0},
        {"type": "list[int]", "pattern": "^1$"},
    ],
)
def test_constraints_that_cannot_apply_fail_loudly(spec: dict[str, object]) -> None:
    # A constraint that would be silently ignored is as dangerous as a typo.
    with pytest.raises(PolicyError, match="does not apply"):
        Policy.from_dict({"tools": {"t": {"mode": "allow", "args": {"x": spec}}}})
