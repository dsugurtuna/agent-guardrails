"""Declarative policy: what each tool may do, and under which limits.

A policy can be written in Python or loaded from YAML. It is validated strictly
(unknown keys are errors) because a typo in a security policy should fail loudly,
not be silently ignored.
"""

from __future__ import annotations

from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)

from ._canonical import canonical_hostname, digest
from .errors import PolicyError

DEFAULT_REDACT_FIELDS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "private_key",
)


class Mode(StrEnum):
    """How a tool call is handled.

    - ``allow``: execute now (still subject to every constraint).
    - ``draft``: never execute; return a preview of what would happen.
    - ``approve``: queue for a human decision; execute later, after re-validation.
    - ``block``: never execute.
    """

    ALLOW = "allow"
    DRAFT = "draft"
    APPROVE = "approve"
    BLOCK = "block"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


ArgType = Literal["str", "int", "float", "bool", "list[str]", "list[int]", "list[float]"]

_PY_TYPES: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list[str]": str,  # item type; the list is built in ArgSpec.to_field
    "list[int]": int,
    "list[float]": float,
}

_STRING_KEYS = ("min_length", "max_length", "pattern")
_NUMBER_KEYS = ("ge", "le")
_LIST_KEYS = ("min_items", "max_items")
_APPLICABLE: dict[str, tuple[str, ...]] = {
    "str": _STRING_KEYS,
    "int": _NUMBER_KEYS,
    "float": _NUMBER_KEYS,
    "bool": (),
    "list[str]": _STRING_KEYS + _LIST_KEYS,
    "list[int]": _NUMBER_KEYS + _LIST_KEYS,
    "list[float]": _NUMBER_KEYS + _LIST_KEYS,
}


class ArgSpec(_Strict):
    """A small, YAML-friendly argument schema. Use ``args_model`` in Python for more.

    For list types, ``choices``, the string constraints (``min_length``,
    ``max_length``, ``pattern``) and the numeric bounds (``ge``, ``le``) apply to each
    item; ``min_items`` and ``max_items`` apply to the list. A constraint that cannot
    apply to the type is an error, not silently ignored.
    """

    type: ArgType = "str"
    required: bool = True
    default: Any = None
    description: str | None = None
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)
    pattern: str | None = None
    choices: list[str | int | float] | None = None
    ge: float | None = None
    le: float | None = None
    min_items: int | None = Field(default=None, ge=0)
    max_items: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _constraints_apply(self) -> ArgSpec:
        allowed = _APPLICABLE[self.type]
        misplaced = [
            key
            for key in _STRING_KEYS + _NUMBER_KEYS + _LIST_KEYS
            if getattr(self, key) is not None and key not in allowed
        ]
        if misplaced:
            raise ValueError(
                f"{', '.join(misplaced)} does not apply to type '{self.type}' "
                f"(allowed: {', '.join(allowed) or 'choices only'})"
            )
        return self

    def to_field(self) -> tuple[Any, Any]:
        value_type: Any = _PY_TYPES[self.type]
        if self.choices is not None:
            value_type = Literal[tuple(self.choices)]
        value_constraints = {
            key: getattr(self, key)
            for key in _STRING_KEYS + _NUMBER_KEYS
            if getattr(self, key) is not None
        }
        py_type: Any
        if self.type.startswith("list"):
            item: Any = Annotated[value_type, Field(**value_constraints)]
            list_constraints: dict[str, Any] = {}
            if self.min_items is not None:
                list_constraints["min_length"] = self.min_items
            if self.max_items is not None:
                list_constraints["max_length"] = self.max_items
            py_type = Annotated[list[item], Field(description=self.description, **list_constraints)]
        else:
            py_type = Annotated[
                value_type, Field(description=self.description, **value_constraints)
            ]
        if self.required:
            return (py_type, ...)
        return (py_type | None, self.default)


class RecipientRule(_Strict):
    """Who a message may be addressed to.

    ``allowed_domains`` entries match exactly (``example.com``), match any subdomain
    (``*.example.com``), or match everything (``*``, which must be written on purpose).
    Entries must be ASCII hostnames (internationalised domains in ``xn--`` form), so a
    typo such as ``*example.com`` fails loudly instead of silently matching nothing.
    """

    fields: list[str] = Field(min_length=1)
    allowed_domains: list[str]
    max_recipients: int | None = Field(default=None, ge=1)

    @field_validator("allowed_domains")
    @classmethod
    def _canonical(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for entry in value:
            text = entry.strip()
            if text == "*":
                out.append(text)
                continue
            wildcard = text.startswith("*.")
            host = canonical_hostname(text[2:] if wildcard else text)
            if host is None:
                raise ValueError(
                    f"allowed_domains entry {entry!r} must be '*', a hostname, or '*.' and "
                    "a hostname (write internationalised domains in their xn-- form)"
                )
            out.append("*." + host if wildcard else host)
        return out


class RateLimit(_Strict):
    max_calls: int = Field(ge=1)
    window_seconds: float = Field(gt=0)
    per_agent: bool = False


class Cost(_Strict):
    """How much one call spends against the agent's budget: a fixed amount or an argument."""

    fixed: float | None = Field(default=None, ge=0)
    field: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Cost:
        if (self.fixed is None) == (self.field is None):
            raise ValueError("cost needs exactly one of 'fixed' or 'field'")
        return self


class Budget(_Strict):
    limit: float = Field(ge=0)
    window_seconds: float | None = Field(default=None, gt=0)


class ToolPolicy(BaseModel):
    """Policy for one tool."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    mode: Mode
    description: str | None = None
    args: dict[str, ArgSpec] | None = None
    args_model: type[BaseModel] | None = Field(default=None, exclude=True)
    recipients: RecipientRule | None = None
    rate_limit: RateLimit | None = None
    cost: Cost | None = None
    approval_ttl_seconds: float = Field(default=3600.0, gt=0)
    dedupe_window_seconds: float = Field(default=3600.0, ge=0)
    max_pending: int | None = Field(default=None, ge=1)
    redact_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_schema(self) -> ToolPolicy:
        if self.args is not None and self.args_model is not None:
            raise ValueError("give either 'args' (YAML schema) or 'args_model' (Python), not both")
        return self

    @cached_property
    def schema_model(self) -> type[BaseModel] | None:
        """The pydantic model used to validate arguments, if any."""
        if self.args_model is not None:
            return self.args_model
        if self.args is None:
            return None
        fields = {name: spec.to_field() for name, spec in self.args.items()}
        model: type[BaseModel] = create_model(  # type: ignore[call-overload]
            "ToolArgs",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        return model

    def fingerprint_data(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        if self.args_model is not None:
            data["args_model_schema"] = self.args_model.model_json_schema()
        return data


class Policy(BaseModel):
    """The whole policy. Tools not listed get ``default_mode`` (``block`` by default)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    default_mode: Mode = Mode.BLOCK
    tools: dict[str, ToolPolicy] = Field(default_factory=dict)
    budgets: dict[str, Budget] = Field(default_factory=dict)
    redact_fields: list[str] = Field(default_factory=lambda: list(DEFAULT_REDACT_FIELDS))

    @field_validator("default_mode")
    @classmethod
    def _no_default_allow(cls, value: Mode) -> Mode:
        if value is Mode.ALLOW:
            raise ValueError(
                "default_mode 'allow' is not permitted: list each tool that may run unattended"
            )
        return value

    @model_validator(mode="after")
    def _schema_fields_exist(self) -> Policy:
        for name, tool in self.tools.items():
            model = tool.schema_model
            if model is None:
                continue
            known = set(model.model_fields)
            referenced: list[str] = []
            if tool.recipients is not None:
                referenced += tool.recipients.fields
            if tool.cost is not None and tool.cost.field is not None:
                referenced.append(tool.cost.field)
            missing = [f for f in referenced if f not in known]
            if missing:
                raise ValueError(f"tool '{name}' refers to unknown argument(s): {missing}")
        return self

    # -- construction -------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise PolicyError(f"invalid policy: {exc}") from exc

    @classmethod
    def from_yaml(cls, source: str | Path) -> Policy:
        """Load from a YAML file (a ``Path``, or a string naming an existing file) or text."""
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        elif "\n" not in source and Path(source).is_file():
            text = Path(source).read_text(encoding="utf-8")
        else:
            text = source
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PolicyError(f"policy is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            hint = " (is the file path right?)" if isinstance(data, str) else ""
            raise PolicyError(f"policy YAML must be a mapping at the top level{hint}")
        return cls.from_dict(data)

    # -- queries ------------------------------------------------------------------

    def tool(self, name: str) -> ToolPolicy | None:
        return self.tools.get(name)

    def mode_for(self, name: str) -> Mode:
        tool = self.tools.get(name)
        return tool.mode if tool is not None else self.default_mode

    def budget_for(self, agent_id: str) -> Budget | None:
        return self.budgets.get(agent_id, self.budgets.get("*"))

    def fingerprint(self) -> str:
        """SHA-256 of the policy's canonical form. Recorded with every decision."""
        data = self.model_dump(mode="json", exclude={"tools"})
        data["tools"] = {name: t.fingerprint_data() for name, t in sorted(self.tools.items())}
        return digest(data)
