"""The guard: every tool call an agent makes passes through :meth:`Guard.call`.

Order of checks for a new call (cheapest and most absolute first):

0. the tool name must be a non-empty string of valid Unicode
1. kill switch
2. mode (``block`` stops here; unknown tools get the policy's ``default_mode``)
3. arguments: bound to the function signature, then validated against the schema
4. recipients (allow-listed domains, maximum count)
5. cost is computed from the arguments
6. then, per mode:
   - ``draft``: return a preview, nothing runs
   - ``approve``: de-duplicate, check queue size, queue for a human
   - ``allow``: in one database transaction, de-duplicate, check the rate limit and
     budget and reserve the execution; re-check the kill switch; run the tool

An approved action goes through the same checks again, against the policy in force
*at execution time*, before it runs (see :meth:`Guard.execute_approved`).
"""

from __future__ import annotations

import inspect
import json
import os
import secrets
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._canonical import canonical_json, dedupe_key, digest, utf8_safe
from .audit import AuditLog, json_safe
from .checks import (
    ValidatedArgs,
    Violation,
    check_recipients,
    compute_cost,
    validate_arguments,
)
from .errors import ApprovalExpiredError, ConfigurationError
from .killswitch import KillSwitch
from .outcomes import TRANSIENT_REASONS, Outcome, Reason, Status
from .policy import Mode, Policy, ToolPolicy
from .redaction import RedactHook, Redactor, scrub
from .store import ActionRecord, ActionState, ActionStore, Tx

HOME_ENV = "AGENT_GUARDRAILS_HOME"
DEFAULT_HOME = ".agent-guardrails"

PreviewFn = Callable[[str, dict[str, Any]], str]


def default_home() -> Path:
    """``$AGENT_GUARDRAILS_HOME`` or ``./.agent-guardrails``: shared with the CLI."""
    return Path(os.environ.get(HOME_ENV, DEFAULT_HOME))


def default_preview(tool: str, args: dict[str, Any]) -> str:
    lines = [f"{tool}("]
    for key in sorted(args):
        lines.append(f"  {key}={json.dumps(args[key], ensure_ascii=False, default=str)},")
    lines.append(")")
    return "\n".join(lines)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _over_budget(total: float, limit: float) -> bool:
    """``total > limit``, compared to nine decimal places.

    Why round? Costs are floats, and binary floating point makes 0.1 + 0.1 + 0.1
    come to 0.30000000000000004, which would refuse the third 0.1 call against a
    budget of 0.3. Nine places is far finer than any currency or credit unit.
    """
    return round(total, 9) > round(limit, 9)


def _new_id() -> str:
    return "act_" + secrets.token_hex(8)


@dataclass(frozen=True)
class _Registered:
    fn: Callable[..., Any]
    signature: inspect.Signature
    preview: PreviewFn | None

    def bind(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """Check the arguments fit the function and fill in its defaults."""
        bound = self.signature.bind(**args)
        bound.apply_defaults()
        out: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            param = self.signature.parameters[name]
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                out.update(value)
            else:
                out[name] = value
        return out


@dataclass(frozen=True)
class _Prepared:
    """A call whose arguments have passed every stateless check."""

    ctx: dict[str, Any]
    policy: Policy
    tool: ToolPolicy
    validated: ValidatedArgs
    cost: float
    key: str
    args_digest: str
    logged: Any


def _register(fn: Callable[..., Any], preview: PreviewFn | None) -> _Registered:
    if inspect.iscoroutinefunction(fn):
        raise ConfigurationError(
            "async tool functions are not supported yet; wrap a synchronous function"
        )
    sig = inspect.signature(fn)
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()):
        raise ConfigurationError("tool functions cannot take *args: arguments must be named")
    return _Registered(fn, sig, preview)


# Tools registered with the module-level @guarded decorator, so a worker process that
# only imports the tool module can still execute approved actions.
_GLOBAL_TOOLS: dict[str, _Registered] = {}
_default_guard: Guard | None = None


class GuardedTool:
    """A tool function wrapped so every call goes through a guard. Returns an Outcome."""

    def __init__(
        self,
        name: str,
        fn: Callable[..., Any],
        guard: Guard | None,
        agent_id: str | None = None,
    ) -> None:
        self.name = name
        self.__wrapped__ = fn
        self.__doc__ = fn.__doc__
        self._guard = guard
        self._agent_id = agent_id
        self._sig = inspect.signature(fn)

    @property
    def guard(self) -> Guard:
        return self._guard if self._guard is not None else get_default_guard()

    def __call__(self, *args: Any, **kwargs: Any) -> Outcome:
        named = dict(kwargs)
        if args:
            try:
                bound = self._sig.bind_partial(*args)
            except TypeError as exc:
                raise TypeError(f"{self.name}: {exc}") from exc
            named.update(bound.arguments)
        return self.guard.call(self.name, named, agent_id=self._agent_id)

    def __repr__(self) -> str:
        return f"<GuardedTool {self.name}>"


class Guard:
    """Applies a :class:`Policy` to tool calls, with durable state and an audit trail.

    By default the queue database, audit log and kill-switch flag live under
    :func:`default_home`, the same place the ``agent-guardrails`` CLI looks.
    """

    def __init__(
        self,
        policy: Policy,
        *,
        home: str | Path | None = None,
        store: ActionStore | None = None,
        audit: AuditLog | None = None,
        kill_switch: KillSwitch | None = None,
        agent_id: str = "default",
        clock: Callable[[], float] | None = None,
        redact_hook: RedactHook | None = None,
    ) -> None:
        if not agent_id:
            raise ConfigurationError("agent_id must be a non-empty string")
        base = Path(home) if home is not None else default_home()
        self._clock = clock or time.time
        self.store = store or ActionStore(base / "queue.db")
        self.audit = audit or AuditLog(base / "audit.jsonl", clock=self._clock)
        self.kill_switch = kill_switch or KillSwitch(flag_file=base / "KILL")
        self.agent_id = agent_id
        self._redact_hook = redact_hook
        self._tools: dict[str, _Registered] = {}
        # The policy and its fingerprint are swapped together, as one tuple, and each
        # call reads them once: a concurrent set_policy must never make a decision
        # taken under one policy be recorded with the other's fingerprint.
        self._active: tuple[Policy, str] = (policy, policy.fingerprint())

    # -- configuration ------------------------------------------------------------

    @property
    def policy(self) -> Policy:
        return self._active[0]

    def set_policy(self, policy: Policy, *, by: str | None = None) -> None:
        """Swap the policy. Queued actions will be re-checked against the new one."""
        new = (policy, policy.fingerprint())
        old = self._active[1]
        self._active = new
        self._audit("policy_changed", old_policy_hash=old, policy_hash=new[1], by=by)

    def register(
        self, name: str, fn: Callable[..., Any], *, preview: PreviewFn | None = None
    ) -> None:
        self._tools[name] = _register(fn, preview)

    def wrap(
        self, fn: Callable[..., Any], *, name: str | None = None, preview: PreviewFn | None = None
    ) -> GuardedTool:
        """Register ``fn`` and return a guarded callable that returns an :class:`Outcome`."""
        tool_name = name or fn.__name__
        self.register(tool_name, fn, preview=preview)
        return GuardedTool(tool_name, fn, self)

    def tool(
        self, name: str | None = None, *, preview: PreviewFn | None = None
    ) -> Callable[[Callable[..., Any]], GuardedTool]:
        """Decorator form of :meth:`wrap`."""

        def decorator(fn: Callable[..., Any]) -> GuardedTool:
            return self.wrap(fn, name=name, preview=preview)

        return decorator

    def _lookup(self, name: str) -> _Registered | None:
        return self._tools.get(name) or _GLOBAL_TOOLS.get(name)

    def _now(self) -> float:
        return self._clock()

    def _redactor(self, policy: Policy, tool: ToolPolicy | None) -> Redactor:
        base = Redactor(policy.redact_fields, self._redact_hook)
        return base.with_fields(tool.redact_fields) if tool is not None else base

    # -- audit helpers ---------------------------------------------------------------

    def _audit(self, event: str, **fields: Any) -> None:
        self.audit.append(event, **{k: v for k, v in fields.items() if v is not None})

    def _block(
        self,
        ctx: dict[str, Any],
        reason: Reason,
        message: str,
        args: Any,
        action_id: str | None = None,
        *,
        hidden: Iterable[str] = (),
    ) -> Outcome:
        """Audit and return a block. ``hidden``: redacted values to keep out of the log.

        The model gets the full message (it sent those values); the audit record gets
        the message with every redacted value removed, since details quote arguments.
        """
        message = utf8_safe(message)  # details can quote malformed input back
        detail = scrub(message, hidden)
        self._audit(
            "blocked", **ctx, action_id=action_id, reason=str(reason), detail=detail, args=args
        )
        return Outcome(Status.BLOCKED, ctx["tool"], message, action_id=action_id, reason=reason)

    def _duplicate(self, ctx: dict[str, Any], dup: ActionRecord, args: Any) -> Outcome:
        if dup.status is ActionState.EXECUTED:
            when = _iso(dup.finished_at or dup.created_at)
            message = f"an identical '{ctx['tool']}' action was already executed at {when}."
        elif dup.status is ActionState.EXECUTING:
            message = f"an identical '{ctx['tool']}' action is being executed right now."
        elif dup.status is ActionState.APPROVED:
            message = f"an identical '{ctx['tool']}' action is already approved and due to run."
        else:
            message = f"an identical '{ctx['tool']}' action is already awaiting approval."
        self._audit("duplicate", **ctx, duplicate_of=dup.id, detail=message, args=args)
        return Outcome(
            Status.DUPLICATE, ctx["tool"], message, action_id=dup.id, details={"of": dup.id}
        )

    # -- the main entry point -------------------------------------------------------

    def call(
        self, name: str, args: Mapping[str, Any] | None = None, *, agent_id: str | None = None
    ) -> Outcome:
        """Apply the policy to one tool call and return what happened."""
        if not isinstance(name, str) or not name or utf8_safe(name) != name:
            # The name comes from the model. Anything that is not usable text cannot be
            # in the policy, and must not reach the store or the hashes: block it.
            label = utf8_safe(name) if isinstance(name, str) else repr(name)
            ctx0 = {"tool": label, "agent_id": agent_id or self.agent_id, "mode": "block"}
            msg = "the tool name is not a valid, non-empty string, so it is blocked."
            logged = self._redactor(self.policy, None).redact(args)
            return self._block(ctx0, Reason.UNKNOWN_TOOL, msg, logged)
        policy, policy_hash = self._active
        tool = policy.tool(name)
        mode = policy.mode_for(name)
        redactor = self._redactor(policy, tool)
        raw: dict[str, Any] = dict(args or {})
        hidden: set[str] = set()
        logged_raw = redactor.redact(raw, hidden)
        ctx: dict[str, Any] = {
            "tool": name,
            "agent_id": agent_id or self.agent_id,
            "mode": str(mode),
            "policy_hash": policy_hash,
        }

        ks = self.kill_switch.status()
        if ks.engaged:
            msg = f"the kill switch is engaged ({ks.reason}); no actions are being taken."
            return self._block(ctx, Reason.KILL_SWITCH, msg, logged_raw)
        if mode is Mode.BLOCK:
            if tool is None:
                msg = f"'{name}' is not in the policy, so it is blocked by default."
                return self._block(ctx, Reason.UNKNOWN_TOOL, msg, logged_raw)
            msg = f"'{name}' is blocked by policy."
            return self._block(ctx, Reason.TOOL_BLOCKED, msg, logged_raw)

        effective = tool if tool is not None else ToolPolicy(mode=mode)
        reg = self._lookup(name)
        prepared = self._prepare(ctx, policy, effective, reg, raw)
        if isinstance(prepared, Violation):
            return self._block(ctx, prepared.reason, prepared.detail, logged_raw, hidden=hidden)

        if mode is Mode.DRAFT:
            render = reg.preview if reg is not None and reg.preview is not None else default_preview
            preview = render(name, prepared.validated.stored)
            self._audit("drafted", **ctx, args=prepared.logged, args_digest=prepared.args_digest)
            return Outcome(
                Status.DRAFTED, name, "draft prepared; nothing was executed", preview=preview
            )
        if mode is Mode.APPROVE:
            return self._enqueue(prepared)
        if reg is None:
            msg = f"'{name}' is allowed by policy but no function is registered for it."
            return self._block(ctx, Reason.NOT_REGISTERED, msg, prepared.logged)
        return self._execute_now(prepared, reg)

    def _prepare(
        self,
        ctx: dict[str, Any],
        policy: Policy,
        tool: ToolPolicy,
        reg: _Registered | None,
        raw: Mapping[str, Any],
    ) -> _Prepared | Violation:
        """Bind, validate and check arguments. Reads and writes no state."""
        if reg is not None:
            try:
                raw = reg.bind(raw)
            except TypeError as exc:
                return Violation(Reason.INVALID_ARGUMENTS, str(exc))
        validated, violation = validate_arguments(tool, raw)
        if violation is not None:
            return violation
        assert validated is not None  # noqa: S101
        violation = check_recipients(tool.recipients, validated.stored)
        if violation is not None:
            return violation
        cost, violation = compute_cost(tool.cost, validated.stored)
        if violation is not None:
            return violation
        fields = tool.recipients.fields if tool.recipients is not None else ()
        return _Prepared(
            ctx=ctx,
            policy=policy,
            tool=tool,
            validated=validated,
            cost=cost,
            key=dedupe_key(ctx["tool"], validated.stored, fields),
            args_digest=digest(validated.stored),
            logged=self._redactor(policy, tool).redact(validated.stored),
        )

    def _check_limits(
        self, tx: Tx, p: _Prepared, now: float, *, exclude_id: str | None = None
    ) -> Violation | ActionRecord | None:
        """Checks that need the database. Returns a violation, a duplicate, or None."""
        name, agent = p.ctx["tool"], p.ctx["agent_id"]
        if p.tool.dedupe_window_seconds > 0:
            dup = tx.find_duplicate(p.key, now - p.tool.dedupe_window_seconds, exclude_id)
            if dup is not None:
                return dup
        if p.tool.rate_limit is not None:
            rl = p.tool.rate_limit
            used = tx.count_runs(name, now - rl.window_seconds, agent if rl.per_agent else None)
            if used >= rl.max_calls:
                return Violation(
                    Reason.RATE_LIMITED,
                    f"'{name}' has run {used} times in the last {rl.window_seconds:g}s; "
                    f"the limit is {rl.max_calls}.",
                )
        budget = p.policy.budget_for(agent)
        if budget is not None and p.cost > 0:
            since = now - budget.window_seconds if budget.window_seconds else None
            spent = tx.spent(agent, since)
            if _over_budget(spent + p.cost, budget.limit):
                return Violation(
                    Reason.BUDGET_EXCEEDED,
                    f"agent '{agent}' has spent {spent:g} of its {budget.limit:g} budget; "
                    f"this call would cost {p.cost:g}.",
                )
        return None

    # -- allow mode --------------------------------------------------------------------

    def _execute_now(self, p: _Prepared, reg: _Registered) -> Outcome:
        now = self._now()
        action_id = _new_id()
        with self.store.transaction() as tx:
            found = self._check_limits(tx, p, now)
            if found is None:
                tx.insert(
                    **self._row(p, action_id, now, keep_secrets=False),
                    status=ActionState.EXECUTING,
                    started_at=now,
                )
        if isinstance(found, ActionRecord):
            return self._duplicate(p.ctx, found, p.logged)
        if isinstance(found, Violation):
            return self._block(p.ctx, found.reason, found.detail, p.logged)
        return self._run(p, action_id, reg, approved_by=None)

    @staticmethod
    def _row(p: _Prepared, action_id: str, now: float, *, keep_secrets: bool) -> dict[str, Any]:
        """Column values for a new action.

        A queued action must keep its full arguments: it runs later, from this row,
        and they are bound to ``args_digest``. An action run at once (allow mode) is
        never executed from its row again, so only its redacted arguments are kept,
        as in the audit log; ``args_digest`` still identifies the full arguments.
        """
        args = p.validated.stored if keep_secrets else json_safe(p.logged)
        return {
            "id": action_id,
            "tool": p.ctx["tool"],
            "agent_id": p.ctx["agent_id"],
            "mode": p.ctx["mode"],
            "args_json": canonical_json(args),
            "args_digest": p.args_digest,
            "dedupe_key": p.key,
            "cost": p.cost,
            "policy_hash": p.ctx["policy_hash"],
            "created_at": now,
        }

    def _run(
        self, p: _Prepared, action_id: str, reg: _Registered, *, approved_by: str | None
    ) -> Outcome:
        """Run a reserved action (status ``executing``). Audits before and after."""
        ctx = p.ctx
        try:
            self._audit(
                "allowed",
                **ctx,
                action_id=action_id,
                args=p.logged,
                args_digest=p.args_digest,
                cost=p.cost or None,
                approved_by=approved_by,
            )
        except Exception:
            # No audit record, no action: fail closed.
            self._set_state(action_id, ActionState.BLOCKED, "audit_unavailable")
            raise
        ks = self.kill_switch.status()
        if ks.engaged:
            # Last-moment check. An approved action goes back to 'approved' so a human
            # can reject it or let it expire; a direct call is simply not run.
            back = ActionState.APPROVED if approved_by is not None else ActionState.BLOCKED
            self._set_state(action_id, back, str(Reason.KILL_SWITCH))
            msg = f"the kill switch is engaged ({ks.reason}); the action was not run."
            return self._block(ctx, Reason.KILL_SWITCH, msg, p.logged, action_id)
        try:
            result = reg.fn(**p.validated.call)
        except Exception as exc:  # the tool's own failure is reported, not raised
            self._set_state(action_id, ActionState.FAILED, str(Reason.TOOL_ERROR))
            self._audit(
                "failed", **ctx, action_id=action_id, reason="tool_error", error=type(exc).__name__
            )
            # Only the exception type goes back to the model: messages can contain secrets.
            return Outcome(
                Status.FAILED,
                ctx["tool"],
                f"the tool raised {type(exc).__name__}; it may or may not have taken effect.",
                action_id=action_id,
                reason=Reason.TOOL_ERROR,
            )
        self._set_state(action_id, ActionState.EXECUTED, None)
        self._audit("executed", **ctx, action_id=action_id, args_digest=p.args_digest)
        return Outcome(Status.EXECUTED, ctx["tool"], "executed", action_id=action_id, result=result)

    def _set_state(self, action_id: str, status: ActionState, reason: str | None) -> None:
        values: dict[str, Any] = {"status": status, "reason": reason}
        if status is ActionState.APPROVED:
            values["started_at"] = None  # handed back: not counted as a run
        else:
            values["finished_at"] = self._now()
        with self.store.transaction() as tx:
            tx.update(action_id, **values)

    # -- approve mode --------------------------------------------------------------------

    def _enqueue(self, p: _Prepared) -> Outcome:
        now = self._now()
        action_id = _new_id()
        expires_at = now + p.tool.approval_ttl_seconds
        name = p.ctx["tool"]
        found: Violation | ActionRecord | None = None
        with self.store.transaction() as tx:
            if p.tool.dedupe_window_seconds > 0:
                found = tx.find_duplicate(p.key, now - p.tool.dedupe_window_seconds)
            if found is None and p.tool.max_pending is not None:
                waiting = tx.pending_count(name)
                if waiting >= p.tool.max_pending:
                    found = Violation(
                        Reason.QUEUE_FULL,
                        f"{waiting} '{name}' actions are already awaiting approval "
                        f"(limit {p.tool.max_pending}).",
                    )
            budget = p.policy.budget_for(p.ctx["agent_id"])
            if found is None and budget is not None and _over_budget(p.cost, budget.limit):
                found = Violation(
                    Reason.BUDGET_EXCEEDED,
                    f"this call would cost {p.cost:g}, more than the whole budget of "
                    f"{budget.limit:g}.",
                )
            if found is None:
                tx.insert(
                    **self._row(p, action_id, now, keep_secrets=True),
                    status=ActionState.PENDING,
                    expires_at=expires_at,
                )
        if isinstance(found, ActionRecord):
            return self._duplicate(p.ctx, found, p.logged)
        if isinstance(found, Violation):
            return self._block(p.ctx, found.reason, found.detail, p.logged)
        self._audit(
            "queued",
            **p.ctx,
            action_id=action_id,
            args=p.logged,
            args_digest=p.args_digest,
            cost=p.cost or None,
            expires_at=_iso(expires_at),
        )
        return Outcome(
            Status.QUEUED,
            name,
            f"queued for human approval; expires at {_iso(expires_at)} if not approved and run.",
            action_id=action_id,
            details={"expires_at": _iso(expires_at)},
        )

    # -- human decisions -----------------------------------------------------------------

    def pending(self) -> list[ActionRecord]:
        self.expire_stale()
        return self.store.list_actions([ActionState.PENDING])

    def expire_stale(self) -> list[ActionRecord]:
        expired = self.store.expire_stale(self._now())
        for rec in expired:
            self._audit(
                "expired", tool=rec.tool, agent_id=rec.agent_id, action_id=rec.id, was=rec.status
            )
        return expired

    def approve(self, action_id: str, *, by: str, note: str | None = None) -> ActionRecord:
        """Record a human approval. This does not run the action: see :meth:`run_approved`."""
        return approve_action(self.store, self.audit, action_id, by=by, now=self._now(), note=note)

    def reject(self, action_id: str, *, by: str, reason: str | None = None) -> ActionRecord:
        return reject_action(self.store, self.audit, action_id, by=by, now=self._now(), note=reason)

    def engage_kill_switch(self, reason: str, *, by: str | None = None) -> None:
        self.kill_switch.engage(reason, by=by)
        self._audit("kill_switch_engaged", reason=reason, by=by)

    def release_kill_switch(self, *, by: str | None = None) -> None:
        status = self.kill_switch.release()
        self._audit("kill_switch_released", by=by, still_engaged_by=status.source)

    # -- executing approved actions ----------------------------------------------------

    def run_approved(self) -> list[Outcome]:
        """Execute every approved action (oldest first), each re-validated first."""
        self.expire_stale()
        approved = self.store.list_actions([ActionState.APPROVED])
        return [self.execute_approved(rec.id) for rec in approved]

    def _expire_if_stale(
        self, rec: ActionRecord, tool: ToolPolicy | None, now: float
    ) -> Outcome | None:
        """Expire an approved action whose time is up; ``None`` if it is still valid.

        The current policy's TTL applies too, so shortening it takes effect on
        approvals already given; lengthening it never revives an old one.
        """
        expires_at = rec.expires_at
        if tool is not None and expires_at is not None:
            expires_at = min(expires_at, rec.created_at + tool.approval_ttl_seconds)
        if expires_at is None or expires_at > now:
            return None
        with self.store.transaction() as tx:
            expired = tx.update(
                rec.id,
                expect=(ActionState.APPROVED,),
                status=ActionState.EXPIRED,
                reason=str(Reason.APPROVAL_EXPIRED),
                finished_at=now,
            )
        if expired:
            self._audit(
                "expired", tool=rec.tool, agent_id=rec.agent_id, action_id=rec.id, was=rec.status
            )
        msg = f"the approval expired at {_iso(expires_at)}; request the action again."
        return Outcome(
            Status.BLOCKED, rec.tool, msg, action_id=rec.id, reason=Reason.APPROVAL_EXPIRED
        )

    def execute_approved(self, action_id: str) -> Outcome:
        """Re-validate an approved action against *current* conditions, then run it.

        Why re-validate? Time passes between approval and execution. The policy may
        have been tightened, the allow-list changed, the budget used up by other
        actions, the kill switch pulled, or the approval may have gone stale. The
        human approved the action in the world as it was; it runs only if it is
        still acceptable in the world as it is (time-of-check vs time-of-use).
        """
        rec = self.store.get(action_id)
        policy, policy_hash = self._active
        tool = policy.tool(rec.tool)
        mode = policy.mode_for(rec.tool)
        hidden: set[str] = set()
        logged = self._redactor(policy, tool).redact(rec.args, hidden)
        ctx: dict[str, Any] = {
            "tool": rec.tool,
            "agent_id": rec.agent_id,
            "mode": str(mode),
            "policy_hash": policy_hash,
        }
        now = self._now()

        def close(reason: Reason, message: str) -> Outcome:
            """Permanently stop this approved action (it will never run)."""
            with self.store.transaction() as tx:
                closed = tx.update(
                    action_id,
                    expect=(ActionState.APPROVED,),
                    status=ActionState.BLOCKED,
                    reason=str(reason),
                    finished_at=now,
                )
            if not closed:  # another worker claimed (or someone rejected) it meanwhile
                status = self.store.get(action_id).status
                msg = (
                    f"action {action_id} is now {status}, not approved; this worker did not run it."
                )
                return self._block(ctx, Reason.NOT_APPROVED, msg, logged, action_id)
            return self._block(ctx, reason, message, logged, action_id, hidden=hidden)

        if rec.status is not ActionState.APPROVED:
            msg = f"action {action_id} is {rec.status}, not approved; it was not run."
            return self._block(ctx, Reason.NOT_APPROVED, msg, logged, action_id)
        ks = self.kill_switch.status()
        if ks.engaged:
            msg = f"the kill switch is engaged ({ks.reason}); the action stays approved but unrun."
            return self._block(ctx, Reason.KILL_SWITCH, msg, logged, action_id)
        stale = self._expire_if_stale(rec, tool, now)
        if stale is not None:
            return stale
        if digest(rec.args) != rec.args_digest:
            return close(
                Reason.ARGUMENTS_CHANGED,
                "the stored arguments no longer match what was queued and approved.",
            )
        if mode not in (Mode.ALLOW, Mode.APPROVE):
            return close(
                Reason.POLICY_CHANGED, f"'{rec.tool}' is now in '{mode}' mode under current policy."
            )
        reg = self._lookup(rec.tool)
        if reg is None:
            msg = f"no function is registered for '{rec.tool}' in this process; it stays approved."
            return self._block(ctx, Reason.NOT_REGISTERED, msg, logged, action_id)
        effective = tool if tool is not None else ToolPolicy(mode=mode)
        prepared = self._prepare(ctx, policy, effective, reg, rec.args)
        if isinstance(prepared, Violation):
            return close(prepared.reason, f"{prepared.detail} (checked against current policy)")
        if prepared.args_digest != rec.args_digest:
            return close(
                Reason.POLICY_CHANGED,
                "the current policy would change the approved arguments; request it again.",
            )

        found: Violation | ActionRecord | None = None
        claimed = False
        with self.store.transaction() as tx:
            if tx.get(action_id).status is ActionState.APPROVED:
                found = self._check_limits(tx, prepared, now, exclude_id=action_id)
                if found is None:
                    claimed = tx.update(
                        action_id,
                        expect=(ActionState.APPROVED,),
                        status=ActionState.EXECUTING,
                        started_at=now,
                        cost=prepared.cost,
                        policy_hash=policy_hash,
                    )
                elif isinstance(found, ActionRecord) or found.reason not in TRANSIENT_REASONS:
                    # Close it in the same transaction as the check: done afterwards, it
                    # could overwrite the claim of a worker that is now running it.
                    reason = "duplicate" if isinstance(found, ActionRecord) else str(found.reason)
                    tx.update(
                        action_id,
                        expect=(ActionState.APPROVED,),
                        status=ActionState.BLOCKED,
                        reason=reason,
                        finished_at=now,
                    )
        if isinstance(found, ActionRecord):
            return self._duplicate(ctx, found, logged)
        if isinstance(found, Violation):
            if found.reason in TRANSIENT_REASONS:
                return self._block(
                    ctx, found.reason, f"{found.detail} It stays approved.", logged, action_id
                )
            return self._block(ctx, found.reason, found.detail, logged, action_id, hidden=hidden)
        if not claimed:
            msg = f"action {action_id} was claimed by another worker; it was not run twice."
            return Outcome(Status.DUPLICATE, rec.tool, msg, action_id=action_id)
        return self._run(prepared, action_id, reg, approved_by=rec.decided_by or "unknown")


# -- functions shared with the CLI (which has no policy or tool functions) -----------


def approve_action(
    store: ActionStore,
    audit: AuditLog,
    action_id: str,
    *,
    by: str,
    now: float,
    note: str | None = None,
) -> ActionRecord:
    try:
        rec = store.approve(action_id, by=by, now=now, note=note)
    except ApprovalExpiredError:
        rec = store.get(action_id)
        audit.append("expired", tool=rec.tool, agent_id=rec.agent_id, action_id=rec.id)
        raise
    fields = {"tool": rec.tool, "agent_id": rec.agent_id, "action_id": rec.id, "by": by}
    extra = {"note": note} if note else {}
    audit.append("approved", **fields, args_digest=rec.args_digest, **extra)
    return rec


def reject_action(
    store: ActionStore,
    audit: AuditLog,
    action_id: str,
    *,
    by: str,
    now: float,
    note: str | None = None,
) -> ActionRecord:
    before = store.get(action_id).status
    rec = store.reject(action_id, by=by, now=now, note=note)
    fields = {"tool": rec.tool, "agent_id": rec.agent_id, "action_id": rec.id, "by": by}
    extra = {"note": note} if note else {}
    audit.append("rejected", **fields, was=str(before), **extra)
    return rec


# -- module-level decorator and default guard ------------------------------------------


def set_default_guard(guard: Guard | None) -> None:
    """Set the guard used by functions decorated with :func:`guarded`."""
    global _default_guard  # noqa: PLW0603
    _default_guard = guard


def get_default_guard() -> Guard:
    if _default_guard is None:
        raise ConfigurationError(
            "no default guard: call set_default_guard(Guard(policy)) before using @guarded tools"
        )
    return _default_guard


def guarded(
    name: str | None = None, *, guard: Guard | None = None, preview: PreviewFn | None = None
) -> Callable[[Callable[..., Any]], GuardedTool]:
    """Decorator: route every call of the function through a guard.

    ``@guarded("send_email")`` uses the default guard (see :func:`set_default_guard`),
    resolved at call time, so tool modules can be imported before the guard exists.
    """

    def decorator(fn: Callable[..., Any]) -> GuardedTool:
        tool_name = name or fn.__name__
        if guard is not None:
            return guard.wrap(fn, name=tool_name, preview=preview)
        _GLOBAL_TOOLS[tool_name] = _register(fn, preview)
        return GuardedTool(tool_name, fn, None)

    return decorator
