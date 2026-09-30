"""Command-line interface for the people who supervise agents.

The CLI works on the same files as the application (queue database, audit log,
kill-switch flag) under ``--home`` / ``$AGENT_GUARDRAILS_HOME`` /
``./.agent-guardrails``. It records human decisions; it never runs tools. Approved
actions are executed by the application, which re-validates them first.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .audit import AuditLog, verify_log
from .errors import GuardrailError
from .guard import HOME_ENV, approve_action, default_home, reject_action
from .killswitch import KillSwitch
from .store import ActionRecord, ActionState, ActionStore


def _iso(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%SZ")


def _record_dict(rec: ActionRecord) -> dict[str, Any]:
    return {
        "id": rec.id,
        "status": str(rec.status),
        "tool": rec.tool,
        "agent_id": rec.agent_id,
        "args": rec.args,
        "args_digest": rec.args_digest,
        "cost": rec.cost,
        "created_at": _iso(rec.created_at),
        "expires_at": _iso(rec.expires_at),
        "decided_by": rec.decided_by,
        "decided_at": _iso(rec.decided_at),
        "decision_note": rec.decision_note,
        "reason": rec.reason,
        "policy_hash": rec.policy_hash,
    }


def _default_user() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):  # no passwd entry, e.g. in some containers
        return "unknown"


class _Ctx:
    def __init__(self, home: Path) -> None:
        self.home = home

    @property
    def store(self) -> ActionStore:
        return ActionStore(self.home / "queue.db")

    @property
    def audit(self) -> AuditLog:
        return AuditLog(self.home / "audit.jsonl")

    @property
    def kill(self) -> KillSwitch:
        return KillSwitch(flag_file=self.home / "KILL")

    def expire(self, store: ActionStore, audit: AuditLog) -> None:
        for rec in store.expire_stale(time.time()):
            audit.append(
                "expired", tool=rec.tool, agent_id=rec.agent_id, action_id=rec.id, was=rec.status
            )


def _cmd_queue_list(ctx: _Ctx, args: argparse.Namespace) -> int:
    store, audit = ctx.store, ctx.audit
    ctx.expire(store, audit)
    statuses = None if args.status == "all" else [ActionState(args.status)]
    records = store.list_actions(statuses)
    if args.json:
        print(json.dumps([_record_dict(r) for r in records], indent=2, ensure_ascii=False))
        return 0
    if not records:
        print(f"No {args.status} actions.")
        return 0
    header = f"{'ID':<22} {'STATUS':<9} {'TOOL':<18} {'AGENT':<12} {'EXPIRES (UTC)':<21} ARGS"
    print(header)
    for r in records:
        preview = json.dumps(r.args, ensure_ascii=False)
        if len(preview) > 48:
            preview = preview[:45] + "..."
        print(
            f"{r.id:<22} {r.status!s:<9} {r.tool:<18} {r.agent_id:<12} "
            f"{_iso(r.expires_at):<21} {preview}"
        )
    print("\nUse 'queue show <id>' to see the full arguments before approving.")
    return 0


def _cmd_queue_show(ctx: _Ctx, args: argparse.Namespace) -> int:
    store = ctx.store
    rec = store.get(store.resolve_id(args.id))
    print(json.dumps(_record_dict(rec), indent=2, ensure_ascii=False))
    return 0


def _cmd_queue_decide(ctx: _Ctx, args: argparse.Namespace) -> int:
    store, audit = ctx.store, ctx.audit
    action_id = store.resolve_id(args.id)
    if args.queue_cmd == "approve":
        rec = approve_action(store, audit, action_id, by=args.by, now=time.time(), note=args.note)
        print(f"Approved {rec.id} ({rec.tool}) as {args.by}.")
        print("It runs when the application next processes approved actions, after re-checks.")
    else:
        rec = reject_action(store, audit, action_id, by=args.by, now=time.time(), note=args.reason)
        print(f"Rejected {rec.id} ({rec.tool}) as {args.by}.")
    return 0


def _cmd_audit_verify(ctx: _Ctx, args: argparse.Namespace) -> int:
    path = Path(args.path) if args.path else ctx.home / "audit.jsonl"
    result = verify_log(path, expected_head=args.expected_head, expected_count=args.expected_count)
    if result.ok:
        print(f"OK: {result.records} records, chain intact. Head: {result.head}")
        if args.expected_head is None:
            print(
                "Note: without --expected-head, truncation of the newest records is not detected."
            )
        return 0
    where = f" at line {result.line}" if result.line else ""
    print(f"FAILED{where}: {result.error} ({result.records} records verified before this)")
    return 1


def _cmd_audit_head(ctx: _Ctx, args: argparse.Namespace) -> int:
    path = Path(args.path) if args.path else ctx.home / "audit.jsonl"
    count, head = AuditLog(path).head()
    print(json.dumps({"records": count, "head": head}))
    return 0


def _cmd_kill(ctx: _Ctx, args: argparse.Namespace) -> int:
    kill = ctx.kill
    if args.kill_cmd == "on":
        kill.engage(args.reason, by=args.by)
        ctx.audit.append("kill_switch_engaged", reason=args.reason, by=args.by)
        print(f"Kill switch ENGAGED ({kill.flag_file}). All guarded actions are blocked.")
        return 0
    if args.kill_cmd == "off":
        status = kill.release()
        ctx.audit.append("kill_switch_released", by=args.by)
        if status.engaged:
            print(f"Flag file removed, but the switch is still engaged via {status.source}.")
            return 1
        print(
            "Kill switch released. Review approved actions before they run: queue list -s approved"
        )
        return 0
    status = kill.status()
    if status.engaged:
        print(f"ENGAGED via {status.source}: {status.reason}")
    else:
        print("not engaged")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-guardrails",
        description="Supervise guarded agent actions: approval queue, audit log, kill switch.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help=f"state directory (default: ${HOME_ENV} or ./.agent-guardrails)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    queue = sub.add_parser("queue", help="list, inspect, approve or reject queued actions")
    qsub = queue.add_subparsers(dest="queue_cmd", required=True)
    qlist = qsub.add_parser("list", help="list actions (pending by default)")
    qlist.add_argument(
        "-s",
        "--status",
        default="pending",
        choices=[*(s.value for s in ActionState), "all"],
    )
    qlist.add_argument("--json", action="store_true")
    qlist.set_defaults(func=_cmd_queue_list)
    qshow = qsub.add_parser("show", help="show one action in full")
    qshow.add_argument("id", help="action id or a unique prefix")
    qshow.set_defaults(func=_cmd_queue_show)
    for name, text in (("approve", "approve a pending action"), ("reject", "reject one")):
        p = qsub.add_parser(name, help=text)
        p.add_argument("id", help="action id or a unique prefix")
        p.add_argument("--by", default=_default_user(), help="who is deciding (default: $USER)")
        if name == "approve":
            p.add_argument("--note", default=None)
        else:
            p.add_argument("--reason", default=None)
        p.set_defaults(func=_cmd_queue_decide)

    audit = sub.add_parser("audit", help="verify the audit log's hash chain")
    asub = audit.add_subparsers(dest="audit_cmd", required=True)
    averify = asub.add_parser("verify", help="recompute the hash chain")
    averify.add_argument("path", nargs="?", help="log path (default: <home>/audit.jsonl)")
    averify.add_argument("--expected-head", default=None, help="anchor hash kept elsewhere")
    averify.add_argument("--expected-count", type=int, default=None)
    averify.set_defaults(func=_cmd_audit_verify)
    ahead = asub.add_parser("head", help="print record count and head hash, for anchoring")
    ahead.add_argument("path", nargs="?")
    ahead.set_defaults(func=_cmd_audit_head)

    kill = sub.add_parser("kill", help="engage, release or inspect the kill switch")
    ksub = kill.add_subparsers(dest="kill_cmd", required=True)
    kon = ksub.add_parser("on", help="block every guarded action")
    kon.add_argument("--reason", default="engaged from CLI")
    kon.add_argument("--by", default=_default_user())
    kon.set_defaults(func=_cmd_kill)
    koff = ksub.add_parser("off", help="release the switch")
    koff.add_argument("--by", default=_default_user())
    koff.set_defaults(func=_cmd_kill)
    ksub.add_parser("status", help="show whether the switch is engaged").set_defaults(
        func=_cmd_kill
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ctx = _Ctx(args.home if args.home is not None else default_home())
    try:
        code: int = args.func(ctx, args)
    except GuardrailError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
