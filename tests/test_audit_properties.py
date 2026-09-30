"""Property-based tests: any change to the audit log's content is detected.

Mutations generated: edit a field, edit a field *and* recompute that record's hash
(a careful forger), delete a record, swap two records, duplicate (replay) a record,
insert a forged record with a valid-looking hash, and recompute the whole chain.
Changes that end at the tail, and whole-chain rewrites, are only detectable with
an anchor (``expected_head``), which is exactly what the docs say.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_guardrails.audit import GENESIS_HASH, AuditLog, record_hash, verify_log

json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**6), max_value=10**6),
    st.text(max_size=20),
)
fields = st.dictionaries(
    st.sampled_from(["tool", "agent_id", "detail", "amount", "args"]),
    st.one_of(json_scalars, st.dictionaries(st.text(max_size=5), json_scalars, max_size=3)),
    max_size=4,
)
events = st.lists(
    st.tuples(st.sampled_from(["allowed", "executed", "queued", "blocked"]), fields),
    min_size=1,
    max_size=12,
)


def _write(tmp: Path, evts: list[tuple[str, dict[str, Any]]]) -> tuple[Path, list[str], str]:
    log = AuditLog(tmp / "audit.jsonl", fsync=False)
    for event, extra in evts:
        log.append(event, **extra)
    count, head = log.head()
    assert count == len(evts)
    return log.path, log.path.read_text(encoding="utf-8").splitlines(), head


def _save(path: Path, lines: list[str]) -> None:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _rehash_from(records: list[dict[str, Any]], start: int) -> None:
    prev = records[start - 1]["hash"] if start > 0 else GENESIS_HASH
    for i in range(start, len(records)):
        records[i]["seq"] = i
        records[i]["prev_hash"] = prev
        records[i]["hash"] = record_hash(records[i])
        prev = records[i]["hash"]


settings_ = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@settings_
@given(evts=events, data=st.data())
def test_any_field_edit_is_detected(evts: list[tuple[str, dict[str, Any]]], data: Any) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path, lines, head = _write(Path(tmp), evts)
        i = data.draw(st.integers(0, len(lines) - 1))
        record = json.loads(lines[i])
        key = data.draw(st.sampled_from(sorted(record)))
        new_value = data.draw(json_scalars.filter(lambda v: v != record[key]))
        record[key] = new_value
        recompute_own_hash = data.draw(st.booleans()) and key != "hash"
        if recompute_own_hash:
            record["hash"] = record_hash(record)
        lines[i] = json.dumps(record)
        _save(path, lines)

        assert not verify_log(path, expected_head=head).ok
        if not recompute_own_hash or i < len(lines) - 1:
            assert not verify_log(path).ok  # detectable without any anchor


@settings_
@given(evts=events.filter(lambda e: len(e) >= 2), data=st.data())
def test_deletion_is_detected(evts: list[tuple[str, dict[str, Any]]], data: Any) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path, lines, head = _write(Path(tmp), evts)
        i = data.draw(st.integers(0, len(lines) - 1))
        del lines[i]
        _save(path, lines)
        assert not verify_log(path, expected_head=head, expected_count=len(evts)).ok
        if i < len(lines):  # anything but the newest record
            assert not verify_log(path).ok


@settings_
@given(evts=events.filter(lambda e: len(e) >= 2), data=st.data())
def test_reordering_is_detected(evts: list[tuple[str, dict[str, Any]]], data: Any) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path, lines, _ = _write(Path(tmp), evts)
        i, j = data.draw(
            st.tuples(st.integers(0, len(lines) - 1), st.integers(0, len(lines) - 1)).filter(
                lambda p: p[0] != p[1]
            )
        )
        lines[i], lines[j] = lines[j], lines[i]
        _save(path, lines)
        assert not verify_log(path).ok


@settings_
@given(evts=events, data=st.data())
def test_replay_and_forged_insertions_are_detected(
    evts: list[tuple[str, dict[str, Any]]], data: Any
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path, lines, head = _write(Path(tmp), evts)
        records = [json.loads(line) for line in lines]
        at = data.draw(st.integers(0, len(lines)))
        if data.draw(st.booleans()):
            forged = dict(records[data.draw(st.integers(0, len(records) - 1))])  # replay
        else:
            prev = records[at - 1]["hash"] if at > 0 else GENESIS_HASH
            forged = {
                "seq": at,
                "ts": "2030-01-01T00:00:00Z",
                "event": "executed",
                "tool": "wire_money",
                "prev_hash": prev,
            }
            forged["hash"] = record_hash(forged)  # a well-formed record in its own right
        lines.insert(at, json.dumps(forged))
        _save(path, lines)
        assert not verify_log(path, expected_head=head).ok


@settings_
@given(evts=events, data=st.data())
def test_full_rewrite_needs_the_anchor(evts: list[tuple[str, dict[str, Any]]], data: Any) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path, lines, head = _write(Path(tmp), evts)
        records = [json.loads(line) for line in lines]
        i = data.draw(st.integers(0, len(records) - 1))
        records[i]["event"] = "rewritten-" + str(records[i]["event"])
        _rehash_from(records, i)  # attacker recomputes every later hash
        _save(path, [json.dumps(r) for r in records])
        assert verify_log(path).ok  # the chain alone is consistent again...
        assert not verify_log(path, expected_head=head).ok  # ...but the anchor catches it


@settings_
@given(evts=events)
def test_untouched_log_always_verifies(evts: list[tuple[str, dict[str, Any]]]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path, _, head = _write(Path(tmp), evts)
        result = verify_log(path, expected_head=head, expected_count=len(evts))
        assert result.ok and result.records == len(evts)
