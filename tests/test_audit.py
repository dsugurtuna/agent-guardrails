from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from agent_guardrails import AuditIntegrityError, AuditLog, verify_log
from agent_guardrails.audit import GENESIS_HASH, record_hash


def _log(tmp_path: Path, n: int = 5) -> AuditLog:
    log = AuditLog(tmp_path / "audit.jsonl", fsync=False)
    for i in range(n):
        log.append("executed", tool="t", n=i)
    return log


def test_chain_structure(tmp_path: Path) -> None:
    log = _log(tmp_path, 3)
    records = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert [r["seq"] for r in records] == [0, 1, 2]
    assert records[0]["prev_hash"] == GENESIS_HASH
    for prev, cur in itertools.pairwise(records):
        assert cur["prev_hash"] == prev["hash"]
    for r in records:
        assert r["hash"] == record_hash(r)
    result = verify_log(log.path)
    assert result.ok and result.records == 3 and result.head == records[-1]["hash"]
    assert log.head() == (3, records[-1]["hash"])


def test_empty_and_missing(tmp_path: Path) -> None:
    missing = verify_log(tmp_path / "nope.jsonl")
    assert not missing.ok
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert verify_log(empty).ok
    assert AuditLog(tmp_path / "nope.jsonl").head() == (0, GENESIS_HASH)


def test_edit_is_detected(tmp_path: Path) -> None:
    log = _log(tmp_path)
    lines = log.path.read_text().splitlines()
    lines[2] = lines[2].replace('"n":2', '"n":20')
    log.path.write_text("\n".join(lines) + "\n")
    result = verify_log(log.path)
    assert not result.ok
    assert result.line == 3
    assert "hash does not match" in (result.error or "")


def test_deletion_and_reordering_are_detected(tmp_path: Path) -> None:
    log = _log(tmp_path)
    lines = log.path.read_text().splitlines()
    deleted = lines[:1] + lines[2:]
    log.path.write_text("\n".join(deleted) + "\n")
    assert not verify_log(log.path).ok
    swapped = [lines[0], lines[2], lines[1], *lines[3:]]
    log.path.write_text("\n".join(swapped) + "\n")
    assert not verify_log(log.path).ok


def test_truncation_needs_an_anchor(tmp_path: Path) -> None:
    log = _log(tmp_path)
    count, head = log.head()
    lines = log.path.read_text().splitlines()
    log.path.write_text("\n".join(lines[:-1]) + "\n")
    assert verify_log(log.path).ok  # the chain alone cannot see a cut tail
    assert not verify_log(log.path, expected_head=head).ok
    assert not verify_log(log.path, expected_count=count).ok


def test_partial_last_line(tmp_path: Path) -> None:
    log = _log(tmp_path, 2)
    with log.path.open("a") as f:
        f.write('{"seq": 2, "trunc')
    result = verify_log(log.path)
    assert not result.ok and "incomplete" in (result.error or "")
    with pytest.raises(AuditIntegrityError):
        log.append("executed")


def test_refuses_to_chain_onto_a_tampered_tail(tmp_path: Path) -> None:
    log = _log(tmp_path, 2)
    lines = log.path.read_text().splitlines()
    lines[-1] = lines[-1].replace('"tool":"t"', '"tool":"x"')
    log.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(AuditIntegrityError):
        log.append("executed")


def test_non_json_line(tmp_path: Path) -> None:
    log = _log(tmp_path, 2)
    with log.path.open("a") as f:
        f.write("not json\n")
    assert verify_log(log.path).line == 3


def test_reserved_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved"):
        AuditLog(tmp_path / "a.jsonl").append("x", hash="0")


def test_key_order_in_file_does_not_matter(tmp_path: Path) -> None:
    log = _log(tmp_path, 2)
    lines = log.path.read_text().splitlines()
    record = json.loads(lines[0])
    lines[0] = json.dumps(dict(reversed(list(record.items()))))
    log.path.write_text("\n".join(lines) + "\n")
    assert verify_log(log.path).ok


def test_unicode_line_separators_cannot_split_a_record(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", fsync=False)
    log.append("executed", detail="a\u2028b\x85c\u2029d", **{"k\x85": "v"})
    text = log.path.read_text(encoding="utf-8")
    assert text.isascii()
    assert len(text.splitlines()) == 1
    assert json.loads(text)["detail"] == "a\u2028b\x85c\u2029d"
    assert verify_log(log.path).ok


def test_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    # Python's json keeps the *last* duplicate, so a forged first value leaves the
    # parsed record (and its hash) unchanged while a person or grep reads the forgery.
    log = _log(tmp_path, 2)
    lines = log.path.read_text().splitlines()
    assert '"event":"executed"' in lines[0]
    lines[0] = '{"event":"approved_by_ceo",' + lines[0][1:]
    log.path.write_text("\n".join(lines) + "\n")
    result = verify_log(log.path)
    assert not result.ok
    assert result.line == 1
    assert "duplicate key" in (result.error or "")
    lines = log.path.read_text().splitlines()
    lines[0], lines[1] = (
        lines[0].replace('"event":"approved_by_ceo",', ""),
        ('{"n":7,' + lines[1][1:]),
    )
    log.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(AuditIntegrityError, match="duplicate key"):
        log.append("executed")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_reported_not_raised(tmp_path: Path, constant: str) -> None:
    log = _log(tmp_path, 2)
    lines = log.path.read_text().splitlines()
    lines[0] = lines[0].replace('"n":0', f'"n":{constant}')
    lines[1] = lines[1].replace('"n":1', f'"n":{constant}')
    log.path.write_text("\n".join(lines) + "\n")
    result = verify_log(log.path)  # must return a failure, not raise ValueError
    assert not result.ok and result.line == 1
    with pytest.raises(AuditIntegrityError):
        log.append("executed")


def test_anchor_survives_honest_growth_but_not_truncation(tmp_path: Path) -> None:
    # The realistic use of an anchor: record (count, head) somewhere else today,
    # check tomorrow that the log still starts with exactly those records.
    log = _log(tmp_path, 5)
    anchor = log.head()
    for i in range(3):
        log.append("executed", tool="t", n=100 + i)  # honest growth since the anchor
    assert verify_log(log.path, anchor=anchor).ok
    assert log.verify(anchor=anchor).ok
    assert not verify_log(log.path, expected_head=anchor[1]).ok  # exact check: log grew

    lines = log.path.read_text().splitlines()
    log.path.write_text("\n".join(lines[:4]) + "\n")  # cut back past the anchor
    cut = verify_log(log.path, anchor=anchor)
    assert not cut.ok and "anchor" in (cut.error or "")
    for i in range(6):  # ...and the writer carries on, so the log is longer again
        log.append("executed", tool="t", n=200 + i)
    assert verify_log(log.path).ok  # the chain alone cannot tell
    regrown = verify_log(log.path, anchor=anchor)
    assert not regrown.ok and "anchor" in (regrown.error or "")


def test_anchor_detects_a_rewritten_prefix(tmp_path: Path) -> None:
    log = _log(tmp_path, 3)
    anchor = log.head()
    records = [json.loads(line) for line in log.path.read_text().splitlines()]
    records[0]["n"] = 99
    prev = GENESIS_HASH
    for record in records:  # a careful forger recomputes every hash
        record["prev_hash"] = prev
        record["hash"] = record_hash(record)
        prev = record["hash"]
    log.path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert verify_log(log.path).ok
    assert not verify_log(log.path, anchor=anchor).ok
    assert verify_log(log.path, anchor=(0, GENESIS_HASH)).ok
    assert not verify_log(log.path, anchor=(0, "f" * 64)).ok
