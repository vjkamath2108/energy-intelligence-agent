"""Tests for src/audit/audit_log.py

Coverage:
- entry_from_recommendation: fields mapped correctly from DispatchRecommendation
- AuditLogger.record: appends to file, creates parent dirs
- AuditLogger.read_all: returns entries in order; empty list when file absent
- Round-trip: serialise → deserialise preserves all fields including timezone
- Append-only: multiple records accumulate; earlier entries are not overwritten
- Immutability: AuditLogEntry is frozen — assignment raises
- Concurrency: concurrent record() calls do not corrupt the log
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.audit.audit_log import AuditLogger, entry_from_recommendation
from src.models import AuditLogEntry, DispatchRecommendation, RetrievedDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(doc_id: str = "doc-1", similarity: float = 0.85) -> RetrievedDocument:
    return RetrievedDocument(
        doc_id=doc_id,
        content="upward balancing constraint dispatch",
        source="regulation/dno-p28.pdf",
        chunk_index=0,
        similarity_score=similarity,
        metadata={},
    )


def _recommendation(
    action: str = "Issue upward dispatch to DRAX1G for 200 MW",
    confidence: float = 0.83,
    hallucination_flagged: bool = False,
    doc_ids: list[str] | None = None,
) -> DispatchRecommendation:
    docs = [_doc(did) for did in (doc_ids or ["doc-1"])]
    return DispatchRecommendation(
        query="What action should we take on the North constraint?",
        action=action,
        confidence=confidence,
        supporting_documents=docs,
        reasoning="North region imbalance confirmed by retrieved dispatch document.",
        requires_human_review=False,
        hallucination_flagged=hallucination_flagged,
    )


# ---------------------------------------------------------------------------
# entry_from_recommendation
# ---------------------------------------------------------------------------


def test_entry_maps_recommendation_id() -> None:
    rec = _recommendation()
    entry = entry_from_recommendation(rec)
    assert entry.recommendation_id == rec.recommendation_id


def test_entry_maps_query() -> None:
    rec = _recommendation()
    entry = entry_from_recommendation(rec)
    assert entry.query == rec.query


def test_entry_maps_action() -> None:
    rec = _recommendation()
    entry = entry_from_recommendation(rec)
    assert entry.action_taken == rec.action


def test_entry_maps_confidence() -> None:
    rec = _recommendation(confidence=0.72)
    entry = entry_from_recommendation(rec)
    assert entry.confidence == pytest.approx(0.72)


def test_entry_maps_hallucination_result_false() -> None:
    rec = _recommendation(hallucination_flagged=False)
    entry = entry_from_recommendation(rec)
    assert entry.hallucination_detection_result is False


def test_entry_maps_hallucination_result_true() -> None:
    rec = _recommendation(hallucination_flagged=True)
    entry = entry_from_recommendation(rec)
    assert entry.hallucination_detection_result is True


def test_entry_maps_retrieved_doc_ids() -> None:
    rec = _recommendation(doc_ids=["doc-a", "doc-b", "doc-c"])
    entry = entry_from_recommendation(rec)
    assert entry.retrieved_doc_ids == ["doc-a", "doc-b", "doc-c"]


def test_entry_has_utc_timestamp() -> None:
    entry = entry_from_recommendation(_recommendation())
    assert entry.timestamp.tzinfo is not None
    assert entry.timestamp.tzinfo == timezone.utc


def test_entry_is_frozen() -> None:
    entry = entry_from_recommendation(_recommendation())
    with pytest.raises((AttributeError, TypeError, ValidationError)):
        entry.action_taken = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AuditLogger.record + read_all
# ---------------------------------------------------------------------------


async def test_record_creates_file(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    al = AuditLogger(log_path)
    entry = entry_from_recommendation(_recommendation())

    await al.record(entry)

    assert log_path.exists()


async def test_record_creates_parent_dirs(tmp_path: Path) -> None:
    log_path = tmp_path / "deep" / "nested" / "audit.jsonl"
    al = AuditLogger(log_path)
    await al.record(entry_from_recommendation(_recommendation()))
    assert log_path.exists()


async def test_read_all_empty_when_file_absent(tmp_path: Path) -> None:
    al = AuditLogger(tmp_path / "nonexistent.jsonl")
    assert await al.read_all() == []


async def test_record_then_read_roundtrip(tmp_path: Path) -> None:
    al = AuditLogger(tmp_path / "audit.jsonl")
    rec = _recommendation()
    entry = entry_from_recommendation(rec)

    await al.record(entry)
    entries = await al.read_all()

    assert len(entries) == 1
    restored = entries[0]
    assert restored.entry_id == entry.entry_id
    assert restored.recommendation_id == entry.recommendation_id
    assert restored.query == entry.query
    assert restored.action_taken == entry.action_taken
    assert restored.confidence == pytest.approx(entry.confidence)
    assert restored.hallucination_detection_result == entry.hallucination_detection_result
    assert restored.retrieved_doc_ids == entry.retrieved_doc_ids


async def test_timestamp_timezone_survives_roundtrip(tmp_path: Path) -> None:
    al = AuditLogger(tmp_path / "audit.jsonl")
    entry = entry_from_recommendation(_recommendation())
    await al.record(entry)

    restored = (await al.read_all())[0]
    assert restored.timestamp == entry.timestamp
    assert restored.timestamp.tzinfo is not None


async def test_append_only_multiple_records(tmp_path: Path) -> None:
    """Three records accumulate; earlier entries are not overwritten."""
    al = AuditLogger(tmp_path / "audit.jsonl")
    entries = [entry_from_recommendation(_recommendation(action=f"action-{i}")) for i in range(3)]

    for e in entries:
        await al.record(e)

    all_entries = await al.read_all()
    assert len(all_entries) == 3
    assert [e.action_taken for e in all_entries] == ["action-0", "action-1", "action-2"]


async def test_second_logger_instance_appends_not_overwrites(tmp_path: Path) -> None:
    """A fresh AuditLogger opened on the same path must not truncate existing data."""
    log_path = tmp_path / "audit.jsonl"

    al1 = AuditLogger(log_path)
    await al1.record(entry_from_recommendation(_recommendation(action="first")))

    al2 = AuditLogger(log_path)
    await al2.record(entry_from_recommendation(_recommendation(action="second")))

    entries = await al2.read_all()
    assert len(entries) == 2
    assert entries[0].action_taken == "first"
    assert entries[1].action_taken == "second"


async def test_each_line_is_valid_json(tmp_path: Path) -> None:
    """JSONL format: every line in the file must be independently parseable JSON."""
    log_path = tmp_path / "audit.jsonl"
    al = AuditLogger(log_path)

    for i in range(5):
        await al.record(entry_from_recommendation(_recommendation(action=f"action-{i}")))

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line in lines:
        parsed = json.loads(line)
        assert "entry_id" in parsed
        assert "recommendation_id" in parsed
        assert "timestamp" in parsed


async def test_concurrent_writes_do_not_corrupt(tmp_path: Path) -> None:
    """20 concurrent record() calls must all land as separate valid lines."""
    log_path = tmp_path / "audit.jsonl"
    al = AuditLogger(log_path)

    entries = [
        entry_from_recommendation(_recommendation(action=f"concurrent-{i}"))
        for i in range(20)
    ]
    await asyncio.gather(*(al.record(e) for e in entries))

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20

    # Every line must be valid JSON with a recognisable entry_id
    recovered_ids = set()
    for line in lines:
        obj = json.loads(line)
        recovered_ids.add(obj["entry_id"])

    assert recovered_ids == {e.entry_id for e in entries}
