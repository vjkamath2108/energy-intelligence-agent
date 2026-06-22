"""Immutable append-only audit log for EIA dispatch decisions.

Every DispatchRecommendation that leaves the pipeline is recorded here before
being surfaced to the operator. Entries are serialised as JSONL (one JSON
object per line) for compliance review and incident reconstruction.

Append-only invariants enforced at every layer:
  - File is opened exclusively in "a" mode — writes never overwrite existing data.
  - No delete, truncate, or seek-to-start operations are exposed by this module.
  - AuditLogEntry is frozen=True (Pydantic v2) — immutable after creation.
  - asyncio.Lock serialises concurrent writes within a single process to prevent
    interleaved JSON lines from corrupting the log.

Storage format: JSONL — one compact JSON object per line, UTF-8 encoded.
Path: AUDIT_LOG_PATH env var, default logs/audit.jsonl.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from src.models import AuditLogEntry, DispatchRecommendation

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path(os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl"))


def entry_from_recommendation(rec: DispatchRecommendation) -> AuditLogEntry:
    """Create an AuditLogEntry from a completed DispatchRecommendation.

    Captures the full decision trace: query, retrieved doc IDs, confidence,
    hallucination detection outcome, and the action surfaced to the operator.

    Args:
        rec: The recommendation produced by ClaudeRecommender or OllamaRecommender.

    Returns:
        Immutable AuditLogEntry ready to be passed to AuditLogger.record().
    """
    return AuditLogEntry(
        recommendation_id=rec.recommendation_id,
        query=rec.query,
        retrieved_doc_ids=[d.doc_id for d in rec.supporting_documents],
        confidence=rec.confidence,
        hallucination_detection_result=rec.hallucination_flagged,
        action_taken=rec.action,
    )


class AuditLogger:
    """Append-only JSONL audit logger for dispatch decisions.

    Thread-safe within a single asyncio event loop via an internal Lock.
    File I/O runs in a thread-pool executor to avoid blocking the loop.

    Args:
        log_path: Path to the JSONL audit log file. Parent directories are
                  created automatically on first write. Defaults to the
                  AUDIT_LOG_PATH env var or logs/audit.jsonl.

    Usage:
        logger = AuditLogger()
        entry = entry_from_recommendation(recommendation)
        await logger.record(entry)
        all_entries = await logger.read_all()   # compliance review
    """

    def __init__(self, log_path: Path | str = _DEFAULT_LOG_PATH) -> None:
        self._path = Path(log_path)
        self._lock = asyncio.Lock()

    async def record(self, entry: AuditLogEntry) -> None:
        """Append one entry to the audit log as a single JSON line.

        Creates the log file and any missing parent directories on first call.
        The write is atomic at the line level: a complete JSON object followed
        by a newline is always written before releasing the lock.

        Args:
            entry: Immutable audit entry to append.
        """
        line = entry.model_dump_json() + "\n"

        async with self._lock:
            await asyncio.to_thread(self._append_line, line)

        logger.debug(
            "Audit entry recorded: entry_id=%s recommendation_id=%s",
            entry.entry_id,
            entry.recommendation_id,
        )

    def _append_line(self, line: str) -> None:
        """Synchronous write executed in a thread-pool executor.

        Args:
            line: Complete JSON line (including trailing newline) to append.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)

    async def read_all(self) -> list[AuditLogEntry]:
        """Read and deserialise all audit entries in append order.

        Intended for compliance review, incident reconstruction, and tests.
        Returns an empty list when the log file does not yet exist.

        Returns:
            All AuditLogEntry objects in the order they were recorded.

        Raises:
            pydantic.ValidationError: If a log line is malformed (log corruption).
        """
        return await asyncio.to_thread(self._read_all_sync)

    def _read_all_sync(self) -> list[AuditLogEntry]:
        """Synchronous read executed in a thread-pool executor.

        Returns:
            Parsed AuditLogEntry list; empty list when file is absent.
        """
        if not self._path.exists():
            return []

        entries: list[AuditLogEntry] = []
        with self._path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(AuditLogEntry.model_validate_json(line))
                except Exception:
                    logger.error(
                        "Corrupt audit log entry at %s line %d — skipping",
                        self._path,
                        lineno,
                    )
                    raise

        return entries
