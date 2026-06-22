"""Append-only RAG pipeline results log for the Streamlit metrics dashboard.

Writes one JSON record per dispatch() call to results/rag_log.jsonl.
Fields captured per record:
  logged_at           — UTC ISO timestamp
  recommendation_id   — UUID from DispatchRecommendation
  query               — operator's original query (truncated to 120 chars)
  backend             — "bedrock" or "ollama"
  latency_ms          — wall-clock time for the full dispatch() call
  confidence          — calibrated confidence score [0, 1]
  n_retrieved_docs    — total docs returned by HybridRetriever.search()
  n_cited_docs        — docs cited in the final recommendation
  citation_coverage   — n_cited / n_retrieved, or 0 when n_retrieved == 0
  hallucination_flagged — True when safety layer rejected the response
  requires_human_review — True when confidence < 0.60 or any safety flag
  action              — first 120 chars of the recommended action

This log is intentionally separate from the audit log (logs/audit.jsonl).
The audit log is compliance-grade and immutable; this log is a rolling
metrics feed for the dashboard and is safe to truncate or archive.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.models import DispatchRecommendation

logger = logging.getLogger(__name__)

RAG_LOG_PATH = Path("results/rag_log.jsonl")


def append_dispatch(
    rec: DispatchRecommendation,
    backend: str,
    latency_ms: float,
    n_retrieved_docs: int,
) -> None:
    """Append one dispatch result record to the RAG metrics log.

    Creates the log file and parent directories on first call.
    Silently swallows I/O errors — logging must not block the pipeline.

    Args:
        rec: The DispatchRecommendation returned by dispatch().
        backend: "bedrock" or "ollama".
        latency_ms: Wall-clock time for the full dispatch() call in milliseconds.
        n_retrieved_docs: Total docs returned by HybridRetriever.search().
    """
    n_cited = len(rec.supporting_documents)
    citation_coverage = (n_cited / n_retrieved_docs) if n_retrieved_docs > 0 else 0.0

    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "recommendation_id": rec.recommendation_id,
        "query": rec.query[:120],
        "backend": backend,
        "latency_ms": round(latency_ms, 1),
        "confidence": round(rec.confidence, 4),
        "n_retrieved_docs": n_retrieved_docs,
        "n_cited_docs": n_cited,
        "citation_coverage": round(citation_coverage, 4),
        "hallucination_flagged": rec.hallucination_flagged,
        "requires_human_review": rec.requires_human_review,
        "action": rec.action[:120],
    }

    try:
        RAG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RAG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.debug("RAG log entry appended: %s", rec.recommendation_id)
    except OSError as e:
        logger.warning("Failed to write RAG log entry: %s", e)
