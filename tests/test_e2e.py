"""End-to-end integration tests for the EIA RAG pipeline and Stream 1 anomaly detection.

All tests run without external dependencies (no AWS, no PostgreSQL, no Redis):
  - EIA: FakeRetriever (canned docs) + FakeRecommender (real safety/confidence layers, no LLM)
  - Stream 1: real IsolationForestScorer with synthetic log events
  - Audit log and RAG log write to pytest tmp_path

Performance gate: full dispatch() orchestration overhead p95 < 2s across 10 runs.
LLM latency is excluded from this gate — the fake recommender returns instantly.
Real-backend latency benchmarks are in results/bedrock_vs_ollama.md.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.energy_agent import EnergyIntelligenceAgent
from src.audit.audit_log import AuditLogger
from src.models import DispatchRecommendation, RetrievedDocument
from src.recommendation_engine.confidence import ConfidenceScorer, ConfidenceSignals
from src.recommendation_engine.safety import check_grounding
from stream1_log_anomaly.src.detection.anomaly_scorer import AnomalyScorer
from stream1_log_anomaly.src.models import LogEvent


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_CONTEXT_DOCS: list[RetrievedDocument] = [
    RetrievedDocument(
        doc_id="nget-p28-001",
        content=(
            "Grid Code ECC.6.3.2: synchronous generators above 50MW must provide "
            "primary frequency response within 10 seconds of frequency deviation "
            "exceeding 0.2Hz from nominal 50Hz. Mandatory primary frequency "
            "response sustained for minimum 20 seconds. BSUoS charges calculated "
            "half-hourly settlement period applied to licensed generators balancing "
            "electricity transmission."
        ),
        source="regulation/nget-p28.pdf",
        chunk_index=0,
        similarity_score=0.95,
    ),
    RetrievedDocument(
        doc_id="bsuos-tariff-001",
        content=(
            "Balancing Services Use of System BSUoS tariff settlement period SP23 "
            "National Grid ESO balancing electricity transmission costs licensed "
            "generators suppliers half-hourly calculated basis 4.82 per MWh."
        ),
        source="regulation/bsuos-tariff-2026.pdf",
        chunk_index=0,
        similarity_score=0.88,
    ),
]

# Normal event latencies — varied to give IsolationForest a realistic distribution
_NORMAL_LATENCIES = [
    50, 120, 80, 200, 45, 100, 160, 90, 55, 130,
    75, 180, 110, 60, 140, 95, 70, 155, 85, 115,
]


class FakeRecommender:
    """Deterministic stand-in for ClaudeRecommender / OllamaRecommender.

    Exercises real check_grounding + ConfidenceScorer so that safety and
    calibration layers are covered by the E2E path. Only the LLM call is faked.
    """

    model_id = "fake-model-v1"

    async def recommend(
        self,
        query: str,
        context_docs: list[RetrievedDocument],
    ) -> DispatchRecommendation:
        """Return a grounded recommendation using corpus vocabulary."""
        if not context_docs:
            action = "FALLBACK: no context available — manual review required"
            reasoning = "No retrieved documents provided."
        else:
            # Seed action with corpus vocabulary so term_overlap passes >= 0.30
            corpus_sample = " ".join(d.content for d in context_docs)[:160]
            action = (
                f"Confirm primary frequency response obligation under Grid Code ECC "
                f"balancing electricity transmission: {corpus_sample}"
            )
            reasoning = (
                f"Retrieved {len(context_docs)} context documents confirm the "
                f"regulatory framework for this dispatch query."
            )

        safety = await check_grounding(action, reasoning, context_docs, embedder=None)
        scorer = ConfidenceScorer()
        signals = ConfidenceSignals(
            llm_confidence=0.85,
            safety_result=safety,
            supporting_documents=context_docs,
        )
        confidence, requires_review = scorer.score(signals)

        return DispatchRecommendation(
            query=query,
            action=action,
            confidence=confidence,
            supporting_documents=context_docs,
            reasoning=reasoning,
            requires_human_review=requires_review or not safety.is_grounded,
            hallucination_flagged=not safety.is_grounded,
        )


class FakeRetriever:
    """Returns canned context documents without a database connection."""

    async def search(self, query: str) -> list[RetrievedDocument]:
        return list(_CONTEXT_DOCS)


def _make_agent(audit_path: Path) -> EnergyIntelligenceAgent:
    """Build a wired agent with fake retriever + fake recommender + real audit logger.

    Patches ClaudeRecommender during construction to prevent boto3 from loading
    AWS credentials. FakeRecommender is injected immediately after.
    """
    audit_logger = AuditLogger(log_path=audit_path)
    with patch("src.agent.energy_agent.ClaudeRecommender"):
        agent = EnergyIntelligenceAgent(
            backend="bedrock",
            retriever=FakeRetriever(),  # type: ignore[arg-type]
            audit_logger=audit_logger,
        )
    agent._recommender = FakeRecommender()  # type: ignore[assignment]
    return agent


# ---------------------------------------------------------------------------
# EIA E2E: dispatch pipeline
# ---------------------------------------------------------------------------


async def test_dispatch_returns_valid_recommendation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dispatch() returns a fully-formed DispatchRecommendation on the happy path."""
    import src.instrumentation.rag_log as rl
    monkeypatch.setattr(rl, "RAG_LOG_PATH", tmp_path / "rag.jsonl")

    agent = _make_agent(tmp_path / "audit.jsonl")
    result = await agent.dispatch("What primary frequency response obligation applies?")

    assert isinstance(result, DispatchRecommendation)
    assert result.query == "What primary frequency response obligation applies?"
    assert result.action != ""
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.hallucination_flagged, bool)
    assert isinstance(result.requires_human_review, bool)
    assert len(result.recommendation_id) > 0


async def test_dispatch_recommendation_is_grounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FakeRecommender produces a grounded response (hallucination_flagged=False)."""
    import src.instrumentation.rag_log as rl
    monkeypatch.setattr(rl, "RAG_LOG_PATH", tmp_path / "rag.jsonl")

    agent = _make_agent(tmp_path / "audit.jsonl")
    result = await agent.dispatch("BSUoS charge for SP23?")

    assert result.hallucination_flagged is False, (
        f"Expected grounded response but got hallucination_flagged=True. "
        f"Action: {result.action[:80]!r}"
    )
    assert len(result.supporting_documents) == len(_CONTEXT_DOCS)


async def test_dispatch_writes_audit_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dispatch() writes exactly one audit entry per call, keyed by recommendation_id."""
    import src.instrumentation.rag_log as rl
    monkeypatch.setattr(rl, "RAG_LOG_PATH", tmp_path / "rag.jsonl")

    audit_path = tmp_path / "audit.jsonl"
    agent = _make_agent(audit_path)
    result = await agent.dispatch("Grid frequency deviation — dispatch query?")

    entries = await AuditLogger(log_path=audit_path).read_all()
    assert len(entries) == 1
    assert entries[0].recommendation_id == result.recommendation_id
    assert entries[0].query == "Grid frequency deviation — dispatch query?"
    assert entries[0].confidence == pytest.approx(result.confidence, abs=1e-4)


async def test_dispatch_writes_rag_log_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dispatch() appends a complete record to the RAG metrics JSONL log."""
    import json
    import src.instrumentation.rag_log as rl

    rag_path = tmp_path / "rag.jsonl"
    monkeypatch.setattr(rl, "RAG_LOG_PATH", rag_path)

    agent = _make_agent(tmp_path / "audit.jsonl")
    result = await agent.dispatch("Dispatch unit 3 — frequency stable?")

    assert rag_path.exists()
    lines = [ln for ln in rag_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["recommendation_id"] == result.recommendation_id
    assert record["n_retrieved_docs"] == len(_CONTEXT_DOCS)
    assert record["n_cited_docs"] == len(_CONTEXT_DOCS)
    assert "latency_ms" in record
    assert "confidence" in record
    assert "citation_coverage" in record


async def test_dispatch_fallback_on_retrieval_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retriever failure → fallback DispatchRecommendation with requires_human_review=True."""
    import src.instrumentation.rag_log as rl
    monkeypatch.setattr(rl, "RAG_LOG_PATH", tmp_path / "rag.jsonl")

    bad_retriever = MagicMock()
    bad_retriever.search = AsyncMock(side_effect=RuntimeError("PostgreSQL unavailable"))

    with patch("src.agent.energy_agent.ClaudeRecommender"):
        agent = EnergyIntelligenceAgent(
            backend="bedrock",
            retriever=bad_retriever,
            audit_logger=AuditLogger(log_path=tmp_path / "audit.jsonl"),
        )
    agent._recommender = FakeRecommender()  # type: ignore[assignment]

    result = await agent.dispatch("Query that triggers DB failure")

    assert result.requires_human_review is True
    assert result.confidence == pytest.approx(0.0)
    assert "FALLBACK" in result.action


async def test_dispatch_multiple_queries_all_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N sequential dispatches produce N audit entries with unique recommendation IDs."""
    import src.instrumentation.rag_log as rl
    monkeypatch.setattr(rl, "RAG_LOG_PATH", tmp_path / "rag.jsonl")

    audit_path = tmp_path / "audit.jsonl"
    agent = _make_agent(audit_path)

    queries = ["BSUoS charge query", "TNUoS tariff query", "BM dispatch query"]
    results = [await agent.dispatch(q) for q in queries]

    entries = await AuditLogger(log_path=audit_path).read_all()
    assert len(entries) == len(queries)

    dispatched_ids = {r.recommendation_id for r in results}
    audited_ids = {e.recommendation_id for e in entries}
    assert dispatched_ids == audited_ids


# ---------------------------------------------------------------------------
# EIA E2E: latency gate
# ---------------------------------------------------------------------------


async def test_dispatch_orchestration_latency_p95_under_2s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Orchestration layer (excluding LLM) stays under 2s p95 across 10 runs.

    Covers agent dispatch loop, safety check, confidence scoring, audit write,
    and RAG log write — everything except actual Bedrock/Ollama inference time.
    """
    import src.instrumentation.rag_log as rl
    monkeypatch.setattr(rl, "RAG_LOG_PATH", tmp_path / "rag.jsonl")

    agent = _make_agent(tmp_path / "audit.jsonl")
    n = 10
    latencies_ms: list[float] = []

    for i in range(n):
        start = time.monotonic()
        await agent.dispatch(f"Latency probe query {i}")
        latencies_ms.append((time.monotonic() - start) * 1000)

    latencies_ms.sort()
    p95_ms = latencies_ms[int(n * 0.95)]
    p50_ms = latencies_ms[n // 2]

    print(f"\nOrchestration latency — P50: {p50_ms:.1f}ms  P95: {p95_ms:.1f}ms")
    assert p95_ms < 2000.0, (
        f"Orchestration P95 {p95_ms:.0f}ms exceeds 2000ms gate. "
        f"Latencies: {[round(l, 1) for l in latencies_ms]}"
    )


# ---------------------------------------------------------------------------
# Stream 1 E2E: anomaly detection pipeline
# ---------------------------------------------------------------------------


def _normal_event(latency_ms: int, index: int = 0) -> LogEvent:
    """Create a normal log event with the given latency."""
    return LogEvent(
        ts=f"2026-06-22T10:00:{index:02d}",
        level="INFO",
        service="api-gateway",
        latency_ms=latency_ms,
        status_code=200,
    )


def _anomaly_event() -> LogEvent:
    """Create an unmistakably anomalous event: very high latency + ERROR + 500."""
    return LogEvent(
        ts="2026-06-22T10:01:00",
        level="ERROR",
        service="payment-svc",
        latency_ms=9500,
        status_code=500,
    )


def _fill_buffer(scorer: AnomalyScorer) -> None:
    """Feed BUFFER_SIZE normal events to fill the scorer's training buffer."""
    for i, latency in enumerate(_NORMAL_LATENCIES[: scorer.buffer_size]):
        scorer.score(_normal_event(latency_ms=latency, index=i))


def test_scorer_buffers_first_n_events() -> None:
    """AnomalyScorer returns None for every event while filling the initial buffer."""
    scorer = AnomalyScorer()
    for i, latency in enumerate(_NORMAL_LATENCIES[: scorer.buffer_size]):
        result = scorer.score(_normal_event(latency_ms=latency, index=i))
        assert result is None, f"Expected None at buffer position {i + 1}"


def test_scorer_returns_result_after_buffer_full() -> None:
    """First event after the buffer fills returns a valid AnomalyResult."""
    scorer = AnomalyScorer()
    _fill_buffer(scorer)

    result = scorer.score(_normal_event(latency_ms=80))
    assert result is not None
    assert isinstance(result.anomaly_score, float)
    assert isinstance(result.is_anomaly, bool)
    assert result.window_id == 0


def test_injected_anomaly_is_flagged() -> None:
    """High-latency ERROR/500 event is correctly flagged as anomaly after training on normal data."""
    scorer = AnomalyScorer()
    _fill_buffer(scorer)

    result = scorer.score(_anomaly_event())

    assert result is not None
    assert result.is_anomaly is True, (
        f"Expected anomaly to be detected. "
        f"Score={result.anomaly_score:.4f} for latency=9500ms ERROR 500."
    )


def test_anomaly_result_carries_original_event() -> None:
    """AnomalyResult.event is the exact LogEvent that was scored."""
    scorer = AnomalyScorer()
    _fill_buffer(scorer)

    event = _anomaly_event()
    result = scorer.score(event)

    assert result is not None
    assert result.event is event


def test_anomaly_result_has_all_canonical_fields() -> None:
    """AnomalyResult includes all fields defined in the canonical model spec."""
    scorer = AnomalyScorer()
    _fill_buffer(scorer)

    result = scorer.score(_anomaly_event())

    assert result is not None
    assert hasattr(result, "event")
    assert hasattr(result, "anomaly_score")
    assert hasattr(result, "is_anomaly")
    assert hasattr(result, "window_id")


def test_normal_events_low_false_positive_rate() -> None:
    """Normal events produce < 20% false positives when trained on normal data.

    IsolationForest with contamination=0.10 may flag up to 10% of in-distribution
    samples. We allow up to 20% false positives to account for model variance with
    a small buffer of 20 events.
    """
    scorer = AnomalyScorer()
    _fill_buffer(scorer)

    n_scored = 20
    false_positives = 0
    for i, latency in enumerate(_NORMAL_LATENCIES[:n_scored]):
        result = scorer.score(_normal_event(latency_ms=latency, index=scorer.buffer_size + i))
        if result is not None and result.is_anomaly:
            false_positives += 1

    fp_rate = false_positives / n_scored
    assert fp_rate <= 0.20, (
        f"False positive rate {fp_rate:.0%} exceeds 20% ({false_positives}/{n_scored} events flagged)"
    )


def test_anomaly_score_more_negative_for_anomalies() -> None:
    """Anomaly events have a more negative Isolation Forest score than normal events.

    The IF score_samples() contract: more negative = more anomalous.
    """
    scorer = AnomalyScorer()
    _fill_buffer(scorer)

    normal_result = scorer.score(_normal_event(latency_ms=80))
    anomaly_result = scorer.score(_anomaly_event())

    assert normal_result is not None
    assert anomaly_result is not None
    assert anomaly_result.anomaly_score < normal_result.anomaly_score, (
        f"Expected anomaly score ({anomaly_result.anomaly_score:.4f}) to be more negative "
        f"than normal score ({normal_result.anomaly_score:.4f})"
    )
