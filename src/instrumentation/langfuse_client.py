"""Langfuse instrumentation client for the EIA RAG pipeline (Langfuse SDK v4).

Extends the Stream 1 Langfuse client pattern to cover the full EIA stack:
retrieval, generation, anomaly detection, and explanation.

Four context managers expose Langfuse observations:
  trace_dispatch        — root trace for one dispatch() call
  trace_retrieval       — child span for HybridRetriever.search()
  trace_generation      — generation span for ClaudeRecommender/OllamaRecommender
  trace_anomaly_detection — span for AnomalyScorer.score() in the streaming path

Design:
- Single Langfuse instance (singleton) shared across the pipeline.
- Graceful degradation: observability is never in the critical path. If Langfuse
  is unreachable or misconfigured the pipeline continues unaffected.
- All context managers yield a dict that callers populate with output metadata.
- Disabled automatically when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are
  absent or when LANGFUSE_ENABLED=false. OTEL exporter noise suppressed.
- Latency measured by time.monotonic() inside each context manager so that
  Langfuse receives accurate wall-clock durations regardless of SDK internals.

Usage:
    async with trace_dispatch(query) as span:
        ...
        span["confidence"] = rec.confidence

    async with trace_retrieval(query, k=5) as span:
        docs = await retriever.search(query)
        span["n_docs"] = len(docs)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Suppress Langfuse OTEL exporter noise before the SDK import when tracing
# is disabled. SDK registers a background exporter at import time regardless
# of whether it is initialised.
_tracing_disabled = (
    not os.getenv("LANGFUSE_PUBLIC_KEY")
    or not os.getenv("LANGFUSE_SECRET_KEY")
    or os.getenv("LANGFUSE_ENABLED", "true").lower() == "false"
)
if _tracing_disabled:
    os.environ["OTEL_SDK_DISABLED"] = "true"

from langfuse import Langfuse  # noqa: E402

_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse | None:
    """Return the shared Langfuse client, or None when tracing is disabled.

    Initialises on first call. Returns None when keys are absent or
    LANGFUSE_ENABLED=false so that all callers degrade to no-ops.

    Returns:
        Configured Langfuse instance, or None.
    """
    global _langfuse

    if _langfuse is not None:
        return _langfuse

    if os.getenv("LANGFUSE_ENABLED", "true").lower() == "false":
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning("Langfuse keys not set — EIA tracing disabled.")
        return None

    try:
        _langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("Langfuse EIA client initialised — host=%s", host)
        return _langfuse
    except Exception as e:
        logger.warning("Langfuse init failed — EIA tracing disabled: %s", e)
        return None


def _flush(lf: Langfuse | None) -> None:
    """Flush pending spans to Langfuse. Swallows errors silently."""
    if lf is None:
        return
    try:
        lf.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def trace_dispatch(query: str) -> AsyncIterator[dict[str, object]]:
    """Root trace for one EIA dispatch() call.

    Wraps the full retrieve → recommend pipeline. Nested trace_retrieval
    and trace_generation spans attach as children via OTEL context propagation.

    Args:
        query: The operator's natural-language dispatch query.

    Yields:
        dict that callers populate with output metadata (e.g. confidence,
        recommendation_id, requires_human_review).
    """
    span_data: dict[str, object] = {}
    lf = get_langfuse()

    if lf is None:
        yield span_data
        return

    start = time.monotonic()
    try:
        with lf.start_as_current_observation(
            name="eia-dispatch",
            input=query,
            metadata={"pipeline": "EIA-RAG"},
        ) as span:
            yield span_data
            elapsed = (time.monotonic() - start) * 1000
            try:
                span.update(
                    output=span_data,
                    metadata={"latency_ms": round(elapsed, 1)},
                )
            except Exception as e:
                logger.debug("Langfuse trace_dispatch update failed: %s", e)
    finally:
        _flush(lf)


@asynccontextmanager
async def trace_retrieval(query: str, k: int = 5) -> AsyncIterator[dict[str, object]]:
    """Child span for HybridRetriever.search().

    Captures query, requested k, latency, and whatever output metadata the
    caller populates (n_docs, sources, etc.).

    Args:
        query: Search query string.
        k: Number of results requested.

    Yields:
        dict for caller output metadata (e.g. n_docs, sources, rrf_scores).
    """
    span_data: dict[str, object] = {}
    lf = get_langfuse()

    if lf is None:
        yield span_data
        return

    start = time.monotonic()
    try:
        with lf.start_as_current_observation(
            name="hybrid-retrieval",
            input={"query": query, "k": k},
            metadata={"retriever": "BM25+semantic+RRF"},
        ) as span:
            yield span_data
            elapsed = (time.monotonic() - start) * 1000
            try:
                span.update(
                    output=span_data,
                    metadata={"latency_ms": round(elapsed, 1)},
                )
            except Exception as e:
                logger.debug("Langfuse trace_retrieval update failed: %s", e)
    finally:
        _flush(lf)


@asynccontextmanager
async def trace_generation(
    query: str,
    backend: str,
    model: str,
    n_context_docs: int,
) -> AsyncIterator[dict[str, object]]:
    """Generation span for ClaudeRecommender or OllamaRecommender.

    Recorded as a Langfuse generation observation so that model, tokens,
    and cost are tracked and visible in the dashboard per-model breakdown.

    Callers should populate span_data with:
      "action"             — recommended dispatch action
      "confidence"         — calibrated confidence score
      "hallucination_flagged" — bool
      "input_tokens"       — int (optional, 0 when not captured)
      "output_tokens"      — int (optional, 0 when not captured)

    Args:
        query: Operator's original query.
        backend: "bedrock" or "ollama".
        model: Model ID string (e.g. "anthropic.claude-sonnet-4-5", "llama3.2:3b").
        n_context_docs: Number of retrieved docs passed to the LLM.

    Yields:
        dict for caller to populate with generation output and token counts.
    """
    span_data: dict[str, object] = {}
    lf = get_langfuse()

    if lf is None:
        yield span_data
        return

    start = time.monotonic()
    try:
        with lf.start_as_current_observation(
            name="eia-generation",
            as_type="generation",
            model=model,
            input={"query": query, "n_context_docs": n_context_docs},
            metadata={"backend": backend},
        ) as span:
            yield span_data
            elapsed = (time.monotonic() - start) * 1000
            try:
                usage: dict[str, int] | None = None
                if "input_tokens" in span_data or "output_tokens" in span_data:
                    usage = {
                        "input": int(span_data.get("input_tokens", 0)),
                        "output": int(span_data.get("output_tokens", 0)),
                    }
                span.update(
                    output={k: v for k, v in span_data.items()
                            if k not in ("input_tokens", "output_tokens")},
                    usage=usage,
                    metadata={"latency_ms": round(elapsed, 1)},
                )
            except Exception as e:
                logger.debug("Langfuse trace_generation update failed: %s", e)
    finally:
        _flush(lf)


@asynccontextmanager
async def trace_anomaly_detection(
    window_size: int,
) -> AsyncIterator[dict[str, object]]:
    """Span for AnomalyScorer.score() in the streaming pipeline.

    Captures sliding window size, anomaly score, and detection result.
    Nested inside a trace_dispatch or standalone trace depending on caller.

    Args:
        window_size: Number of events in the current scoring window.

    Yields:
        dict for caller to populate with is_anomaly, anomaly_score, event_id.
    """
    span_data: dict[str, object] = {}
    lf = get_langfuse()

    if lf is None:
        yield span_data
        return

    start = time.monotonic()
    try:
        with lf.start_as_current_observation(
            name="anomaly-detection",
            input={"window_size": window_size},
            metadata={"scorer": "IsolationForest"},
        ) as span:
            yield span_data
            elapsed = (time.monotonic() - start) * 1000
            try:
                span.update(
                    output=span_data,
                    metadata={"latency_ms": round(elapsed, 1)},
                )
            except Exception as e:
                logger.debug("Langfuse trace_anomaly_detection update failed: %s", e)
    finally:
        _flush(lf)
