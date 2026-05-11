"""
Langfuse instrumentation client.

Provides a single initialised Langfuse instance and a trace_explanation
context manager used by both BedrockExplainer and OllamaExplainer.

Design decisions:
- Single Langfuse instance shared across the pipeline — not one per call.
- trace_explanation wraps the full explain() call — input, output, latency,
  model name all captured in one span.
- Backend name passed explicitly — traces are labelled bedrock or ollama
  so you can filter and compare in the dashboard.
- Graceful degradation — if Langfuse is unreachable, the pipeline continues.
  Observability must never be in the critical path.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

logger = logging.getLogger(__name__)

_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse | None:
    """
    Return the shared Langfuse instance.

    Returns None if Langfuse is not configured or unreachable —
    callers must handle None gracefully.
    """
    global _langfuse

    if _langfuse is not None:
        return _langfuse

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    if not public_key or not secret_key:
        logger.warning("Langfuse keys not set — tracing disabled.")
        return None

    try:
        _langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("Langfuse client initialised — host=%s", host)
        return _langfuse
    except Exception as e:
        logger.warning("Langfuse init failed — tracing disabled: %s", e)
        return None


@asynccontextmanager
async def trace_explanation(
    service: str,
    level: str,
    latency_ms: int,
    backend: str,
    prompt: str,
) -> AsyncIterator[dict]:
    """
    Async context manager that creates a Langfuse trace for one explanation.

    Usage:
        async with trace_explanation(...) as span:
            explanation = await explainer.explain(event, context)
            span["output"] = explanation.model_dump()

    Args:
        service: Service name from the anomalous event.
        level: Log level from the anomalous event (ERROR, WARN etc).
        latency_ms: Latency from the anomalous event.
        backend: "bedrock" or "ollama".
        prompt: The rendered prompt sent to the LLM.
    """
    lf = get_langfuse()
    span_data: dict = {}

    if lf is None:
        yield span_data
        return

    trace = None
    generation = None

    try:
        trace = lf.trace(
            name="anomaly-explanation",
            metadata={
                "service": service,
                "level": level,
                "latency_ms": latency_ms,
                "backend": backend,
            },
        )

        generation = trace.generation(
            name="llm-explanation",
            model=backend,
            input=prompt,
        )

        yield span_data  # caller fills span_data["output"]

        # Update generation with output after yield
        if "output" in span_data:
            generation.end(output=span_data["output"])
            trace.update(output=span_data["output"])
        else:
            generation.end()

    except Exception as e:
        logger.warning("Langfuse trace failed — pipeline continues: %s", e)
        yield span_data
    finally:
        try:
            if lf:
                lf.flush()
        except Exception:
            pass