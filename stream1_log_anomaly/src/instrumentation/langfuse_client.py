"""
Langfuse instrumentation client (Langfuse SDK v4).

Provides a shared Langfuse instance and a trace_explanation async context
manager used by BedrockExplainer and OllamaExplainer.

Design decisions:
- Single Langfuse instance shared across the pipeline — not one per call.
- trace_explanation wraps the full explain() call: input, output, latency,
  and model name captured in one generation observation.
- Backend name passed explicitly — traces labelled bedrock/ollama so you can
  filter and compare in the Langfuse dashboard.
- Graceful degradation — if Langfuse is unreachable the pipeline continues.
  Observability must never be in the critical path.

v4 API notes:
- lf.trace() / lf.generation() removed; use lf.start_as_current_observation()
  with as_type="generation" for a generation span.
- Call .set_trace_io() on the span to attach trace-level input/output metadata.
- lf.flush() retained for forcing delivery before process exit.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

# Suppress Langfuse OTEL exporter noise when keys are not configured.
# The SDK registers a background exporter at import time regardless of
# whether it is initialised — this env var disables it cleanly before
# the Langfuse import fires.
if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
    os.environ["OTEL_SDK_DISABLED"] = "true"

from contextlib import asynccontextmanager  # noqa: E402
from typing import AsyncIterator  # noqa: E402

from langfuse import Langfuse  # noqa: E402

logger = logging.getLogger(__name__)

_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse | None:
    global _langfuse

    if _langfuse is not None:
        return _langfuse

    # Explicit kill switch — set LANGFUSE_ENABLED=false to disable tracing
    # without removing keys from .env
    if os.getenv("LANGFUSE_ENABLED", "true").lower() == "false":
        logger.info("Langfuse disabled via LANGFUSE_ENABLED=false.")
        return None

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
    model: str | None = None,
) -> AsyncIterator[dict[str, object]]:
    """
    Async context manager that records one anomaly explanation as a Langfuse
    generation observation (Langfuse SDK v4).

    Usage:
        async with trace_explanation(...) as span:
            explanation = await explainer.explain(event, context)
            span["output"] = explanation.model_dump()

    Args:
        service: Service name from the anomalous log event.
        level: Log level of the anomalous event (ERROR, WARN, etc.).
        latency_ms: Observed latency of the anomalous event.
        backend: "bedrock" or "ollama" — stored as metadata tag.
        prompt: The rendered prompt sent to the LLM.
        model: Specific model name (e.g. "llama3.2:3b", "claude-sonnet-4-5").
               Passed to Langfuse model field for per-model filtering.
               Defaults to backend name if not provided.
    """
    lf = get_langfuse()
    span_data: dict[str, object] = {}
    model_label = model or backend

    if lf is None:
        yield span_data
        return

    try:
        with lf.start_as_current_observation(
            name="anomaly-explanation",
            as_type="generation",
            model=model_label,
            input=prompt,
            metadata={
                "service": service,
                "level": level,
                "latency_ms": latency_ms,
                "backend": backend,
            },
        ) as generation:

            yield span_data  # caller fills span_data["output"]

            if "output" in span_data:
                generation.update(output=span_data["output"])

    except Exception as e:
        logger.warning("Langfuse trace failed — pipeline continues: %s", e)

    finally:
        try:
            if lf:
                lf.flush()
        except Exception:
            pass