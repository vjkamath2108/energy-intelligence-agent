"""Tests for src/instrumentation/langfuse_client.py and decorators.py.

All tests run without Langfuse keys — verifying that every context manager
and decorator degrades gracefully to a no-op. No network calls are made.

Coverage:
- get_langfuse() returns None when keys are absent.
- trace_dispatch / trace_retrieval / trace_generation / trace_anomaly_detection
  yield a dict, are awaitable, and do not swallow caller exceptions.
- Caller can read and write span_data dict inside the context managers.
- @traced_span preserves return value for both sync and async functions.
- @traced_span propagates caller exceptions (does not swallow them).
"""

from __future__ import annotations

import pytest

from src.instrumentation.decorators import traced_span
from src.instrumentation.langfuse_client import (
    get_langfuse,
    trace_anomaly_detection,
    trace_dispatch,
    trace_generation,
    trace_retrieval,
)


# ---------------------------------------------------------------------------
# get_langfuse — no-op when keys absent
# ---------------------------------------------------------------------------


def test_get_langfuse_returns_none_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_langfuse() returns None when LANGFUSE_PUBLIC_KEY is absent."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    import src.instrumentation.langfuse_client as lf_mod
    lf_mod._langfuse = None  # reset singleton

    result = lf_mod.get_langfuse()
    assert result is None


# ---------------------------------------------------------------------------
# trace_dispatch
# ---------------------------------------------------------------------------


async def test_trace_dispatch_yields_dict() -> None:
    """trace_dispatch yields a mutable dict when Langfuse is disabled."""
    async with trace_dispatch("test query") as span:
        assert isinstance(span, dict)
        span["key"] = "value"

    assert span["key"] == "value"


async def test_trace_dispatch_propagates_exception() -> None:
    """Exceptions inside trace_dispatch are not swallowed."""
    with pytest.raises(ValueError, match="pipeline error"):
        async with trace_dispatch("test query"):
            raise ValueError("pipeline error")


# ---------------------------------------------------------------------------
# trace_retrieval
# ---------------------------------------------------------------------------


async def test_trace_retrieval_yields_dict() -> None:
    """trace_retrieval yields a mutable dict when Langfuse is disabled."""
    async with trace_retrieval("BSUoS query", k=5) as span:
        span["n_docs"] = 3
        span["sources"] = ["regulation/p28.pdf"]

    assert span["n_docs"] == 3
    assert span["sources"] == ["regulation/p28.pdf"]


async def test_trace_retrieval_propagates_exception() -> None:
    """Retrieval exceptions propagate through the context manager."""
    with pytest.raises(RuntimeError, match="db unavailable"):
        async with trace_retrieval("query"):
            raise RuntimeError("db unavailable")


# ---------------------------------------------------------------------------
# trace_generation
# ---------------------------------------------------------------------------


async def test_trace_generation_yields_dict() -> None:
    """trace_generation yields a mutable dict with generation metadata."""
    async with trace_generation(
        query="dispatch query",
        backend="bedrock",
        model="anthropic.claude-sonnet-4-5",
        n_context_docs=3,
    ) as span:
        span["confidence"] = 0.85
        span["hallucination_flagged"] = False
        span["input_tokens"] = 512
        span["output_tokens"] = 128

    assert span["confidence"] == pytest.approx(0.85)
    assert span["input_tokens"] == 512


async def test_trace_generation_propagates_exception() -> None:
    """Generation exceptions propagate through the context manager."""
    with pytest.raises(RuntimeError, match="bedrock throttled"):
        async with trace_generation("query", "bedrock", "claude-sonnet", 2):
            raise RuntimeError("bedrock throttled")


# ---------------------------------------------------------------------------
# trace_anomaly_detection
# ---------------------------------------------------------------------------


async def test_trace_anomaly_detection_yields_dict() -> None:
    """trace_anomaly_detection yields a mutable dict."""
    async with trace_anomaly_detection(window_size=30) as span:
        span["is_anomaly"] = True
        span["anomaly_score"] = -0.42

    assert span["is_anomaly"] is True
    assert span["anomaly_score"] == pytest.approx(-0.42)


async def test_trace_anomaly_detection_propagates_exception() -> None:
    """Scorer exceptions propagate through the context manager."""
    with pytest.raises(ValueError, match="bad window"):
        async with trace_anomaly_detection(window_size=0):
            raise ValueError("bad window")


# ---------------------------------------------------------------------------
# @traced_span decorator — async
# ---------------------------------------------------------------------------


async def test_traced_span_async_returns_value() -> None:
    """@traced_span preserves the return value of an async function."""
    @traced_span("test-async-span")
    async def add(a: int, b: int) -> int:
        return a + b

    result = await add(3, 4)
    assert result == 7


async def test_traced_span_async_propagates_exception() -> None:
    """@traced_span does not swallow exceptions from async functions."""
    @traced_span("test-async-span")
    async def fail() -> None:
        raise ValueError("async failure")

    with pytest.raises(ValueError, match="async failure"):
        await fail()


async def test_traced_span_async_passes_args() -> None:
    """@traced_span passes positional and keyword args correctly."""
    @traced_span("test-args-span")
    async def greet(name: str, *, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}"

    result = await greet("operator", greeting="Hi")
    assert result == "Hi, operator"


# ---------------------------------------------------------------------------
# @traced_span decorator — sync
# ---------------------------------------------------------------------------


def test_traced_span_sync_returns_value() -> None:
    """@traced_span preserves the return value of a sync function."""
    @traced_span("test-sync-span")
    def multiply(a: int, b: int) -> int:
        return a * b

    assert multiply(6, 7) == 42


def test_traced_span_sync_propagates_exception() -> None:
    """@traced_span does not swallow exceptions from sync functions."""
    @traced_span("test-sync-span")
    def fail() -> None:
        raise TypeError("sync failure")

    with pytest.raises(TypeError, match="sync failure"):
        fail()


def test_traced_span_preserves_function_name() -> None:
    """@traced_span preserves __name__ and __doc__ via functools.wraps."""
    @traced_span("test-span")
    def my_function() -> None:
        """My docstring."""

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring."
