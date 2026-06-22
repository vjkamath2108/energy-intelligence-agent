"""Langfuse span decorators for EIA pipeline components.

Provides @traced_span — a decorator factory that wraps any sync or async
function with a Langfuse observation span. Useful for decorating standalone
functions (e.g. scoring helpers, constraint checkers) without modifying their
call sites.

For class methods where call-site context matters (retriever, recommender),
prefer the context managers in langfuse_client.py — they allow the caller to
attach structured output to the span after the function returns.

Usage:
    from src.instrumentation.decorators import traced_span

    @traced_span("constraint-validation")
    def check_constraints(action: str, docs: list[RetrievedDocument]) -> ValidationResult:
        ...

    @traced_span("embedding-query", as_type="span")
    async def embed_query(text: str) -> list[float]:
        ...
"""

import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from src.instrumentation.langfuse_client import get_langfuse

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def traced_span(span_name: str, *, as_type: str = "span") -> Callable[[F], F]:
    """Decorator factory that wraps a function with a Langfuse span.

    Works for both synchronous and asynchronous functions. Degrades
    gracefully to a no-op when Langfuse is disabled or unavailable.
    Observability failures are logged at DEBUG level and never propagate.

    Args:
        span_name: Name of the Langfuse observation span.
        as_type: Langfuse observation type — "span" (default) or "generation".
                 Use "generation" for LLM call wrappers.

    Returns:
        Decorator that preserves the wrapped function's signature.

    Example:
        @traced_span("anomaly-scoring")
        def score(self, event: LogEvent) -> AnomalyResult | None:
            ...
    """

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                lf = get_langfuse()
                if lf is None:
                    return await fn(*args, **kwargs)

                start = time.monotonic()
                try:
                    with lf.start_as_current_observation(  # type: ignore[call-overload]
                        name=span_name,
                        as_type=as_type,
                    ) as span:
                        result = await fn(*args, **kwargs)
                        elapsed = (time.monotonic() - start) * 1000
                        try:
                            span.update(metadata={"latency_ms": round(elapsed, 1)})
                        except Exception:
                            pass
                        return result
                except Exception as e:
                    logger.debug(
                        "traced_span(%r) Langfuse error — calling unwrapped: %s",
                        span_name,
                        e,
                    )
                    return await fn(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                lf = get_langfuse()
                if lf is None:
                    return fn(*args, **kwargs)

                start = time.monotonic()
                try:
                    with lf.start_as_current_observation(  # type: ignore[call-overload]
                        name=span_name,
                        as_type=as_type,
                    ) as span:
                        result = fn(*args, **kwargs)
                        elapsed = (time.monotonic() - start) * 1000
                        try:
                            span.update(metadata={"latency_ms": round(elapsed, 1)})
                        except Exception:
                            pass
                        return result
                except Exception as e:
                    logger.debug(
                        "traced_span(%r) Langfuse error — calling unwrapped: %s",
                        span_name,
                        e,
                    )
                    return fn(*args, **kwargs)

            return sync_wrapper  # type: ignore[return-value]

    return decorator
