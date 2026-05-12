"""
Ollama explainer for anomaly events.

Calls a local Ollama instance to generate human-readable explanations
when Isolation Forest flags an anomaly.

Design decisions:
- Identical interface to BedrockExplainer — swap by flag, not by rewriting call sites.
- Uses a shared httpx.AsyncClient (created at instantiation) — not per-call.
- Startup health check: fails fast with a clear error if Ollama is not running.
- Model availability check: verifies OLLAMA_MODEL is pulled before first call.
- stream: false — avoids streaming response parsing complexity.
- Same JSON prompt as BedrockExplainer — imported from prompts.py.
- Same retry + reprompt logic on invalid JSON.
- Zero AWS calls — fully local inference, no data leaves the machine.
"""

import json
import logging
import os
import time

import httpx
from pydantic import ValidationError

from stream1_log_anomaly.src.models import LogEvent, Explanation
from stream1_log_anomaly.src.explanation.prompts import build_prompt
from stream1_log_anomaly.src.instrumentation.langfuse_client import trace_explanation
from stream1_log_anomaly.src.instrumentation.anomaly_log import append_explanation

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "30.0"))
MAX_RETRIES = 2


def _parse_explanation(raw_text: str) -> Explanation:
    """
    Parse and validate raw Ollama response into Explanation model.

    Strips markdown fences if present.
    Appends closing brace if model truncated response.
    """
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1]
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()

    if clean.startswith("{") and not clean.endswith("}"):
        clean = clean + "}"

    try:
        data = json.loads(clean)
        return Explanation(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Failed to parse explanation: {e}\nRaw: {raw_text}") from e


class OllamaExplainer:
    """
    Generates human-readable anomaly explanations via local Ollama instance.

    Interface is identical to BedrockExplainer — swap by flag, not by
    rewriting call sites.

    Performs a startup health check and model availability check on first
    instantiation. Fails fast with a clear error message if Ollama is not
    running or the configured model is not available.

    Usage:
        explainer = OllamaExplainer()
        await explainer.check_health()   # call once at startup
        explanation = await explainer.explain(anomalous_event, context_events)
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=OLLAMA_BASE_URL,
            timeout=OLLAMA_TIMEOUT_S,
        )

    async def check_health(self) -> None:
        """
        Verify Ollama is running and the configured model is available.

        Call this once at application startup — not on every explain() call.

        Raises:
            RuntimeError: If Ollama is not reachable or model is not pulled.
        """
        # 1. Check Ollama is reachable
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama returned unexpected status {e.response.status_code} "
                f"on health check. Response: {e.response.text}"
            ) from e
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Ollama health check timed out after {OLLAMA_TIMEOUT_S}s. "
                f"Is the server overloaded?"
            )

        # 2. Check the configured model is pulled
        data = response.json()
        available_models: list[str] = [m["name"] for m in data.get("models", [])]

        # Ollama model names can include tags: "llama3.2:3b" or just "llama3.2"
        # Normalise both sides for comparison
        def _normalise(name: str) -> str:
            return name.split(":")[0].lower()

        configured = _normalise(OLLAMA_MODEL)
        if not any(_normalise(m) == configured for m in available_models):
            available_str = ", ".join(available_models) if available_models else "(none)"
            raise RuntimeError(
                f"Model '{OLLAMA_MODEL}' is not available in Ollama. "
                f"Available models: {available_str}. "
                f"Pull it with: ollama pull {OLLAMA_MODEL}"
            )

        logger.info(
            "Ollama health check passed — model '%s' available at %s",
            OLLAMA_MODEL,
            OLLAMA_BASE_URL,
        )

    async def _call_ollama(self, prompt: str) -> tuple[str, bool]:
        """
        Call local Ollama API and return raw text response.

        Args:
            prompt: Fully rendered user prompt.

        Returns:
            Tuple of (raw_text, is_retry=False). is_retry flag set by caller.

        Raises:
            RuntimeError: If Ollama call fails after timeout or HTTP error.
        """
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 512,
            },
        }

        try:
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            raw: str = data["response"]
            logger.debug(
                "Ollama (%s) response: %s...",
                OLLAMA_MODEL,
                raw[:120].replace("\n", " "),
            )
            return raw, False
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Ollama call timed out after {OLLAMA_TIMEOUT_S}s "
                f"(model: {OLLAMA_MODEL})"
            ) from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}") from e

    async def explain(
        self,
        anomalous_event: LogEvent,
        context_events: list[LogEvent],
    ) -> Explanation:
        """
        Generate a structured explanation for an anomalous log event.

        Retries once with stricter prompt if model returns invalid JSON.
        Traces every call to Langfuse and appends to anomaly_log.jsonl.
        Logs model name and response latency on every call.

        Args:
            anomalous_event: Event flagged by Isolation Forest.
            context_events: Surrounding ±5 events for causality.

        Returns:
            Validated Explanation with severity, summary, cause, action.

        Raises:
            RuntimeError: If explanation cannot be generated after retry.
        """
        prompt = build_prompt(anomalous_event, context_events)
        start = time.monotonic()
        is_retry = False

        async with trace_explanation(
            service=anomalous_event.service,
            level=anomalous_event.level,
            latency_ms=anomalous_event.latency_ms,
            backend="ollama",
            model=OLLAMA_MODEL,
            prompt=prompt,
        ) as span:
            raw, _ = await self._call_ollama(prompt)

            try:
                explanation = _parse_explanation(raw)
            except ValueError:
                is_retry = True
                logger.warning(
                    "Invalid JSON from Ollama (%s) — retrying with stricter prompt",
                    OLLAMA_MODEL,
                )
                strict_prompt = (
                    prompt
                    + "\n\nCRITICAL: Return ONLY a valid JSON object. "
                    "No markdown. No explanation. No text before or after the JSON."
                )
                raw, _ = await self._call_ollama(strict_prompt)
                explanation = _parse_explanation(raw)

            elapsed_ms = (time.monotonic() - start) * 1000

            span["output"] = explanation.model_dump()
            span["metadata"] = {
                "model": OLLAMA_MODEL,
                "backend": "ollama",
                "latency_ms": round(elapsed_ms),
                "is_retry": is_retry,
            }

        logger.info(
            "Ollama explanation — model=%s latency=%.0fms retry=%s severity=%s",
            OLLAMA_MODEL,
            elapsed_ms,
            is_retry,
            explanation.severity,
        )

        append_explanation(anomalous_event, explanation, "ollama", elapsed_ms)
        return explanation

    async def aclose(self) -> None:
        """Close the underlying httpx client. Call on application shutdown."""
        await self._client.aclose()


if __name__ == "__main__":
    import asyncio
    import json
    from pathlib import Path
    from stream1_log_anomaly.src.detection.anomaly_scorer import AnomalyScorer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    async def _smoke_test() -> None:
        events = [
            LogEvent(**json.loads(line))
            for line in Path("stream1_log_anomaly/logs/sample.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]

        scorer = AnomalyScorer()
        explainer = OllamaExplainer()

        # Health check at startup — exits cleanly if Ollama not running
        await explainer.check_health()

        for i, event in enumerate(events):
            result = scorer.score(event)
            if result is None or not result.is_anomaly:
                continue

            context = events[max(0, i - 5) : i + 6]

            print(f"\n{'='*60}")
            print(f"ANOMALY at event [{i:03d}] — {event.service} {event.level}")
            print(f"{'='*60}")

            explanation = await explainer.explain(event, context)

            print(f"Severity:  {explanation.severity}")
            print(f"Summary:   {explanation.summary}")
            print(f"Cause:     {explanation.likely_cause}")
            print(f"Action:    {explanation.recommended_action}")

        await explainer.aclose()

    asyncio.run(_smoke_test())