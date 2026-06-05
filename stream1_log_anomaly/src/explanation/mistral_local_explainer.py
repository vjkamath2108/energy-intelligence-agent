"""
Mistral Local explainer for anomaly events.

Calls Mistral 7B via a local Ollama instance to generate human-readable
explanations when Isolation Forest flags an anomaly.

Design decisions:
- Identical interface to BedrockExplainer and MistralBedrockExplainer.
- Zero AWS calls — fully sovereign, no data leaves the machine.
- Completes the three-way benchmark bouquet:
    claude-bedrock   → frontier managed (Claude Sonnet)
    mistral-bedrock  → open-weight managed (Mistral 7B via AWS)
    mistral-local    → open-weight sovereign (Mistral 7B via Ollama)
- Uses OLLAMA_MODEL env var, defaulting to mistral:latest.
- Startup health check verifies Ollama is running and model is pulled.
- Same retry + reprompt logic on invalid JSON as other explainers.
- Same Langfuse tracing with backend="mistral-local" for clean separation
  in benchmark comparisons.
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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "60.0"))  # Mistral 7B slower than 3B locally
MAX_RETRIES = 2


def _parse_explanation(raw_text: str) -> Explanation:
    """
    Parse and validate raw Mistral response into Explanation model.

    Strips markdown fences if present.
    Appends closing brace if model truncated response.
    Normalises severity — Mistral occasionally returns "<MEDIUM>" or " HIGH".
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
        if "severity" in data and isinstance(data["severity"], str):
            data["severity"] = data["severity"].strip().strip("<>").strip().upper()
        return Explanation(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Failed to parse explanation: {e}\nRaw: {raw_text}") from e


class MistralLocalExplainer:
    """
    Generates anomaly explanations via Mistral 7B running locally in Ollama.

    Sovereign inference — zero external calls, no data leaves the machine.
    Interface identical to BedrockExplainer and MistralBedrockExplainer.

    Usage:
        explainer = MistralLocalExplainer()
        await explainer.check_health()   # once at startup
        explanation = await explainer.explain(anomalous_event, context_events)
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=OLLAMA_BASE_URL,
            timeout=OLLAMA_TIMEOUT_S,
        )

    async def check_health(self) -> None:
        """
        Verify Ollama is running and mistral:latest is pulled.

        Raises:
            RuntimeError: If Ollama is not reachable or model is not available.
        """
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                "Is Ollama running? Start with: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama returned status {e.response.status_code} on health check."
            ) from e
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Ollama health check timed out after {OLLAMA_TIMEOUT_S}s."
            )

        data = response.json()
        available: list[str] = [m["name"] for m in data.get("models", [])]

        def _normalise(name: str) -> str:
            return name.split(":")[0].lower()

        configured = _normalise(OLLAMA_MODEL)
        if not any(_normalise(m) == configured for m in available):
            available_str = ", ".join(available) if available else "(none)"
            raise RuntimeError(
                f"Model '{OLLAMA_MODEL}' not available in Ollama. "
                f"Available: {available_str}. "
                f"Pull it with: ollama pull {OLLAMA_MODEL}"
            )

        logger.info(
            "Mistral Local health check passed — model '%s' available at %s",
            OLLAMA_MODEL,
            OLLAMA_BASE_URL,
        )

    async def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama Mistral and return raw text response."""
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
                "temperature": 0.1,  # Low temp for structured JSON — matches Bedrock setting
            },
        }

        try:
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            raw: str = data["response"]
            logger.debug(
                "Mistral Local (%s) response: %s...",
                OLLAMA_MODEL,
                raw[:120].replace("\n", " "),
            )
            return raw
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Mistral Local timed out after {OLLAMA_TIMEOUT_S}s. "
                "Mistral 7B is slower than 3B locally — consider increasing OLLAMA_TIMEOUT_S."
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

        Retries once with stricter prompt if Mistral returns invalid JSON.
        Traces to Langfuse with backend="mistral-local" for benchmark separation.

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
            backend="mistral-local",
            model=OLLAMA_MODEL,
            prompt=prompt,
        ) as span:
            raw = await self._call_ollama(prompt)

            try:
                explanation = _parse_explanation(raw)
            except ValueError:
                is_retry = True
                logger.warning(
                    "Invalid JSON from Mistral Local (%s) — retrying with stricter prompt",
                    OLLAMA_MODEL,
                )
                strict_prompt = (
                    prompt
                    + "\n\nCRITICAL: Return ONLY a valid JSON object. "
                    "No markdown. No explanation. No text before or after the JSON."
                )
                raw = await self._call_ollama(strict_prompt)
                try:
                    explanation = _parse_explanation(raw)
                except ValueError as e:
                    raise RuntimeError(f"Failed after retry: {e}") from e

            elapsed_ms = (time.monotonic() - start) * 1000

            span["output"] = explanation.model_dump()
            span["metadata"] = {
                "model": OLLAMA_MODEL,
                "backend": "mistral-local",
                "latency_ms": round(elapsed_ms),
                "is_retry": is_retry,
            }

        logger.info(
            "Mistral Local explanation — model=%s latency=%.0fms retry=%s severity=%s",
            OLLAMA_MODEL,
            elapsed_ms,
            is_retry,
            explanation.severity,
        )

        append_explanation(anomalous_event, explanation, "mistral-local", elapsed_ms)
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
        explainer = MistralLocalExplainer()

        await explainer.check_health()

        for i, event in enumerate(events):
            result = scorer.score(event)
            if result is None or not result.is_anomaly:
                continue

            context = events[max(0, i - 5) : i + 6]

            print(f"\n{'='*60}")
            print(f"ANOMALY at event [{i:03d}] — {event.service} {event.level}")
            print(f"{'='*60}")

            start = time.monotonic()
            explanation = await explainer.explain(event, context)
            elapsed = (time.monotonic() - start) * 1000

            print(f"Severity:  {explanation.severity}")
            print(f"Summary:   {explanation.summary}")
            print(f"Cause:     {explanation.likely_cause}")
            print(f"Action:    {explanation.recommended_action}")
            print(f"Latency:   {elapsed:.0f}ms")  # ← add this

        await explainer.aclose()

    asyncio.run(_smoke_test())