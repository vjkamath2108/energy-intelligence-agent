"""
Ollama explainer for anomaly events.

Calls a local Ollama instance to generate human-readable explanations
when Isolation Forest flags an anomaly.

Design decisions:
- Identical interface to BedrockExplainer — swap by flag, not by rewriting call sites.
- Uses httpx.AsyncClient for non-blocking HTTP calls.
- Same JSON prompt as BedrockExplainer — imported from prompts.py.
- Same retry + reprompt logic on invalid JSON.
- Zero AWS calls — fully local inference, no data leaves the machine.
"""

import json
import logging
import os

import httpx
from pydantic import ValidationError

from stream1_log_anomaly.src.explanation.prompts import build_prompt
from stream1_log_anomaly.src.models import LogEvent
from stream1_log_anomaly.src.explanation.bedrock_explainer import Explanation

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
MAX_RETRIES = 2


async def _call_ollama(prompt: str) -> str:
    """
    Call local Ollama API and return raw text response.

    Args:
        prompt: Fully rendered user prompt.

    Returns:
        Raw text content from Ollama.

    Raises:
        RuntimeError: If Ollama call fails.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,  # get full response in one shot, not streaming chunks
        "options": {
            "num_predict": 512,  # increase from default
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["response"]
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama call failed: {e}") from e

def _parse_explanation(raw_text: str) -> Explanation:
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1]
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()

    # Small models sometimes truncate the closing brace — add it if missing
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

    Usage:
        explainer = OllamaExplainer()
        explanation = await explainer.explain(anomalous_event, context_events)
    """

    async def explain(
        self,
        anomalous_event: LogEvent,
        context_events: list[LogEvent],
    ) -> Explanation:
        """
        Generate a structured explanation for an anomalous log event.

        Retries once with stricter prompt if model returns invalid JSON.

        Args:
            anomalous_event: Event flagged by Isolation Forest.
            context_events: Surrounding ±5 events for causality.

        Returns:
            Validated Explanation with severity, summary, cause, action.

        Raises:
            RuntimeError: If explanation cannot be generated after retry.
        """
        prompt = build_prompt(anomalous_event, context_events)

        raw = await _call_ollama(prompt)
        logger.debug("Ollama raw response: %s", raw[:200])

        try:
            return _parse_explanation(raw)
        except ValueError:
            logger.warning("Invalid JSON from Ollama — retrying with stricter prompt")
            strict_prompt = prompt + "\n\nYou MUST return only valid JSON. Nothing else."
            raw = await _call_ollama(strict_prompt)
            return _parse_explanation(raw)


if __name__ == "__main__":
    import asyncio
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

    asyncio.run(_smoke_test())