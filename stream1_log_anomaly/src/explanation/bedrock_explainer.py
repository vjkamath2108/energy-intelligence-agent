"""
Bedrock explainer for anomaly events.

Calls Claude via AWS Bedrock to generate a human-readable explanation
when Isolation Forest flags an anomaly.

Design decisions:
- LLM explains, never scores. Isolation Forest owns the anomaly decision.
- Structured JSON output enforced via prompt + Pydantic validation.
- Retry on invalid JSON: reprompt once before raising.
- Exponential backoff on rate limits (ThrottlingException).
- Context window: anomalous event + surrounding ±5 events for causality.
"""

import json
import logging
import os
import time
from typing import Any

import asyncio
import boto3
from botocore.exceptions import ClientError
from pydantic import ValidationError

from stream1_log_anomaly.src.models import LogEvent, Explanation
from stream1_log_anomaly.src.explanation.prompts import build_prompt
from stream1_log_anomaly.src.instrumentation.langfuse_client import trace_explanation
from stream1_log_anomaly.src.instrumentation.anomaly_log import append_explanation

logger = logging.getLogger(__name__)

BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)
BEDROCK_REGION = "us-west-2"
MAX_RETRIES = 3
BASE_BACKOFF = 1.0


def _call_bedrock_sync(client: Any, prompt: str) -> str:
    """
    Call Bedrock Claude and return the raw text response.

    Retries on ThrottlingException with exponential backoff.

    Args:
        client: Boto3 Bedrock runtime client.
        prompt: Fully rendered user prompt.

    Returns:
        Raw text content from Claude.

    Raises:
        RuntimeError: If all retries exhausted.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}],
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps(body),
            )
            raw = json.loads(response["body"].read())
            return raw["content"][0]["text"]

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ThrottlingException" and attempt < MAX_RETRIES - 1:
                wait = BASE_BACKOFF * (2**attempt)
                logger.warning(
                    "Bedrock throttled — retrying in %.1fs (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Bedrock call failed after {MAX_RETRIES} attempts")


async def _call_bedrock(client: Any, prompt: str) -> str:
    """
    Async wrapper around _call_bedrock_sync.

    Runs the blocking boto3 call in a thread pool executor so it does
    not block the asyncio event loop during Bedrock latency (typically 2-3s).
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _call_bedrock_sync, client, prompt)


def _parse_explanation(raw_text: str) -> Explanation:
    """
    Parse and validate raw Bedrock response into Explanation model.

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


class BedrockExplainer:
    """
    Generates human-readable anomaly explanations via AWS Bedrock.

    Interface is identical to OllamaExplainer — swap by flag, not by
    rewriting call sites.

    Usage:
        explainer = BedrockExplainer()
        explanation = await explainer.explain(anomalous_event, context_events)
    """

    def __init__(self, region: str = BEDROCK_REGION) -> None:
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
        )

    async def explain(
        self,
        anomalous_event: LogEvent,
        context_events: list[LogEvent],
    ) -> Explanation:
        """
        Generate a structured explanation for an anomalous log event.

        Retries once with a stricter prompt if Claude returns invalid JSON.
        Traces every call to Langfuse and appends to anomaly_log.jsonl.

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

        async with trace_explanation(
            service=anomalous_event.service,
            level=anomalous_event.level,
            latency_ms=anomalous_event.latency_ms,
            backend="bedrock",
            prompt=prompt,
        ) as span:
            raw = await _call_bedrock(self._client, prompt)
            logger.debug("Bedrock raw response: %s", raw[:200])

            try:
                explanation = _parse_explanation(raw)
            except ValueError:
                logger.warning("Invalid JSON from Bedrock — retrying with stricter prompt")
                strict_prompt = prompt + "\n\nYou MUST return only valid JSON. Nothing else."
                raw = await _call_bedrock(self._client, strict_prompt)
                explanation = _parse_explanation(raw)

            span["output"] = explanation.model_dump()

        elapsed_ms = (time.monotonic() - start) * 1000
        append_explanation(anomalous_event, explanation, "bedrock", elapsed_ms)

        return explanation


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
        explainer = BedrockExplainer()

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