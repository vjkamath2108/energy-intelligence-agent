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
import time
from typing import Any

import os
import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict, ValidationError
from stream1_log_anomaly.src.models import LogEvent


logger = logging.getLogger(__name__)


# us-west-2 requires regional inference profile prefix.
# Raw model IDs (anthropic.claude-*) are rejected — use us.* prefix.
# Full versioned IDs required — short aliases don't work with invoke_model.
# BEDROCK_MODEL_ID = "us.anthropic.claude-opus-4-5-20251101-v1:0"
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID","us.anthropic.claude-sonnet-4-5-20250929-v1:0")

BEDROCK_REGION = "us-west-2"
MAX_RETRIES = 3
BASE_BACKOFF = 1.0


class Explanation(BaseModel):
    """
    Structured LLM explanation of an anomaly event.

    Attributes:
        severity: HIGH, MEDIUM, or LOW.
        summary: One sentence describing what happened.
        likely_cause: Most probable root cause.
        recommended_action: What an engineer should do next.
    """

    model_config = ConfigDict(frozen=True)

    severity: str
    summary: str
    likely_cause: str
    recommended_action: str


def _build_prompt(
    anomalous_event: dict,
    context_events: list[dict],
) -> str:
    """
    Build the user prompt for the explainer.

    Args:
        anomalous_event: The event flagged as anomalous by Isolation Forest.
        context_events: Surrounding ±5 events for causality context.
    """
    context_str = json.dumps([e.model_dump() for e in context_events], indent=2)
    event_str = json.dumps(anomalous_event.model_dump(), indent=2)

    return f"""You are an expert site reliability engineer analysing log anomalies.

Surrounding context events (±5 around the anomaly):
{context_str}

Anomalous event flagged by Isolation Forest:
{event_str}

Respond ONLY with a JSON object. No preamble, no markdown, no explanation outside the JSON.Do not use markdown code fences. Do not write ```json.
Start your response with {{ and end with }}.

{{
  "severity": "<HIGH|MEDIUM|LOW>",
  "summary": "<one sentence: what happened>",
  "likely_cause": "<most probable root cause>",
  "recommended_action": "<what an engineer should do next>"
}}"""


def _call_bedrock(client: Any, prompt: str) -> str:
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


def _parse_explanation(raw_text: str) -> Explanation:
    clean = raw_text.strip()
    if clean.startswith("```"):
        # Strip opening fence (```json or ```)
        clean = clean.split("\n", 1)[1]
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()
    
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
        explanation = explainer.explain(anomalous_event, context_events)
    """

    def __init__(self, region: str = BEDROCK_REGION) -> None:
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
        )

    def explain(
        self,
        anomalous_event: dict,
        context_events: list[dict],
    ) -> Explanation:
        """
        Generate a structured explanation for an anomalous log event.

        Retries once with a stricter prompt if Claude returns invalid JSON.

        Args:
            anomalous_event: Event flagged by Isolation Forest.
            context_events: Surrounding ±5 events for causality.

        Returns:
            Validated Explanation with severity, summary, cause, action.

        Raises:
            RuntimeError: If explanation cannot be generated after retry.
        """
        prompt = _build_prompt(anomalous_event, context_events)

        raw = _call_bedrock(self._client, prompt)
        logger.debug("Bedrock raw response: %s", raw[:200])

        try:
            return _parse_explanation(raw)
        except ValueError:
            logger.warning("Invalid JSON from Bedrock — retrying with stricter prompt")
            strict_prompt = prompt + "\n\nYou MUST return only valid JSON. Nothing else."
            raw = _call_bedrock(self._client, strict_prompt)
            return _parse_explanation(raw)

if __name__ == "__main__":
    import json
    from pathlib import Path
    from stream1_log_anomaly.src.detection.anomaly_scorer import AnomalyScorer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    events = [
        json.loads(line)
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

        # ±5 context window, clamped to list bounds
        context = events[max(0, i - 5) : i + 6]

        print(f"\n{'='*60}")
        print(f"ANOMALY at event [{i:03d}] — {event.get('service')} {event.get('level')}")
        print(f"{'='*60}")

        explanation = explainer.explain(event, context)

        print(f"Severity:  {explanation.severity}")
        print(f"Summary:   {explanation.summary}")
        print(f"Cause:     {explanation.likely_cause}")
        print(f"Action:    {explanation.recommended_action}")