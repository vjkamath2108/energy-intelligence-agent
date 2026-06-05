"""
Mistral Bedrock explainer for anomaly events.

Calls Mistral 7B Instruct via AWS Bedrock to generate human-readable
explanations when Isolation Forest flags an anomaly.

Design decisions:
- Identical interface to BedrockExplainer and OllamaExplainer — swap by flag.
- Mistral 7B on Bedrock uses invoke_model with [INST] prompt format,
  NOT the Converse/Messages API used by Claude.
- Response extracted from outputs[0].text, not content[0].text.
- Severity normalisation copied from OllamaExplainer — Mistral occasionally
  returns "<MEDIUM>" or " HIGH" formatting artifacts.
- Same retry + reprompt logic on invalid JSON.
- Same Langfuse tracing and anomaly_log append pattern.
"""

import json
import logging
import os
import time
from typing import Any, cast

import asyncio
import boto3  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]
from pydantic import ValidationError

from stream1_log_anomaly.src.models import LogEvent, Explanation
from stream1_log_anomaly.src.explanation.prompts import build_prompt
from stream1_log_anomaly.src.instrumentation.langfuse_client import trace_explanation
from stream1_log_anomaly.src.instrumentation.anomaly_log import append_explanation

logger = logging.getLogger(__name__)

MISTRAL_BEDROCK_MODEL_ID = os.getenv(
    "MISTRAL_BEDROCK_MODEL_ID",
    "mistral.mistral-7b-instruct-v0:2",
)
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "eu-west-2")
MAX_RETRIES = 3
BASE_BACKOFF = 1.0


def _build_mistral_prompt(prompt: str) -> str:
    """
    Wrap prompt in Mistral [INST] format required by Bedrock invoke_model.

    Mistral 7B expects: <s>[INST] {user_message} [/INST]
    """
    return f"<s>[INST] {prompt} [/INST]"


def _call_mistral_bedrock_sync(client: Any, prompt: str) -> str:
    """
    Call Mistral 7B via Bedrock invoke_model and return raw text response.

    NOTE: Mistral on Bedrock uses a different request/response schema to Claude:
    - Request body: {"prompt": "...", "max_tokens": N}  (not "messages")
    - Response body: {"outputs": [{"text": "..."}]}  (not "content")

    Retries on ThrottlingException with exponential backoff.
    """
    body = {
        "prompt": _build_mistral_prompt(prompt),
        "max_tokens": 512,
        "temperature": 0.1,  # Low temp for structured JSON output
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = client.invoke_model(
                modelId=MISTRAL_BEDROCK_MODEL_ID,
                body=json.dumps(body),
            )
            raw = json.loads(response["body"].read())
            # Mistral response schema: {"outputs": [{"text": "...", "stop_reason": "..."}]}
            return cast(str, raw["outputs"][0]["text"])

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ThrottlingException" and attempt < MAX_RETRIES - 1:
                wait = BASE_BACKOFF * (2 ** attempt)
                logger.warning(
                    "Mistral Bedrock throttled — retrying in %.1fs (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Mistral Bedrock call failed after {MAX_RETRIES} attempts")


async def _call_mistral_bedrock(client: Any, prompt: str) -> str:
    """Async wrapper — runs blocking boto3 call in thread pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _call_mistral_bedrock_sync, client, prompt)


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


class MistralBedrockExplainer:
    """
    Generates human-readable anomaly explanations via Mistral 7B on AWS Bedrock.

    Interface is identical to BedrockExplainer and OllamaExplainer —
    swap by flag, not by rewriting call sites.

    Usage:
        explainer = MistralBedrockExplainer()
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

        Retries once with stricter prompt if Mistral returns invalid JSON.
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
        is_retry = False

        async with trace_explanation(
            service=anomalous_event.service,
            level=anomalous_event.level,
            latency_ms=anomalous_event.latency_ms,
            backend="mistral-bedrock",
            model=MISTRAL_BEDROCK_MODEL_ID,
            prompt=prompt,
        ) as span:
            raw = await _call_mistral_bedrock(self._client, prompt)
            logger.debug(
                "Mistral Bedrock raw response: %s...",
                raw[:200].replace("\n", " "),
            )

            try:
                explanation = _parse_explanation(raw)
            except ValueError:
                is_retry = True
                logger.warning(
                    "Invalid JSON from Mistral Bedrock — retrying with stricter prompt"
                )
                strict_prompt = (
                    prompt
                    + "\n\nCRITICAL: Return ONLY a valid JSON object. "
                    "No markdown. No explanation. No text before or after the JSON."
                )
                raw = await _call_mistral_bedrock(self._client, strict_prompt)
                try:
                    explanation = _parse_explanation(raw)
                except ValueError as e:
                    raise RuntimeError(f"Failed after retry: {e}") from e

            elapsed_ms = (time.monotonic() - start) * 1000

            span["output"] = explanation.model_dump()
            span["metadata"] = {
                "model": MISTRAL_BEDROCK_MODEL_ID,
                "backend": "mistral-bedrock",
                "latency_ms": round(elapsed_ms),
                "is_retry": is_retry,
            }

        logger.info(
            "Mistral Bedrock explanation — model=%s latency=%.0fms retry=%s severity=%s",
            MISTRAL_BEDROCK_MODEL_ID,
            elapsed_ms,
            is_retry,
            explanation.severity,
        )

        append_explanation(anomalous_event, explanation, "mistral-bedrock", elapsed_ms)
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
        explainer = MistralBedrockExplainer()

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