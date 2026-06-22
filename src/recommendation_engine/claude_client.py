"""AWS Bedrock Claude wrapper for EIA dispatch recommendations.

Calls anthropic.claude-sonnet-4-5 via the Bedrock Messages API and returns
a validated DispatchRecommendation with safety checks applied.

Design decisions:
- boto3 invoke_model is synchronous; run in thread pool executor to avoid
  blocking the asyncio event loop during Bedrock latency (typically 1-3s).
- Retry on ThrottlingException / ServiceUnavailableException with exponential
  backoff (max 3 attempts: 1s, 2s, 4s delays).
- JSON retry: if Claude returns non-JSON, reprompt once with the parse error.
- Safety checks (term overlap + optional semantic similarity) run before the
  DispatchRecommendation is constructed — all flags set at creation time.
- temperature=0.0 for reproducibility in safety-critical dispatch contexts.
- Interface identical to OllamaRecommender — swap backend by flag, not by
  rewriting call sites.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, cast

import boto3  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from src.models import DispatchRecommendation, RetrievedDocument
from src.recommendation_engine._parsing import ParsedLLMOutput, parse_llm_output
from src.recommendation_engine.confidence import (
    ConfidenceScorer,
    ConfidenceSignals,
)
from src.recommendation_engine.prompts import build_user_message, load_system_prompt
from src.recommendation_engine.safety import (
    EmbedderProtocol,
    SafetyResult,
    check_grounding,
)
from src.safety.constraint_validator import ValidationResult, check_constraints

logger = logging.getLogger(__name__)

_BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-5")
_BEDROCK_REGION = os.getenv("AWS_REGION", "eu-west-2")
_MAX_RETRIES = 3
_BASE_BACKOFF = 1.0
_MAX_TOKENS = 1024


def _call_bedrock_sync(
    client: Any,
    model_id: str,
    system_prompt: str,
    user_message: str,
) -> str:
    """Call Bedrock Claude synchronously and return the raw response text.

    Retries on ThrottlingException / ServiceUnavailableException with
    exponential backoff.

    Args:
        client: boto3 bedrock-runtime client.
        model_id: Bedrock model ID (e.g. 'anthropic.claude-sonnet-4-5').
        system_prompt: System prompt string for the Messages API.
        user_message: User message string.

    Returns:
        Raw text content from Claude's response.

    Raises:
        ClientError: For non-retryable Bedrock errors.
        RuntimeError: If all retries are exhausted.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": _MAX_TOKENS,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
        "temperature": 0.0,
    }

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.invoke_model(modelId=model_id, body=json.dumps(body))
            result = json.loads(response["body"].read())
            return cast(str, result["content"][0]["text"])
        except ClientError as e:
            code = e.response["Error"]["Code"]
            retryable = code in ("ThrottlingException", "ServiceUnavailableException")
            if retryable and attempt < _MAX_RETRIES - 1:
                wait = _BASE_BACKOFF * (2 ** attempt)
                logger.warning(
                    "Bedrock throttled — retrying in %.1fs (%d/%d)",
                    wait, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Bedrock call failed after {_MAX_RETRIES} attempts")  # pragma: no cover


@dataclass
class ClaudeRecommender:
    """Generates DispatchRecommendation via AWS Bedrock Claude.

    Interface is identical to OllamaRecommender — swap by flag at startup,
    not by rewriting call sites.

    Args:
        model_id: Bedrock model ID. Defaults to BEDROCK_MODEL_ID env var.
        region: AWS region. Defaults to AWS_REGION env var.
        embedder: Optional embedder for semantic grounding check. When None,
                  only term overlap is used for hallucination detection.
        prompt_version: Prompt config version to load from prompt_configs/.

    Usage:
        recommender = ClaudeRecommender(embedder=TitanEmbedder())
        rec = await recommender.recommend("BSUoS dispatch query", context_docs)
    """

    model_id: str = field(default_factory=lambda: _BEDROCK_MODEL_ID)
    region: str = field(default_factory=lambda: _BEDROCK_REGION)
    embedder: EmbedderProtocol | None = None
    prompt_version: str = "v1"
    scorer: ConfidenceScorer = field(default_factory=ConfidenceScorer)

    def __post_init__(self) -> None:
        self._client = boto3.client("bedrock-runtime", region_name=self.region)
        self._system_prompt = load_system_prompt(self.prompt_version)

    async def _call_bedrock(self, user_message: str) -> str:
        """Run the synchronous Bedrock call in the event loop's thread pool.

        Args:
            user_message: Formatted user message string.

        Returns:
            Raw text response from Claude.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            _call_bedrock_sync,
            self._client,
            self.model_id,
            self._system_prompt,
            user_message,
        )

    async def recommend(
        self,
        query: str,
        context_docs: list[RetrievedDocument],
    ) -> DispatchRecommendation:
        """Generate a grounded DispatchRecommendation for the given query.

        Calls Bedrock Claude, parses JSON output, runs safety checks, and
        returns a fully-formed DispatchRecommendation with all flags set.

        Args:
            query: Operator's natural-language dispatch query.
            context_docs: Retrieved documents from the hybrid retriever.

        Returns:
            DispatchRecommendation with hallucination_flagged and
            requires_human_review set from safety check results.

        Raises:
            RuntimeError: If Bedrock call fails after retries.
            ValueError: If Claude returns invalid JSON after one retry.
        """
        user_message = build_user_message(query, context_docs)
        raw = await self._call_bedrock(user_message)

        try:
            parsed: ParsedLLMOutput = parse_llm_output(raw, context_docs)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("JSON parse failed — retrying with stricter prompt: %s", e)
            retry_msg = (
                user_message
                + f"\n\nYou MUST return only valid JSON. Previous error: {e}"
            )
            raw = await self._call_bedrock(retry_msg)
            parsed = parse_llm_output(raw, context_docs)

        safety: SafetyResult = await check_grounding(
            parsed["action"], parsed["reasoning"], context_docs, self.embedder
        )

        signals = ConfidenceSignals(
            llm_confidence=parsed["confidence"],
            safety_result=safety,
            supporting_documents=parsed["supporting_documents"],
        )
        confidence, requires_review = self.scorer.score(signals)

        constraint_result: ValidationResult = check_constraints(
            parsed["action"], context_docs
        )
        if not constraint_result.passed:
            logger.warning(
                "Regulatory constraint violations detected — flagging for human review: %s",
                [v.reason for v in constraint_result.violations],
            )

        return DispatchRecommendation(
            query=query,
            action=parsed["action"],
            confidence=confidence,
            supporting_documents=parsed["supporting_documents"],
            reasoning=parsed["reasoning"],
            requires_human_review=(
                requires_review or not safety.is_grounded or not constraint_result.passed
            ),
            hallucination_flagged=not safety.is_grounded,
        )
