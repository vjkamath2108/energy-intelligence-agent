"""Ollama inference backend for EIA dispatch recommendations.

Calls a local Ollama instance to generate DispatchRecommendation when
operating in sovereign (no-cloud) mode. Interface is identical to
ClaudeRecommender — swap by CLI flag at startup, not by rewriting call sites.

Design decisions:
- Uses httpx.AsyncClient with a shared session (created at instantiation).
- Startup health check verifies Ollama is running and model is pulled — fails
  fast before the first recommend() call.
- format: "json" in the Ollama payload enables JSON mode (Ollama >= 0.1.9).
- No embedder for semantic grounding: sovereign mode avoids all Bedrock calls;
  term overlap check is applied via safety.check_grounding(embedder=None).
- Same retry-on-invalid-JSON pattern as claude_client.
- Zero AWS calls — fully local inference, no data leaves the machine.
"""

import json
import logging
import os
from dataclasses import dataclass, field

import httpx

from src.models import DispatchRecommendation, RetrievedDocument
from src.recommendation_engine._parsing import ParsedLLMOutput, parse_llm_output
from src.recommendation_engine.confidence import (
    ConfidenceScorer,
    ConfidenceSignals,
)
from src.recommendation_engine.prompts import build_user_message, load_system_prompt
from src.recommendation_engine.safety import (
    SafetyResult,
    check_grounding,
)
from src.safety.constraint_validator import ValidationResult, check_constraints

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
_OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "60.0"))
_MAX_TOKENS = 1024


@dataclass
class OllamaRecommender:
    """Generates DispatchRecommendation via a local Ollama instance.

    Interface is identical to ClaudeRecommender — swap by flag at startup,
    not by rewriting call sites. Zero AWS calls — fully local inference.

    Args:
        base_url: Ollama server URL. Defaults to OLLAMA_BASE_URL env var.
        model: Ollama model name. Defaults to OLLAMA_MODEL env var.
        prompt_version: Prompt config version to load from prompt_configs/.

    Usage:
        recommender = OllamaRecommender()
        await recommender.check_health()   # once at startup
        rec = await recommender.recommend("BSUoS dispatch query", context_docs)
        await recommender.aclose()         # on shutdown
    """

    base_url: str = field(default_factory=lambda: _OLLAMA_BASE_URL)
    model: str = field(default_factory=lambda: _OLLAMA_MODEL)
    prompt_version: str = "v1"
    scorer: ConfidenceScorer = field(default_factory=ConfidenceScorer)

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=_OLLAMA_TIMEOUT_S
        )
        self._system_prompt = load_system_prompt(self.prompt_version)

    async def check_health(self) -> None:
        """Verify Ollama is running and the configured model is available.

        Call once at application startup — not on every recommend() call.

        Raises:
            RuntimeError: If Ollama is unreachable or the model is not pulled.
        """
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Start it with: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama health check returned {e.response.status_code}"
            ) from e

        models = [m["name"] for m in response.json().get("models", [])]
        configured = self.model.split(":")[0].lower()
        if not any(m.split(":")[0].lower() == configured for m in models):
            available = ", ".join(models) or "(none)"
            raise RuntimeError(
                f"Model '{self.model}' not available in Ollama. "
                f"Available: {available}. Pull with: ollama pull {self.model}"
            )
        logger.info("Ollama health check passed — model '%s' available", self.model)

    async def _call_ollama(self, user_message: str) -> str:
        """Call Ollama chat API and return the raw response text.

        Args:
            user_message: Formatted user message string.

        Returns:
            Raw text response from Ollama.

        Raises:
            RuntimeError: On Ollama connection timeout or HTTP error.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "format": "json",
            "options": {"num_predict": _MAX_TOKENS, "temperature": 0.0},
        }
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            return str(response.json()["message"]["content"])
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Ollama call timed out after {_OLLAMA_TIMEOUT_S}s "
                f"(model: {self.model})"
            ) from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}") from e

    async def recommend(
        self,
        query: str,
        context_docs: list[RetrievedDocument],
    ) -> DispatchRecommendation:
        """Generate a grounded DispatchRecommendation using local Ollama inference.

        Identical output schema to ClaudeRecommender. Term overlap safety check
        is applied; semantic similarity check is skipped (no Bedrock calls).

        Args:
            query: Operator's natural-language dispatch query.
            context_docs: Retrieved documents from the hybrid retriever.

        Returns:
            DispatchRecommendation with hallucination_flagged and
            requires_human_review set from safety check results.

        Raises:
            RuntimeError: If Ollama call fails.
            ValueError: If Ollama returns invalid JSON after one retry.
        """
        user_message = build_user_message(query, context_docs)
        raw = await self._call_ollama(user_message)

        try:
            parsed: ParsedLLMOutput = parse_llm_output(raw, context_docs)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("JSON parse failed — retrying: %s", e)
            retry_msg = (
                user_message + "\n\nReturn ONLY a valid JSON object. No other text."
            )
            raw = await self._call_ollama(retry_msg)
            parsed = parse_llm_output(raw, context_docs)

        safety: SafetyResult = await check_grounding(
            parsed["action"], parsed["reasoning"], context_docs, embedder=None
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
                requires_review
                or not safety.is_grounded
                or not constraint_result.passed
            ),
            hallucination_flagged=not safety.is_grounded,
        )

    async def aclose(self) -> None:
        """Close the underlying httpx client. Call on application shutdown."""
        await self._client.aclose()
