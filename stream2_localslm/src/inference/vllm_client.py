"""vLLM inference client for production sovereign deployments on EKS.

Targets the vLLM OpenAI-compatible REST API (/v1/chat/completions,
/v1/models). In production, this points at a vLLM deployment inside the
client's VPC on EKS — zero data egress, no external API dependency.

Design decisions:
- OpenAI-compatible API: vLLM exposes the same schema as the OpenAI SDK,
  so callers can swap to hosted OpenAI by changing base_url without any
  code changes beyond the URL.
- No API key by default: internal VPC deployments typically authenticate
  via IAM/IRSA or network policy, not bearer tokens. The api_key field
  is optional and only set when vLLM is deployed behind an API gateway.
- Same chat() interface as OllamaClient: callers use either backend
  interchangeably.
- health check uses /v1/models (OpenAI standard) rather than a
  vLLM-specific endpoint for portability.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
_DEFAULT_MODEL = os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
_DEFAULT_TIMEOUT_S = float(os.getenv("VLLM_TIMEOUT_S", "30.0"))
_DEFAULT_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "1024"))


@dataclass
class VLLMClient:
    """Async vLLM inference client using the OpenAI-compatible REST API.

    Suitable for sovereign EKS deployments where Mistral or other open-weight
    models are served inside the client VPC with zero external calls.

    Args:
        base_url: vLLM server base URL. Defaults to VLLM_BASE_URL env var.
        model: Model identifier as registered in vLLM.
        timeout_s: HTTP request timeout in seconds.
        temperature: Sampling temperature. Use 0.0 for deterministic output.
        max_tokens: Maximum tokens in the completion.
        api_key: Optional bearer token for gateway-protected deployments.
    """

    base_url: str = field(default_factory=lambda: _DEFAULT_BASE_URL)
    model: str = field(default_factory=lambda: _DEFAULT_MODEL)
    timeout_s: float = field(default_factory=lambda: _DEFAULT_TIMEOUT_S)
    temperature: float = 0.0
    max_tokens: int = field(default_factory=lambda: _DEFAULT_MAX_TOKENS)
    api_key: str | None = None

    def __post_init__(self) -> None:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout_s,
        )

    async def check_health(self) -> None:
        """Verify vLLM is reachable and the configured model is loaded.

        Uses the OpenAI-compatible /v1/models endpoint. Call once at startup.

        Raises:
            RuntimeError: If vLLM is unreachable or the model is not loaded.
        """
        try:
            response = await self._client.get("/v1/models")
            response.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to vLLM at {self.base_url}. "
                "Is the vLLM server running?"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"vLLM health check returned {e.response.status_code}"
            ) from e
        except httpx.TimeoutException:
            raise RuntimeError(
                f"vLLM health check timed out after {self.timeout_s}s"
            )

        models = [m["id"] for m in response.json().get("data", [])]
        if self.model not in models:
            available = ", ".join(models) or "(none)"
            raise RuntimeError(
                f"Model '{self.model}' not loaded in vLLM. "
                f"Loaded models: {available}"
            )

        logger.info(
            "vLLM health check passed — model '%s' available at %s",
            self.model,
            self.base_url,
        )

    async def chat(self, system: str, user: str) -> str:
        """Send a chat-style request to vLLM via /v1/chat/completions.

        Interface is identical to OllamaClient.chat() — callers swap backends
        by swapping the client instance, not by changing call sites.

        Args:
            system: System prompt defining the model's role and output format.
            user: User message containing the specific request.

        Returns:
            Raw text content of the model's response.

        Raises:
            RuntimeError: On HTTP error or request timeout.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            response = await self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            content: str = response.json()["choices"][0]["message"]["content"]
            logger.debug(
                "vLLM chat — model=%s chars=%d", self.model, len(content)
            )
            return content
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"vLLM chat timed out after {self.timeout_s}s (model: {self.model})"
            ) from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"vLLM HTTP error: {e}") from e

    async def list_models(self) -> list[str]:
        """Return IDs of all models currently loaded in vLLM.

        Returns:
            List of model ID strings.

        Raises:
            RuntimeError: If vLLM is unreachable.
        """
        try:
            response = await self._client.get("/v1/models")
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Could not list vLLM models: {e}") from e
        return [m["id"] for m in response.json().get("data", [])]

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Call on application shutdown."""
        await self._client.aclose()
