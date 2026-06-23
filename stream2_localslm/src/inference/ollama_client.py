"""Generic Ollama inference client for sovereign (zero-egress) deployments.

Provides chat and generate interfaces to a local or network-local Ollama
instance. Domain-agnostic — callers supply their own prompts. Stream 1
explanation wrappers and the EIA core OllamaRecommender both sit above
this transport layer.

Design decisions:
- httpx.AsyncClient with a shared session — not created per-call.
- Startup health check verifies Ollama is reachable and model is pulled.
- stream=False / format="json" in chat mode — avoids partial-JSON parsing.
- Retry logic is the caller's responsibility; this client raises on failure.
- Zero AWS calls — all inference is local, no data leaves the machine.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
_DEFAULT_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "60.0"))
_DEFAULT_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))


@dataclass
class OllamaClient:
    """Generic async Ollama inference client.

    Supports both the /api/chat (system + user messages) and /api/generate
    (single-string prompt) interfaces. Callers choose the interface that
    matches their prompt structure.

    Args:
        base_url: Ollama server base URL. Defaults to OLLAMA_BASE_URL env var.
        model: Model name as registered in Ollama (e.g. "mistral:latest").
        timeout_s: HTTP request timeout in seconds.
        temperature: Sampling temperature applied to all calls.
        num_predict: Maximum tokens to generate per call.
    """

    base_url: str = field(default_factory=lambda: _DEFAULT_BASE_URL)
    model: str = field(default_factory=lambda: _DEFAULT_MODEL)
    timeout_s: float = field(default_factory=lambda: _DEFAULT_TIMEOUT_S)
    temperature: float = 0.0
    num_predict: int = field(default_factory=lambda: _DEFAULT_NUM_PREDICT)

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_s,
        )

    async def check_health(self) -> None:
        """Verify Ollama is reachable and the configured model is available.

        Call once at application startup — not before every inference call.

        Raises:
            RuntimeError: If Ollama is unreachable, returns an error status,
                or the configured model is not pulled.
        """
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start with: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama health check returned {e.response.status_code}"
            ) from e
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Ollama health check timed out after {self.timeout_s}s"
            )

        available = [m["name"] for m in response.json().get("models", [])]
        configured = self.model.split(":")[0].lower()
        if not any(m.split(":")[0].lower() == configured for m in available):
            available_str = ", ".join(available) or "(none)"
            raise RuntimeError(
                f"Model '{self.model}' not available in Ollama. "
                f"Available: {available_str}. "
                f"Pull with: ollama pull {self.model}"
            )

        logger.info(
            "Ollama health check passed — model '%s' available at %s",
            self.model,
            self.base_url,
        )

    async def chat(self, system: str, user: str) -> str:
        """Send a chat-style request (system + user messages) to Ollama.

        Uses /api/chat with format="json" to encourage structured output.
        Temperature is fixed at self.temperature for reproducibility.

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
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            content: str = response.json()["message"]["content"]
            logger.debug(
                "Ollama chat — model=%s chars=%d", self.model, len(content)
            )
            return content
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Ollama chat timed out after {self.timeout_s}s (model: {self.model})"
            ) from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}") from e

    async def generate(self, prompt: str) -> str:
        """Send a single-string prompt to Ollama via /api/generate.

        Use this when the prompt is a single block rather than
        separate system/user messages.

        Args:
            prompt: Complete prompt string.

        Returns:
            Raw text response from the model.

        Raises:
            RuntimeError: On HTTP error or request timeout.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        try:
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            raw: str = response.json()["response"]
            logger.debug(
                "Ollama generate — model=%s chars=%d", self.model, len(raw)
            )
            return raw
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"Ollama generate timed out after {self.timeout_s}s (model: {self.model})"
            ) from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}") from e

    async def list_models(self) -> list[str]:
        """Return names of all models currently pulled in Ollama.

        Returns:
            List of model name strings (e.g. ["mistral:latest", "llama3.2:3b"]).

        Raises:
            RuntimeError: If Ollama is unreachable.
        """
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Could not list Ollama models: {e}") from e
        return [m["name"] for m in response.json().get("models", [])]

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Call on application shutdown."""
        await self._client.aclose()
