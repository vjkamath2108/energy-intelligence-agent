"""Unit tests for stream2_localslm sovereign inference clients.

All HTTP calls are mocked — no Ollama or vLLM instance required.
Tests cover the happy path, error handling, and health check logic
for both OllamaClient and VLLMClient.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from stream2_localslm.src.inference.ollama_client import OllamaClient
from stream2_localslm.src.inference.vllm_client import VLLMClient
from stream2_localslm.src.inference.benchmarks import (
    BenchmarkReport,
    RunResult,
    run_benchmark,
)


# ---------------------------------------------------------------------------
# OllamaClient — health check
# ---------------------------------------------------------------------------


async def test_ollama_health_check_passes() -> None:
    """check_health succeeds when Ollama is reachable and model is available."""
    client = OllamaClient(model="mistral:latest")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"models": [{"name": "mistral:latest"}]}

    with patch.object(client._client, "get", new=AsyncMock(return_value=mock_response)):
        await client.check_health()

    await client.aclose()


async def test_ollama_health_check_connect_error() -> None:
    """check_health raises RuntimeError when Ollama is not reachable."""
    client = OllamaClient(model="mistral:latest")

    with patch.object(
        client._client,
        "get",
        new=AsyncMock(side_effect=httpx.ConnectError("refused")),
    ):
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            await client.check_health()

    await client.aclose()


async def test_ollama_health_check_model_not_available() -> None:
    """check_health raises RuntimeError when the configured model is not pulled."""
    client = OllamaClient(model="mistral:latest")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"models": [{"name": "llama3.2:3b"}]}

    with patch.object(client._client, "get", new=AsyncMock(return_value=mock_response)):
        with pytest.raises(RuntimeError, match="not available in Ollama"):
            await client.check_health()

    await client.aclose()


async def test_ollama_health_check_timeout() -> None:
    """check_health raises RuntimeError on timeout."""
    client = OllamaClient(model="mistral:latest")

    with patch.object(
        client._client,
        "get",
        new=AsyncMock(side_effect=httpx.TimeoutException("timeout")),
    ):
        with pytest.raises(RuntimeError, match="timed out"):
            await client.check_health()

    await client.aclose()


# ---------------------------------------------------------------------------
# OllamaClient — chat
# ---------------------------------------------------------------------------


async def test_ollama_chat_returns_content() -> None:
    """chat() calls /api/chat and returns the message content string."""
    client = OllamaClient(model="mistral:latest")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "message": {"content": '{"action": "dispatch unit 3"}'}
    }

    with patch.object(
        client._client, "post", new=AsyncMock(return_value=mock_response)
    ) as mock_post:
        result = await client.chat("You are an expert.", "What should I do?")

    assert result == '{"action": "dispatch unit 3"}'
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
    assert payload["model"] == "mistral:latest"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert payload["stream"] is False

    await client.aclose()


async def test_ollama_chat_timeout_raises() -> None:
    """chat() raises RuntimeError on timeout."""
    client = OllamaClient(model="mistral:latest")

    with patch.object(
        client._client,
        "post",
        new=AsyncMock(side_effect=httpx.TimeoutException("timeout")),
    ):
        with pytest.raises(RuntimeError, match="timed out"):
            await client.chat("system", "user")

    await client.aclose()


async def test_ollama_chat_http_error_raises() -> None:
    """chat() raises RuntimeError on HTTP error."""
    client = OllamaClient(model="mistral:latest")

    with patch.object(
        client._client,
        "post",
        new=AsyncMock(side_effect=httpx.HTTPError("server error")),
    ):
        with pytest.raises(RuntimeError, match="HTTP error"):
            await client.chat("system", "user")

    await client.aclose()


# ---------------------------------------------------------------------------
# OllamaClient — generate
# ---------------------------------------------------------------------------


async def test_ollama_generate_returns_response() -> None:
    """generate() calls /api/generate and returns the response string."""
    client = OllamaClient(model="llama3.2:3b")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "The BSUoS tariff is 4.82 per MWh."}

    with patch.object(
        client._client, "post", new=AsyncMock(return_value=mock_response)
    ):
        result = await client.generate("What is the BSUoS tariff?")

    assert result == "The BSUoS tariff is 4.82 per MWh."
    await client.aclose()


# ---------------------------------------------------------------------------
# OllamaClient — list_models
# ---------------------------------------------------------------------------


async def test_ollama_list_models_returns_names() -> None:
    """list_models() returns model name strings from /api/tags."""
    client = OllamaClient()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "models": [{"name": "mistral:latest"}, {"name": "llama3.2:3b"}]
    }

    with patch.object(client._client, "get", new=AsyncMock(return_value=mock_response)):
        models = await client.list_models()

    assert models == ["mistral:latest", "llama3.2:3b"]
    await client.aclose()


# ---------------------------------------------------------------------------
# VLLMClient — health check
# ---------------------------------------------------------------------------


async def test_vllm_health_check_passes() -> None:
    """check_health succeeds when vLLM is reachable and model is loaded."""
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    client = VLLMClient(model=model_id)
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"data": [{"id": model_id}]}

    with patch.object(client._client, "get", new=AsyncMock(return_value=mock_response)):
        await client.check_health()

    await client.aclose()


async def test_vllm_health_check_connect_error() -> None:
    """check_health raises RuntimeError when vLLM server is not reachable."""
    client = VLLMClient(model="mistralai/Mistral-7B-Instruct-v0.3")

    with patch.object(
        client._client,
        "get",
        new=AsyncMock(side_effect=httpx.ConnectError("refused")),
    ):
        with pytest.raises(RuntimeError, match="Cannot connect to vLLM"):
            await client.check_health()

    await client.aclose()


async def test_vllm_health_check_model_not_loaded() -> None:
    """check_health raises RuntimeError when the model is not loaded in vLLM."""
    client = VLLMClient(model="mistralai/Mistral-7B-Instruct-v0.3")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"data": [{"id": "some-other-model"}]}

    with patch.object(client._client, "get", new=AsyncMock(return_value=mock_response)):
        with pytest.raises(RuntimeError, match="not loaded in vLLM"):
            await client.check_health()

    await client.aclose()


# ---------------------------------------------------------------------------
# VLLMClient — chat
# ---------------------------------------------------------------------------


async def test_vllm_chat_returns_content() -> None:
    """chat() calls /v1/chat/completions and returns the message content."""
    client = VLLMClient(model="mistralai/Mistral-7B-Instruct-v0.3")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Dispatch unit 3 at 100 MW."}}]
    }

    with patch.object(
        client._client, "post", new=AsyncMock(return_value=mock_response)
    ) as mock_post:
        result = await client.chat("You are an expert.", "Should we dispatch unit 3?")

    assert result == "Dispatch unit 3 at 100 MW."
    mock_post.assert_called_once()
    call_json = mock_post.call_args[1]["json"]
    assert call_json["model"] == "mistralai/Mistral-7B-Instruct-v0.3"
    assert call_json["messages"][0]["role"] == "system"
    assert call_json["messages"][1]["role"] == "user"
    assert call_json["temperature"] == 0.0

    await client.aclose()


async def test_vllm_chat_timeout_raises() -> None:
    """chat() raises RuntimeError on timeout."""
    client = VLLMClient(model="mistralai/Mistral-7B-Instruct-v0.3")

    with patch.object(
        client._client,
        "post",
        new=AsyncMock(side_effect=httpx.TimeoutException("timeout")),
    ):
        with pytest.raises(RuntimeError, match="timed out"):
            await client.chat("system", "user")

    await client.aclose()


async def test_vllm_api_key_added_to_headers() -> None:
    """VLLMClient with api_key sets Authorization header on the HTTP client."""
    client = VLLMClient(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        api_key="test-secret-token",
    )
    assert client._client.headers.get("authorization") == "Bearer test-secret-token"
    await client.aclose()


async def test_vllm_list_models_returns_ids() -> None:
    """list_models() returns model ID strings from /v1/models."""
    client = VLLMClient(model="mistralai/Mistral-7B-Instruct-v0.3")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"id": "mistralai/Mistral-7B-Instruct-v0.3"},
            {"id": "mistralai/Mistral-7B-v0.1"},
        ]
    }

    with patch.object(client._client, "get", new=AsyncMock(return_value=mock_response)):
        models = await client.list_models()

    assert "mistralai/Mistral-7B-Instruct-v0.3" in models
    await client.aclose()


# ---------------------------------------------------------------------------
# BenchmarkReport — aggregate stats
# ---------------------------------------------------------------------------


def test_benchmark_report_success_rate() -> None:
    """success_rate is correct fraction of successful runs."""
    report = BenchmarkReport(backend="ollama", model="mistral:latest")
    report.runs = [
        RunResult("ollama", "mistral:latest", "p1", 1200.0, 100, True),
        RunResult("ollama", "mistral:latest", "p2", 1500.0, 120, True),
        RunResult("ollama", "mistral:latest", "p3", 0.0, 0, False, error="timeout"),
    ]
    assert report.success_rate == pytest.approx(2 / 3)


def test_benchmark_report_p50_latency() -> None:
    """p50_latency_ms returns median of successful run latencies."""
    report = BenchmarkReport(backend="ollama", model="mistral:latest")
    report.runs = [
        RunResult("ollama", "mistral:latest", "p1", 1000.0, 100, True),
        RunResult("ollama", "mistral:latest", "p2", 2000.0, 120, True),
        RunResult("ollama", "mistral:latest", "p3", 3000.0, 130, True),
        RunResult("ollama", "mistral:latest", "p4", 0.0, 0, False, error="err"),
    ]
    assert report.p50_latency_ms == pytest.approx(2000.0)


def test_benchmark_report_p95_latency_single_run() -> None:
    """p95_latency_ms returns the only value when there is one successful run."""
    report = BenchmarkReport(backend="vllm", model="mistral")
    report.runs = [RunResult("vllm", "mistral", "p1", 800.0, 50, True)]
    assert report.p95_latency_ms == pytest.approx(800.0)


def test_benchmark_report_empty_report() -> None:
    """Empty report returns None for latency stats and 0.0 for success rate."""
    report = BenchmarkReport(backend="ollama", model="mistral:latest")
    assert report.success_rate == 0.0
    assert report.p50_latency_ms is None
    assert report.p95_latency_ms is None
    assert report.mean_chars_out is None


# ---------------------------------------------------------------------------
# run_benchmark — integration (fully mocked)
# ---------------------------------------------------------------------------


async def test_run_benchmark_ollama_only() -> None:
    """run_benchmark with include_vllm=False runs only Ollama and returns one report."""
    mock_content = "The BSUoS tariff applies as follows..."

    with (
        patch(
            "stream2_localslm.src.inference.benchmarks.OllamaClient"
        ) as MockOllama,
    ):
        mock_ollama_instance = AsyncMock()
        mock_ollama_instance.model = "mistral:latest"
        mock_ollama_instance.check_health = AsyncMock()
        mock_ollama_instance.chat = AsyncMock(return_value=mock_content)
        mock_ollama_instance.aclose = AsyncMock()
        MockOllama.return_value = mock_ollama_instance

        reports = await run_benchmark(
            ollama_model="mistral:latest",
            vllm_model="mistralai/Mistral-7B-Instruct-v0.3",
            runs_per_prompt=1,
            include_vllm=False,
        )

    assert len(reports) == 1
    assert reports[0].backend == "ollama"
    assert reports[0].success_rate == 1.0
    mock_ollama_instance.check_health.assert_called_once()
