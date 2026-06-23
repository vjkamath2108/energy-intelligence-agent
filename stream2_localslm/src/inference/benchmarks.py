"""Inference backend benchmark runner — Ollama vs vLLM latency and throughput.

Runs a configurable prompt suite against one or both sovereign backends and
emits a structured report. Designed for repeatable comparison during model
selection and hardware sizing decisions.

Usage:
    # Ollama only (default)
    python -m stream2_localslm.src.inference.benchmarks

    # Both Ollama and vLLM
    python -m stream2_localslm.src.inference.benchmarks --vllm

    # Custom prompt count and model
    python -m stream2_localslm.src.inference.benchmarks --runs 20 --model mistral:latest
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import statistics
import time
from dataclasses import dataclass, field

from stream2_localslm.src.inference.ollama_client import OllamaClient
from stream2_localslm.src.inference.vllm_client import VLLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark prompt suite — energy domain queries
# ---------------------------------------------------------------------------

BENCHMARK_PROMPTS: list[tuple[str, str]] = [
    (
        "bsuos-sp23",
        "What BSUoS charge applies to settlement period SP23 for a 50 MW generator "
        "and how is it calculated under the balancing cost recovery mechanism?",
    ),
    (
        "tnuos-zone1",
        "Calculate the TNUoS generation tariff liability for a Zone 1 (North Scotland) "
        "generator with 120 MW registered capacity at the 2026/27 tariff rate.",
    ),
    (
        "frequency-response",
        "Under Grid Code ECC.6.3.2, what is the primary frequency response obligation "
        "for a 100 MW CCGT when frequency deviates by 0.5 Hz from nominal?",
    ),
    (
        "cm-penalty",
        "A capacity provider failed to deliver during a system stress event. "
        "Explain the Capacity Market penalty calculation under Section 9.4 for 2026/27.",
    ),
    (
        "bm-dispatch",
        "What are the settlement implications when BM unit SCOT-W01 receives an "
        "accepted offer dispatch instruction to increase output by 30 MW?",
    ),
]

_SYSTEM_PROMPT = (
    "You are an energy market expert specialising in UK electricity regulation. "
    "Answer the question concisely and accurately, citing specific rules where relevant. "
    "Respond in plain text, no formatting."
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    """Single benchmark run result."""

    backend: str
    model: str
    prompt_name: str
    latency_ms: float
    chars_out: int
    success: bool
    error: str = ""


@dataclass
class BenchmarkReport:
    """Aggregate benchmark report across all runs."""

    backend: str
    model: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def successful_runs(self) -> list[RunResult]:
        """Return only successful runs."""
        return [r for r in self.runs if r.success]

    @property
    def success_rate(self) -> float:
        """Fraction of runs that succeeded."""
        if not self.runs:
            return 0.0
        return len(self.successful_runs) / len(self.runs)

    @property
    def p50_latency_ms(self) -> float | None:
        """Median latency across successful runs."""
        latencies = [r.latency_ms for r in self.successful_runs]
        return statistics.median(latencies) if latencies else None

    @property
    def p95_latency_ms(self) -> float | None:
        """95th-percentile latency across successful runs."""
        latencies = sorted(r.latency_ms for r in self.successful_runs)
        if not latencies:
            return None
        idx = max(0, int(len(latencies) * 0.95) - 1)
        return latencies[idx]

    @property
    def mean_chars_out(self) -> float | None:
        """Mean output length across successful runs."""
        chars = [r.chars_out for r in self.successful_runs]
        return statistics.mean(chars) if chars else None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


async def _run_ollama(
    client: OllamaClient,
    prompts: list[tuple[str, str]],
    runs_per_prompt: int,
) -> BenchmarkReport:
    """Run all prompts against Ollama and collect results.

    Args:
        client: Initialised OllamaClient (health check already done).
        prompts: List of (name, user_text) tuples.
        runs_per_prompt: How many times to run each prompt (for latency stats).

    Returns:
        BenchmarkReport with per-run results.
    """
    report = BenchmarkReport(backend="ollama", model=client.model)

    for prompt_name, user_text in prompts:
        for _ in range(runs_per_prompt):
            start = time.monotonic()
            try:
                content = await client.chat(_SYSTEM_PROMPT, user_text)
                latency_ms = (time.monotonic() - start) * 1000
                report.runs.append(
                    RunResult(
                        backend="ollama",
                        model=client.model,
                        prompt_name=prompt_name,
                        latency_ms=latency_ms,
                        chars_out=len(content),
                        success=True,
                    )
                )
                logger.info(
                    "ollama  %-20s  %.0fms  %d chars",
                    prompt_name,
                    latency_ms,
                    len(content),
                )
            except RuntimeError as e:
                latency_ms = (time.monotonic() - start) * 1000
                report.runs.append(
                    RunResult(
                        backend="ollama",
                        model=client.model,
                        prompt_name=prompt_name,
                        latency_ms=latency_ms,
                        chars_out=0,
                        success=False,
                        error=str(e),
                    )
                )
                logger.warning("ollama  %s  FAILED: %s", prompt_name, e)

    return report


async def _run_vllm(
    client: VLLMClient,
    prompts: list[tuple[str, str]],
    runs_per_prompt: int,
) -> BenchmarkReport:
    """Run all prompts against vLLM and collect results.

    Args:
        client: Initialised VLLMClient (health check already done).
        prompts: List of (name, user_text) tuples.
        runs_per_prompt: How many times to run each prompt.

    Returns:
        BenchmarkReport with per-run results.
    """
    report = BenchmarkReport(backend="vllm", model=client.model)

    for prompt_name, user_text in prompts:
        for _ in range(runs_per_prompt):
            start = time.monotonic()
            try:
                content = await client.chat(_SYSTEM_PROMPT, user_text)
                latency_ms = (time.monotonic() - start) * 1000
                report.runs.append(
                    RunResult(
                        backend="vllm",
                        model=client.model,
                        prompt_name=prompt_name,
                        latency_ms=latency_ms,
                        chars_out=len(content),
                        success=True,
                    )
                )
                logger.info(
                    "vllm    %-20s  %.0fms  %d chars",
                    prompt_name,
                    latency_ms,
                    len(content),
                )
            except RuntimeError as e:
                latency_ms = (time.monotonic() - start) * 1000
                report.runs.append(
                    RunResult(
                        backend="vllm",
                        model=client.model,
                        prompt_name=prompt_name,
                        latency_ms=latency_ms,
                        chars_out=0,
                        success=False,
                        error=str(e),
                    )
                )
                logger.warning("vllm    %s  FAILED: %s", prompt_name, e)

    return report


def _print_report(report: BenchmarkReport) -> None:
    """Print a formatted benchmark summary to stdout.

    Args:
        report: Completed BenchmarkReport to display.
    """
    print(f"\n{'─' * 60}")
    print(f"  {report.backend.upper()}  |  model: {report.model}")
    print(f"{'─' * 60}")
    print(f"  Runs total   : {len(report.runs)}")
    print(f"  Success rate : {report.success_rate:.0%}")
    if report.p50_latency_ms is not None:
        print(f"  Latency P50  : {report.p50_latency_ms:.0f}ms")
    if report.p95_latency_ms is not None:
        print(f"  Latency P95  : {report.p95_latency_ms:.0f}ms")
    if report.mean_chars_out is not None:
        print(f"  Mean output  : {report.mean_chars_out:.0f} chars")

    failures = [r for r in report.runs if not r.success]
    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for f in failures:
            print(f"    [{f.prompt_name}] {f.error}")
    print()


async def run_benchmark(
    ollama_model: str,
    vllm_model: str,
    runs_per_prompt: int,
    include_vllm: bool,
) -> list[BenchmarkReport]:
    """Run the full benchmark suite and return all reports.

    Args:
        ollama_model: Ollama model name to benchmark.
        vllm_model: vLLM model ID to benchmark (if include_vllm is True).
        runs_per_prompt: Number of repetitions per prompt for latency stats.
        include_vllm: Whether to include the vLLM backend in the run.

    Returns:
        List of BenchmarkReport, one per backend.
    """
    reports: list[BenchmarkReport] = []

    ollama = OllamaClient(model=ollama_model)
    try:
        await ollama.check_health()
        report = await _run_ollama(ollama, BENCHMARK_PROMPTS, runs_per_prompt)
        reports.append(report)
    finally:
        await ollama.aclose()

    if include_vllm:
        vllm_base = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        vllm = VLLMClient(base_url=vllm_base, model=vllm_model)
        try:
            await vllm.check_health()
            report = await _run_vllm(vllm, BENCHMARK_PROMPTS, runs_per_prompt)
            reports.append(report)
        finally:
            await vllm.aclose()

    return reports


def main() -> None:
    """CLI entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Sovereign inference benchmark — Ollama vs vLLM"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "mistral:latest"),
        help="Ollama model name (default: mistral:latest)",
    )
    parser.add_argument(
        "--vllm-model",
        default=os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
        help="vLLM model ID",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Include vLLM backend in the benchmark",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Repetitions per prompt (default: 3)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s  %(message)s",
    )

    print("\nSovereign Inference Benchmark")
    print(f"  Ollama model : {args.model}")
    if args.vllm:
        print(f"  vLLM model   : {args.vllm_model}")
    print(f"  Prompts      : {len(BENCHMARK_PROMPTS)}")
    print(f"  Runs each    : {args.runs}")

    reports = asyncio.run(
        run_benchmark(
            ollama_model=args.model,
            vllm_model=args.vllm_model,
            runs_per_prompt=args.runs,
            include_vllm=args.vllm,
        )
    )

    for report in reports:
        _print_report(report)


if __name__ == "__main__":
    main()
