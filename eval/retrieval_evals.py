"""Retrieval evaluation framework: Precision@K, NDCG@K, Hit Rate@K, latency.

Compares three retrieval strategies:
  bm25     — PostgreSQL tsvector keyword search
  semantic — Titan V2 pgvector cosine search
  hybrid   — BM25 + semantic fused with RRF

Gate: hybrid Precision@5 >= 0.70. Exits with code 1 on failure.

Usage:
    python -m eval.retrieval_evals                      # all three, k=5
    python -m eval.retrieval_evals --retriever hybrid   # one retriever
    python -m eval.retrieval_evals --k 10               # change k
    python -m eval.retrieval_evals --quiet              # suppress per-query rows

Prerequisites:
    DB populated: python -m src.data_pipeline.ingest
    Embeddings:   python -m src.retriever.embeddings (embed_all_chunks + build_hnsw_index)
"""

import argparse
import asyncio
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Awaitable, Callable, TypeAlias

from dotenv import load_dotenv

from src.db import close_pool
from src.models import RetrievedDocument
from src.retriever.bm25 import BM25Retriever
from src.retriever.embeddings import TitanEmbedder, search_similar
from src.retriever.hybrid_retriever import HybridRetriever

load_dotenv()
logger = logging.getLogger(__name__)

_PRECISION_GATE = 0.70
_K_DEFAULT = 5

SearchFn: TypeAlias = Callable[[str, int], Awaitable[list[RetrievedDocument]]]


# ---------------------------------------------------------------------------
# Labeled eval dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalQuery:
    """A single labeled evaluation query with expected source prefixes."""

    query: str
    relevant_prefixes: list[str]  # doc.source must start with one of these
    tags: list[str] = field(default_factory=list)


# 15 queries spanning tariffs/regulations/decisions across three advantage profiles.
# Relevant prefix matching: doc.source.startswith(prefix)
# BM25-favoured  — exact acronyms appear verbatim in that source type
# semantic-favoured — paraphrase/intent, acronyms absent from query
# mixed — benefits from both signals
EVAL_QUERIES: list[EvalQuery] = [
    # BM25-favoured: exact energy domain acronyms
    EvalQuery(
        "TNUoS tariff 2024 generation charges",
        ["tariffs/tnuos"],
        ["bm25-favoured", "tariffs"],
    ),
    EvalQuery(
        "BSUoS balancing services use of system charges",
        ["tariffs/bsuos"],
        ["bm25-favoured", "tariffs"],
    ),
    EvalQuery(
        "DUoS red band amber green unit charges",
        ["tariffs/duos"],
        ["bm25-favoured", "tariffs"],
    ),
    EvalQuery(
        "Capacity Market T-4 T-1 agreement penalty stress event",
        ["tariffs/capacity-market"],
        ["bm25-favoured", "tariffs"],
    ),
    EvalQuery(
        "BM unit dispatch instruction gate closure settlement period",
        ["decisions/dispatch", "decisions/balancing-action"],
        ["bm25-favoured", "decisions"],
    ),
    # Semantic-favoured: paraphrase / intent
    EvalQuery(
        "how much does the system operator spend balancing the electricity grid",
        ["tariffs/bsuos"],
        ["semantic-favoured", "tariffs"],
    ),
    EvalQuery(
        "technical rules for connecting power stations to the national grid",
        ["regulations/"],
        ["semantic-favoured", "regulations"],
    ),
    EvalQuery(
        "what happens when generators cannot meet their frequency obligations",
        ["regulations/"],
        ["semantic-favoured", "regulations"],
    ),
    EvalQuery(
        "financial penalties for failing to deliver agreed electricity output",
        ["tariffs/capacity-market"],
        ["semantic-favoured", "tariffs"],
    ),
    EvalQuery(
        "electricity network access costs for large industrial power consumers",
        ["tariffs/tnuos", "tariffs/duos"],
        ["semantic-favoured", "tariffs"],
    ),
    # Mixed: exact technical terms plus conceptual context
    EvalQuery(
        "primary frequency response delivered within 10 seconds maintained 20 seconds",
        ["regulations/"],
        ["mixed", "regulations"],
    ),
    EvalQuery(
        "reactive power capability diagram absorption generation MVA rated output",
        ["regulations/"],
        ["mixed", "regulations"],
    ),
    EvalQuery(
        "half-hourly metered data BSC submission data aggregator D plus 1",
        ["regulations/"],
        ["mixed", "regulations"],
    ),
    EvalQuery(
        "upward balancing action constraint zone offer accepted MWh imbalance price",
        ["decisions/"],
        ["mixed", "decisions"],
    ),
    EvalQuery(
        "system frequency deviation constraint cost post-event review SQSS N-1",
        ["decisions/"],
        ["mixed", "decisions"],
    ),
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _is_relevant(doc: RetrievedDocument, prefixes: list[str]) -> bool:
    return any(doc.source.startswith(p) for p in prefixes)


def precision_at_k(
    results: list[RetrievedDocument], prefixes: list[str], k: int
) -> float:
    """Fraction of top-k results that are relevant.

    Args:
        results: Ranked retrieval results.
        prefixes: Source path prefixes that count as relevant.
        k: Cut-off rank.

    Returns:
        Float in [0, 1].
    """
    top_k = results[:k]
    if not top_k:
        return 0.0
    return sum(1.0 for d in top_k if _is_relevant(d, prefixes)) / k


def ndcg_at_k(
    results: list[RetrievedDocument], prefixes: list[str], k: int
) -> float:
    """Normalised Discounted Cumulative Gain at k (binary relevance).

    Args:
        results: Ranked retrieval results.
        prefixes: Source path prefixes that count as relevant.
        k: Cut-off rank.

    Returns:
        Float in [0, 1].
    """
    top_k = results[:k]
    if not top_k:
        return 0.0
    dcg = sum(
        float(_is_relevant(d, prefixes)) / math.log2(i + 2)
        for i, d in enumerate(top_k)
    )
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(len(top_k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def hit_rate_at_k(
    results: list[RetrievedDocument], prefixes: list[str], k: int
) -> float:
    """1.0 if at least one relevant result appears in top-k, else 0.0.

    Args:
        results: Ranked retrieval results.
        prefixes: Source path prefixes that count as relevant.
        k: Cut-off rank.

    Returns:
        1.0 or 0.0.
    """
    return 1.0 if any(_is_relevant(d, prefixes) for d in results[:k]) else 0.0


# ---------------------------------------------------------------------------
# Per-retriever evaluation runner
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Metrics for a single query against one retriever."""

    query: str
    precision: float
    ndcg: float
    hit_rate: float
    latency_ms: float
    n_returned: int


@dataclass
class RetrieverReport:
    """Aggregated results for one retriever across all eval queries."""

    name: str
    results: list[QueryResult] = field(default_factory=list)

    @property
    def mean_precision(self) -> float:
        return mean(r.precision for r in self.results) if self.results else 0.0

    @property
    def mean_ndcg(self) -> float:
        return mean(r.ndcg for r in self.results) if self.results else 0.0

    @property
    def mean_hit_rate(self) -> float:
        return mean(r.hit_rate for r in self.results) if self.results else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return mean(r.latency_ms for r in self.results) if self.results else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        sorted_lats = sorted(r.latency_ms for r in self.results)
        # With 15 queries p95 = index 14 (the max); still useful as an upper bound
        idx = max(0, int(len(sorted_lats) * 0.95) - 1)
        return sorted_lats[idx]


async def _eval_retriever(
    name: str,
    search_fn: SearchFn,
    queries: list[EvalQuery],
    k: int,
    quiet: bool = False,
) -> RetrieverReport:
    """Run all eval queries against one retriever and collect metrics.

    Args:
        name: Display name for the retriever.
        search_fn: Async callable (query, k) -> list[RetrievedDocument].
        queries: Labeled eval queries with ground truth prefixes.
        k: Number of results to retrieve per query.
        quiet: Suppress per-query debug output.

    Returns:
        RetrieverReport with per-query and aggregate metrics.
    """
    report = RetrieverReport(name=name)
    for eq in queries:
        t0 = time.monotonic()
        try:
            docs = await search_fn(eq.query, k)
        except Exception:
            logger.exception("Retriever %r failed on query %r", name, eq.query[:50])
            docs = []
        latency_ms = (time.monotonic() - t0) * 1000

        p = precision_at_k(docs, eq.relevant_prefixes, k)
        n = ndcg_at_k(docs, eq.relevant_prefixes, k)
        h = hit_rate_at_k(docs, eq.relevant_prefixes, k)

        report.results.append(
            QueryResult(query=eq.query, precision=p, ndcg=n, hit_rate=h,
                        latency_ms=latency_ms, n_returned=len(docs))
        )
        if not quiet:
            logger.debug(
                "[%s] P@%d=%.2f NDCG=%.2f HR=%.0f %5.0fms  %s",
                name, k, p, n, h, latency_ms, eq.query[:55],
            )
    return report


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def _print_summary_table(reports: list[RetrieverReport], k: int) -> None:
    """Print ASCII aggregate comparison table to stdout.

    Args:
        reports: One RetrieverReport per retriever.
        k: Cutoff used during eval (for column headers).
    """
    col = [22, 10, 10, 10, 16, 14]
    headers = ["Retriever", f"P@{k}", f"NDCG@{k}", f"HR@{k}", "Latency mean", "Latency p95"]
    sep = "+" + "+".join("-" * c for c in col) + "+"
    header_row = "|" + "|".join(h.center(c) for h, c in zip(headers, col)) + "|"
    print("\n" + sep)
    print(header_row)
    print(sep)
    for r in reports:
        gate_flag = (
            "  ← GATE FAIL"
            if r.name == "hybrid" and r.mean_precision < _PRECISION_GATE
            else ("  ← GATE PASS" if r.name == "hybrid" else "")
        )
        row = "|" + "|".join([
            r.name.center(col[0]),
            f"{r.mean_precision:.3f}".center(col[1]),
            f"{r.mean_ndcg:.3f}".center(col[2]),
            f"{r.mean_hit_rate:.3f}".center(col[3]),
            f"{r.mean_latency_ms:.0f} ms".center(col[4]),
            f"{r.p95_latency_ms:.0f} ms".center(col[5]),
        ]) + f"|{gate_flag}"
        print(row)
    print(sep)


def _print_per_query_table(report: RetrieverReport, k: int) -> None:
    """Print per-query breakdown for one retriever.

    Args:
        report: Results for a single retriever.
        k: Cutoff (for label).
    """
    print(f"\n### {report.name} — per-query breakdown (k={k})")
    print(f"  {'P@k':>5}  {'NDCG':>5}  {'HR':>3}  {'ms':>6}  query")
    print("  " + "-" * 72)
    for r in report.results:
        print(
            f"  {r.precision:5.2f}  {r.ndcg:5.2f}  {r.hit_rate:3.0f}"
            f"  {r.latency_ms:6.0f}  {r.query[:58]}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _main(k: int, retrievers: list[str], quiet: bool) -> int:
    """Run retrieval eval across requested retrievers and return exit code.

    Args:
        k: Number of results per query.
        retrievers: Ordered list of retriever names to evaluate.
        quiet: Suppress per-query debug logging.

    Returns:
        0 on gate pass, 1 on gate fail.
    """
    bm25 = BM25Retriever()
    embedder = TitanEmbedder()
    hybrid = HybridRetriever(bm25=bm25, embedder=embedder)

    search_fns: dict[str, SearchFn] = {
        "bm25": lambda q, k_: bm25.search(q, k=k_),
        "semantic": lambda q, k_: search_similar(q, embedder, k=k_, threshold=0.0),
        "hybrid": lambda q, k_: hybrid.search(q, k=k_),
    }

    reports: list[RetrieverReport] = []
    try:
        for name in retrievers:
            logger.info("Evaluating retriever=%r on %d queries …", name, len(EVAL_QUERIES))
            report = await _eval_retriever(name, search_fns[name], EVAL_QUERIES, k, quiet)
            reports.append(report)
            if not quiet:
                _print_per_query_table(report, k)
    finally:
        await close_pool()

    _print_summary_table(reports, k)

    hybrid_report = next((r for r in reports if r.name == "hybrid"), None)
    if hybrid_report is None:
        return 0

    if hybrid_report.mean_precision < _PRECISION_GATE:
        print(
            f"\n❌ GATE FAIL: hybrid Precision@{k} = {hybrid_report.mean_precision:.3f}"
            f" < {_PRECISION_GATE}"
        )
        return 1

    print(
        f"\n✅ GATE PASS: hybrid Precision@{k} = {hybrid_report.mean_precision:.3f}"
        f" ≥ {_PRECISION_GATE}"
    )
    return 0


def main() -> None:
    """CLI entry point for retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="Retrieval eval — Precision@K gate for EIA hybrid retriever"
    )
    parser.add_argument("--k", type=int, default=_K_DEFAULT, help="Cutoff rank (default 5)")
    parser.add_argument(
        "--retriever",
        choices=["bm25", "semantic", "hybrid", "all"],
        default="all",
        help="Which retriever to evaluate (default: all)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-query rows")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if not args.quiet else logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    retrievers = (
        ["bm25", "semantic", "hybrid"] if args.retriever == "all" else [args.retriever]
    )
    sys.exit(asyncio.run(_main(args.k, retrievers, args.quiet)))


if __name__ == "__main__":
    main()
