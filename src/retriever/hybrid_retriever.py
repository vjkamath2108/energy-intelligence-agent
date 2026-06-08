"""
Hybrid retriever: BM25 + semantic search fused with Reciprocal Rank Fusion (RRF).

Why hybrid over either alone:
  - BM25 (tsvector) excels at exact energy domain terms: "BSUoS", "TNUoS",
    "phraseto_tsquery". Misses paraphrase and intent.
  - Semantic (pgvector cosine) excels at intent and paraphrase: "grid
    frequency obligations" matches "primary frequency response requirements".
    Misses rare domain acronyms not seen during embedding training.
  - RRF fusion captures both signals without requiring score normalisation.
    Scores are not comparable across retrievers; ranks are.

Why RRF over score normalisation:
  - ts_rank_cd and cosine similarity are on different scales with different
    distributions. Normalising requires empirical calibration per corpus.
  - RRF is parameter-free beyond k=60 (standard default). Robust to outlier
    scores. Proven in TREC and production RAG systems.

RRF formula:
  score(doc) = Σ 1 / (k + rank_i)   for each retriever i that returned doc

  where k=60 is the standard constant that dampens the impact of top-ranked
  documents, preventing a single retriever from dominating.

Public API:
  HybridRetriever.search()       — fused BM25 + semantic search
  HybridRetriever.apply_schema() — delegate to BM25Retriever.apply_schema()

Design decisions:
  - BM25 and semantic searches run concurrently via asyncio.gather — not
    sequentially. Wall-clock time = max(bm25_latency, semantic_latency),
    not sum. Critical for the <500ms p95 gate.
  - fetch_k > k for each retriever: cast a wider net before fusion so RRF
    has more candidates to rerank. Default fetch_k = k * 3.
  - threshold=0.0 for semantic search in hybrid mode: BM25 already filters
    on keyword relevance; applying a cosine threshold here would drop valid
    candidates that BM25 can rescue via fusion.
  - doc_id is the deduplication key. If BM25 and semantic both return the
    same chunk, RRF accumulates both rank contributions — correct behaviour.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

from src.retriever.bm25 import BM25Retriever
from src.retriever.embeddings import TitanEmbedder, search_similar
from src.models import RetrievedDocument

logger = logging.getLogger(__name__)

# RRF rank constant — standard value from Cormack et al. (2009).
# Higher k reduces the reward gap between rank 1 and rank 2.
_RRF_K = 60


@dataclass
class HybridRetriever:
    """
    Fuses BM25 keyword search and semantic vector search via RRF.

    Both retrievers are called concurrently. Results are merged by doc_id
    and reranked by accumulated RRF score.

    Args:
        bm25: BM25Retriever instance (default: new instance with 'english').
        embedder: TitanEmbedder instance for query embedding.
        fetch_k_multiplier: Per-retriever fetch size = k * multiplier.
                            Wider candidate pool improves fusion quality.
                            Default 3 balances recall vs DB load.

    Usage:
        retriever = HybridRetriever(embedder=TitanEmbedder())
        results = await retriever.search("BSUoS balancing charges", k=5)
    """

    bm25: BM25Retriever = field(default_factory=BM25Retriever)
    embedder: TitanEmbedder = field(default_factory=TitanEmbedder)
    fetch_k_multiplier: int = 3

    async def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[RetrievedDocument]:
        """
        Hybrid search: BM25 + semantic, fused with RRF.

        Runs both retrievers concurrently. Deduplicates by doc_id.
        Returns top-k results ranked by RRF score.

        Args:
            query: Natural-language or keyword query string.
            k: Number of final results to return after fusion.

        Returns:
            List of RetrievedDocument ordered by descending RRF score.
            similarity_score field contains the RRF score (not raw BM25
            or cosine score — those are intermediate and discarded).
        """
        if not query.strip():
            return []

        fetch_k = k * self.fetch_k_multiplier
        start = time.monotonic()

        # Run both retrievers concurrently — not sequentially.
        # Wall-clock time = max(bm25, semantic), not sum.
        bm25_results, semantic_results = await asyncio.gather(
            self.bm25.search(query, k=fetch_k),
            search_similar(
                query,
                self.embedder,
                k=fetch_k,
                threshold=0.0,  # no threshold — let RRF decide
            ),
        )

        elapsed_ms = (time.monotonic() - start) * 1000

        logger.debug(
            "Retrieval latency=%.0fms bm25=%d semantic=%d query=%r",
            elapsed_ms,
            len(bm25_results),
            len(semantic_results),
            query[:60],
        )

        fused = _rrf_fuse(bm25_results, semantic_results, k=k)

        logger.info(
            "HybridRetriever query=%r bm25=%d semantic=%d fused=%d latency=%.0fms",
            query[:60],
            len(bm25_results),
            len(semantic_results),
            len(fused),
            elapsed_ms,
        )

        return fused

    async def apply_schema(self) -> None:
        """
        Apply BM25 DDL additions (fts column + GIN index) idempotently.

        Delegates to BM25Retriever.apply_schema(). Safe to call on startup.
        Semantic search schema (embedding column + HNSW index) is handled
        by embeddings.build_hnsw_index() — call that separately.
        """
        await self.bm25.apply_schema()


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def _rrf_fuse(
    bm25_results: list[RetrievedDocument],
    semantic_results: list[RetrievedDocument],
    k: int,
) -> list[RetrievedDocument]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.

    RRF score for a document = Σ 1 / (_RRF_K + rank_i)
    where rank_i is the 1-based position in retriever i's result list.
    Documents not returned by a retriever contribute 0 for that retriever.

    Deduplicates by doc_id. The RetrievedDocument with the most content
    (longest content string) is kept when the same doc_id appears in both
    lists — in practice content is identical, this is a safety fallback.

    Args:
        bm25_results: Ranked results from BM25Retriever.search().
        semantic_results: Ranked results from search_similar().
        k: Number of top results to return.

    Returns:
        Top-k RetrievedDocument sorted by descending RRF score.
        similarity_score is set to the RRF score for downstream logging.
    """
    # Accumulate RRF scores keyed by doc_id
    rrf_scores: dict[str, float] = {}

    # Track the best RetrievedDocument object per doc_id
    # (content is identical across retrievers; keep the richer metadata)
    docs: dict[str, RetrievedDocument] = {}

    for rank, doc in enumerate(bm25_results, start=1):
        rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0.0) + 1.0 / (_RRF_K + rank)
        if doc.doc_id not in docs or len(doc.content) > len(docs[doc.doc_id].content):
            docs[doc.doc_id] = doc

    for rank, doc in enumerate(semantic_results, start=1):
        rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0.0) + 1.0 / (_RRF_K + rank)
        if doc.doc_id not in docs or len(doc.content) > len(docs[doc.doc_id].content):
            docs[doc.doc_id] = doc

    # Sort by RRF score descending, take top k
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for doc_id, rrf_score in ranked:
        doc = docs[doc_id]
        results.append(
            RetrievedDocument(
                doc_id=doc.doc_id,
                content=doc.content,
                source=doc.source,
                chunk_index=doc.chunk_index,
                similarity_score=round(rrf_score, 6),  # RRF score replaces raw score
                metadata=doc.metadata,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

_TEST_QUERIES = [
    "BSUoS balancing services charges",          # favours BM25 — exact acronym
    "grid frequency stability obligations",       # favours semantic — paraphrase
    "TNUoS tariff 2024",                          # mixed — acronym + year
    "dispatch instruction balancing mechanism",   # mixed — domain terms + intent
]


async def _smoke_test() -> None:
    import os
    from src.db import close_pool

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    retriever = HybridRetriever(embedder=TitanEmbedder())

    try:
        await retriever.apply_schema()

        for query in _TEST_QUERIES:
            print(f"\n{'='*60}")
            print(f"Query: {query!r}")
            print(f"{'='*60}")

            results = await retriever.search(query, k=5)

            if not results:
                print("  (no results)")
                continue

            for i, doc in enumerate(results, start=1):
                print(
                    f"  [{i}] rrf={doc.similarity_score:.6f}"
                    f"  {doc.source} chunk={doc.chunk_index}"
                )
                print(f"       {doc.content[:120].replace(chr(10), ' ')}...")

    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_smoke_test())