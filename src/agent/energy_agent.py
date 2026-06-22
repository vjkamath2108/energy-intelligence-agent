"""EnergyIntelligenceAgent — orchestrates the EIA RAG pipeline.

Composes HybridRetriever, ClaudeRecommender/OllamaRecommender, and AuditLogger
into a single dispatch(query) call. Always returns a DispatchRecommendation —
never raises; failures produce a fallback with requires_human_review=True.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from src.audit.audit_log import AuditLogger, entry_from_recommendation
from src.instrumentation.langfuse_client import (
    trace_dispatch,
    trace_generation,
    trace_retrieval,
)
from src.instrumentation.rag_log import append_dispatch
from src.models import DispatchRecommendation, RetrievedDocument
from src.recommendation_engine.claude_client import ClaudeRecommender
from src.recommendation_engine.ollama_client import OllamaRecommender
from src.retriever.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class EnergyIntelligenceAgent:
    """Orchestrates retrieve → recommend → audit for energy dispatch queries.

    Args:
        backend: 'bedrock' uses AWS Bedrock Claude; 'ollama' uses local Ollama.
        retriever: HybridRetriever instance. Injectable for testing.
        audit_logger: AuditLogger instance. Injectable for testing.
    """

    backend: Literal["bedrock", "ollama"] = "bedrock"
    retriever: HybridRetriever = field(default_factory=HybridRetriever)
    audit_logger: AuditLogger = field(default_factory=AuditLogger)

    def __post_init__(self) -> None:
        self._recommender: ClaudeRecommender | OllamaRecommender = (
            ClaudeRecommender() if self.backend == "bedrock" else OllamaRecommender()
        )

    def _model_label(self) -> str:
        """Return a human-readable model identifier for Langfuse traces.

        Returns:
            Model ID string from the active recommender.
        """
        return str(
            getattr(self._recommender, "model_id", None)
            or getattr(self._recommender, "model", None)
            or "unknown"
        )

    async def dispatch(self, query: str) -> DispatchRecommendation:
        """Run the RAG pipeline and return a grounded dispatch recommendation.

        Traces retrieval, generation, and the full dispatch call to Langfuse
        when keys are configured. Tracing failures are silently swallowed —
        observability is never in the critical path.

        Args:
            query: Operator's natural-language dispatch query.

        Returns:
            DispatchRecommendation — grounded or fallback with requires_human_review=True.
        """
        docs: list[RetrievedDocument] = []
        rec: DispatchRecommendation
        _pipeline_start = time.monotonic()

        async with trace_dispatch(query) as dispatch_span:
            try:
                async with trace_retrieval(query) as ret_span:
                    docs = await self.retriever.search(query)
                    ret_span["n_docs"] = len(docs)
                    ret_span["sources"] = list({d.source for d in docs})

                async with trace_generation(
                    query,
                    backend=self.backend,
                    model=self._model_label(),
                    n_context_docs=len(docs),
                ) as gen_span:
                    rec = await self._recommender.recommend(query, docs)
                    gen_span["action"] = rec.action
                    gen_span["confidence"] = rec.confidence
                    gen_span["hallucination_flagged"] = rec.hallucination_flagged
                    gen_span["requires_human_review"] = rec.requires_human_review

            except Exception as e:
                logger.error(
                    "Pipeline failure — returning fallback recommendation: %s: %s",
                    type(e).__name__,
                    e,
                )
                rec = DispatchRecommendation(
                    query=query,
                    action="FALLBACK: system unavailable — manual review required",
                    confidence=0.0,
                    supporting_documents=[],
                    reasoning=f"{type(e).__name__}: {e}",
                    requires_human_review=True,
                    hallucination_flagged=False,
                )

            dispatch_span["recommendation_id"] = rec.recommendation_id
            dispatch_span["confidence"] = rec.confidence
            dispatch_span["requires_human_review"] = rec.requires_human_review

        latency_ms = (time.monotonic() - _pipeline_start) * 1000
        append_dispatch(rec, self.backend, latency_ms, len(docs))

        try:
            entry = entry_from_recommendation(rec)
            await self.audit_logger.record(entry)
        except Exception as audit_exc:
            logger.error(
                "Audit write failed — recommendation not recorded: %s: %s",
                type(audit_exc).__name__,
                audit_exc,
            )

        return rec
