"""EnergyIntelligenceAgent — orchestrates the EIA RAG pipeline.

Composes HybridRetriever, ClaudeRecommender/OllamaRecommender, and AuditLogger
into a single dispatch(query) call. Always returns a DispatchRecommendation —
never raises; failures produce a fallback with requires_human_review=True.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

from src.audit.audit_log import AuditLogger, entry_from_recommendation
from src.models import DispatchRecommendation
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

    async def dispatch(self, query: str) -> DispatchRecommendation:
        """Run the RAG pipeline and return a grounded dispatch recommendation.

        Args:
            query: Operator's natural-language dispatch query.

        Returns:
            DispatchRecommendation — grounded or fallback with requires_human_review=True.
        """
        raise NotImplementedError
