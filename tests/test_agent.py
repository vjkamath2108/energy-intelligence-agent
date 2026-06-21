"""Unit tests for EnergyIntelligenceAgent orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.energy_agent import EnergyIntelligenceAgent
from src.models import DispatchRecommendation, RetrievedDocument
from src.recommendation_engine.claude_client import ClaudeRecommender
from src.recommendation_engine.ollama_client import OllamaRecommender


def test_bedrock_backend_creates_claude_recommender() -> None:
    """backend='bedrock' wires ClaudeRecommender into _recommender."""
    with patch("src.agent.energy_agent.ClaudeRecommender") as mock_cls:
        agent = EnergyIntelligenceAgent(backend="bedrock", retriever=MagicMock())
        mock_cls.assert_called_once_with()
        assert agent._recommender is mock_cls.return_value


def test_ollama_backend_creates_ollama_recommender() -> None:
    """backend='ollama' wires OllamaRecommender into _recommender."""
    with patch("src.agent.energy_agent.OllamaRecommender") as mock_cls:
        agent = EnergyIntelligenceAgent(backend="ollama", retriever=MagicMock())
        mock_cls.assert_called_once_with()
        assert agent._recommender is mock_cls.return_value


async def test_dispatch_happy_path_returns_recommendation_and_audits() -> None:
    """Successful pipeline: retriever → recommender → audit → return rec."""
    mock_doc = RetrievedDocument(
        doc_id="doc-1",
        content="Grid frequency constraint: 50Hz ±0.5Hz",
        source="regulation/nget-p28.pdf",
        chunk_index=0,
        similarity_score=0.92,
    )
    mock_rec = DispatchRecommendation(
        query="Should we dispatch unit 3?",
        action="Dispatch unit 3 at 100MW",
        confidence=0.85,
        supporting_documents=[mock_doc],
        reasoning="Grid frequency within nominal range.",
        requires_human_review=False,
        hallucination_flagged=False,
    )

    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[mock_doc])

    mock_audit = MagicMock()
    mock_audit.record = AsyncMock()

    with patch("src.agent.energy_agent.ClaudeRecommender") as mock_recommender_cls:
        mock_recommender = MagicMock()
        mock_recommender.recommend = AsyncMock(return_value=mock_rec)
        mock_recommender_cls.return_value = mock_recommender

        agent = EnergyIntelligenceAgent(
            backend="bedrock",
            retriever=mock_retriever,
            audit_logger=mock_audit,
        )
        result = await agent.dispatch("Should we dispatch unit 3?")

    assert result is mock_rec
    mock_retriever.search.assert_awaited_once_with("Should we dispatch unit 3?")
    mock_recommender.recommend.assert_awaited_once_with(
        "Should we dispatch unit 3?", [mock_doc]
    )
    mock_audit.record.assert_awaited_once()
