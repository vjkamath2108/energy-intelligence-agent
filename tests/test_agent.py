"""Unit tests for EnergyIntelligenceAgent orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.energy_agent import EnergyIntelligenceAgent
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
