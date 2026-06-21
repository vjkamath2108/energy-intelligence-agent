# EnergyIntelligenceAgent Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `EnergyIntelligenceAgent` — a dataclass orchestrator in `src/agent/energy_agent.py` that composes HybridRetriever, ClaudeRecommender/OllamaRecommender, and AuditLogger into a single `dispatch(query) → DispatchRecommendation` call.

**Architecture:** Thin composition layer using Option A (dataclass with `__post_init__`). The agent owns backend selection (`bedrock` vs `ollama`), always returns a typed `DispatchRecommendation`, and audits every decision — including fallbacks. Safety, confidence, and constraint checks already live inside the recommender; the orchestrator does not re-implement them.

**Tech Stack:** Python 3.11+, asyncio, Pydantic v2, pytest with `asyncio_mode = auto`, `unittest.mock`.

## Global Constraints

- Python 3.11+ — use `X | Y` unions, not `Optional[X]`
- All functions must have type hints and a one-line docstring
- `asyncio` throughout — no `threading`, no `asyncio.run()` inside methods
- No `print()` — use `logging`
- `DispatchRecommendation` is `frozen=True` — construct once, never mutate
- `pytest.ini` sets `asyncio_mode = auto` — async test functions need no decorator

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/agent/energy_agent.py` | **Create** | `EnergyIntelligenceAgent` dataclass — the only new source file |
| `tests/test_agent.py` | **Create** | 4 unit tests: backend selection ×2, happy path, retrieval failure, recommender failure |

No other files are touched. All business logic (safety, confidence, constraints, audit serialisation) stays in the components it already lives in.

---

## Task 1: Scaffold class + backend selection

**Files:**
- Create: `src/agent/energy_agent.py`
- Create: `tests/test_agent.py`

**Interfaces:**
- Consumes: `ClaudeRecommender` from `src.recommendation_engine.claude_client`, `OllamaRecommender` from `src.recommendation_engine.ollama_client`, `HybridRetriever` from `src.retriever.hybrid_retriever`, `AuditLogger` from `src.audit.audit_log`
- Produces: `EnergyIntelligenceAgent(backend, retriever, audit_logger)` — the class other tasks and `main.py` will import

- [ ] **Step 1: Write the two failing backend selection tests**

Create `tests/test_agent.py`:

```python
"""Unit tests for EnergyIntelligenceAgent orchestrator."""

from unittest.mock import patch

from src.agent.energy_agent import EnergyIntelligenceAgent
from src.recommendation_engine.claude_client import ClaudeRecommender
from src.recommendation_engine.ollama_client import OllamaRecommender


def test_bedrock_backend_creates_claude_recommender() -> None:
    """backend='bedrock' wires ClaudeRecommender into _recommender."""
    with patch("src.agent.energy_agent.ClaudeRecommender") as mock_cls:
        agent = EnergyIntelligenceAgent(backend="bedrock")
        mock_cls.assert_called_once_with()
        assert agent._recommender is mock_cls.return_value


def test_ollama_backend_creates_ollama_recommender() -> None:
    """backend='ollama' wires OllamaRecommender into _recommender."""
    with patch("src.agent.energy_agent.OllamaRecommender") as mock_cls:
        agent = EnergyIntelligenceAgent(backend="ollama")
        mock_cls.assert_called_once_with()
        assert agent._recommender is mock_cls.return_value
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_agent.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agent.energy_agent'`

- [ ] **Step 3: Create `src/agent/energy_agent.py` with the class scaffold**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py::test_bedrock_backend_creates_claude_recommender tests/test_agent.py::test_ollama_backend_creates_ollama_recommender -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/energy_agent.py tests/test_agent.py
git commit -m "feat(agent): scaffold EnergyIntelligenceAgent with backend selection"
```

---

## Task 2: Happy path `dispatch()`

**Files:**
- Modify: `src/agent/energy_agent.py` — implement `dispatch()` body
- Modify: `tests/test_agent.py` — add happy path test

**Interfaces:**
- Consumes: `HybridRetriever.search(query: str, k: int = 5) -> list[RetrievedDocument]`, `ClaudeRecommender.recommend(query: str, context_docs: list[RetrievedDocument]) -> DispatchRecommendation`, `AuditLogger.record(entry: AuditLogEntry) -> None`, `entry_from_recommendation(rec: DispatchRecommendation) -> AuditLogEntry`
- Produces: `EnergyIntelligenceAgent.dispatch(query: str) -> DispatchRecommendation` — tested and working for the success path

- [ ] **Step 1: Add the happy path test to `tests/test_agent.py`**

Append to the existing file after the two backend tests:

```python
from unittest.mock import AsyncMock, MagicMock

from src.models import DispatchRecommendation, RetrievedDocument


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
```

- [ ] **Step 2: Run to verify test fails**

```bash
pytest tests/test_agent.py::test_dispatch_happy_path_returns_recommendation_and_audits -v
```

Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: Implement `dispatch()` in `src/agent/energy_agent.py`**

Replace the `dispatch` method body (remove `raise NotImplementedError`):

```python
    async def dispatch(self, query: str) -> DispatchRecommendation:
        """Run the RAG pipeline and return a grounded dispatch recommendation.

        Args:
            query: Operator's natural-language dispatch query.

        Returns:
            DispatchRecommendation — grounded or fallback with requires_human_review=True.
        """
        try:
            docs = await self.retriever.search(query)
            rec = await self._recommender.recommend(query, docs)
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

        entry = entry_from_recommendation(rec)
        await self.audit_logger.record(entry)
        return rec
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_agent.py::test_dispatch_happy_path_returns_recommendation_and_audits -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/energy_agent.py tests/test_agent.py
git commit -m "feat(agent): implement dispatch() happy path"
```

---

## Task 3: Graceful degradation tests

**Files:**
- Modify: `tests/test_agent.py` — add 2 failure tests

**Interfaces:**
- Consumes: `EnergyIntelligenceAgent.dispatch()` as implemented in Task 2
- Produces: Verified that `dispatch()` returns `requires_human_review=True` on any exception, and the audit is always recorded

- [ ] **Step 1: Add retrieval failure and recommender failure tests to `tests/test_agent.py`**

Append to the existing file:

```python
async def test_dispatch_retrieval_failure_returns_fallback_and_audits() -> None:
    """Retrieval exception → fallback DispatchRecommendation, audit still recorded."""
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(side_effect=RuntimeError("DB unreachable"))

    mock_audit = MagicMock()
    mock_audit.record = AsyncMock()

    with patch("src.agent.energy_agent.ClaudeRecommender"):
        agent = EnergyIntelligenceAgent(
            backend="bedrock",
            retriever=mock_retriever,
            audit_logger=mock_audit,
        )
        result = await agent.dispatch("Should we dispatch unit 3?")

    assert result.requires_human_review is True
    assert result.confidence == 0.0
    assert result.query == "Should we dispatch unit 3?"
    assert "FALLBACK" in result.action
    assert "RuntimeError" in result.reasoning
    mock_audit.record.assert_awaited_once()


async def test_dispatch_recommender_failure_returns_fallback_and_audits() -> None:
    """Recommender exception → fallback DispatchRecommendation, audit still recorded."""
    mock_doc = RetrievedDocument(
        doc_id="doc-1",
        content="Grid frequency constraint",
        source="regulation/nget-p28.pdf",
        chunk_index=0,
        similarity_score=0.9,
    )
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[mock_doc])

    mock_audit = MagicMock()
    mock_audit.record = AsyncMock()

    with patch("src.agent.energy_agent.ClaudeRecommender") as mock_recommender_cls:
        mock_recommender = MagicMock()
        mock_recommender.recommend = AsyncMock(
            side_effect=RuntimeError("Bedrock throttled beyond retries")
        )
        mock_recommender_cls.return_value = mock_recommender

        agent = EnergyIntelligenceAgent(
            backend="bedrock",
            retriever=mock_retriever,
            audit_logger=mock_audit,
        )
        result = await agent.dispatch("Should we dispatch unit 3?")

    assert result.requires_human_review is True
    assert result.confidence == 0.0
    assert "FALLBACK" in result.action
    assert "RuntimeError" in result.reasoning
    mock_audit.record.assert_awaited_once()
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_agent.py::test_dispatch_retrieval_failure_returns_fallback_and_audits tests/test_agent.py::test_dispatch_recommender_failure_returns_fallback_and_audits -v
```

Expected: FAIL — both assertions on `requires_human_review` fail (method raises `NotImplementedError` is already gone, but confirm they fail due to wrong behaviour if dispatch() was not yet correct)

> Note: if Task 2 is complete, these may already pass. Run them to confirm either way.

- [ ] **Step 3: Run the full test file to confirm all 5 tests pass**

```bash
pytest tests/test_agent.py -v
```

Expected output:
```
tests/test_agent.py::test_bedrock_backend_creates_claude_recommender PASSED
tests/test_agent.py::test_ollama_backend_creates_ollama_recommender PASSED
tests/test_agent.py::test_dispatch_happy_path_returns_recommendation_and_audits PASSED
tests/test_agent.py::test_dispatch_retrieval_failure_returns_fallback_and_audits PASSED
tests/test_agent.py::test_dispatch_recommender_failure_returns_fallback_and_audits PASSED

5 passed
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_agent.py
git commit -m "test(agent): add degradation tests — retrieval failure and recommender failure"
```

---

## Gate

```bash
pytest tests/test_agent.py -v
```

All 5 tests pass. `dispatch()` returns a valid, fully-typed `DispatchRecommendation` end-to-end — grounded on the happy path, fallback with `requires_human_review=True` on any exception, always audited.
