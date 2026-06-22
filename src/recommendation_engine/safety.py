"""Hallucination detection for EIA dispatch recommendations.

Two grounding checks run before a DispatchRecommendation is returned:

  1. Term overlap (always): fraction of meaningful words in the action/reasoning
     that appear verbatim in the retrieved context. Gate: >= 30%.

  2. Semantic similarity (when embedder is provided): cosine distance between
     the embedded claim and the embedded corpus. Gate: >= 0.50.
     Skipped in sovereign/test mode when no embedder is passed.

is_grounded = True iff all applicable checks pass.
A failing check sets hallucination_flagged=True and requires_human_review=True
on the final DispatchRecommendation.
"""

import logging
import math
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.models import RetrievedDocument

logger = logging.getLogger(__name__)

HUMAN_REVIEW_THRESHOLD: float = 0.60  # W4.1 will calibrate against validation set
TERM_OVERLAP_THRESHOLD: float = 0.30
SEMANTIC_THRESHOLD: float = 0.50

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "in",
        "of",
        "to",
        "and",
        "for",
        "that",
        "this",
        "with",
        "be",
        "are",
        "was",
        "were",
        "will",
        "would",
        "can",
        "should",
        "have",
        "has",
        "had",
        "by",
        "from",
        "at",
        "as",
        "on",
        "it",
        "its",
        "or",
        "not",
        "but",
        "we",
        "our",
        "you",
        "they",
        "their",
        "which",
        "when",
        "where",
        "how",
        "what",
        "who",
        "all",
        "any",
        "may",
        "if",
        "so",
        "do",
        "did",
        "does",
        "been",
        "being",
        "get",
        "got",
    }
)


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Structural protocol for embedding backends (Bedrock Titan, Ollama, etc.)."""

    async def embed(self, text: str) -> list[float]:
        """Embed text into a fixed-dimension vector."""
        ...


@dataclass(frozen=True)
class SafetyResult:
    """Outcome of the hallucination grounding check.

    is_grounded is True when all applicable checks pass their thresholds.
    frozen=True: immutable after creation so it can be attached to audit records.
    """

    is_grounded: bool
    term_overlap_ratio: float
    semantic_similarity: float | None  # None when no embedder provided
    flagged_reasons: tuple[str, ...]


def _tokenise(text: str) -> set[str]:
    """Extract meaningful lowercase tokens from text, excluding stopwords.

    Args:
        text: Input text to tokenise.

    Returns:
        Set of lowercase tokens with length >= 3, excluding common stopwords.
    """
    tokens = {w for w in re.findall(r"\b[a-z][a-z0-9]{2,}\b", text.lower())}
    return tokens - _STOPWORDS


def term_overlap(text: str, corpus: str) -> float:
    """Fraction of meaningful tokens in text that appear in corpus.

    Args:
        text: The claim to check (action + reasoning).
        corpus: Concatenated retrieved document content.

    Returns:
        Float in [0, 1]. Returns 1.0 when text has no meaningful tokens
        (vacuously grounded — cannot flag a claim with no content).
    """
    meaningful = _tokenise(text)
    if not meaningful:
        return 1.0
    corpus_tokens = _tokenise(corpus)
    return len(meaningful & corpus_tokens) / len(meaningful)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors, clipped to [0, 1].

    Titan V2 produces unit-normalised vectors (normalize=True), so cosine
    equals the dot product. Implemented fully for correctness with other
    embedders.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity clipped to [0, 1].
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))


async def check_grounding(
    action: str,
    reasoning: str,
    context_docs: list[RetrievedDocument],
    embedder: EmbedderProtocol | None = None,
) -> SafetyResult:
    """Check whether a recommendation is grounded in the retrieved context.

    Term overlap always runs. Semantic similarity runs only when an embedder
    is provided — pass None to skip Bedrock calls in sovereign/test contexts.

    Args:
        action: The recommended dispatch action string.
        reasoning: The LLM's chain-of-thought reasoning string.
        context_docs: Retrieved document chunks as grounding evidence.
        embedder: Optional embedding backend for semantic similarity check.

    Returns:
        SafetyResult with is_grounded, term_overlap_ratio, semantic_similarity,
        and flagged_reasons describing any failed checks.
    """
    if not context_docs:
        return SafetyResult(
            is_grounded=False,
            term_overlap_ratio=0.0,
            semantic_similarity=None,
            flagged_reasons=("no retrieved documents to ground against",),
        )

    corpus = " ".join(d.content for d in context_docs)

    # Term overlap checks the action only — the action is the atomic claim that
    # must be grounded. Reasoning is a verbose narrative explanation and
    # naturally uses vocabulary beyond what appears in the source docs, which
    # would dilute the ratio and produce false positives.
    overlap = term_overlap(action, corpus)
    reasons: list[str] = []

    if overlap < TERM_OVERLAP_THRESHOLD:
        reasons.append(
            f"term overlap {overlap:.2f} < threshold {TERM_OVERLAP_THRESHOLD}"
        )

    sem_sim: float | None = None
    if embedder is not None:
        # Use action + reasoning for semantic similarity — the full output gives
        # the embedder more signal than the action alone.
        sem_text = f"{action} {reasoning}"
        claim_vec = await embedder.embed(sem_text[:2000])
        corpus_vec = await embedder.embed(corpus[:2000])
        sem_sim = _cosine(claim_vec, corpus_vec)
        if sem_sim < SEMANTIC_THRESHOLD:
            reasons.append(
                f"semantic similarity {sem_sim:.2f} < threshold {SEMANTIC_THRESHOLD}"
            )

    is_grounded = not reasons
    if not is_grounded:
        logger.warning("Grounding check failed: %s", "; ".join(reasons))

    return SafetyResult(
        is_grounded=is_grounded,
        term_overlap_ratio=overlap,
        semantic_similarity=sem_sim,
        flagged_reasons=tuple(reasons),
    )
