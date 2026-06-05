"""Chunking strategies for energy domain documents.

Three implementations, all stateless (same input → same output):

  FixedSizeChunker    — sliding window over whitespace-split tokens
  SlidingWindowChunker — overlapping windows with configurable stride
  SemanticChunker     — splits at sentence/paragraph/regulatory boundaries

All return list[RetrievedDocument] using models from src/models.py.
"""

import re
import uuid
from abc import ABC, abstractmethod

from src.models import RetrievedDocument

# Regulatory section boundary patterns (energy domain specific)
_SECTION_RE = re.compile(
    r"(?m)^(?:"
    r"(?:Section|Article|Clause|Appendix|Schedule|Part)\s+\d+"
    r"|(?:#+\s)"
    r"|\d+\.\s+[A-Z]"         # top-level only — "1. Overview" not "1.1 sub-clause"
    r")"
)

# Sentence boundary: end of sentence followed by whitespace and capital letter
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Merge small segments until they reach this token count (~400 chars).
# Prevents paragraph splitting from producing hundreds of sub-100-char chunks.
_MIN_MERGE_TOKENS = 80


class BaseChunker(ABC):
    """Abstract base for all chunking strategies.

    All implementations must be stateless — no mutable instance state
    may be modified during or after a call to chunk().
    """

    @abstractmethod
    def chunk(self, content: str, source: str) -> list[RetrievedDocument]:
        """Split content into a list of RetrievedDocument chunks.

        Args:
            content: The full document text.
            source:  Origin path used as provenance, e.g. 'regulations/p28.pdf'.

        Returns:
            Ordered list of RetrievedDocument instances, one per chunk.
        """


def _tokens(text: str) -> list[str]:
    """Split text into whitespace-delimited tokens."""
    return text.split()


def _make_chunk(
    tokens: list[str],
    source: str,
    chunk_index: int,
    extra_meta: dict[str, str] | None = None,
) -> RetrievedDocument:
    """Construct a RetrievedDocument from a token list."""
    return RetrievedDocument(
        doc_id=str(uuid.uuid4()),
        content=" ".join(tokens),
        source=source,
        chunk_index=chunk_index,
        similarity_score=0.0,
        metadata=extra_meta or {},
    )


# ---------------------------------------------------------------------------
# Strategy 1 — Fixed Size
# ---------------------------------------------------------------------------


class FixedSizeChunker(BaseChunker):
    """Splits documents into fixed-length token windows with overlap.

    Simple and fast but blindly cuts mid-sentence and mid-clause, which
    breaks the regulatory context needed for accurate retrieval.

    Args:
        chunk_size: Target number of tokens per chunk.
        overlap:    Number of tokens shared between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, content: str, source: str) -> list[RetrievedDocument]:
        """Split content by token count with fixed overlap.

        Args:
            content: Full document text.
            source:  Document provenance path.

        Returns:
            List of fixed-size RetrievedDocument chunks.
        """
        tokens = _tokens(content)
        if not tokens:
            return []

        chunks: list[RetrievedDocument] = []
        step = self.chunk_size - self.overlap
        start = 0

        while start < len(tokens):
            window = tokens[start : start + self.chunk_size]
            chunks.append(
                _make_chunk(
                    window,
                    source,
                    len(chunks),
                    {"chunker": "fixed", "start_token": str(start)},
                )
            )
            if len(window) < self.chunk_size:
                break
            start += step

        return chunks


# ---------------------------------------------------------------------------
# Strategy 2 — Sliding Window
# ---------------------------------------------------------------------------


class SlidingWindowChunker(BaseChunker):
    """Overlapping sliding windows with configurable stride.

    Produces more chunks than FixedSizeChunker for the same document, with
    higher average overlap. Useful when retrieval recall matters more than
    storage efficiency, but still ignores semantic boundaries.

    Args:
        window_size: Number of tokens per window.
        step:        Number of tokens to advance between windows (stride).
    """

    def __init__(self, window_size: int = 256, step: int = 128) -> None:
        if step <= 0:
            raise ValueError("step must be positive")
        self.window_size = window_size
        self.step = step

    def chunk(self, content: str, source: str) -> list[RetrievedDocument]:
        """Produce overlapping windows with the configured stride.

        Args:
            content: Full document text.
            source:  Document provenance path.

        Returns:
            List of sliding-window RetrievedDocument chunks.
        """
        tokens = _tokens(content)
        if not tokens:
            return []

        chunks: list[RetrievedDocument] = []
        start = 0

        while start < len(tokens):
            window = tokens[start : start + self.window_size]
            chunks.append(
                _make_chunk(
                    window,
                    source,
                    len(chunks),
                    {
                        "chunker": "sliding",
                        "start_token": str(start),
                        "step": str(self.step),
                    },
                )
            )
            if len(window) < self.window_size:
                break
            start += self.step

        return chunks


# ---------------------------------------------------------------------------
# Strategy 3 — Semantic (winner)
# ---------------------------------------------------------------------------


class SemanticChunker(BaseChunker):
    """Splits at regulatory section boundaries and sentence endings.

    For energy domain documents this is the correct strategy:
    regulatory text is structured around Articles, Clauses and Sections
    that define independent obligations. Cutting across these boundaries
    destroys the semantic unit retrieved during RAG.

    Split priority (highest to lowest):
      1. Regulatory section headers  (Section N, Article N, Clause N, …)
      2. Paragraph breaks            (blank lines)
      3. Sentence endings            (period/! /? followed by capital)

    If a candidate segment exceeds max_chunk_size tokens after splitting
    on the above boundaries, it is further split at sentence boundaries.

    Args:
        max_chunk_size: Soft upper bound on tokens per chunk.
    """

    def __init__(self, max_chunk_size: int = 800) -> None:
        self.max_chunk_size = max_chunk_size

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chunk(self, content: str, source: str) -> list[RetrievedDocument]:
        """Split content at regulatory boundaries, capped at max_chunk_size.

        Args:
            content: Full document text.
            source:  Document provenance path.

        Returns:
            List of semantically coherent RetrievedDocument chunks.
        """
        segments = self._split_semantic(content)
        merged = self._merge_small(segments)
        chunks: list[RetrievedDocument] = []

        for seg in merged:
            seg = seg.strip()
            if not seg:
                continue
            tokens = _tokens(seg)
            if len(tokens) <= self.max_chunk_size:
                if tokens:
                    chunks.append(
                        _make_chunk(
                            tokens, source, len(chunks), {"chunker": "semantic"}
                        )
                    )
            else:
                # Oversized segment: split further at sentence boundaries
                for sub in self._split_sentences(seg):
                    sub_tokens = _tokens(sub)
                    if sub_tokens:
                        chunks.append(
                            _make_chunk(
                                sub_tokens,
                                source,
                                len(chunks),
                                {"chunker": "semantic", "sub_split": "sentence"},
                            )
                        )

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_semantic(self, text: str) -> list[str]:
        """Split text at regulatory headers and paragraph breaks.

        Args:
            text: Raw document text.

        Returns:
            List of raw text segments (may still exceed max_chunk_size).
        """
        # Find all section boundary positions
        boundaries = [m.start() for m in _SECTION_RE.finditer(text)]

        if not boundaries:
            # Fall back to paragraph splitting
            return self._split_paragraphs(text)

        segments: list[str] = []
        prev = 0
        for pos in boundaries:
            if pos > prev:
                seg = text[prev:pos]
                segments.extend(self._split_paragraphs(seg))
            prev = pos

        # Remainder after last boundary
        if prev < len(text):
            segments.extend(self._split_paragraphs(text[prev:]))

        return segments

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text on blank lines.

        Args:
            text: Text block to split.

        Returns:
            Non-empty paragraph strings.
        """
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    def _merge_small(self, segments: list[str]) -> list[str]:
        """Merge consecutive small segments until they reach _MIN_MERGE_TOKENS.

        Prevents blank-line splitting from producing hundreds of sub-100-char
        fragments. Stops merging into the current group when adding the next
        segment would exceed max_chunk_size.

        Args:
            segments: Raw segments from _split_semantic.

        Returns:
            List of segments where each is at least _MIN_MERGE_TOKENS tokens,
            unless it is the last (or only) segment in a boundary group.
        """
        merged: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for seg in segments:
            t = len(_tokens(seg))
            if current_tokens + t > self.max_chunk_size and current_parts:
                merged.append("\n\n".join(current_parts))
                current_parts = [seg]
                current_tokens = t
            else:
                current_parts.append(seg)
                current_tokens += t
                if current_tokens >= _MIN_MERGE_TOKENS:
                    merged.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0

        if current_parts:
            merged.append("\n\n".join(current_parts))

        return merged

    def _split_sentences(self, text: str) -> list[str]:
        """Split an oversized paragraph into sentence-level chunks.

        Accumulates sentences until max_chunk_size would be exceeded,
        then starts a new chunk.

        Args:
            text: Oversized paragraph text.

        Returns:
            List of sentence-group strings each within max_chunk_size.
        """
        sentences = _SENTENCE_END_RE.split(text)
        groups: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            t = len(_tokens(sentence))
            if current_tokens + t > self.max_chunk_size and current:
                groups.append(" ".join(current))
                current = [sentence]
                current_tokens = t
            else:
                current.append(sentence)
                current_tokens += t

        if current:
            groups.append(" ".join(current))

        return groups
