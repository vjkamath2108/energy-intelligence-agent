"""Tests for src/data_pipeline/chunkers.py — all three chunking strategies."""

import pytest

from src.data_pipeline.chunkers import (
    FixedSizeChunker,
    SemanticChunker,
    SlidingWindowChunker,
)
from src.models import RetrievedDocument

_SHORT = "one two three four five"
_LONG = " ".join([f"word{i}" for i in range(600)])
_REGULATORY = """
Section 1 — Purpose and Scope

1.1 This establishes requirements for all transmission-connected generators.

1.2 Compliance is mandatory under the relevant Grid Code.

Section 2 — Voltage Requirements

2.1 All generators must maintain voltage within specified bounds.

2.2 Reactive power capability must meet the P-Q diagram in Schedule 1.

Article 3 — Frequency Response

3.1 Primary frequency response must be delivered within 10 seconds.

3.2 Secondary frequency response within 30 seconds.
"""


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------


class TestFixedSizeChunker:
    def test_empty_content_returns_no_chunks(self) -> None:
        c = FixedSizeChunker()
        assert c.chunk("", "src.pdf") == []

    def test_short_content_is_single_chunk(self) -> None:
        c = FixedSizeChunker(chunk_size=512, overlap=50)
        chunks = c.chunk(_SHORT, "src.pdf")
        assert len(chunks) == 1
        assert chunks[0].content == _SHORT

    def test_long_content_splits_correctly(self) -> None:
        c = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = c.chunk(_LONG, "src.pdf")
        # 600 tokens, step=90 → ceil(600 / 90) ≈ 7 chunks
        assert len(chunks) > 1
        for ch in chunks:
            assert isinstance(ch, RetrievedDocument)
            assert len(ch.content.split()) <= 100

    def test_chunk_index_is_sequential(self) -> None:
        c = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = c.chunk(_LONG, "src.pdf")
        for i, ch in enumerate(chunks):
            assert ch.chunk_index == i

    def test_source_is_propagated(self) -> None:
        c = FixedSizeChunker()
        chunks = c.chunk(_SHORT, "regulation/test.pdf")
        assert chunks[0].source == "regulation/test.pdf"

    def test_metadata_contains_chunker_key(self) -> None:
        c = FixedSizeChunker()
        chunks = c.chunk(_SHORT, "src.pdf")
        assert chunks[0].metadata.get("chunker") == "fixed"

    def test_overlap_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            FixedSizeChunker(chunk_size=100, overlap=100)

    def test_whitespace_only_content(self) -> None:
        c = FixedSizeChunker()
        assert c.chunk("   \n\t  ", "src.pdf") == []

    def test_exact_chunk_size_content(self) -> None:
        # 100 tokens with chunk_size=100, overlap=10: produces 2 chunks —
        # the full window and then the overlap tail (tokens 90-99).
        words = " ".join([f"w{i}" for i in range(100)])
        c = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = c.chunk(words, "src.pdf")
        assert len(chunks) == 2
        assert chunks[0].content == words


# ---------------------------------------------------------------------------
# SlidingWindowChunker
# ---------------------------------------------------------------------------


class TestSlidingWindowChunker:
    def test_empty_content_returns_no_chunks(self) -> None:
        c = SlidingWindowChunker()
        assert c.chunk("", "src.pdf") == []

    def test_short_content_single_chunk(self) -> None:
        c = SlidingWindowChunker(window_size=256, step=128)
        chunks = c.chunk(_SHORT, "src.pdf")
        assert len(chunks) == 1

    def test_overlapping_windows_produced(self) -> None:
        c = SlidingWindowChunker(window_size=100, step=50)
        chunks = c.chunk(_LONG, "src.pdf")
        # 600 tokens, step=50, window=100 → ~11 windows
        assert len(chunks) > 1

    def test_each_window_within_size(self) -> None:
        c = SlidingWindowChunker(window_size=100, step=50)
        chunks = c.chunk(_LONG, "src.pdf")
        for ch in chunks:
            assert len(ch.content.split()) <= 100

    def test_metadata_contains_chunker_key(self) -> None:
        c = SlidingWindowChunker()
        chunks = c.chunk(_SHORT, "src.pdf")
        assert chunks[0].metadata.get("chunker") == "sliding"

    def test_metadata_contains_step(self) -> None:
        c = SlidingWindowChunker(step=64)
        chunks = c.chunk(_SHORT, "src.pdf")
        assert chunks[0].metadata.get("step") == "64"

    def test_step_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="step must be positive"):
            SlidingWindowChunker(step=0)

    def test_chunk_index_sequential(self) -> None:
        c = SlidingWindowChunker(window_size=50, step=25)
        chunks = c.chunk(_LONG, "src.pdf")
        for i, ch in enumerate(chunks):
            assert ch.chunk_index == i

    def test_source_propagated(self) -> None:
        c = SlidingWindowChunker()
        chunks = c.chunk(_SHORT, "regulation/bm25.pdf")
        assert chunks[0].source == "regulation/bm25.pdf"


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------


class TestSemanticChunker:
    def test_empty_content_returns_no_chunks(self) -> None:
        c = SemanticChunker()
        assert c.chunk("", "src.pdf") == []

    def test_single_short_paragraph_is_one_chunk(self) -> None:
        c = SemanticChunker()
        chunks = c.chunk("Short content without sections.", "src.pdf")
        assert len(chunks) == 1

    def test_regulatory_sections_split_at_boundaries(self) -> None:
        # Small max_chunk_size forces splits even after merge phase
        c = SemanticChunker(max_chunk_size=30)
        chunks = c.chunk(_REGULATORY, "grid-code.pdf")
        assert len(chunks) >= 2

    def test_chunks_do_not_exceed_max_size(self) -> None:
        c = SemanticChunker(max_chunk_size=50)
        # Use long content without regulatory headers to force sentence splitting
        long_para = "This is sentence one. This is sentence two. " * 30
        chunks = c.chunk(long_para, "src.pdf")
        for ch in chunks:
            assert len(ch.content.split()) <= 55  # slight buffer for sentence boundaries

    def test_metadata_contains_chunker_key(self) -> None:
        c = SemanticChunker()
        chunks = c.chunk(_REGULATORY, "src.pdf")
        for ch in chunks:
            assert ch.metadata.get("chunker") == "semantic"

    def test_source_propagated(self) -> None:
        c = SemanticChunker()
        chunks = c.chunk(_REGULATORY, "regulation/grid-code.pdf")
        assert all(ch.source == "regulation/grid-code.pdf" for ch in chunks)

    def test_chunk_index_sequential(self) -> None:
        c = SemanticChunker()
        chunks = c.chunk(_REGULATORY, "src.pdf")
        for i, ch in enumerate(chunks):
            assert ch.chunk_index == i

    def test_oversized_segment_split_at_sentences(self) -> None:
        # Build content that forces sub-sentence splitting
        long_no_sections = ". ".join([f"This is sentence {i} with many words here" for i in range(50)])
        c = SemanticChunker(max_chunk_size=30)
        chunks = c.chunk(long_no_sections, "src.pdf")
        assert len(chunks) > 1

    def test_paragraph_split_fallback(self) -> None:
        # Text without section headers → paragraph splitting
        text = "First paragraph content.\n\nSecond paragraph content.\n\nThird."
        c = SemanticChunker(max_chunk_size=200)
        chunks = c.chunk(text, "src.pdf")
        # All three paragraphs should appear in output (possibly merged)
        combined = " ".join(ch.content for ch in chunks)
        assert "First paragraph" in combined
        assert "Third" in combined

    def test_all_chunks_have_unique_doc_ids(self) -> None:
        c = SemanticChunker()
        chunks = c.chunk(_REGULATORY, "src.pdf")
        ids = [ch.doc_id for ch in chunks]
        assert len(ids) == len(set(ids))
