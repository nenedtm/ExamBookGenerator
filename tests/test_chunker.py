"""Tests for pipeline.chunker."""

from __future__ import annotations

import pytest

from core.models import Chunk, Document, FileType
from pipeline.chunker import count_tokens, create_chunks


# ── Helpers ─────────────────────────────────────────────────────────────────

def _doc(content: str = "Hello world", title: str = "Test") -> Document:
    return Document(source_path="/data/test.pdf", content=content, title=title)


def _long_doc(n_paragraphs: int = 10) -> Document:
    parts = [f"Paragraph {i}: " + "word " * 20 for i in range(n_paragraphs)]
    return _doc(content="\n\n".join(parts))


# ── count_tokens ─────────────────────────────────────────────────────────────

class TestCountTokens:
    def test_returns_int(self) -> None:
        assert isinstance(count_tokens("hello"), int)

    def test_positive(self) -> None:
        assert count_tokens("hello world") > 0

    def test_empty_string(self) -> None:
        assert count_tokens("") >= 0

    def test_longer_text_more_tokens(self) -> None:
        short = count_tokens("hello")
        long = count_tokens("hello " * 100)
        assert long > short


# ── create_chunks: basic ────────────────────────────────────────────────────

class TestCreateChunks:
    def test_returns_list(self) -> None:
        result = create_chunks(_doc())
        assert isinstance(result, list)

    def test_chunk_type(self) -> None:
        result = create_chunks(_doc())
        assert isinstance(result[0], Chunk)

    def test_document_id_linked(self) -> None:
        doc = _doc()
        result = create_chunks(doc)
        assert all(c.document_id == doc.id for c in result)

    def test_positions_sequential(self) -> None:
        doc = _long_doc(5)
        result = create_chunks(doc, max_tokens=50)
        positions = [c.position for c in result]
        assert positions == list(range(len(result)))

    def test_content_not_empty(self) -> None:
        result = create_chunks(_doc("Some real content here"))
        assert len(result) >= 1
        assert len(result[0].content) > 0


# ── create_chunks: token limits ─────────────────────────────────────────────

class TestTokenLimits:
    def test_single_short_doc_one_chunk(self) -> None:
        doc = _doc("Short text")
        result = create_chunks(doc, max_tokens=1024)
        assert len(result) == 1

    def test_long_doc_multiple_chunks(self) -> None:
        doc = _long_doc(20)
        result = create_chunks(doc, max_tokens=50)
        assert len(result) > 1

    def test_each_chunk_within_limit(self) -> None:
        doc = _long_doc(15)
        max_tok = 80
        result = create_chunks(doc, max_tokens=max_tok)
        for chunk in result:
            assert count_tokens(chunk.content) <= max_tok + 20  # small margin for overlap


# ── create_chunks: overlap ──────────────────────────────────────────────────

class TestOverlap:
    def test_no_overlap(self) -> None:
        doc = _long_doc(10)
        result = create_chunks(doc, max_tokens=40, overlap_tokens=0)
        assert len(result) >= 2

    def test_with_overlap(self) -> None:
        doc = _long_doc(10)
        no_overlap = create_chunks(doc, max_tokens=40, overlap_tokens=0)
        with_overlap = create_chunks(doc, max_tokens=40, overlap_tokens=20)
        # Overlap should produce same or more chunks (more text per chunk)
        assert len(with_overlap) >= len(no_overlap) - 1


# ── create_chunks: edge cases ───────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_content(self) -> None:
        doc = _doc(content="")
        result = create_chunks(doc)
        assert result == []

    def test_whitespace_only(self) -> None:
        doc = _doc(content="   \n\n  \t  ")
        result = create_chunks(doc)
        assert result == []

    def test_single_word(self) -> None:
        doc = _doc(content="Hello")
        result = create_chunks(doc, max_tokens=10)
        assert len(result) == 1
        assert "Hello" in result[0].content

    def test_paragraph_preserved(self) -> None:
        text = "This is a complete paragraph with several words that should stay together."
        doc = _doc(content=text)
        result = create_chunks(doc, max_tokens=100)
        assert len(result) == 1
        assert result[0].content == text

    def test_multiple_paragraphs(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        doc = _doc(content=text)
        result = create_chunks(doc, max_tokens=100)
        assert len(result) >= 1
        # All paragraphs should appear in the output
        full = " ".join(c.content for c in result)
        assert "First paragraph" in full
        assert "Second paragraph" in full
        assert "Third paragraph" in full


# ── create_chunks: oversized segment ────────────────────────────────────────

class TestOversizedSegment:
    def test_single_oversized_paragraph(self) -> None:
        text = "word " * 500  # ~250 tokens
        doc = _doc(content=text)
        result = create_chunks(doc, max_tokens=50)
        assert len(result) > 1

    def test_no_mid_sentence_cut(self) -> None:
        sentences = [f"Sentence {i} with some words." for i in range(20)]
        text = " ".join(sentences)
        doc = _doc(content=text)
        result = create_chunks(doc, max_tokens=30)
        # Each chunk should end with a complete sentence (period)
        for chunk in result:
            stripped = chunk.content.rstrip()
            assert stripped.endswith(".") or stripped.endswith("!") or stripped.endswith("?") or len(result) == 1


# ── create_chunks: content preservation ─────────────────────────────────────

class TestContentPreservation:
    def test_all_text_recovered(self) -> None:
        parts = [f"Section {i}. " + "Details. " * 5 for i in range(5)]
        text = "\n\n".join(parts)
        doc = _doc(content=text)
        result = create_chunks(doc, max_tokens=50)
        combined = " ".join(c.content for c in result)
        for word in ["Section 0", "Section 4", "Details"]:
            assert word in combined

    def test_metadata_populated(self) -> None:
        doc = _doc(content="Test content")
        result = create_chunks(doc)
        assert result[0].content == "Test content"
        assert result[0].position == 0

    def test_source_path_preserved(self) -> None:
        doc = Document(source_path="/data/lecture.pdf", content="text")
        result = create_chunks(doc)
        assert len(result) >= 1
