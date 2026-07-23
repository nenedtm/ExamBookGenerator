"""Tests for pipeline.deduplicator."""

from __future__ import annotations

import pytest

from core.models import Document, FileType
from pipeline.deduplicator import _content_hash, _pick_best, _similarity, deduplicate


# ── Helpers ─────────────────────────────────────────────────────────────────

def _doc(
    source_path: str = "/data/file.pdf",
    content: str = "Some content here",
    file_type: FileType = FileType.PDF,
) -> Document:
    return Document(source_path=source_path, content=content, file_type=file_type)


# ── _content_hash ────────────────────────────────────────────────────────────

class TestContentHash:
    def test_same_content_same_hash(self) -> None:
        a = _doc(content="Hello world")
        b = _doc(content="Hello world")
        assert _content_hash(a) == _content_hash(b)

    def test_different_content_different_hash(self) -> None:
        a = _doc(content="Alpha")
        b = _doc(content="Beta")
        assert _content_hash(a) != _content_hash(b)

    def test_empty_content(self) -> None:
        a = _doc(content="")
        b = _doc(content="")
        assert _content_hash(a) == _content_hash(b)

    def test_is_sha256(self) -> None:
        h = _content_hash(_doc())
        assert len(h) == 64


# ── _similarity ──────────────────────────────────────────────────────────────

class TestSimilarity:
    def test_identical(self) -> None:
        a = _doc(content="Same text")
        b = _doc(content="Same text")
        assert _similarity(a, b) == 1.0

    def test_empty_both(self) -> None:
        a = _doc(content="")
        b = _doc(content="")
        assert _similarity(a, b) == 1.0

    def test_one_empty(self) -> None:
        a = _doc(content="Something")
        b = _doc(content="")
        assert _similarity(a, b) == 0.0

    def test_different(self) -> None:
        a = _doc(content="Completely different text here")
        b = _doc(content="Unrelated content over there")
        assert _similarity(a, b) < 0.5


# ── _pick_best ───────────────────────────────────────────────────────────────

class TestPickBest:
    def test_longest_content_wins(self) -> None:
        short = _doc(source_path="/a.pdf", content="Short")
        long = _doc(source_path="/b.pdf", content="Longer content here")
        best = _pick_best([short, long])
        assert best.source_path == "/b.pdf"

    def test_longest_path_breaks_tie(self) -> None:
        a = _doc(source_path="/a.pdf", content="Same")
        b = _doc(source_path="/longer_name.pdf", content="Same")
        best = _pick_best([a, b])
        assert best.source_path == "/longer_name.pdf"


# ── deduplicate — exact duplicates ──────────────────────────────────────────

class TestExactDedup:
    def test_identical_removed(self) -> None:
        docs = [
            _doc(source_path="/a.pdf", content="Same text"),
            _doc(source_path="/b.pdf", content="Same text"),
        ]
        result = deduplicate(docs)
        assert len(result) == 1

    def test_best_kept(self) -> None:
        docs = [
            _doc(source_path="/a.pdf", content="Same text"),
            _doc(source_path="/b.pdf", content="Same text"),
        ]
        result = deduplicate(docs)
        assert len(result) == 1
        assert result[0].source_path == "/a.pdf"

    def test_all_unique(self) -> None:
        docs = [
            _doc(source_path="/a.pdf", content="Alpha"),
            _doc(source_path="/b.pdf", content="Beta"),
            _doc(source_path="/c.pdf", content="Gamma"),
        ]
        result = deduplicate(docs)
        assert len(result) == 3

    def test_three_identical(self) -> None:
        docs = [
            _doc(source_path="/a.pdf", content="Dup"),
            _doc(source_path="/b.pdf", content="Dup"),
            _doc(source_path="/c.pdf", content="Dup"),
        ]
        result = deduplicate(docs)
        assert len(result) == 1


# ── deduplicate — near-duplicates ───────────────────────────────────────────

class TestNearDedup:
    def test_near_duplicate_removed(self) -> None:
        base = "Introduction to linear algebra. Vectors and matrices."
        docs = [
            _doc(source_path="/a.pdf", content=base),
            _doc(source_path="/b.pdf", content=base + " Extra."),
        ]
        # SequenceMatcher ratio for these is ~0.936
        result = deduplicate(docs, similarity_threshold=0.90)
        assert len(result) == 1

    def test_different_kept(self) -> None:
        docs = [
            _doc(source_path="/a.pdf", content="Topic A: Calculus"),
            _doc(source_path="/b.pdf", content="Topic B: Algebra"),
        ]
        result = deduplicate(docs, similarity_threshold=0.92)
        assert len(result) == 2

    def test_longest_content_kept(self) -> None:
        short = "Intro to physics"
        long = "Intro to physics."  # slightly longer (ratio ~0.89)
        docs = [
            _doc(source_path="/short.pdf", content=short),
            _doc(source_path="/long.pdf", content=long),
        ]
        result = deduplicate(docs, similarity_threshold=0.85)
        assert len(result) == 1
        assert result[0].source_path == "/long.pdf"


# ── deduplicate — combined scenarios ────────────────────────────────────────

class TestCombined:
    def test_exact_and_near(self) -> None:
        a = "The quick brown fox jumps over the lazy dog"
        b = "The quick brown fox jumps over the lazy dog!"  # near dup
        c = "Completely unrelated content here"
        docs = [
            _doc(source_path="/a.pdf", content=a),
            _doc(source_path="/b.pdf", content=a),      # exact dup of a
            _doc(source_path="/c.pdf", content=b),      # near dup of a
            _doc(source_path="/d.pdf", content=c),
        ]
        result = deduplicate(docs, similarity_threshold=0.90)
        assert len(result) == 2

    def test_empty_input(self) -> None:
        assert deduplicate([]) == []

    def test_single_document(self) -> None:
        docs = [_doc()]
        result = deduplicate(docs)
        assert len(result) == 1

    def test_all_identical(self) -> None:
        docs = [_doc(source_path=f"/{i}.pdf", content="Same") for i in range(10)]
        result = deduplicate(docs)
        assert len(result) == 1

    def test_file_type_not_lost(self) -> None:
        docs = [
            _doc(source_path="/a.pdf", content="Text", file_type=FileType.PDF),
            _doc(source_path="/b.pdf", content="Text", file_type=FileType.PDF),
        ]
        result = deduplicate(docs)
        assert result[0].file_type == FileType.PDF

    def test_unicode_content(self) -> None:
        docs = [
            _doc(source_path="/a.pdf", content="Ciao, testo con accenti: à, è"),
            _doc(source_path="/b.pdf", content="Ciao, testo con accenti: à, è"),
        ]
        result = deduplicate(docs)
        assert len(result) == 1

    def test_empty_content_docs(self) -> None:
        docs = [
            _doc(source_path="/a.pdf", content=""),
            _doc(source_path="/b.pdf", content=""),
        ]
        result = deduplicate(docs)
        assert len(result) == 1


# ── deduplicate — threshold tuning ──────────────────────────────────────────

class TestThreshold:
    def test_strict_threshold_keeps_more(self) -> None:
        a = "Introduction to calculus and derivatives"
        b = "Introduction to calculus and integrals"
        docs = [
            _doc(source_path="/a.pdf", content=a),
            _doc(source_path="/b.pdf", content=b),
        ]
        strict = deduplicate(docs, similarity_threshold=1.0)
        relaxed = deduplicate(docs, similarity_threshold=0.70)
        assert len(strict) >= len(relaxed)

    def test_threshold_one_only_exact(self) -> None:
        docs = [
            _doc(source_path="/a.pdf", content="Hello"),
            _doc(source_path="/b.pdf", content="Hello!"),
        ]
        result = deduplicate(docs, similarity_threshold=1.0)
        assert len(result) == 2
