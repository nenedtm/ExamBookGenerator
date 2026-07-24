"""Tests for pipeline.merge."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from core.models import IndexEntry, Topic
from pipeline.merge import (
    _add_numbering,
    _consolidate_entries,
    _extract_h2_headings,
    _extract_title,
    _normalise_image_paths,
    _sort_chapters_by_syllabus,
    _strip_local_toc,
    _write_output,
    merge_chapters,
)
from utils.config import ConfigManager


# ── Helpers ──────────────────────────────────────────────────────────────────

CHAPTER_1 = """# Linear Algebra

- [Linear Algebra](#linear-algebra)
- [Vector Spaces](#vector-spaces)

## Linear Algebra

Linear algebra studies vectors and matrices.

### Vector Spaces

A vector space is a set with operations.

## Eigenvalues

An eigenvalue λ satisfies Av = λv.
"""

CHAPTER_2 = """# Calculus

- [Calculus](#calculus)
- [Derivatives](#derivatives)

## Calculus

Calculus studies change and accumulation.

### Derivatives

The derivative measures instantaneous rate of change.
"""


def _cfg(**overrides) -> ConfigManager:
    cfg = ConfigManager()
    for k, v in overrides.items():
        cfg.set(k, v)
    return cfg


def _entries() -> list[IndexEntry]:
    return [
        IndexEntry(title="Linear Algebra", anchor="linear-algebra", level=1, order=0),
        IndexEntry(title="Vector Spaces", anchor="vector-spaces", level=2, order=1),
        IndexEntry(title="Eigenvalues", anchor="eigenvalues", level=2, order=2),
        IndexEntry(title="Calculus", anchor="calculus", level=1, order=3),
        IndexEntry(title="Derivatives", anchor="derivatives", level=2, order=4),
    ]


def _topics() -> list[Topic]:
    return [
        Topic(name="Linear Algebra", description="Vectors", related_documents=["d1"], order_source="pedagogical"),
        Topic(name="Calculus", description="Derivatives", related_documents=["d2"], order_source="pedagogical"),
    ]


def _syllabus_topics() -> list[Topic]:
    return [
        Topic(name="Calculus", description="Derivatives", related_documents=["d2"],
              order_source="syllabus", syllabus_position=1),
        Topic(name="Linear Algebra", description="Vectors", related_documents=["d1"],
              order_source="syllabus", syllabus_position=0),
    ]


# ── _extract_title ───────────────────────────────────────────────────────────

class TestExtractTitle:
    def test_extracts_h1(self) -> None:
        assert _extract_title("# My Chapter\n\nText") == "My Chapter"

    def test_no_title(self) -> None:
        assert _extract_title("Just text") == "Untitled"


# ── _strip_local_toc ────────────────────────────────────────────────────────

class TestStripLocalToc:
    def test_removes_toc_block(self) -> None:
        result = _strip_local_toc(CHAPTER_1)
        assert "- [Linear Algebra]" not in result
        assert "- [Vector Spaces]" not in result
        # Title and content remain
        assert "# Linear Algebra" in result
        assert "## Linear Algebra" in result

    def test_no_toc_unchanged(self) -> None:
        md = "# Title\n\n## Section\n\nText."
        assert _strip_local_toc(md) == md

    def test_preserves_content_after_toc(self) -> None:
        result = _strip_local_toc(CHAPTER_1)
        assert "Linear algebra studies" in result
        assert "eigenvalue" in result.lower()


# ── _extract_h2_headings ─────────────────────────────────────────────────────

class TestExtractH2Headings:
    def test_extracts_h2(self) -> None:
        headings = _extract_h2_headings(CHAPTER_1)
        assert headings == ["Linear Algebra", "Eigenvalues"]


# ── _add_numbering ───────────────────────────────────────────────────────────

class TestAddNumbering:
    def test_adds_number(self) -> None:
        result = _add_numbering("# Linear Algebra\n\nText", 3)
        assert result.startswith("# 3. Linear Algebra")

    def test_no_h1_unchanged(self) -> None:
        md = "Just text, no heading"
        assert _add_numbering(md, 1) == md


# ── _normalise_image_paths ───────────────────────────────────────────────────

class TestNormaliseImagePaths:
    def test_prepends_output(self) -> None:
        result = _normalise_image_paths("![Fig](images/fig.png)")
        assert "output/images/fig.png" in result

    def test_leaves_absolute_alone(self) -> None:
        result = _normalise_image_paths("![Fig](output/images/fig.png)")
        assert "output/images/fig.png" in result

    def test_leaves_urls_alone(self) -> None:
        result = _normalise_image_paths("![Fig](https://example.com/img.png)")
        assert "https://example.com/img.png" in result


# ── _sort_chapters_by_syllabus ──────────────────────────────────────────────

class TestSortChaptersBySyllabus:
    def test_sorts_by_syllabus_position(self) -> None:
        chapters = [CHAPTER_2, CHAPTER_1]  # Calculus first, LA second
        topics = _syllabus_topics()  # LA=0, Calc=1
        result = _sort_chapters_by_syllabus(chapters, topics)
        assert _extract_title(result[0]) == "Linear Algebra"
        assert _extract_title(result[1]) == "Calculus"

    def test_no_syllabus_preserves_order(self) -> None:
        chapters = [CHAPTER_1, CHAPTER_2]
        result = _sort_chapters_by_syllabus(chapters, _topics())
        assert _extract_title(result[0]) == "Linear Algebra"
        assert _extract_title(result[1]) == "Calculus"

    def test_no_topics_preserves_order(self) -> None:
        chapters = [CHAPTER_1, CHAPTER_2]
        result = _sort_chapters_by_syllabus(chapters, None)
        assert _extract_title(result[0]) == "Linear Algebra"


# ── _consolidate_entries ─────────────────────────────────────────────────────

class TestConsolidateEntries:
    def test_builds_from_chapters(self) -> None:
        result = _consolidate_entries([CHAPTER_1, CHAPTER_2], [])
        titles = [e.title for e in result if e.level == 1]
        assert "Linear Algebra" in titles
        assert "Calculus" in titles

    def test_ignores_provided_when_chapters_present(self) -> None:
        entries = _entries()
        result = _consolidate_entries([CHAPTER_1, CHAPTER_2], entries)
        # Should rebuild from actual headings, not use provided entries
        titles = [e.title for e in result]
        assert "Linear Algebra" in titles
        assert "Calculus" in titles
        # Sub-sections from provided entries should NOT appear
        assert "Vector Spaces" not in titles

    def test_falls_back_to_provided_when_no_chapters(self) -> None:
        entries = _entries()
        result = _consolidate_entries([], entries)
        assert result == entries


# ── _write_output ────────────────────────────────────────────────────────────

class TestWriteOutput:
    def test_writes_file(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        try:
            path = _write_output("Hello", "test.md")
            assert path.exists()
            assert path.read_text() == "Hello"
        finally:
            os.chdir(Path(__file__).resolve().parent.parent)


# ── merge_chapters — full mode ───────────────────────────────────────────────

class TestMergeFull:
    def test_writes_exam_manual(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        try:
            cfg = _cfg(**{"structure.include_toc": True, "generation.scope": "full"})
            path = merge_chapters(
                [CHAPTER_1, CHAPTER_2], _entries(), cfg, topics=_topics(),
            )
            assert path.name == "Exam_Manual.md"
            content = path.read_text()

            # Metadata present
            assert "Generated by ExamBookGenerator" in content
            assert "Scope: full" in content

            # Title present
            assert "# Exam Manual" in content

            # Global TOC present (only top-level chapters)
            assert "- [1. Linear Algebra](#1-linear-algebra)" in content
            assert "- [2. Calculus](#2-calculus)" in content

            # Chapter numbering
            assert "# 1." in content
            assert "# 2." in content

            # Local TOCs stripped
            local_toc_count = content.count("- [Vector Spaces]")
            # Should NOT appear in the output (local TOC stripped, not in global)
            assert local_toc_count == 0

        finally:
            os.chdir(Path(__file__).resolve().parent.parent)

    def test_empty_chapters(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        try:
            cfg = _cfg(**{"structure.include_toc": True, "generation.scope": "full"})
            path = merge_chapters([], [], cfg)
            content = path.read_text()
            assert "No chapters generated" in content
        finally:
            os.chdir(Path(__file__).resolve().parent.parent)

    def test_syllabus_ordering(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        try:
            cfg = _cfg(**{"structure.include_toc": True, "generation.scope": "full"})
            path = merge_chapters(
                [CHAPTER_2, CHAPTER_1],  # wrong order
                _entries(), cfg,
                topics=_syllabus_topics(),  # LA=0, Calc=1
            )
            content = path.read_text()
            # LA should be chapter 1 (position 0), Calculus chapter 2
            la_pos = content.index("Linear Algebra")
            calc_pos = content.index("Calculus")
            assert la_pos < calc_pos
        finally:
            os.chdir(Path(__file__).resolve().parent.parent)

    def test_image_paths_normalised(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        try:
            md = "# Ch\n\n![Fig](images/fig.png)\n\nText."
            cfg = _cfg(**{"structure.include_toc": True, "generation.scope": "full"})
            path = merge_chapters([md], [], cfg)
            content = path.read_text()
            assert "output/images/fig.png" in content
        finally:
            os.chdir(Path(__file__).resolve().parent.parent)


# ── merge_chapters — topic mode ──────────────────────────────────────────────

class TestMergeTopic:
    def test_writes_topic_file(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        try:
            cfg = _cfg(**{"structure.include_toc": True, "generation.scope": "topic"})
            path = merge_chapters(
                [CHAPTER_1], _entries(), cfg, focus_topic="Linear Algebra",
            )
            assert "linear-algebra" in path.name
            assert path.name.endswith(".md")
            content = path.read_text()
            assert "Scope: topic" in content
            assert "# Linear Algebra" in content
        finally:
            os.chdir(Path(__file__).resolve().parent.parent)

    def test_empty_topic(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        try:
            cfg = _cfg(**{"structure.include_toc": True, "generation.scope": "topic"})
            path = merge_chapters([], [], cfg, focus_topic="Vector Spaces")
            assert "vector-spaces" in path.name
            content = path.read_text()
            assert "No content generated" in content
        finally:
            os.chdir(Path(__file__).resolve().parent.parent)

    def test_local_toc_preserved_in_topic(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        try:
            cfg = _cfg(**{"structure.include_toc": True, "generation.scope": "topic"})
            path = merge_chapters(
                [CHAPTER_1], _entries(), cfg, focus_topic="Linear Algebra",
            )
            content = path.read_text()
            # Local TOC should be stripped (as per _strip_local_toc)
            # but the chapter content should still be there
            assert "Linear algebra studies" in content
        finally:
            os.chdir(Path(__file__).resolve().parent.parent)


# ── Integration ──────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_manual_structure(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        try:
            cfg = _cfg(**{"structure.include_toc": True, "generation.scope": "full"})
            path = merge_chapters(
                [CHAPTER_1, CHAPTER_2], _entries(), cfg, topics=_topics(),
            )
            content = path.read_text()
            lines = content.split("\n")

            # Skip HTML comment block and blank lines
            in_comment = False
            content_lines = []
            for l in lines:
                stripped = l.strip()
                if stripped.startswith("<!--"):
                    in_comment = True
                if in_comment:
                    if stripped.endswith("-->"):
                        in_comment = False
                    continue
                if stripped:
                    content_lines.append(stripped)

            assert content_lines[0] == "# Exam Manual"

            # TOC should follow title with numbered chapter headings
            toc_line = next(
                l for l in content_lines[1:10]
                if l.startswith("- [")
            )
            assert "1-linear-algebra" in toc_line or "linear-algebra" in toc_line

        finally:
            os.chdir(Path(__file__).resolve().parent.parent)
