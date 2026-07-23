"""Tests for pipeline.outline_generator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.models import IndexEntry, Topic
from pipeline.outline_generator import (
    OutlineGenerator,
    OutlineGeneratorError,
    _build_entries,
    _parse_outline_response,
    _render_outline_md,
    _unique_anchors,
    slugify,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _two_topics() -> list[Topic]:
    return [
        Topic(
            name="Linear Algebra",
            description="Vector spaces and matrices",
            subtopic_count=4,
            order_source="syllabus",
            syllabus_position=0,
        ),
        Topic(
            name="Calculus",
            description="Derivatives and integrals",
            subtopic_count=3,
            order_source="syllabus",
            syllabus_position=1,
        ),
    ]


def _single_topic() -> list[Topic]:
    return [
        Topic(
            name="Linear Algebra",
            description="Vector spaces and matrices",
            subtopic_count=4,
            order_source="manual",
        ),
    ]


def _llm_outline_response() -> str:
    return json.dumps({
        "chapters": [
            {
                "title": "Linear Algebra",
                "sections": ["Vector Spaces", "Eigenvalues"],
            },
            {
                "title": "Calculus",
                "sections": ["Derivatives", "Integrals"],
            },
        ]
    })


def _llm_single_chapter_response() -> str:
    return json.dumps({
        "chapters": [
            {
                "title": "Linear Algebra",
                "sections": ["Vector Spaces", "Eigenvalues", "Applications"],
            },
        ]
    })


# ── slugify ──────────────────────────────────────────────────────────────────

class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("Vector Spaces") == "vector-spaces"

    def test_lowercases(self) -> None:
        assert slugify("LINEAR ALGEBRA") == "linear-algebra"

    def test_strips_special_chars(self) -> None:
        assert slugify("What is C++?") == "what-is-c"

    def test_collapses_whitespace(self) -> None:
        assert slugify("  lots   of   spaces  ") == "lots-of-spaces"

    def test_empty_string(self) -> None:
        assert slugify("") == ""

    def test_unicode(self) -> None:
        result = slugify("Intégrales et Dérivées")
        assert "int" in result
        assert result == "integrales-et-derivees"

    def test_hyphens_preserved(self) -> None:
        assert slugify("well-known facts") == "well-known-facts"

    def test_numbers_preserved(self) -> None:
        assert slugify("Chapter 3: Basics") == "chapter-3-basics"


# ── _unique_anchors ──────────────────────────────────────────────────────────

class TestUniqueAnchors:
    def test_no_duplicates(self) -> None:
        entries = [
            IndexEntry(title="A", anchor="a", level=1, order=0),
            IndexEntry(title="B", anchor="b", level=1, order=1),
        ]
        result = _unique_anchors(entries)
        assert [e.anchor for e in result] == ["a", "b"]

    def test_duplicates_get_suffix(self) -> None:
        entries = [
            IndexEntry(title="A", anchor="x", level=1, order=0),
            IndexEntry(title="A2", anchor="x", level=2, order=1),
            IndexEntry(title="A3", anchor="x", level=2, order=2),
        ]
        result = _unique_anchors(entries)
        assert [e.anchor for e in result] == ["x", "x-2", "x-3"]

    def test_empty_list(self) -> None:
        assert _unique_anchors([]) == []


# ── _parse_outline_response ──────────────────────────────────────────────────

class TestParseOutlineResponse:
    def test_valid_json(self) -> None:
        result = _parse_outline_response(_llm_outline_response())
        assert len(result) == 2
        assert result[0]["title"] == "Linear Algebra"

    def test_markdown_fences_stripped(self) -> None:
        raw = "```json\n" + _llm_outline_response() + "\n```"
        result = _parse_outline_response(raw)
        assert len(result) == 2

    def test_surrounding_text(self) -> None:
        raw = "Here is the outline:\n" + _llm_outline_response() + "\nDone."
        result = _parse_outline_response(raw)
        assert len(result) == 2

    def test_missing_chapters_key_raises(self) -> None:
        with pytest.raises(OutlineGeneratorError, match="missing the 'chapters' array"):
            _parse_outline_response('{"other": []}')

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(OutlineGeneratorError, match="could not extract JSON"):
            _parse_outline_response("not json at all {{{")

    def test_empty_chapters_list(self) -> None:
        result = _parse_outline_response('{"chapters": []}')
        assert result == []

    def test_chapters_without_sections(self) -> None:
        raw = json.dumps({"chapters": [{"title": "Intro"}]})
        result = _parse_outline_response(raw)
        assert result[0]["title"] == "Intro"
        assert result[0].get("sections", []) == [] or "sections" not in result[0]


# ── _build_entries ───────────────────────────────────────────────────────────

class TestBuildEntries:
    def test_chapter_and_sections(self) -> None:
        chapters = [
            {"title": "Algebra", "sections": ["Groups", "Rings"]},
        ]
        entries = _build_entries(chapters)
        assert len(entries) == 3
        assert entries[0].level == 1
        assert entries[0].title == "Algebra"
        assert entries[0].anchor == "algebra"
        assert entries[1].level == 2
        assert entries[1].title == "Groups"
        assert entries[2].level == 2
        assert entries[2].title == "Rings"

    def test_orders_are_sequential(self) -> None:
        chapters = [
            {"title": "Ch1", "sections": ["S1", "S2"]},
            {"title": "Ch2", "sections": []},
        ]
        entries = _build_entries(chapters)
        orders = [e.order for e in entries]
        assert orders == list(range(len(entries)))

    def test_empty_sections_skipped(self) -> None:
        chapters = [{"title": "Only Chapter", "sections": []}]
        entries = _build_entries(chapters)
        assert len(entries) == 1
        assert entries[0].level == 1

    def test_empty_title_skipped(self) -> None:
        chapters = [{"title": "", "sections": ["Something"]}]
        entries = _build_entries(chapters)
        # Chapter with empty title is skipped, so no entries
        assert len(entries) == 0

    def test_empty_section_title_skipped(self) -> None:
        chapters = [{"title": "Chapter", "sections": ["", "Valid"]}]
        entries = _build_entries(chapters)
        assert len(entries) == 2  # chapter + 1 valid section


# ── _render_outline_md ───────────────────────────────────────────────────────

class TestRenderOutlineMd:
    def test_renders_correctly(self) -> None:
        entries = [
            IndexEntry(title="Chapter 1", anchor="chapter-1", level=1, order=0),
            IndexEntry(title="Section A", anchor="section-a", level=2, order=1),
            IndexEntry(title="Chapter 2", anchor="chapter-2", level=1, order=2),
        ]
        md = _render_outline_md(entries)
        lines = md.strip().split("\n")
        assert lines[0] == "- [Chapter 1](#chapter-1)"
        assert lines[1] == "  - [Section A](#section-a)"
        assert lines[2] == "- [Chapter 2](#chapter-2)"

    def test_empty_entries(self) -> None:
        assert _render_outline_md([]) == ""


# ── OutlineGenerator — full mode ─────────────────────────────────────────────

class TestOutlineGeneratorFull:
    def test_generates_outline(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_outline_response()

        gen = OutlineGenerator(mock_client)
        md, entries = gen.generate(
            _two_topics(),
            scope="full",
            output_path=tmp_path / "outline.md",
        )

        assert len(entries) == 6  # 2 chapters + 4 sections
        assert entries[0].level == 1  # "Linear Algebra"
        assert entries[0].title == "Linear Algebra"
        assert entries[1].level == 2  # "Vector Spaces"
        assert entries[2].level == 2  # "Eigenvalues"
        assert entries[3].level == 1  # "Calculus"
        assert entries[4].level == 2  # "Derivatives"
        assert entries[5].level == 2  # "Integrals"
        assert md  # non-empty markdown

    def test_respects_topic_order(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_outline_response()

        gen = OutlineGenerator(mock_client)
        _, entries = gen.generate(
            _two_topics(),
            scope="full",
            output_path=tmp_path / "outline.md",
        )

        # First chapter must be Linear Algebra (topic 0), second must be Calculus (topic 1)
        chapter_titles = [e.title for e in entries if e.level == 1]
        assert chapter_titles == ["Linear Algebra", "Calculus"]

    def test_writes_outline_md(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_outline_response()

        gen = OutlineGenerator(mock_client)
        gen.generate(
            _two_topics(),
            scope="full",
            output_path=tmp_path / "outline.md",
        )

        assert (tmp_path / "outline.md").exists()
        content = (tmp_path / "outline.md").read_text()
        assert "Linear Algebra" in content

    def test_anchors_unique(self, tmp_path: Path) -> None:
        # Craft a response where two sections produce the same slug
        response = json.dumps({
            "chapters": [
                {"title": "Intro", "sections": ["The Basics"]},
                {"title": "Advanced", "sections": ["The Basics"]},
            ]
        })
        mock_client = MagicMock()
        mock_client.generate.return_value = response

        gen = OutlineGenerator(mock_client)
        _, entries = gen.generate(
            [Topic(name="A"), Topic(name="B")],
            scope="full",
            output_path=tmp_path / "outline.md",
        )

        anchors = [e.anchor for e in entries]
        assert len(anchors) == len(set(anchors)), "Anchors must be unique"


# ── OutlineGenerator — topic (focus) mode ────────────────────────────────────

class TestOutlineGeneratorFocus:
    def test_single_topic_outline(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_single_chapter_response()

        gen = OutlineGenerator(mock_client)
        md, entries = gen.generate(
            _single_topic(),
            scope="topic",
            output_path=tmp_path / "outline.md",
        )

        assert len(entries) == 4  # 1 chapter + 3 sections
        assert entries[0].level == 1
        assert entries[0].title == "Linear Algebra"
        assert all(e.level == 2 for e in entries[1:])


# ── OutlineGenerator — empty input ───────────────────────────────────────────

class TestOutlineGeneratorEmpty:
    def test_empty_topics_returns_empty(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        gen = OutlineGenerator(mock_client)
        md, entries = gen.generate([], output_path=tmp_path / "outline.md")

        assert md == ""
        assert entries == []
        mock_client.generate.assert_not_called()


# ── OutlineGenerator — error handling ────────────────────────────────────────

class TestOutlineGeneratorErrors:
    def test_invalid_llm_response_raises(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = "I don't understand."

        gen = OutlineGenerator(mock_client)
        with pytest.raises(OutlineGeneratorError, match="could not extract JSON"):
            gen.generate(
                _two_topics(),
                output_path=tmp_path / "outline.md",
            )

    def test_llm_chapters_wrong_type(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = '{"chapters": "not a list"}'

        gen = OutlineGenerator(mock_client)
        with pytest.raises(OutlineGeneratorError, match="missing the 'chapters' array"):
            gen.generate(
                _two_topics(),
                output_path=tmp_path / "outline.md",
            )


# ── OutlineGenerator — format_topics ─────────────────────────────────────────

class TestFormatTopics:
    def test_formats_correctly(self) -> None:
        topics = _two_topics()
        result = OutlineGenerator._format_topics(topics)
        assert "1. Linear Algebra" in result
        assert "2. Calculus" in result
        assert "Description: Vector spaces" in result
        assert "Subtopics: 4" in result
