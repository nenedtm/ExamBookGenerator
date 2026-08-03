"""Tests for pipeline.chapter_generator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.models import Chunk, Document, ExtractedImage, IndexEntry, OutlineChapter, Topic
from llm.ollama_client import OllamaError
from pipeline.chapter_generator import (
    ChapterGeneratorError,
    _align_outline_headings,
    _build_chapter_index,
    _find_insertion_index,
    _insert_images,
    _missing_outline_sections,
    _parse_chapter_response,
    _slugify,
    build_sources_block,
    generate_chapter,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

SIMPLE_TEMPLATE = "# {{title}}\n\n{{toc}}\n\n{{content}}"


def _make_topic(
    name: str = "Linear Algebra",
    description: str = "Vector spaces and matrices",
    related_docs: list[str] | None = None,
) -> Topic:
    return Topic(
        name=name,
        description=description,
        related_documents=related_docs or ["doc-1"],
    )


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(document_id="doc-1", content="Vector spaces are fundamental...", position=0),
        Chunk(document_id="doc-1", content="Eigenvalues and eigenvectors...", position=1),
    ]


def _llm_chapter_response() -> str:
    return json.dumps({
        "title": "Linear Algebra",
        "content": (
            "## Linear Algebra\n\n"
            "This chapter covers vector spaces.\n\n"
            "### Vector Spaces\n\n"
            "A vector space is a set with addition and scalar multiplication.\n\n"
            "### Eigenvalues\n\n"
            "An eigenvalue λ satisfies Av = λv.\n"
        ),
        "sections": ["Vector Spaces", "Eigenvalues"],
    })


def _llm_focus_response() -> str:
    return json.dumps({
        "title": "Focus: Vector Spaces",
        "content": (
            "## Focus: Vector Spaces\n\n"
            "A deep dive into vector spaces.\n\n"
            "### Definition\n\n"
            "A vector space V over a field F...\n\n"
            "### Properties\n\n"
            "Closure, associativity, identity.\n"
        ),
        "sections": ["Definition", "Properties"],
    })


def _image() -> ExtractedImage:
    img = ExtractedImage(
        source_document_id="doc-1",
        file_path="output/assets/images/matrix.png",
        caption="Matrix diagram",
        width=300,
        height=200,
    )
    img.ai_description = "A matrix multiplication diagram."
    return img


def _mock_client(llm_response: str | None = None) -> MagicMock:
    client = MagicMock()
    client.generate.return_value = llm_response or _llm_chapter_response()
    return client


# ── build_sources_block ──────────────────────────────────────────────────────


class TestBuildSourcesBlock:
    def test_lists_documents_in_order_of_appearance(self) -> None:
        chunks = [
            Chunk(document_id="doc-b", content="...", position=0),
            Chunk(document_id="doc-a", content="...", position=1),
            Chunk(document_id="doc-a", content="...", position=2),
        ]
        docs = [
            Document(id="doc-a", title="Notes A", source_path="/data/a.md"),
            Document(id="doc-b", title="Notes B", source_path="/data/b.md"),
        ]
        block = build_sources_block(chunks, docs)
        assert block == (
            "1. Notes B — `/data/b.md`\n"
            "2. Notes A — `/data/a.md`"
        )

    def test_deduplicates_documents(self) -> None:
        chunks = [Chunk(document_id="doc-a", content="x", position=0)]
        docs = [
            Document(id="doc-a", title="A", source_path="/a.md"),
            Document(id="doc-a", title="A", source_path="/a.md"),
        ]
        assert build_sources_block(chunks, docs) == "1. A — `/a.md`"

    def test_excludes_syllabus_documents(self) -> None:
        chunks = [Chunk(document_id="doc-a", content="x", position=0)]
        docs = [Document(id="doc-a", title="A", source_path="/a.md", is_syllabus=True)]
        assert build_sources_block(chunks, docs) == ""

    def test_empty_when_no_chunks_or_documents(self) -> None:
        assert build_sources_block([], None) == ""
        assert build_sources_block([Chunk(document_id="d", content="x", position=0)], None) == ""

    def test_unknown_document_id_skipped(self) -> None:
        chunks = [Chunk(document_id="ghost", content="x", position=0)]
        docs = [Document(id="doc-a", title="A", source_path="/a.md")]
        assert build_sources_block(chunks, docs) == ""


# ── _slugify ─────────────────────────────────────────────────────────────────

class TestSlugify:
    def test_basic(self) -> None:
        assert _slugify("Vector Spaces") == "vector-spaces"

    def test_special_chars(self) -> None:
        assert _slugify("What is C++?") == "what-is-c"

    def test_empty(self) -> None:
        assert _slugify("") == ""


# ── _parse_chapter_response ──────────────────────────────────────────────────

class TestParseChapterResponse:
    def test_valid_json(self) -> None:
        result = _parse_chapter_response(_llm_chapter_response())
        assert result["title"] == "Linear Algebra"
        assert "## Linear Algebra" in result["content"]
        assert result["sections"] == ["Vector Spaces", "Eigenvalues"]

    def test_markdown_fences_stripped(self) -> None:
        raw = "```json\n" + _llm_chapter_response() + "\n```"
        result = _parse_chapter_response(raw)
        assert result["title"] == "Linear Algebra"

    def test_surrounding_text(self) -> None:
        raw = "Here:\n" + _llm_chapter_response() + "\nDone."
        result = _parse_chapter_response(raw)
        assert result["title"] == "Linear Algebra"

    def test_missing_title_raises(self) -> None:
        raw = json.dumps({"content": "body", "sections": []})
        with pytest.raises(ChapterGeneratorError, match="missing required key 'title'"):
            _parse_chapter_response(raw)

    def test_missing_content_raises(self) -> None:
        raw = json.dumps({"title": "T", "sections": []})
        with pytest.raises(ChapterGeneratorError, match="missing required key 'content'"):
            _parse_chapter_response(raw)

    def test_empty_title_raises(self) -> None:
        raw = json.dumps({"title": "", "content": "body"})
        with pytest.raises(ChapterGeneratorError, match="missing required key 'title'"):
            _parse_chapter_response(raw)

    def test_no_sections_defaults_to_empty(self) -> None:
        raw = json.dumps({"title": "T", "content": "body"})
        result = _parse_chapter_response(raw)
        assert result["sections"] == []

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ChapterGeneratorError, match="unrecognisable JSON"):
            _parse_chapter_response("not json at all {{{")


# ── _missing_outline_sections ────────────────────────────────────────────────

class TestMissingOutlineSections:
    def test_all_present(self) -> None:
        content = "## T\n\nIntro.\n\n### Vector Spaces\n\nText.\n\n### Eigenvalues\n\nMore."
        assert _missing_outline_sections(content, ["Vector Spaces", "Eigenvalues"]) == []

    def test_case_and_punctuation_insensitive(self) -> None:
        content = "## T\n\n### vEcToR spaces!\n\nText."
        assert _missing_outline_sections(content, ["Vector Spaces"]) == []

    def test_missing_sections_reported(self) -> None:
        content = "## T\n\n### Vector Spaces\n\nText."
        assert _missing_outline_sections(content, ["Vector Spaces", "Eigenvalues"]) == ["Eigenvalues"]


# ── _align_outline_headings ──────────────────────────────────────────────────

class TestAlignOutlineHeadings:
    def test_renames_title_heading(self) -> None:
        content = "## LLM Invented Title\n\nIntro.\n\n### Something\n\nText."
        aligned = _align_outline_headings(content, "Real Title", ["Something"])
        assert "## Real Title" in aligned
        assert "### Something" in aligned

    def test_renames_sections_to_exact_outline_titles(self) -> None:
        content = (
            "## T\n\nIntro.\n\n"
            "### Simulazione del processo Polya\n\nText.\n\n"
            "### Reti Bayesiane\n\nText."
        )
        aligned = _align_outline_headings(
            content,
            "Processi scambiabili",
            [
                "Simulazione della passeggiata aleatoria di Polya",
                "Cenni sulle reti Bayesiane",
            ],
        )
        assert "### Simulazione della passeggiata aleatoria di Polya" in aligned
        assert "### Cenni sulle reti Bayesiane" in aligned

    def test_unrelated_headings_kept_untouched(self) -> None:
        content = "## T\n\nIntro.\n\n### The Mathematics of Random Cube Solving\n\nText."
        aligned = _align_outline_headings(content, "Real Title", ["Introduzione euristica"])
        assert "## Real Title" in aligned
        assert "### The Mathematics of Random Cube Solving" in aligned

    def test_no_headings_returns_unchanged(self) -> None:
        content = "Just prose with no headings at all."
        assert _align_outline_headings(content, "Title", ["Section"]) == content


# ── _build_chapter_index ─────────────────────────────────────────────────────

class TestBuildChapterIndex:
    def test_chapter_and_sections(self) -> None:
        entries = _build_chapter_index("Linear Algebra", ["Vector Spaces", "Eigenvalues"])
        assert len(entries) == 3
        assert entries[0].level == 1
        assert entries[0].title == "Linear Algebra"
        assert entries[0].anchor == "linear-algebra"
        assert entries[1].level == 2
        assert entries[1].title == "Vector Spaces"
        assert entries[2].level == 2
        assert entries[2].title == "Eigenvalues"

    def test_orders_sequential(self) -> None:
        entries = _build_chapter_index("Ch", ["S1", "S2"])
        assert [e.order for e in entries] == [0, 1, 2]

    def test_empty_sections(self) -> None:
        entries = _build_chapter_index("Chapter", [])
        assert len(entries) == 1

    def test_empty_title_skipped(self) -> None:
        entries = _build_chapter_index("", ["Section"])
        assert len(entries) == 1  # only the section

    def test_empty_section_skipped(self) -> None:
        entries = _build_chapter_index("Ch", ["", "Valid"])
        assert len(entries) == 2  # chapter + 1 valid section


# ── _find_insertion_index ────────────────────────────────────────────────────

class TestFindInsertionIndex:
    def test_after_introduction(self) -> None:
        lines = [
            "## Title",
            "",
            "Intro paragraph.",
            "### Section",
            "More text.",
        ]
        idx = _find_insertion_index(lines, "after introduction")
        assert idx == 2  # after "## Title" + blank line

    def test_after_section(self) -> None:
        lines = [
            "## Title",
            "",
            "### Vector Spaces",
            "Content here.",
            "### Eigenvalues",
            "More.",
        ]
        idx = _find_insertion_index(lines, "after section: Vector Spaces")
        assert idx == 3  # after "### Vector Spaces" + "Content here."

    def test_before_conclusion(self) -> None:
        lines = [
            "## Title",
            "Text.",
            "### Summary",
            "Final words.",
        ]
        idx = _find_insertion_index(lines, "before conclusion")
        assert idx == 2  # before "### Summary"

    def test_unknown_placement(self) -> None:
        lines = ["## Title", "Text."]
        idx = _find_insertion_index(lines, "somewhere weird")
        assert idx == len(lines)


# ── _insert_images ───────────────────────────────────────────────────────────

class TestInsertImages:
    def test_inserts_image_markdown(self) -> None:
        content = "## Title\n\nIntro.\n\n### Section\n\nText."
        img = _image()
        result = _insert_images(content, [(img, "after introduction")])
        assert "![Matrix diagram](output/assets/images/matrix.png)" in result

    def test_empty_matched(self) -> None:
        content = "## Title\n\nText."
        assert _insert_images(content, []) == content

    def test_multiple_images(self) -> None:
        content = "## Title\n\nIntro.\n\n### Section\n\nText.\n\n### Conclusion\n\nDone."
        img1 = _image()
        img2 = ExtractedImage(
            source_document_id="doc-1",
            file_path="img2.png",
            caption="Second",
            width=100, height=100,
        )
        img2.ai_description = "Another image"
        result = _insert_images(content, [
            (img1, "after introduction"),
            (img2, "after section: Section"),
        ])
        assert "![Matrix diagram]" in result
        assert "![Second]" in result


# ── generate_chapter — full mode ─────────────────────────────────────────────

class TestGenerateChapterFull:
    def test_returns_chapter_md(self, tmp_path: Path) -> None:
        client = _mock_client()
        topic = _make_topic()
        chunks = _make_chunks()

        md, title, img_ids = generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=5,
            scope="full",
            client=client,
        )

        assert title == "Linear Algebra"
        assert "Linear Algebra" in md
        assert "# Linear Algebra" in md
        assert "- [Linear Algebra](#linear-algebra)" in md
        assert "- [Vector Spaces](#vector-spaces)" in md
        assert "- [Eigenvalues](#eigenvalues)" in md
        assert img_ids == []

    def test_uses_chapter_prompt(self) -> None:
        client = _mock_client()
        topic = _make_topic()
        chunks = _make_chunks()

        generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=5, scope="full", client=client,
        )

        call_args = client.generate.call_args[0][0]
        assert "Write a comprehensive" in call_args
        assert "Linear Algebra" in call_args

    def test_sources_rendered_in_output(self) -> None:
        client = _mock_client()
        topic = _make_topic()
        chunks = _make_chunks()
        docs = [Document(id="doc-1", title="Linear Algebra Notes", source_path="/data/la.md")]
        template = "# {{title}}\n\n{{content}}\n\n## References\n\n{{sources}}"

        md, _, _ = generate_chapter(
            topic, chunks, template,
            depth_level=5, scope="full",
            client=client, documents=docs,
        )

        assert "## References" in md
        assert "1. Linear Algebra Notes — `/data/la.md`" in md

    def test_sources_passed_to_prompt(self) -> None:
        client = _mock_client()
        topic = _make_topic()
        chunks = _make_chunks()
        docs = [Document(id="doc-1", title="Linear Algebra Notes", source_path="/data/la.md")]

        generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=5, scope="full",
            client=client, documents=docs,
        )

        call_args = client.generate.call_args[0][0]
        assert "SOURCES TO CITE" in call_args
        assert "Linear Algebra Notes" in call_args
        assert "Cite them inline in the text as [1], [2]" in call_args

    def test_no_sources_without_documents(self) -> None:
        client = _mock_client()
        topic = _make_topic()
        chunks = _make_chunks()

        generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=5, scope="full", client=client,
        )

        call_args = client.generate.call_args[0][0]
        assert "SOURCES TO CITE" not in call_args

    def test_custom_index_entries_used(self) -> None:
        client = _mock_client()
        topic = _make_topic()
        chunks = _make_chunks()
        custom_entries = [
            IndexEntry(title="Custom", anchor="custom", level=1, order=0),
        ]

        md, _, _ = generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=5, scope="full",
            index_entries=custom_entries, client=client,
        )

        assert "- [Custom](#custom)" in md
        # LLM sections should NOT appear in TOC when custom entries are given
        assert "- [Vector Spaces](#vector-spaces)" not in md
        assert "- [Eigenvalues](#eigenvalues)" not in md

    def test_outline_chapter_forces_title(self) -> None:
        """The outline title must be used even if the LLM invents another one."""
        llm_response = json.dumps({
            "title": "LLM Invented Title",
            "content": (
                "## LLM Invented Title\n\n"
                "Introduction paragraph with real body text.\n\n"
                "More body text explaining the topic in detail.\n\n"
                "### Distribuzione invariante\n\n"
                "Body text about the invariant distribution with details.\n\n"
                "### Teorema limite\n\n"
                "Body text about the limit theorem with details.\n"
            ),
            "sections": ["Distribuzione invariante", "Teorema limite"],
        })
        client = _mock_client(llm_response)
        topic = _make_topic()
        chunks = _make_chunks()
        outline = OutlineChapter(
            title="Catene di Markov",
            sections=["Distribuzione invariante", "Teorema limite"],
            topic_indices=[0],
        )

        md, title, _ = generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=5, scope="full",
            outline_chapter=outline, client=client,
        )

        assert title == "Catene di Markov"
        assert "# Catene di Markov" in md

    def test_outline_chapter_retries_until_sections_present(self) -> None:
        """A response missing required sections triggers a retry."""
        bad = json.dumps({
            "title": "X",
            "content": (
                "## X\n\nIntro.\n\n"
                "### Vector Spaces\n\n"
                "Paragraph one with real body text.\n\n"
                "Paragraph two with real body text.\n\n"
                "Paragraph three with real body text.\n"
            ),
            "sections": ["Vector Spaces"],
        })
        good = _llm_chapter_response()
        client = _mock_client()
        client.generate.side_effect = [bad, good]
        topic = _make_topic()
        chunks = _make_chunks()
        outline = OutlineChapter(
            title="Linear Algebra",
            sections=["Vector Spaces", "Eigenvalues"],
            topic_indices=[0],
        )

        md, title, _ = generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=5, scope="full",
            outline_chapter=outline, client=client,
        )

        assert client.generate.call_count == 2
        assert title == "Linear Algebra"
        # Second (compliant) response accepted
        assert "### Eigenvalues" in md
        assert "# Linear Algebra" in md


# ── generate_chapter — topic (focus) mode ────────────────────────────────────

class TestGenerateChapterFocus:
    def test_uses_focus_prompt(self) -> None:
        client = _mock_client(_llm_focus_response())
        topic = _make_topic(name="Vector Spaces")
        chunks = _make_chunks()

        md, title, img_ids = generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=8,
            scope="topic",
            client=client,
        )

        call_args = client.generate.call_args[0][0]
        assert "single specific topic" in call_args
        assert "Vector Spaces" in call_args
        assert title == "Focus: Vector Spaces"

    def test_focus_generates_local_toc(self) -> None:
        client = _mock_client(_llm_focus_response())
        topic = _make_topic(name="Vector Spaces")
        chunks = _make_chunks()

        md, _, _ = generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=8,
            scope="topic",
            client=client,
        )

        assert "- [Focus: Vector Spaces](#focus-vector-spaces)" in md
        assert "- [Definition](#definition)" in md
        assert "- [Properties](#properties)" in md

    def test_focus_sources_passed_to_prompt_and_output(self) -> None:
        client = _mock_client(_llm_focus_response())
        topic = _make_topic(name="Vector Spaces")
        chunks = _make_chunks()
        docs = [Document(id="doc-1", title="Vector Notes", source_path="/data/v.md")]
        template = "# {{title}}\n\n{{content}}\n\n## References\n\n{{sources}}"

        md, _, _ = generate_chapter(
            topic, chunks, template,
            depth_level=8,
            scope="topic",
            client=client, documents=docs,
        )

        call_args = client.generate.call_args[0][0]
        assert "SOURCES TO CITE" in call_args
        assert "Vector Notes" in call_args
        assert "1. Vector Notes — `/data/v.md`" in md


# ── generate_chapter — image insertion ───────────────────────────────────────

class TestGenerateChapterImages:
    def test_images_inserted(self) -> None:
        client = _mock_client()
        topic = _make_topic()
        chunks = _make_chunks()
        img = _image()

        # Mock image matching to return a result
        with patch("pipeline.chapter_generator.select_images_for_chapter") as mock_select:
            mock_select.return_value = [(img, "after section: Vector Spaces")]
            md, title, img_ids = generate_chapter(
                topic, chunks, SIMPLE_TEMPLATE,
                depth_level=5, scope="full",
                candidate_images=[img], client=client,
            )

        assert img.id in img_ids
        assert "![Matrix diagram]" in md

    def test_no_images_when_empty_candidates(self) -> None:
        client = _mock_client()
        topic = _make_topic()
        chunks = _make_chunks()

        md, _, img_ids = generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=5, scope="full",
            candidate_images=[], client=client,
        )

        assert img_ids == []


# ── generate_chapter — error handling ────────────────────────────────────────

class TestGenerateChapterErrors:
    def test_invalid_llm_response_raises(self) -> None:
        client = _mock_client("I don't understand.")
        topic = _make_topic()
        chunks = _make_chunks()

        with pytest.raises(ChapterGeneratorError, match="Failed to generate meaningful content"):
            generate_chapter(
                topic, chunks, SIMPLE_TEMPLATE,
                depth_level=5, scope="full", client=client,
            )

    def test_image_matching_failure_continues(self) -> None:
        client = _mock_client()
        topic = _make_topic()
        chunks = _make_chunks()
        img = _image()

        with patch("pipeline.chapter_generator.select_images_for_chapter") as mock_select:
            mock_select.side_effect = OllamaError("vision down")
            md, title, img_ids = generate_chapter(
                topic, chunks, SIMPLE_TEMPLATE,
                depth_level=5, scope="full",
                candidate_images=[img], client=client,
            )

        # Should still produce a chapter without images
        assert title == "Linear Algebra"
        assert img_ids == []


# ── generate_chapter — missing_from_notes placeholder ────────────────────────

class TestGenerateChapterMissing:
    def test_placeholder_chapter_for_missing_topic(self) -> None:
        client = _mock_client()
        topic = _make_topic()
        topic.missing_from_notes = True
        chunks = _make_chunks()  # chunks exist but topic is marked missing

        md, title, img_ids = generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=5, scope="full", client=client,
        )

        # Should NOT call the LLM
        client.generate.assert_not_called()
        # Should return a placeholder
        assert title == "Linear Algebra"
        assert "no matching content was found" in md.lower()
        assert "Linear Algebra" in md
        assert img_ids == []

    def test_placeholder_uses_description(self) -> None:
        client = _mock_client()
        topic = _make_topic(description="Vector spaces and linear transformations")
        topic.missing_from_notes = True

        md, title, _ = generate_chapter(
            topic, [], SIMPLE_TEMPLATE,
            depth_level=5, scope="full", client=client,
        )

        assert "Vector spaces and linear transformations" in md

    def test_extra_in_notes_adds_banner(self) -> None:
        client = _mock_client()
        topic = _make_topic()
        topic.extra_in_notes = True
        chunks = _make_chunks()

        md, title, img_ids = generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=5, scope="full", client=client,
        )

        # LLM should be called (not a placeholder)
        client.generate.assert_called_once()
        assert title == "Linear Algebra"
        assert "not part of the official exam syllabus" in md.lower()
        assert img_ids == []


# ── Integration: full end-to-end ─────────────────────────────────────────────

class TestIntegration:
    def test_full_mode_end_to_end(self, tmp_path: Path) -> None:
        client = _mock_client()
        topic = _make_topic()
        chunks = _make_chunks()

        md, title, img_ids = generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=5, scope="full", client=client,
        )

        # Title present
        assert "# Linear Algebra" in md
        # TOC present
        assert "- [Linear Algebra](#linear-algebra)" in md
        assert "- [Vector Spaces](#vector-spaces)" in md
        assert "- [Eigenvalues](#eigenvalues)" in md
        # Content present
        assert "vector spaces" in md.lower()
        assert "eigenvalue" in md.lower()
        # No images (none provided)
        assert img_ids == []

    def test_focus_mode_end_to_end(self) -> None:
        client = _mock_client(_llm_focus_response())
        topic = _make_topic(name="Vector Spaces")
        chunks = _make_chunks()

        md, title, img_ids = generate_chapter(
            topic, chunks, SIMPLE_TEMPLATE,
            depth_level=8, scope="topic", client=client,
        )

        assert title == "Focus: Vector Spaces"
        assert "# Focus: Vector Spaces" in md
        assert "- [Definition](#definition)" in md
        assert "- [Properties](#properties)" in md
        assert "deep dive" in md.lower()


