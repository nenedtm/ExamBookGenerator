"""Tests for pipeline.validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.models import ExtractedImage, Topic
from pipeline.validator import (
    _check_duplicate_content,
    _check_empty_chapters,
    _check_image_references,
    _check_language,
    _check_markdown_structure,
    _check_missing_sections,
    _check_syllabus_order,
    _check_toc_structure,
    _check_topic_focus,
    _extract_all_headings,
    _extract_plain_text,
    _extract_title,
    _heading_to_slug,
    _split_chapters,
    _topic_keywords,
    validate_manual,
)
from utils.config import ConfigManager


# ── Helpers ──────────────────────────────────────────────────────────────────

VALID_MANUAL = """# Exam Manual

- [Linear Algebra](#linear-algebra)
- [Vector Spaces](#vector-spaces)

## Linear Algebra

Linear algebra is the study of vectors and matrices.

### Vector Spaces

A vector space is a set with addition and scalar multiplication.

## Calculus

Calculus studies rates of change and accumulation.

### Derivatives

The derivative measures instantaneous rate of change.
"""

FOCUS_MANUAL = """# Focus: Vector Spaces

- [Focus: Vector Spaces](#focus-vector-spaces)
- [Definition](#definition)

## Focus: Vector Spaces

Vector spaces are fundamental structures in linear algebra.

### Definition

A vector space V over a field F is a set with two operations.
"""


def _topics() -> list[Topic]:
    return [
        Topic(
            name="Linear Algebra",
            description="Vectors and matrices",
            related_documents=["d1"],
            order_source="pedagogical",
        ),
        Topic(
            name="Calculus",
            description="Derivatives and integrals",
            related_documents=["d2"],
            order_source="pedagogical",
        ),
    ]


def _syllabus_topics() -> list[Topic]:
    return [
        Topic(
            name="Linear Algebra",
            description="Vectors",
            related_documents=["d1"],
            order_source="syllabus",
            syllabus_position=0,
        ),
        Topic(
            name="Calculus",
            description="Derivatives",
            related_documents=["d2"],
            order_source="syllabus",
            syllabus_position=1,
        ),
    ]


def _cfg(**overrides) -> ConfigManager:
    cfg = ConfigManager()
    for k, v in overrides.items():
        cfg.set(k, v)
    return cfg


# ── _extract_title ───────────────────────────────────────────────────────────

class TestExtractTitle:
    def test_extracts_first_h1(self) -> None:
        assert _extract_title("# My Manual\n\nText") == "My Manual"

    def test_no_title(self) -> None:
        assert _extract_title("Just text, no heading") == ""


# ── _split_chapters ──────────────────────────────────────────────────────────

class TestSplitChapters:
    def test_splits_at_h2(self) -> None:
        chapters = _split_chapters("## Ch1\nBody1\n\n## Ch2\nBody2")
        assert len(chapters) == 2
        assert chapters[0]["title"] == "Ch1"
        assert chapters[0]["body"] == "Body1"
        assert chapters[1]["title"] == "Ch2"
        assert chapters[1]["body"] == "Body2"

    def test_no_chapters(self) -> None:
        assert _split_chapters("# Title\nJust text") == []


# ── _extract_all_headings ────────────────────────────────────────────────────

class TestExtractAllHeadings:
    def test_extracts_all_levels(self) -> None:
        text = "# H1\n## H2\n### H3\nText\n#### H4"
        headings = _extract_all_headings(text)
        assert headings == ["H1", "H2", "H3", "H4"]


# ── _heading_to_slug ─────────────────────────────────────────────────────────

class TestHeadingToSlug:
    def test_basic(self) -> None:
        assert _heading_to_slug("Vector Spaces") == "vector-spaces"

    def test_special_chars(self) -> None:
        assert _heading_to_slug("What is C++?") == "what-is-c"


# ── _topic_keywords ──────────────────────────────────────────────────────────

class TestTopicKeywords:
    def test_extracts_words(self) -> None:
        kw = _topic_keywords("Linear Algebra and Matrices")
        assert "linear" in kw
        assert "algebra" in kw
        assert "matrices" in kw

    def test_filters_short(self) -> None:
        kw = _topic_keywords("AI is")
        assert kw == set()


# ── _extract_plain_text ──────────────────────────────────────────────────────

class TestExtractPlainText:
    def test_strips_markdown(self) -> None:
        text = "# Header\n**bold** and *italic*\n[link](url)\n![img](path)"
        plain = _extract_plain_text(text)
        assert "Header" in plain
        assert "**" not in plain
        assert "![img]" not in plain


# ── _check_markdown_structure ────────────────────────────────────────────────

class TestCheckMarkdownStructure:
    def test_valid(self) -> None:
        result = _check_markdown_structure("# Title\n\n## Section\n\nText")
        assert result["status"] == "pass"

    def test_unclosed_fence(self) -> None:
        result = _check_markdown_structure("```python\ncode\n")
        assert result["status"] == "fail"
        assert any("unclosed" in e.lower() for e in result["errors"])

    def test_indented_header(self) -> None:
        result = _check_markdown_structure("# Title\n\n    ## Indented")
        assert result["status"] == "fail"
        assert any("indented" in e.lower() for e in result["errors"])


# ── _check_language ──────────────────────────────────────────────────────────

class TestCheckLanguage:
    def test_english_passes(self) -> None:
        text = (
            "This is the chapter about linear algebra. "
            "The vector space is a fundamental concept in mathematics. "
            "It is used in many areas of science and engineering. "
            "The derivative measures the rate of change of a function. "
            "This concept is important for understanding the material."
        )
        result = _check_language(text)
        assert result["status"] == "pass"

    def test_non_english_warns(self) -> None:
        text = (
            "Questo è il capitolo sull'algebra lineare. "
            "Lo spazio vettoriale è un concetto fondamentale. "
            "Viene usato in molte aree della matematica. "
            "La derivata misura il tasso di variazione di una funzione. "
            "Questo concetto è importante per capire il materiale."
        )
        result = _check_language(text)
        assert result["status"] == "warning"

    def test_empty_text_warns(self) -> None:
        result = _check_language("")
        assert result["status"] == "warning"


# ── _check_empty_chapters ────────────────────────────────────────────────────

class TestCheckEmptyChapters:
    def test_no_empty(self) -> None:
        chapters = [{"title": "Ch1", "body": "Some content here."}]
        result = _check_empty_chapters(chapters)
        assert result["status"] == "pass"

    def test_empty_found(self) -> None:
        chapters = [
            {"title": "Ch1", "body": "Content"},
            {"title": "Ch2", "body": ""},
        ]
        result = _check_empty_chapters(chapters)
        assert result["status"] == "fail"
        assert len(result["errors"]) == 1
        assert "Ch2" in result["errors"][0]


# ── _check_duplicate_content ─────────────────────────────────────────────────

class TestCheckDuplicateContent:
    def test_no_duplicates(self) -> None:
        chapters = [
            {"title": "A", "body": "This is unique content about topic A."},
            {"title": "B", "body": "This is different content about topic B."},
        ]
        result = _check_duplicate_content(chapters)
        assert result["status"] == "pass"

    def test_duplicates_detected(self) -> None:
        body = "This is a sentence about linear algebra. And another sentence."
        chapters = [
            {"title": "A", "body": body},
            {"title": "B", "body": body},
        ]
        result = _check_duplicate_content(chapters)
        assert result["status"] == "fail"
        assert "duplicate" in result["errors"][0].lower()


# ── _check_image_references ──────────────────────────────────────────────────

class TestCheckImageReferences:
    def test_valid_reference(self, tmp_path: Path) -> None:
        img = tmp_path / "fig.png"
        img.write_bytes(b"\x89PNG")
        text = f"# Title\n\n![Fig]({img})"
        images = [ExtractedImage(source_document_id="d1", file_path=str(img), width=10, height=10)]
        result = _check_image_references(text, images)
        assert result["status"] == "pass"

    def test_missing_image(self) -> None:
        text = "# Title\n\n![Fig](nonexistent.png)"
        result = _check_image_references(text, [])
        assert result["status"] == "fail"
        assert "nonexistent.png" in result["errors"][0]


# ── _check_toc_structure (v3) ────────────────────────────────────────────────

class TestCheckTocStructure:
    def test_valid_toc(self) -> None:
        headings = ["Exam Manual", "Linear Algebra", "Vector Spaces", "Calculus"]
        text = (
            "# Exam Manual\n\n"
            "- [Linear Algebra](#linear-algebra)\n"
            "- [Vector Spaces](#vector-spaces)\n\n"
            "## Linear Algebra\n\nText\n\n### Vector Spaces\n\nText\n\n## Calculus\n\nText"
        )
        result = _check_toc_structure(text, headings)
        assert result["status"] == "pass"

    def test_broken_anchor(self) -> None:
        headings = ["Title"]
        text = (
            "# Title\n\n"
            "- [Missing](#nonexistent)\n\n"
            "## Title\n\nText"
        )
        result = _check_toc_structure(text, headings)
        assert result["status"] == "fail"
        assert any("#nonexistent" in e for e in result["errors"])

    def test_no_toc(self) -> None:
        headings = ["Title"]
        text = "# Title\n\nJust text, no TOC."
        result = _check_toc_structure(text, headings)
        assert result["status"] == "fail"
        assert any("No TOC" in e for e in result["errors"])

    def test_indented_toc_entries(self) -> None:
        headings = ["Ch", "Sec"]
        text = (
            "# Title\n\n"
            "- [Ch](#ch)\n"
            "  - [Sec](#sec)\n\n"
            "## Ch\n\n### Sec\n\nText"
        )
        result = _check_toc_structure(text, headings)
        assert result["status"] == "pass"


# ── _check_topic_focus (v3) ─────────────────────────────────────────────────

class TestCheckTopicFocus:
    def test_stays_on_topic(self) -> None:
        text = (
            "Vector spaces are fundamental in linear algebra. "
            "A vector space V over a field F is defined by axioms. "
            "The dimension of a vector space is the size of a basis. "
            "Vector spaces generalize Euclidean spaces to arbitrary dimensions."
        )
        topics = [Topic(name="Calculus", description="Derivatives")]
        result = _check_topic_focus(text, "Vector Spaces", topics)
        assert result["status"] == "pass"

    def test_topic_drift_warns(self) -> None:
        text = (
            "Derivatives measure rates of change in calculus. "
            "The integral accumulates quantities over intervals. "
            "These calculus concepts are essential for analysis."
        )
        topics = [Topic(name="Linear Algebra", description="Vectors")]
        result = _check_topic_focus(text, "Vector Spaces", topics)
        # "vector spaces" keywords not present, but "linear algebra" might be
        # depending on what _topic_keywords extracts. The check should warn.
        assert result["status"] == "warning"

    def test_other_topic_presence_warns(self) -> None:
        text = (
            "Vector spaces are important. "
            "Calculus derivatives are also covered here in detail. "
            "Integrals and differentiation are useful concepts."
        )
        topics = [Topic(name="Calculus", description="Derivatives and integrals")]
        result = _check_topic_focus(text, "Vector Spaces", topics)
        # "Calculus" keywords are present → warning
        assert result["status"] == "warning"


# ── _check_syllabus_order (v3) ──────────────────────────────────────────────

class TestCheckSyllabusOrder:
    def test_correct_order(self) -> None:
        meta = [
            {"title": "Linear Algebra", "order": 0, "syllabus_position": 0},
            {"title": "Calculus", "order": 1, "syllabus_position": 1},
        ]
        result = _check_syllabus_order(meta)
        assert result["status"] == "pass"

    def test_wrong_order(self) -> None:
        meta = [
            {"title": "Calculus", "order": 0, "syllabus_position": 1},
            {"title": "Linear Algebra", "order": 1, "syllabus_position": 0},
        ]
        result = _check_syllabus_order(meta)
        assert result["status"] == "fail"
        assert any("syllabus order" in e.lower() for e in result["errors"])

    def test_no_syllabus_positions(self) -> None:
        meta = [{"title": "Ch1", "order": 0}]
        result = _check_syllabus_order(meta)
        assert result["status"] == "pass"


# ── validate_manual — full integration ───────────────────────────────────────

class TestValidateManual:
    def test_valid_manual_passes(self, tmp_path: Path) -> None:
        cfg = _cfg(**{"structure.include_toc": True, "generation.scope": "full"})
        result = validate_manual(VALID_MANUAL, _topics(), cfg)
        assert result["overall"] in ("pass", "warning")
        assert result["summary"]["total_checks"] > 0
        assert "checks" in result

    def test_empty_chapter_fails(self) -> None:
        manual = "# Title\n\n- [Ch1](#ch1)\n\n## Ch1\n\nText.\n\n## Ch2\n\n"
        cfg = _cfg(**{"structure.include_toc": True, "generation.scope": "full"})
        result = validate_manual(manual, _topics(), cfg)
        failed_checks = [c for c in result["checks"] if c["status"] == "fail"]
        check_names = [c["check"] for c in failed_checks]
        assert "empty_chapters" in check_names

    def test_broken_anchor_fails(self) -> None:
        manual = "# Title\n\n- [Missing](#nonexistent)\n\n## Title\n\nText."
        cfg = _cfg(**{"structure.include_toc": True, "generation.scope": "full"})
        result = validate_manual(manual, _topics(), cfg)
        failed_checks = [c for c in result["checks"] if c["status"] == "fail"]
        check_names = [c["check"] for c in failed_checks]
        assert "toc_structure" in check_names

    def test_no_toc_when_disabled(self) -> None:
        manual = "# Title\n\n## Section\n\nText."
        cfg = _cfg(**{"structure.include_toc": False, "generation.scope": "full"})
        result = validate_manual(manual, _topics(), cfg)
        check_names = [c["check"] for c in result["checks"]]
        assert "toc_structure" not in check_names

    def test_syllabus_order_checked(self) -> None:
        manual = (
            "# Manual\n\n"
            "- [Linear Algebra](#linear-algebra)\n"
            "- [Calculus](#calculus)\n\n"
            "## Linear Algebra\n\nText.\n\n## Calculus\n\nText."
        )
        chapters_meta = [
            {"title": "Linear Algebra", "order": 0, "syllabus_position": 0},
            {"title": "Calculus", "order": 1, "syllabus_position": 1},
        ]
        cfg = _cfg(**{"structure.include_toc": True, "generation.scope": "full"})
        result = validate_manual(
            manual, _syllabus_topics(), cfg, chapters_meta=chapters_meta
        )
        check_names = [c["check"] for c in result["checks"]]
        assert "syllabus_order" in check_names

    def test_focus_mode_checks_topic(self) -> None:
        cfg = _cfg(**{
            "structure.include_toc": True,
            "generation.scope": "topic",
            "generation.focus_topic": "Vector Spaces",
        })
        result = validate_manual(FOCUS_MANUAL, _topics(), cfg)
        check_names = [c["check"] for c in result["checks"]]
        assert "topic_focus" in check_names

    def test_writes_validation_json(self, tmp_path: Path) -> None:
        import os
        os.chdir(tmp_path)
        try:
            cfg = _cfg(**{"structure.include_toc": True, "generation.scope": "full"})
            validate_manual(VALID_MANUAL, _topics(), cfg)
            assert (tmp_path / "output" / "validation.json").exists()
        finally:
            os.chdir(Path(__file__).resolve().parent.parent)
