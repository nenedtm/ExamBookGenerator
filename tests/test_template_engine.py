"""Tests for pipeline.template_engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.models import IndexEntry
from pipeline.template_engine import (
    TemplateError,
    TemplateNotFoundError,
    TemplateRenderError,
    apply_template,
    load_template,
    render_images,
    render_toc,
    verify_anchors,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _simple_template() -> str:
    return "# {{title}}\n\n{{toc}}\n\n{{content}}\n\n---\n\n{{sources}}\n\n{{images}}"


def _entries() -> list[IndexEntry]:
    return [
        IndexEntry(title="Linear Algebra", anchor="linear-algebra", level=1, order=0),
        IndexEntry(title="Vector Spaces", anchor="vector-spaces", level=2, order=1),
        IndexEntry(title="Calculus", anchor="calculus", level=1, order=2),
    ]


# ── load_template ────────────────────────────────────────────────────────────

class TestLoadTemplate:
    def test_loads_existing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "template.md"
        p.write_text("# {{title}}\n{{content}}", encoding="utf-8")
        result = load_template(p)
        assert "# {{title}}" in result
        assert "{{content}}" in result

    def test_raises_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(TemplateNotFoundError):
            load_template(tmp_path / "nonexistent.md")

    def test_loads_default_template(self) -> None:
        result = load_template(Path("template.md"))
        assert "{{title}}" in result
        assert "{{toc}}" in result
        assert "{{content}}" in result
        assert "{{sources}}" in result
        assert "{{images}}" in result


# ── render_toc ───────────────────────────────────────────────────────────────

class TestRenderToc:
    def test_level_1_entries(self) -> None:
        entries = [
            IndexEntry(title="Chapter 1", anchor="chapter-1", level=1, order=0),
            IndexEntry(title="Chapter 2", anchor="chapter-2", level=1, order=1),
        ]
        toc = render_toc(entries)
        assert toc == "- [Chapter 1](#chapter-1)\n- [Chapter 2](#chapter-2)"

    def test_level_2_indentation(self) -> None:
        entries = [
            IndexEntry(title="Chapter", anchor="ch", level=1, order=0),
            IndexEntry(title="Section", anchor="sec", level=2, order=1),
        ]
        toc = render_toc(entries)
        assert "- [Chapter](#ch)" in toc
        assert "  - [Section](#sec)" in toc

    def test_empty_entries(self) -> None:
        assert render_toc([]) == ""

    def test_realistic_example(self) -> None:
        toc = render_toc(_entries())
        lines = toc.split("\n")
        assert lines[0] == "- [Linear Algebra](#linear-algebra)"
        assert lines[1] == "  - [Vector Spaces](#vector-spaces)"
        assert lines[2] == "- [Calculus](#calculus)"


# ── render_images ────────────────────────────────────────────────────────────

class TestRenderImages:
    def test_renders_images_with_caption(self) -> None:
        imgs = [{"caption": "Matrix diagram", "path": "images/matrix.png"}]
        block = render_images(imgs)
        assert "![Matrix diagram](images/matrix.png)" in block
        assert "*Matrix diagram*" in block

    def test_renders_images_without_caption(self) -> None:
        imgs = [{"caption": "", "path": "images/img.png"}]
        block = render_images(imgs)
        assert "[](images/img.png)" in block
        assert block.count("*") == 0  # no italic line for empty caption

    def test_empty_list(self) -> None:
        assert render_images([]) == ""

    def test_multiple_images(self) -> None:
        imgs = [
            {"caption": "A", "path": "a.png"},
            {"caption": "B", "path": "b.png"},
        ]
        block = render_images(imgs)
        assert block.count("![A](a.png)") == 1
        assert block.count("![B](b.png)") == 1


# ── apply_template — basic substitution ──────────────────────────────────────

class TestApplyTemplateBasic:
    def test_replaces_title(self) -> None:
        tpl = "# {{title}}\n\n{{content}}"
        out = apply_template(tpl, title="Intro", content="Hello")
        assert out == "# Intro\n\nHello"

    def test_replaces_all_variables(self) -> None:
        tpl = "# {{title}}\n\n{{toc}}\n\n{{content}}\n\n---\n\n{{sources}}\n\n{{images}}"
        out = apply_template(
            tpl,
            title="My Book",
            content="Body text",
            index_entries=[],
            sources="Ref 1",
            images=[{"caption": "Pic", "path": "pic.png"}],
            include_toc=False,
        )
        assert "# My Book" in out
        assert "Body text" in out
        assert "Ref 1" in out
        assert "![Pic](pic.png)" in out

    def test_raises_on_missing_title(self) -> None:
        with pytest.raises(TemplateRenderError, match="title"):
            apply_template("{{content}}", content="x")

    def test_raises_on_missing_content(self) -> None:
        with pytest.raises(TemplateRenderError, match="content"):
            apply_template("{{title}}", title="x")


# ── apply_template — TOC behavior ────────────────────────────────────────────

class TestApplyTemplateToc:
    def test_toc_present_when_include_toc_true(self) -> None:
        tpl = "# {{title}}\n\n{{toc}}\n\n{{content}}"
        out = apply_template(
            tpl,
            title="T",
            content="Body",
            index_entries=_entries(),
            include_toc=True,
        )
        assert "- [Linear Algebra](#linear-algebra)" in out
        assert "- [Calculus](#calculus)" in out

    def test_toc_absent_when_include_toc_false(self) -> None:
        tpl = "# {{title}}\n\n{{toc}}\n\n{{content}}"
        out = apply_template(
            tpl,
            title="T",
            content="Body",
            index_entries=_entries(),
            include_toc=False,
        )
        assert "linear-algebra" not in out
        assert "#linear-algebra" not in out

    def test_toc_absent_when_no_entries(self) -> None:
        tpl = "# {{title}}\n\n{{toc}}\n\n{{content}}"
        out = apply_template(
            tpl,
            title="T",
            content="Body",
            index_entries=[],
            include_toc=True,
        )
        # Should not leave a blank line between title and content
        assert out == "# T\n\nBody"

    def test_toc_between_title_and_content(self) -> None:
        tpl = "# {{title}}\n\n{{toc}}\n\n{{content}}"
        out = apply_template(
            tpl,
            title="T",
            content="Body",
            index_entries=_entries(),
            include_toc=True,
        )
        title_pos = out.index("# T")
        toc_pos = out.index("- [Linear Algebra]")
        content_pos = out.index("Body")
        assert title_pos < toc_pos < content_pos


# ── apply_template — images ──────────────────────────────────────────────────

class TestApplyTemplateImages:
    def test_images_present(self) -> None:
        tpl = "# {{title}}\n\n{{content}}\n\n{{images}}"
        out = apply_template(
            tpl,
            title="T",
            content="Body",
            images=[{"caption": "Fig 1", "path": "f1.png"}],
        )
        assert "![Fig 1](f1.png)" in out

    def test_images_empty(self) -> None:
        tpl = "# {{title}}\n\n{{content}}\n\n{{images}}"
        out = apply_template(
            tpl,
            title="T",
            content="Body",
            images=[],
        )
        assert "![Fig 1]" not in out

    def test_images_absent_when_none(self) -> None:
        tpl = "# {{title}}\n\n{{content}}\n\n{{images}}"
        out = apply_template(
            tpl,
            title="T",
            content="Body",
            images=None,
        )
        assert "![Fig 1]" not in out


# ── apply_template — sources ─────────────────────────────────────────────────

class TestApplyTemplateSources:
    def test_sources_present(self) -> None:
        tpl = "# {{title}}\n\n{{content}}\n\n---\n\n{{sources}}"
        out = apply_template(
            tpl,
            title="T",
            content="Body",
            sources="1. Author\n2. Another",
        )
        assert "1. Author" in out
        assert "2. Another" in out

    def test_sources_empty(self) -> None:
        tpl = "# {{title}}\n\n{{content}}\n\n---\n\n{{sources}}"
        out = apply_template(
            tpl,
            title="T",
            content="Body",
            sources="",
        )
        assert "Body" in out


# ── verify_anchors ───────────────────────────────────────────────────────────

class TestVerifyAnchors:
    def test_all_anchors_found(self) -> None:
        content = "## Linear Algebra\n\nText\n\n## Calculus\n\nMore"
        entries = _entries()[:2]  # linear-algebra, vector-spaces → only linear-algebra heading present
        missing = verify_anchors(content, entries)
        # vector-spaces is not a heading in content, so it should be missing
        assert missing == ["vector-spaces"]

    def test_no_anchors_missing(self) -> None:
        content = "## Linear Algebra\n\n## Vector Spaces\n\n## Calculus"
        entries = _entries()
        missing = verify_anchors(content, entries)
        assert missing == []

    def test_empty_entries(self) -> None:
        assert verify_anchors("## Anything", []) == []

    def test_slug_matching(self) -> None:
        content = "## What is C++?"
        entries = [IndexEntry(title="What is C++?", anchor="what-is-c", level=1, order=0)]
        missing = verify_anchors(content, entries)
        assert missing == []  # slug "what-is-c" matches heading "What is C++?"


# ── Integration: full template rendering ─────────────────────────────────────

class TestIntegration:
    def test_full_render(self, tmp_path: Path) -> None:
        tpl = load_template(Path("template.md"))
        out = apply_template(
            tpl,
            title="Exam Manual",
            content="## Linear Algebra\n\nText about LA.\n\n## Calculus\n\nText about Calc.",
            index_entries=_entries(),
            sources="1. Notes",
            images=[{"caption": "Diagram", "path": "images/diag.png"}],
            include_toc=True,
        )
        assert "# Exam Manual" in out
        assert "- [Linear Algebra](#linear-algebra)" in out
        assert "  - [Vector Spaces](#vector-spaces)" in out
        assert "- [Calculus](#calculus)" in out
        assert "## Linear Algebra" in out
        assert "## Calculus" in out
        assert "1. Notes" in out
        assert "![Diagram](images/diag.png)" in out

    def test_toc_heading_order(self) -> None:
        tpl = load_template(Path("template.md"))
        out = apply_template(
            tpl,
            title="Manual",
            content="## Algebra\n\nText\n\n## Geometry\n\nMore",
            index_entries=[
                IndexEntry(title="Algebra", anchor="algebra", level=1, order=0),
                IndexEntry(title="Geometry", anchor="geometry", level=1, order=1),
            ],
            include_toc=True,
        )
        title_idx = out.index("# Manual")
        toc_idx = out.index("- [Algebra]")
        heading_idx = out.index("## Algebra")
        assert title_idx < toc_idx < heading_idx
