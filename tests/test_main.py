"""Tests for main.py — CLI and pipeline orchestration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.models import Document, ExtractedImage, FileType
from main import (
    _parse_all,
    _parse_document,
    build_parser,
    run_pipeline,
)


# ── build_parser ─────────────────────────────────────────────────────────────

class TestBuildParser:
    def test_input_defaults_to_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.input is None

    def test_minimal_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--input", "/tmp/material"])
        assert args.input == "/tmp/material"
        assert args.scope is None
        assert args.topic is None
        assert args.no_interactive is False
        assert args.no_images is False

    def test_all_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--input", "/tmp/material",
            "--template", "my_template.md",
            "--model", "llama3",
            "--output", "/tmp/out",
            "--depth", "7",
            "--no-images",
            "--syllabus", "/tmp/syllabus.md",
            "--scope", "topic",
            "--topic", "Linear Algebra",
            "--focus-depth", "9",
            "--no-interactive",
        ])
        assert args.input == "/tmp/material"
        assert args.template == "my_template.md"
        assert args.model == "llama3"
        assert args.output == "/tmp/out"
        assert args.depth == 7
        assert args.no_images is True
        assert args.syllabus == "/tmp/syllabus.md"
        assert args.scope == "topic"
        assert args.topic == "Linear Algebra"
        assert args.focus_depth == 9
        assert args.no_interactive is True

    def test_scope_choices(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--input", "/tmp", "--scope", "invalid"])


# ── _parse_document ──────────────────────────────────────────────────────────

class TestParseDocument:
    def test_txt_file(self, tmp_path: Path) -> None:
        txt = tmp_path / "notes.txt"
        txt.write_text("Hello world", encoding="utf-8")
        doc = Document(source_path=str(txt), file_type=FileType.TXT)
        result_doc, images = _parse_document(doc)
        assert result_doc.content == "Hello world"
        assert images == []

    def test_md_file(self, tmp_path: Path) -> None:
        md = tmp_path / "notes.md"
        md.write_text("# Title\n\nContent", encoding="utf-8")
        doc = Document(source_path=str(md), file_type=FileType.MARKDOWN)
        result_doc, images = _parse_document(doc)
        assert "# Title" in result_doc.content

    def test_image_type_skipped(self) -> None:
        doc = Document(source_path="/fake/img.png", file_type=FileType.IMAGE)
        result_doc, images = _parse_document(doc)
        assert result_doc.content == ""
        assert images == []

    def test_unknown_type_skipped(self) -> None:
        doc = Document(source_path="/fake/file.xyz", file_type=FileType.UNKNOWN)
        result_doc, images = _parse_document(doc)
        assert images == []

    def test_parse_error_returns_doc(self) -> None:
        doc = Document(source_path="/nonexistent/file.pdf", file_type=FileType.PDF)
        result_doc, images = _parse_document(doc)
        # Should not raise — error is caught internally
        assert result_doc is doc
        assert images == []


# ── _parse_all ───────────────────────────────────────────────────────────────

class TestParseAll:
    def test_batch_parsing(self, tmp_path: Path) -> None:
        txt = tmp_path / "a.txt"
        txt.write_text("Content A", encoding="utf-8")
        txt2 = tmp_path / "b.txt"
        txt2.write_text("Content B", encoding="utf-8")

        docs = [
            Document(source_path=str(txt), file_type=FileType.TXT),
            Document(source_path=str(txt2), file_type=FileType.TXT),
        ]
        parsed, images = _parse_all(docs)
        assert len(parsed) == 2
        assert all(d.content for d in parsed)

    def test_skips_image_documents(self) -> None:
        docs = [
            Document(source_path="/fake/img.png", file_type=FileType.IMAGE),
            Document(source_path="/fake/notes.txt", file_type=FileType.TXT),
        ]
        parsed, images = _parse_all(docs)
        # IMAGE type is skipped by _parse_all
        assert len(parsed) == 1


# ── run_pipeline — full integration (mocked) ────────────────────────────────

class TestRunPipeline:
    def _make_args(
        self,
        input_dir: str = "/tmp/material",
        **overrides,
    ) -> MagicMock:
        args = MagicMock()
        args.input = input_dir
        args.template = "template.md"
        args.model = None
        args.output = None
        args.depth = None
        args.no_images = False
        args.syllabus = None
        args.scope = "full"
        args.topic = None
        args.focus_depth = None
        args.no_interactive = True
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_full_mode_integration(self, tmp_path: Path) -> None:
        # Setup material directory
        material = tmp_path / "material"
        material.mkdir()
        (material / "notes.txt").write_text(
            "Linear algebra studies vector spaces and matrices. "
            "Calculus studies derivatives and integrals. "
            "These are fundamental topics in mathematics.",
            encoding="utf-8",
        )

        # Setup template
        template = tmp_path / "template.md"
        template.write_text("# {{title}}\n\n{{toc}}\n\n{{content}}", encoding="utf-8")

        args = self._make_args(
            input_dir=str(material),
            template=str(template),
            no_interactive=True,
            scope="full",
        )

        # Mock all LLM-dependent steps
        mock_topics = [
            MagicMock(name="Linear Algebra", related_documents=["d1"],
                     order_source="pedagogical", subtopic_count=3),
            MagicMock(name="Calculus", related_documents=["d1"],
                     order_source="pedagogical", subtopic_count=2),
        ]
        mock_entries = [
            MagicMock(title="Linear Algebra", anchor="linear-algebra", level=1, order=0),
            MagicMock(title="Calculus", anchor="calculus", level=1, order=1),
        ]
        mock_outline_chapters = [
            MagicMock(title="Linear Algebra", sections=["Vector Spaces", "Eigenvalues"], topic_indices=[0]),
            MagicMock(title="Calculus", sections=["Derivatives", "Integrals"], topic_indices=[1]),
        ]

        with patch("main.TopicAnalyzer") as MockAnalyzer, \
             patch("main.OutlineGenerator") as MockOutline, \
             patch("main.generate_chapter") as mock_gen_ch, \
             patch("main.merge_chapters") as mock_merge, \
             patch("main.validate_manual") as mock_validate, \
             patch("main.OllamaClient") as MockClient, \
             patch("main.deduplicate") as mock_dedup, \
             patch("main.create_chunks") as mock_chunk, \
             patch("main.load_template") as mock_tpl:

            # Configure mocks
            MockClient.from_config.return_value = MagicMock()
            mock_instance = MockAnalyzer.return_value
            mock_instance.analyze.return_value = mock_topics
            MockOutline.return_value.generate.return_value = ("outline", mock_entries, mock_outline_chapters)
            mock_gen_ch.return_value = ("# Chapter\n\nContent.", "Chapter", [])
            mock_validate.return_value = {
                "overall": "pass",
                "checks": [],
                "summary": {"total_checks": 1, "passed": 1, "warnings": 0, "failed": 0},
            }
            mock_tpl.return_value = "# {{title}}\n\n{{content}}"
            mock_dedup.return_value = [Document(source_path="a.txt", content="Some text")]
            mock_chunk.return_value = [MagicMock(document_id="d1", content="chunk")]

            # Create output dir and a dummy file so validation can read it
            out_dir = tmp_path / "output"
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / "Exam_Manual.md"
            out_file.write_text("# Exam Manual\n\nContent.", encoding="utf-8")
            mock_merge.return_value = out_file

            output_path, validation = run_pipeline(args)

            # Verify pipeline was called
            mock_instance.analyze.assert_called_once()
            MockOutline.return_value.generate.assert_called_once()
            assert mock_gen_ch.call_count == 2  # 2 outline chapters
            mock_merge.assert_called_once()
            mock_validate.assert_called_once()
            assert validation["overall"] == "pass"

    def test_topic_mode_requires_topic(self, tmp_path: Path) -> None:
        material = tmp_path / "material"
        material.mkdir()
        (material / "notes.txt").write_text("Some content", encoding="utf-8")

        template = tmp_path / "template.md"
        template.write_text("# {{title}}\n\n{{content}}", encoding="utf-8")

        args = self._make_args(
            input_dir=str(material),
            template=str(template),
            no_interactive=True,
            scope="topic",
            topic=None,  # missing --topic
        )

        with pytest.raises(SystemExit):
            run_pipeline(args)

    def test_missing_input_exits(self) -> None:
        args = self._make_args(input_dir="/nonexistent/path")
        with pytest.raises(SystemExit):
            run_pipeline(args)

    def test_no_files_exits(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        template = tmp_path / "template.md"
        template.write_text("# {{title}}\n\n{{content}}", encoding="utf-8")

        args = self._make_args(
            input_dir=str(empty),
            template=str(template),
            no_interactive=True,
        )

        with pytest.raises(SystemExit):
            run_pipeline(args)
