"""Tests for pipeline.normalizer."""

from __future__ import annotations

import pytest

from core.models import Document, FileType
from pipeline.normalizer import normalize_documents, normalize_text


# ── normalize_text ───────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_empty_string(self) -> None:
        assert normalize_text("") == ""

    def test_strips_whitespace(self) -> None:
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_spaces(self) -> None:
        assert normalize_text("hello   world") == "hello world"

    def test_collapses_tabs(self) -> None:
        assert normalize_text("hello\t\tworld") == "hello world"

    def test_collapses_newlines(self) -> None:
        assert normalize_text("line1\n\n\n\nline2") == "line1\n\nline2"

    def test_removes_control_chars(self) -> None:
        assert normalize_text("hel\x00lo") == "hello"
        assert normalize_text("a\x07b") == "ab"

    def test_preserves_newlines_normalizes_tabs(self) -> None:
        result = normalize_text("line1\nline2\ttab")
        assert "\n" in result
        assert "line2 tab" in result

    def test_unicode_nfc(self) -> None:
        # é as combining vs precomposed
        import unicodedata
        combining = "e\u0301"  # e + combining acute
        precomposed = "\u00e9"  # é
        assert normalize_text(combining) == normalize_text(precomposed)

    def test_multiline_cleanup(self) -> None:
        raw = "  First line  \n   Second line  \n\n\n  Third  "
        result = normalize_text(raw)
        assert result == "First line\nSecond line\n\nThird"

    def test_no_double_blank_lines(self) -> None:
        raw = "a\n\n\n\n\n\nb"
        assert normalize_text(raw) == "a\n\nb"

    def test_null_bytes_removed(self) -> None:
        assert normalize_text("test\x00\x00data") == "testdata"

    def test_only_whitespace(self) -> None:
        assert normalize_text("   \n\n\t  ") == ""


# ── normalize_documents ──────────────────────────────────────────────────────

class TestNormalizeDocuments:
    def test_returns_list(self) -> None:
        result = normalize_documents([{"filename": "a.txt", "text": "hello"}])
        assert isinstance(result, list)

    def test_document_type(self) -> None:
        result = normalize_documents([{"filename": "a.txt", "text": "hello"}])
        assert isinstance(result[0], Document)

    def test_empty_input(self) -> None:
        assert normalize_documents([]) == []

    def test_text_cleaned(self) -> None:
        docs = normalize_documents([{"filename": "a.txt", "text": "  spaced  "}])
        assert docs[0].content == "spaced"

    def test_title_from_filename(self) -> None:
        docs = normalize_documents([{"filename": "lecture_notes.pdf", "text": "text"}])
        assert docs[0].title == "lecture_notes"

    def test_source_path_is_filename(self) -> None:
        docs = normalize_documents([{"filename": "file.docx", "text": "text"}])
        assert docs[0].source_path == "file.docx"

    def test_file_type_txt(self) -> None:
        docs = normalize_documents([{"filename": "notes.txt", "text": "text"}])
        assert docs[0].file_type == FileType.TXT

    def test_file_type_pdf(self) -> None:
        docs = normalize_documents([{"filename": "paper.pdf", "text": "text"}])
        assert docs[0].file_type == FileType.PDF

    def test_file_type_docx(self) -> None:
        docs = normalize_documents([{"filename": "report.docx", "text": "text"}])
        assert docs[0].file_type == FileType.DOCX

    def test_file_type_md(self) -> None:
        docs = normalize_documents([{"filename": "readme.md", "text": "text"}])
        assert docs[0].file_type == FileType.MARKDOWN

    def test_file_type_image(self) -> None:
        docs = normalize_documents([{"filename": "photo.png", "text": "ocr text"}])
        assert docs[0].file_type == FileType.IMAGE

    def test_metadata_char_count(self) -> None:
        docs = normalize_documents([{"filename": "a.txt", "text": "hello"}])
        assert docs[0].metadata["char_count"] == 5

    def test_metadata_word_count(self) -> None:
        docs = normalize_documents([{"filename": "a.txt", "text": "hello world"}])
        assert docs[0].metadata["word_count"] == 2

    def test_metadata_line_count(self) -> None:
        docs = normalize_documents([{"filename": "a.txt", "text": "a\nb\nc"}])
        assert docs[0].metadata["line_count"] == 3

    def test_metadata_original_filename(self) -> None:
        docs = normalize_documents([{"filename": "x.pdf", "text": "text"}])
        assert docs[0].metadata["original_filename"] == "x.pdf"

    def test_multiple_documents(self) -> None:
        raw = [
            {"filename": "a.txt", "text": "aaa"},
            {"filename": "b.md", "text": "bbb"},
            {"filename": "c.pdf", "text": "ccc"},
        ]
        docs = normalize_documents(raw)
        assert len(docs) == 3
        assert docs[0].content == "aaa"
        assert docs[1].content == "bbb"
        assert docs[2].content == "ccc"

    def test_empty_text(self) -> None:
        docs = normalize_documents([{"filename": "empty.txt", "text": ""}])
        assert docs[0].content == ""
        assert docs[0].metadata["char_count"] == 0
        assert docs[0].metadata["word_count"] == 0

    def test_missing_keys_handled(self) -> None:
        docs = normalize_documents([{}])
        assert isinstance(docs[0], Document)
        assert docs[0].content == ""

    def test_markdown_preserved(self) -> None:
        md = "## Title\n\n- item 1\n- item 2\n\n**bold text**"
        docs = normalize_documents([{"filename": "notes.md", "text": md}])
        assert "## Title" in docs[0].content
        assert "**bold text**" in docs[0].content

    def test_ocr_text_cleaned(self) -> None:
        ocr = "  Some   noisy   OCR   text  \n\n\n  with gaps  "
        docs = normalize_documents([{"filename": "scan.png", "text": ocr}])
        assert docs[0].content == "Some noisy OCR text\n\nwith gaps"

    def test_unicode_content(self) -> None:
        text = "Ciao, questo è un testo con accenti: à, è, ì, ò, ù"
        docs = normalize_documents([{"filename": "italian.txt", "text": text}])
        assert "à" in docs[0].content
        assert "è" in docs[0].content
