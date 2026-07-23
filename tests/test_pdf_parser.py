"""Tests for parsers.pdf_parser."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pytest
from PIL import Image

from core.models import Document, FileType
from parsers.pdf_parser import parse_pdf


# ── Helpers ─────────────────────────────────────────────────────────────────

def _create_pdf(
    path: Path,
    pages: list[str] | None = None,
    title: str = "",
    image_size: Optional[tuple[int, int]] = (200, 150),
) -> Path:
    """Build a minimal PDF for testing."""
    doc = fitz.open()
    if title:
        doc.set_metadata({"title": title})

    text_pages = pages or ["Hello world"]
    for text in text_pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), text)

        if image_size is not None:
            img = Image.new("RGB", image_size, color="green")
            buf = BytesIO()
            img.save(buf, format="PNG")
            page.insert_image(fitz.Rect(100, 200, 100 + image_size[0], 200 + image_size[1]),
                              stream=buf.getvalue())

    doc.save(path)
    doc.close()
    return path


def _corrupt_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n\x00\x01\x02corrupt data here")
    return path


# ── parse_pdf ───────────────────────────────────────────────────────────────

class TestParsePdf:
    def test_returns_document(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "test.pdf")
        doc, images = parse_pdf(pdf)
        assert isinstance(doc, Document)
        assert isinstance(images, list)

    def test_file_type_is_pdf(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "test.pdf")
        doc, _ = parse_pdf(pdf)
        assert doc.file_type == FileType.PDF

    def test_source_path_matches(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "test.pdf")
        doc, _ = parse_pdf(pdf)
        assert doc.source_path == str(pdf)

    def test_text_extracted(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "test.pdf", pages=["Alpha beta gamma"])
        doc, _ = parse_pdf(pdf)
        assert "Alpha beta gamma" in doc.content

    def test_multiple_pages(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "test.pdf", pages=["Page one", "Page two", "Page three"])
        doc, _ = parse_pdf(pdf)
        assert "Page one" in doc.content
        assert "Page two" in doc.content
        assert "Page three" in doc.content

    def test_metadata_page_count(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "test.pdf", pages=["a", "b"])
        doc, _ = parse_pdf(pdf)
        assert doc.metadata["page_count"] == 2

    def test_metadata_title_from_pdf(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "test.pdf", title="My Book")
        doc, _ = parse_pdf(pdf)
        assert doc.metadata.get("title") == "My Book"

    def test_title_overridden_by_metadata(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "test.pdf", title="Meta Title")
        doc, _ = parse_pdf(pdf)
        assert doc.title == "Meta Title"

    def test_images_extracted(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "test.pdf", image_size=(200, 150))
        doc, images = parse_pdf(pdf)
        assert len(images) >= 1
        assert len(doc.images) >= 1
        assert images[0].page_or_slide == 1

    def test_image_has_file_on_disk(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "test.pdf", image_size=(200, 150))
        _, images = parse_pdf(pdf)
        assert Path(images[0].file_path).exists()

    def test_no_images_page(self, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "test.pdf", image_size=None)
        doc, images = parse_pdf(pdf)
        assert images == []
        assert doc.images == []

    def test_empty_pdf(self, tmp_path: Path) -> None:
        # PyMuPDF cannot save 0-page PDFs; use a page with no text/images
        doc_fit = fitz.open()
        doc_fit.new_page()
        doc_fit.save(tmp_path / "empty.pdf")
        doc_fit.close()
        doc, images = parse_pdf(tmp_path / "empty.pdf")
        assert doc.content.strip() == ""
        assert images == []

    def test_corrupt_pdf(self, tmp_path: Path) -> None:
        _corrupt_pdf(tmp_path / "bad.pdf")
        doc, images = parse_pdf(tmp_path / "bad.pdf")
        assert doc.content == ""
        assert images == []

    def test_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_pdf(Path("/no/such/file.pdf"))

    def test_empty_text_only(self, tmp_path: Path) -> None:
        # PDF with page but no visible text
        doc_fit = fitz.open()
        doc_fit.new_page()
        doc_fit.save(tmp_path / "no_text.pdf")
        doc_fit.close()
        doc, images = parse_pdf(tmp_path / "no_text.pdf")
        assert doc.content.strip() == ""

    def test_caption_near_image(self, tmp_path: Path) -> None:
        doc_fit = fitz.open()
        page = doc_fit.new_page(width=595, height=842)
        page.insert_text((100, 190), "Figure 1: Test diagram")
        img = Image.new("RGB", (200, 150), color="red")
        buf = BytesIO()
        img.save(buf, format="PNG")
        page.insert_image(fitz.Rect(100, 200, 300, 350), stream=buf.getvalue())
        doc_fit.save(tmp_path / "caption.pdf")
        doc_fit.close()
        _, images = parse_pdf(tmp_path / "caption.pdf")
        assert len(images) == 1
        assert images[0].caption is not None
