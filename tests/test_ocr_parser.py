"""Tests for parsers.ocr_parser."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from core.models import Document, FileType
from parsers.ocr_parser import parse_image


# ── Helpers ─────────────────────────────────────────────────────────────────

def _create_image(
    path: Path,
    size: tuple[int, int] = (100, 50),
    color: str = "white",
    fmt: str = "PNG",
) -> Path:
    """Create a minimal image file for testing."""
    img = Image.new("RGB", size, color=color)
    img.save(path, format=fmt)
    return path


def _create_image_with_text(path: Path) -> Path:
    """Create an image with rendered text for OCR testing."""
    img = Image.new("RGB", (400, 100), color="white")
    # Draw some text using Pillow's built-in font
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), "Hello World", fill="black")
    except Exception:
        pass
    img.save(path, format="PNG")
    return path


def _corrupt_image(path: Path) -> Path:
    """Write garbage bytes to simulate a corrupt image."""
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00corrupt image data here")
    return path


# ── parse_image ─────────────────────────────────────────────────────────────

class TestParseImage:
    def test_returns_document(self, tmp_path: Path) -> None:
        p = _create_image(tmp_path / "test.png")
        doc = parse_image(p)
        assert isinstance(doc, Document)

    def test_file_type_is_image(self, tmp_path: Path) -> None:
        p = _create_image(tmp_path / "test.png")
        doc = parse_image(p)
        assert doc.file_type == FileType.IMAGE

    def test_source_path_matches(self, tmp_path: Path) -> None:
        p = _create_image(tmp_path / "test.png")
        doc = parse_image(p)
        assert doc.source_path == str(p)

    def test_title_from_filename(self, tmp_path: Path) -> None:
        p = _create_image(tmp_path / "my_photo.png")
        doc = parse_image(p)
        assert doc.title == "my_photo"

    def test_metadata_dimensions(self, tmp_path: Path) -> None:
        p = _create_image(tmp_path / "test.png", size=(200, 150))
        doc = parse_image(p)
        assert doc.metadata["width"] == 200
        assert doc.metadata["height"] == 150

    @patch("parsers.ocr_parser.pytesseract")
    def test_ocr_called(self, mock_pytesseract: MagicMock, tmp_path: Path) -> None:
        mock_pytesseract.image_to_string.return_value = "Extracted text"
        p = _create_image(tmp_path / "test.png")
        doc = parse_image(p)
        mock_pytesseract.image_to_string.assert_called_once()
        assert doc.content == "Extracted text"

    @patch("parsers.ocr_parser.pytesseract")
    def test_ocr_metadata(self, mock_pytesseract: MagicMock, tmp_path: Path) -> None:
        mock_pytesseract.image_to_string.return_value = "text"
        p = _create_image(tmp_path / "test.png")
        doc = parse_image(p)
        assert doc.metadata["ocr_engine"] == "tesseract"
        assert doc.metadata["ocr_lang"] == "ita+eng"

    @patch("parsers.ocr_parser.pytesseract")
    def test_custom_lang(self, mock_pytesseract: MagicMock, tmp_path: Path) -> None:
        mock_pytesseract.image_to_string.return_value = "text"
        p = _create_image(tmp_path / "test.png")
        doc = parse_image(p, lang="eng")
        assert doc.metadata["ocr_lang"] == "eng"

    def test_empty_ocr_result(self, tmp_path: Path) -> None:
        with patch("parsers.ocr_parser.pytesseract") as mock_pyt:
            mock_pyt.image_to_string.return_value = ""
            p = _create_image(tmp_path / "test.png")
            doc = parse_image(p)
            assert doc.content == ""

    def test_corrupt_image(self, tmp_path: Path) -> None:
        _corrupt_image(tmp_path / "bad.png")
        doc = parse_image(tmp_path / "bad.png")
        assert doc.content == ""
        assert "error" in doc.metadata

    def test_unsupported_format(self, tmp_path: Path) -> None:
        p = tmp_path / "test.bmp"
        p.write_bytes(b"not a real bmp")
        doc = parse_image(p)
        assert doc.content == ""
        assert "error" in doc.metadata

    def test_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_image(Path("/no/such/file.png"))

    @patch("parsers.ocr_parser.pytesseract")
    def test_jpg_support(self, mock_pytesseract: MagicMock, tmp_path: Path) -> None:
        mock_pytesseract.image_to_string.return_value = "jpg text"
        p = _create_image(tmp_path / "test.jpg", fmt="JPEG")
        doc = parse_image(p)
        assert doc.content == "jpg text"

    @patch("parsers.ocr_parser.pytesseract")
    def test_jpeg_support(self, mock_pytesseract: MagicMock, tmp_path: Path) -> None:
        mock_pytesseract.image_to_string.return_value = "jpeg text"
        p = _create_image(tmp_path / "test.jpeg", fmt="JPEG")
        doc = parse_image(p)
        assert doc.content == "jpeg text"

    @patch("parsers.ocr_parser.pytesseract")
    def test_ocr_exception_handled(self, mock_pytesseract: MagicMock, tmp_path: Path) -> None:
        mock_pytesseract.image_to_string.side_effect = RuntimeError("tesseract not found")
        p = _create_image(tmp_path / "test.png")
        doc = parse_image(p)
        assert doc.content == ""
        assert "error" in doc.metadata
