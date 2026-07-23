"""Tests for pipeline.image_extractor."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from core.models import ExtractedImage
from pipeline.image_extractor import deduplicate_images, save_extracted_image


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_png(width: int, height: int, color: str = "red") -> bytes:
    """Generate valid PNG bytes of the given dimensions."""
    img = Image.new("RGB", (width, height), color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(width: int, height: int) -> bytes:
    """Generate valid JPEG bytes of the given dimensions."""
    img = Image.new("RGB", (width, height), color="blue")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _corrupt_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00corrupt"


# ── save_extracted_image ───────────────────────────────────────────────────

class TestSaveExtractedImage:
    def test_valid_png(self, tmp_path: Path) -> None:
        raw = _make_png(200, 150)
        img = save_extracted_image(raw, "doc1", assets_dir=tmp_path)
        assert img is not None
        assert img.source_document_id == "doc1"
        assert img.width == 200
        assert img.height == 150
        assert Path(img.file_path).exists()

    def test_valid_jpeg(self, tmp_path: Path) -> None:
        raw = _make_jpeg(300, 250)
        img = save_extracted_image(raw, "doc1", assets_dir=tmp_path)
        assert img is not None
        assert img.width == 300
        assert "jpeg" in img.file_path

    def test_file_saved_on_disk(self, tmp_path: Path) -> None:
        raw = _make_png(120, 110)
        img = save_extracted_image(raw, "doc1", assets_dir=tmp_path)
        assert Path(img.file_path).is_file()
        assert Path(img.file_path).stat().st_size > 0

    def test_hash_based_filename(self, tmp_path: Path) -> None:
        raw = _make_png(120, 110)
        img = save_extracted_image(raw, "doc1", assets_dir=tmp_path)
        filename = Path(img.file_path).name
        assert len(filename) == 20  # 16 hex + ".png"

    def test_same_bytes_reuse_file(self, tmp_path: Path) -> None:
        raw = _make_png(120, 110)
        img1 = save_extracted_image(raw, "doc1", assets_dir=tmp_path)
        img2 = save_extracted_image(raw, "doc2", assets_dir=tmp_path)
        assert img1.file_path == img2.file_path
        assert Path(img1.file_path).exists()

    def test_corrupt_returns_none(self, tmp_path: Path) -> None:
        result = save_extracted_image(_corrupt_bytes(), "doc1", assets_dir=tmp_path)
        assert result is None

    def test_empty_bytes_returns_none(self, tmp_path: Path) -> None:
        result = save_extracted_image(b"", "doc1", assets_dir=tmp_path)
        assert result is None

    def test_too_small_returns_none(self, tmp_path: Path) -> None:
        raw = _make_png(50, 50)
        result = save_extracted_image(raw, "doc1", assets_dir=tmp_path)
        assert result is None

    def test_custom_min_size(self, tmp_path: Path) -> None:
        raw = _make_png(30, 30)
        result = save_extracted_image(
            raw, "doc1", assets_dir=tmp_path, min_width=10, min_height=10,
        )
        assert result is not None

    def test_page_or_slide(self, tmp_path: Path) -> None:
        raw = _make_png(120, 110)
        img = save_extracted_image(raw, "doc1", page_or_slide=7, assets_dir=tmp_path)
        assert img.page_or_slide == 7

    def test_caption(self, tmp_path: Path) -> None:
        raw = _make_png(120, 110)
        img = save_extracted_image(raw, "doc1", caption="Fig. 1", assets_dir=tmp_path)
        assert img.caption == "Fig. 1"

    def test_missing_source_document_id_raises(self, tmp_path: Path) -> None:
        raw = _make_png(120, 110)
        with pytest.raises(ValueError, match="source_document_id"):
            save_extracted_image(raw, "", assets_dir=tmp_path)

    def test_assets_dir_created_automatically(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "nested" / "assets"
        raw = _make_png(120, 110)
        img = save_extracted_image(raw, "doc1", assets_dir=out_dir)
        assert out_dir.exists()
        assert Path(img.file_path).exists()


# ── deduplicate_images ──────────────────────────────────────────────────────

class TestDeduplicateImages:
    def _img(self, file_path: str, doc_id: str = "d") -> ExtractedImage:
        return ExtractedImage(
            source_document_id=doc_id,
            file_path=file_path,
            width=100,
            height=100,
        )

    def test_no_duplicates(self) -> None:
        imgs = [self._img("a.png"), self._img("b.png"), self._img("c.png")]
        result = deduplicate_images(imgs)
        assert len(result) == 3

    def test_removes_duplicates(self) -> None:
        imgs = [self._img("a.png"), self._img("a.png"), self._img("b.png")]
        result = deduplicate_images(imgs)
        assert len(result) == 2
        assert result[0].file_path == "a.png"
        assert result[1].file_path == "b.png"

    def test_keeps_first_occurrence(self) -> None:
        imgs = [
            self._img("x.png", doc_id="first"),
            self._img("x.png", doc_id="second"),
        ]
        result = deduplicate_images(imgs)
        assert len(result) == 1
        assert result[0].source_document_id == "first"

    def test_empty_list(self) -> None:
        assert deduplicate_images([]) == []

    def test_single_image(self) -> None:
        imgs = [self._img("only.png")]
        result = deduplicate_images(imgs)
        assert len(result) == 1

    def test_all_same(self) -> None:
        imgs = [self._img("same.png") for _ in range(5)]
        result = deduplicate_images(imgs)
        assert len(result) == 1
