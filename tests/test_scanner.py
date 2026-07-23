"""Tests for pipeline.scanner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.models import FileType
from pipeline.scanner import detect_file_type, generate_inventory, scan_directory


# ── Helpers ─────────────────────────────────────────────────────────────────

def _touch(path: Path) -> Path:
    """Create an empty file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


# ── detect_file_type ───────────────────────────────────────────────────────

class TestDetectFileType:
    def test_pdf(self) -> None:
        assert detect_file_type(Path("doc.pdf")) == FileType.PDF

    def test_docx(self) -> None:
        assert detect_file_type(Path("report.docx")) == FileType.DOCX

    def test_pptx(self) -> None:
        assert detect_file_type(Path("slides.pptx")) == FileType.PPTX

    def test_txt(self) -> None:
        assert detect_file_type(Path("notes.txt")) == FileType.TXT

    def test_markdown(self) -> None:
        assert detect_file_type(Path("readme.md")) == FileType.MARKDOWN

    def test_markdown_long(self) -> None:
        assert detect_file_type(Path("doc.markdown")) == FileType.MARKDOWN

    def test_image_png(self) -> None:
        assert detect_file_type(Path("photo.png")) == FileType.IMAGE

    def test_image_jpg(self) -> None:
        assert detect_file_type(Path("photo.jpg")) == FileType.IMAGE

    def test_image_jpeg(self) -> None:
        assert detect_file_type(Path("photo.jpeg")) == FileType.IMAGE

    def test_image_svg(self) -> None:
        assert detect_file_type(Path("icon.svg")) == FileType.IMAGE

    def test_unknown_extension(self) -> None:
        assert detect_file_type(Path("data.xyz")) == FileType.UNKNOWN

    def test_no_extension(self) -> None:
        assert detect_file_type(Path("Makefile")) == FileType.UNKNOWN

    def test_case_insensitive(self) -> None:
        assert detect_file_type(Path("DOC.PDF")) == FileType.PDF


# ── scan_directory ─────────────────────────────────────────────────────────

class TestScanDirectory:
    def test_flat_directory(self, tmp_path: Path) -> None:
        _touch(tmp_path / "a.pdf")
        _touch(tmp_path / "b.docx")
        docs = scan_directory(tmp_path)
        assert len(docs) == 2

    def test_recursive(self, tmp_path: Path) -> None:
        _touch(tmp_path / "top.pdf")
        _touch(tmp_path / "sub" / "nested.docx")
        _touch(tmp_path / "sub" / "deep" / "deep.pptx")
        docs = scan_directory(tmp_path)
        assert len(docs) == 3

    def test_skips_unknown(self, tmp_path: Path) -> None:
        _touch(tmp_path / "readme.pdf")
        _touch(tmp_path / "data.xyz")
        _touch(tmp_path / "noext")
        docs = scan_directory(tmp_path)
        assert len(docs) == 1
        assert docs[0].file_type == FileType.PDF

    def test_empty_directory(self, tmp_path: Path) -> None:
        docs = scan_directory(tmp_path)
        assert docs == []

    def test_documents_have_source_path(self, tmp_path: Path) -> None:
        _touch(tmp_path / "test.pdf")
        docs = scan_directory(tmp_path)
        assert docs[0].source_path == str(tmp_path / "test.pdf")

    def test_documents_have_correct_type(self, tmp_path: Path) -> None:
        _touch(tmp_path / "img.png")
        docs = scan_directory(tmp_path)
        assert docs[0].file_type == FileType.IMAGE

    def test_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            scan_directory(Path("/definitely/does/not/exist"))

    def test_file_not_dir_raises(self, tmp_path: Path) -> None:
        f = _touch(tmp_path / "file.txt")
        with pytest.raises(NotADirectoryError):
            scan_directory(f)

    def test_sorted_output(self, tmp_path: Path) -> None:
        _touch(tmp_path / "z.pdf")
        _touch(tmp_path / "a.docx")
        _touch(tmp_path / "m.txt")
        docs = scan_directory(tmp_path)
        names = [Path(d.source_path).name for d in docs]
        assert names == sorted(names)


# ── generate_inventory ──────────────────────────────────────────────────────

class TestGenerateInventory:
    def test_creates_json(self, tmp_path: Path) -> None:
        _touch(tmp_path / "a.pdf")
        docs = scan_directory(tmp_path)
        out = generate_inventory(docs, output_path=tmp_path / "inv.json")
        assert out.exists()

    def test_json_content(self, tmp_path: Path) -> None:
        _touch(tmp_path / "lecture.docx")
        docs = scan_directory(tmp_path)
        out = generate_inventory(docs, output_path=tmp_path / "inv.json")
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["file_type"] == "docx"
        assert "id" in data[0]
        assert "title" in data[0]
        assert "source_path" in data[0]

    def test_empty_inventory(self, tmp_path: Path) -> None:
        out = generate_inventory([], output_path=tmp_path / "inv.json")
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == []

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = generate_inventory([], output_path=tmp_path / "sub" / "dir" / "inv.json")
        assert out.exists()

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        out = generate_inventory([], output_path=tmp_path / "inv.json")
        assert out.is_absolute()
