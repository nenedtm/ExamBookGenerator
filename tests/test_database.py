"""Tests for storage.database."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.models import Document, FileType
from storage.database import (
    count_documents,
    delete_document,
    load_document,
    load_document_by_id,
    save_document,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_doc(
    source_path: str = "/data/lecture.pdf",
    title: str = "",
    content: str = "Hello world",
    file_type: FileType = FileType.PDF,
) -> Document:
    return Document(
        source_path=source_path,
        title=title,
        content=content,
        file_type=file_type,
        metadata={"author": "Prof. X"},
        images=["img_001"],
    )


def _db(tmp_path: Path) -> str:
    return str(tmp_path / "test.db")


# ── save_document / load_document ────────────────────────────────────────────

class TestSaveDocument:
    def test_returns_document(self, tmp_path: Path) -> None:
        doc = _make_doc()
        result = save_document(doc, db_path=_db(tmp_path))
        assert isinstance(result, Document)

    def test_loadable_after_save(self, tmp_path: Path) -> None:
        doc = _make_doc()
        save_document(doc, db_path=_db(tmp_path))
        loaded = load_document(doc.source_path, db_path=_db(tmp_path))
        assert loaded is not None
        assert loaded.id == doc.id

    def test_content_preserved(self, tmp_path: Path) -> None:
        doc = _make_doc(content="Specific content here")
        save_document(doc, db_path=_db(tmp_path))
        loaded = load_document(doc.source_path, db_path=_db(tmp_path))
        assert loaded.content == "Specific content here"

    def test_metadata_preserved(self, tmp_path: Path) -> None:
        doc = _make_doc()
        save_document(doc, db_path=_db(tmp_path))
        loaded = load_document(doc.source_path, db_path=_db(tmp_path))
        assert loaded.metadata["author"] == "Prof. X"

    def test_images_preserved(self, tmp_path: Path) -> None:
        doc = _make_doc()
        save_document(doc, db_path=_db(tmp_path))
        loaded = load_document(doc.source_path, db_path=_db(tmp_path))
        assert loaded.images == ["img_001"]

    def test_file_type_preserved(self, tmp_path: Path) -> None:
        doc = _make_doc(file_type=FileType.DOCX)
        save_document(doc, db_path=_db(tmp_path))
        loaded = load_document(doc.source_path, db_path=_db(tmp_path))
        assert loaded.file_type == FileType.DOCX

    def test_title_preserved(self, tmp_path: Path) -> None:
        doc = _make_doc(title="Custom Title")
        save_document(doc, db_path=_db(tmp_path))
        loaded = load_document(doc.source_path, db_path=_db(tmp_path))
        assert loaded.title == "Custom Title"


class TestUpsert:
    def test_update_on_duplicate_source_path(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        doc1 = _make_doc(content="Version 1")
        save_document(doc1, db_path=db)

        doc2 = _make_doc(content="Version 2")
        save_document(doc2, db_path=db)

        loaded = load_document(doc1.source_path, db_path=db)
        assert loaded.content == "Version 2"

    def test_count_after_upsert(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_document(_make_doc(source_path="/a.pdf"), db_path=db)
        save_document(_make_doc(source_path="/b.pdf"), db_path=db)
        save_document(_make_doc(source_path="/a.pdf"), db_path=db)  # upsert
        assert count_documents(db_path=db) == 2


class TestLoadNotFound:
    def test_returns_none(self, tmp_path: Path) -> None:
        result = load_document("/no/such/file.pdf", db_path=_db(tmp_path))
        assert result is None


class TestLoadDocumentById:
    def test_found(self, tmp_path: Path) -> None:
        doc = _make_doc()
        save_document(doc, db_path=_db(tmp_path))
        loaded = load_document_by_id(doc.id, db_path=_db(tmp_path))
        assert loaded is not None
        assert loaded.id == doc.id

    def test_not_found(self, tmp_path: Path) -> None:
        result = load_document_by_id("nonexistent", db_path=_db(tmp_path))
        assert result is None


class TestCountDocuments:
    def test_empty_db(self, tmp_path: Path) -> None:
        assert count_documents(db_path=_db(tmp_path)) == 0

    def test_after_inserts(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_document(_make_doc(source_path="/a.pdf"), db_path=db)
        save_document(_make_doc(source_path="/b.pdf"), db_path=db)
        assert count_documents(db_path=db) == 2


class TestDeleteDocument:
    def test_existing(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        doc = _make_doc()
        save_document(doc, db_path=db)
        assert delete_document(doc.source_path, db_path=db) is True
        assert load_document(doc.source_path, db_path=db) is None

    def test_nonexistent(self, tmp_path: Path) -> None:
        assert delete_document("/missing.pdf", db_path=_db(tmp_path)) is False

    def test_count_after_delete(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_document(_make_doc(source_path="/a.pdf"), db_path=db)
        save_document(_make_doc(source_path="/b.pdf"), db_path=db)
        delete_document("/a.pdf", db_path=db)
        assert count_documents(db_path=db) == 1


class TestEdgeCases:
    def test_empty_content(self, tmp_path: Path) -> None:
        doc = _make_doc(content="")
        save_document(doc, db_path=_db(tmp_path))
        loaded = load_document(doc.source_path, db_path=_db(tmp_path))
        assert loaded.content == ""

    def test_empty_metadata(self, tmp_path: Path) -> None:
        doc = Document(source_path="/x.pdf", metadata={}, images=[])
        save_document(doc, db_path=_db(tmp_path))
        loaded = load_document("/x.pdf", db_path=_db(tmp_path))
        assert loaded.metadata == {}
        assert loaded.images == []

    def test_unicode_content(self, tmp_path: Path) -> None:
        doc = _make_doc(content="Ciao, testo con accenti: à, è, ì, ò, ù")
        save_document(doc, db_path=_db(tmp_path))
        loaded = load_document(doc.source_path, db_path=_db(tmp_path))
        assert "à" in loaded.content

    def test_large_content(self, tmp_path: Path) -> None:
        big = "x" * 500_000
        doc = _make_doc(content=big)
        save_document(doc, db_path=_db(tmp_path))
        loaded = load_document(doc.source_path, db_path=_db(tmp_path))
        assert len(loaded.content) == 500_000

    def test_database_auto_created(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "sub" / "dir" / "test.db")
        doc = _make_doc()
        save_document(doc, db_path=db_path)
        assert Path(db_path).exists()
