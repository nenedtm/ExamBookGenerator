"""Tests for core.models."""

from __future__ import annotations

import pytest

from core.models import (
    Chapter,
    Chunk,
    Document,
    ExtractedImage,
    FileType,
    IndexEntry,
    Topic,
    _new_id,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

class TestNewId:
    def test_returns_string(self) -> None:
        assert isinstance(_new_id(), str)

    def test_length(self) -> None:
        assert len(_new_id()) == 12

    def test_unique(self) -> None:
        ids = {_new_id() for _ in range(100)}
        assert len(ids) == 100


# ── FileType ────────────────────────────────────────────────────────────────

class TestFileType:
    def test_members_exist(self) -> None:
        assert FileType.PDF == "pdf"
        assert FileType.DOCX == "docx"
        assert FileType.PPTX == "pptx"
        assert FileType.TXT == "txt"
        assert FileType.MARKDOWN == "markdown"
        assert FileType.IMAGE == "image"
        assert FileType.UNKNOWN == "unknown"


# ── ExtractedImage ──────────────────────────────────────────────────────────

class TestExtractedImage:
    def test_create_with_required_fields(self) -> None:
        img = ExtractedImage(source_document_id="doc1")
        assert img.source_document_id == "doc1"
        assert img.id  # auto-generated
        assert img.file_path == ""
        assert img.page_or_slide is None
        assert img.caption is None
        assert img.ai_description is None
        assert img.width == 0
        assert img.height == 0

    def test_create_with_all_fields(self) -> None:
        img = ExtractedImage(
            source_document_id="doc1",
            file_path="assets/img.png",
            page_or_slide=3,
            caption="Figure 1",
            ai_description="A diagram",
            width=800,
            height=600,
        )
        assert img.page_or_slide == 3
        assert img.width == 800

    def test_missing_source_document_id_raises(self) -> None:
        with pytest.raises(ValueError, match="source_document_id"):
            ExtractedImage(source_document_id="")

    def test_negative_width_raises(self) -> None:
        with pytest.raises(ValueError, match="width"):
            ExtractedImage(source_document_id="d", width=-1)

    def test_negative_height_raises(self) -> None:
        with pytest.raises(ValueError, match="height"):
            ExtractedImage(source_document_id="d", height=-1)


# ── Document ────────────────────────────────────────────────────────────────

class TestDocument:
    def test_create_defaults(self) -> None:
        doc = Document()
        assert doc.id
        assert doc.title == "untitled"
        assert doc.source_path == ""
        assert doc.file_type == FileType.UNKNOWN
        assert doc.content == ""
        assert doc.metadata == {}
        assert doc.images == []

    def test_title_from_source_path(self) -> None:
        doc = Document(source_path="/data/lecture_01.pdf")
        assert doc.title == "lecture_01"

    def test_explicit_title_wins(self) -> None:
        doc = Document(title="My Title", source_path="/data/lecture_01.pdf")
        assert doc.title == "My Title"

    def test_file_type(self) -> None:
        doc = Document(file_type=FileType.PDF)
        assert doc.file_type == FileType.PDF

    def test_metadata(self) -> None:
        doc = Document(metadata={"author": "Prof. X", "pages": 42})
        assert doc.metadata["author"] == "Prof. X"
        assert doc.metadata["pages"] == 42

    def test_images_list(self) -> None:
        doc = Document(images=["img_001", "img_002"])
        assert len(doc.images) == 2


# ── Chunk ───────────────────────────────────────────────────────────────────

class TestChunk:
    def test_create_defaults(self) -> None:
        chunk = Chunk(document_id="doc1")
        assert chunk.id
        assert chunk.document_id == "doc1"
        assert chunk.content == ""
        assert chunk.position == 0

    def test_custom_values(self) -> None:
        chunk = Chunk(document_id="doc1", content="text", position=5)
        assert chunk.content == "text"
        assert chunk.position == 5

    def test_missing_document_id_raises(self) -> None:
        with pytest.raises(ValueError, match="document_id"):
            Chunk(document_id="")

    def test_negative_position_raises(self) -> None:
        with pytest.raises(ValueError, match="position"):
            Chunk(document_id="d", position=-1)


# ── Topic ───────────────────────────────────────────────────────────────────

class TestTopic:
    def test_create_defaults(self) -> None:
        topic = Topic(name="Algebra")
        assert topic.name == "Algebra"
        assert topic.description == ""
        assert topic.related_documents == []
        assert topic.subtopic_count == 0

    def test_custom_values(self) -> None:
        topic = Topic(
            name="Algebra",
            description="Linear algebra basics",
            related_documents=["d1", "d2"],
            subtopic_count=5,
        )
        assert len(topic.related_documents) == 2
        assert topic.subtopic_count == 5

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Topic(name="")

    def test_negative_subtopic_count_raises(self) -> None:
        with pytest.raises(ValueError, match="subtopic_count"):
            Topic(name="t", subtopic_count=-1)


# ── Chapter ─────────────────────────────────────────────────────────────────

class TestChapter:
    def test_create_defaults(self) -> None:
        ch = Chapter(title="Introduction")
        assert ch.title == "Introduction"
        assert ch.content == ""
        assert ch.order == 0
        assert ch.images == []

    def test_custom_values(self) -> None:
        ch = Chapter(title="Ch1", content="# Hello", order=2, images=["img1"])
        assert ch.content == "# Hello"
        assert ch.order == 2
        assert ch.images == ["img1"]

    def test_missing_title_raises(self) -> None:
        with pytest.raises(ValueError, match="title"):
            Chapter(title="")

    def test_negative_order_raises(self) -> None:
        with pytest.raises(ValueError, match="order"):
            Chapter(title="t", order=-1)


# ── v3: Document.is_syllabus ────────────────────────────────────────────────

class TestDocumentIsSyllabus:
    def test_default_false(self) -> None:
        doc = Document()
        assert doc.is_syllabus is False

    def test_can_set_true(self) -> None:
        doc = Document(is_syllabus=True)
        assert doc.is_syllabus is True


# ── v3: Topic order_source / syllabus_position ──────────────────────────────

class TestTopicV3:
    def test_default_order_source(self) -> None:
        topic = Topic(name="Algebra")
        assert topic.order_source == "pedagogical"

    def test_syllabus_order_source(self) -> None:
        topic = Topic(name="Algebra", order_source="syllabus", syllabus_position=3)
        assert topic.order_source == "syllabus"
        assert topic.syllabus_position == 3

    def test_syllabus_position_default_none(self) -> None:
        topic = Topic(name="Algebra")
        assert topic.syllabus_position is None


# ── v3: Chapter.toc_entries ─────────────────────────────────────────────────

class TestChapterV3:
    def test_default_toc_entries(self) -> None:
        ch = Chapter(title="Intro")
        assert ch.toc_entries == []

    def test_custom_toc_entries(self) -> None:
        ch = Chapter(title="Intro", toc_entries=["Overview", "History", "Methods"])
        assert len(ch.toc_entries) == 3


# ── IndexEntry ──────────────────────────────────────────────────────────────

class TestIndexEntry:
    def test_create_with_title(self) -> None:
        entry = IndexEntry(title="Vector Spaces", anchor="vector-spaces", level=1, order=0)
        assert entry.title == "Vector Spaces"
        assert entry.anchor == "vector-spaces"
        assert entry.level == 1
        assert entry.order == 0

    def test_missing_title_raises(self) -> None:
        with pytest.raises(ValueError, match="title"):
            IndexEntry(title="")

    def test_level_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="level"):
            IndexEntry(title="X", level=0)

    def test_negative_order_raises(self) -> None:
        with pytest.raises(ValueError, match="order"):
            IndexEntry(title="X", order=-1)

    def test_defaults(self) -> None:
        entry = IndexEntry(title="Section")
        assert entry.anchor == ""
        assert entry.level == 1
        assert entry.order == 0
