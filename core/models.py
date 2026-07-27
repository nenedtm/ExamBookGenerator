"""Central data models shared by the entire ExamBookGenerator pipeline.

Every other module (parsers, pipeline, llm, storage) imports from here.
These classes carry **no** knowledge of AI, parsers, or filesystem layout —
they are pure data containers with lightweight validation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


def _new_id() -> str:
    """Generate a short, unique identifier."""
    return uuid.uuid4().hex[:12]


# ── Enums ───────────────────────────────────────────────────────────────────

class FileType(str, Enum):
    """Supported source file types."""

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    TXT = "txt"
    MARKDOWN = "markdown"
    IMAGE = "image"
    UNKNOWN = "unknown"


# ── Models ──────────────────────────────────────────────────────────────────

@dataclass
class ExtractedImage:
    """An image extracted from a source document (not AI-generated).

    Attributes
    ----------
    id:
        Unique identifier for this image.
    source_document_id:
        The ``Document.id`` this image was extracted from.
    file_path:
        Relative path where the image is stored under ``output/assets/images/``.
    page_or_slide:
        Page or slide number where the image was found, or *None* if unknown.
    caption:
        Original caption embedded in the document, if any.
    ai_description:
        Description produced by a vision model (populated later by the
        image-matching step).
    width:
        Image width in pixels.
    height:
        Image height in pixels.
    """

    id: str = field(default_factory=_new_id)
    source_document_id: str = ""
    file_path: str = ""
    page_or_slide: int | None = None
    caption: str | None = None
    ai_description: str | None = None
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        if not self.source_document_id:
            raise ValueError("source_document_id is required")
        if self.width < 0:
            raise ValueError(f"width must be >= 0, got {self.width}")
        if self.height < 0:
            raise ValueError(f"height must be >= 0, got {self.height}")


@dataclass
class Document:
    """A document that has been parsed and whose text has been extracted.

    Attributes
    ----------
    id:
        Unique identifier for this document.
    title:
        Human-readable title (derived from filename or metadata).
    source_path:
        Original filesystem path of the source file.
    file_type:
        Kind of source file (PDF, DOCX, …).
    content:
        Extracted plain-text content.
    metadata:
        Arbitrary key-value metadata (author, page count, …).
    images:
        List of ``ExtractedImage.id`` values found in this document.
    """

    id: str = field(default_factory=_new_id)
    title: str = ""
    source_path: str = ""
    file_type: FileType = FileType.UNKNOWN
    content: str = ""
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    is_syllabus: bool = False

    def __post_init__(self) -> None:
        if not self.title:
            self.title = Path(self.source_path).stem if self.source_path else "untitled"


@dataclass
class Chunk:
    """A portion of a ``Document``, typically produced by a text splitter.

    Attributes
    ----------
    id:
        Unique identifier for this chunk.
    document_id:
        The ``Document.id`` this chunk belongs to.
    content:
        The text content of the chunk.
    position:
        Ordinal position of this chunk within its document (0-based).
    """

    id: str = field(default_factory=_new_id)
    document_id: str = ""
    content: str = ""
    position: int = 0

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id is required")
        if self.position < 0:
            raise ValueError(f"position must be >= 0, got {self.position}")


@dataclass
class Topic:
    """A subject or argument extracted from the source material.

    Attributes
    ----------
    name:
        Short, human-readable topic name.
    description:
        Brief summary of what this topic covers.
    related_documents:
        List of ``Document.id`` values that discuss this topic.
    subtopic_count:
        Number of sub-topics used to estimate generated chapter length.
    """

    name: str = ""
    description: str = ""
    related_documents: list[str] = field(default_factory=list)
    subtopic_count: int = 0
    order_source: Literal["syllabus", "pedagogical", "manual"] = "pedagogical"
    syllabus_position: int | None = None
    missing_from_notes: bool = False
    extra_in_notes: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name is required")
        if self.subtopic_count < 0:
            raise ValueError(f"subtopic_count must be >= 0, got {self.subtopic_count}")


@dataclass
class Chapter:
    """A chapter of the final generated manual.

    Attributes
    ----------
    title:
        Chapter heading.
    content:
        Full Markdown text of the chapter.
    order:
        Position of this chapter in the final document (0-based).
    images:
        List of ``ExtractedImage.id`` values actually inserted into this chapter.
    """

    title: str = ""
    content: str = ""
    order: int = 0
    images: list[str] = field(default_factory=list)
    toc_entries: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("title is required")
        if self.order < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")


# ── v3 models ────────────────────────────────────────────────────────────────

@dataclass
class IndexEntry:
    """A single entry in a programmatically-built table of contents.

    Attributes
    ----------
    title:
        Display text for this entry (e.g. ``"Vector Spaces"``).
    anchor:
        Markdown-friendly slug used for internal links (e.g. ``"vector-spaces"``).
    level:
        Nesting depth — 1 for top-level chapters, 2 for sub-sections, etc.
    order:
        Sequential position in the document (0-based).
    """

    title: str = ""
    anchor: str = ""
    level: int = 1
    order: int = 0

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("title is required")
        if self.level < 1:
            raise ValueError(f"level must be >= 1, got {self.level}")
        if self.order < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")


@dataclass
class OutlineChapter:
    """A chapter planned by the outline generator.

    Attributes
    ----------
    title:
        The chapter heading as determined by the outline.
    sections:
        Ordered list of sub-section headings within this chapter.
    topic_indices:
        Indices into the topic list that this outline chapter covers.
        The outline generator may group multiple topics into one chapter.
    """

    title: str = ""
    sections: list[str] = field(default_factory=list)
    topic_indices: list[int] = field(default_factory=list)
