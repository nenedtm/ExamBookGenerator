"""Document normalizer for ExamBookGenerator.

Converts raw parser output ( dicts with ``filename`` / ``text`` ) into
standardised ``Document`` objects, cleaning and normalising the text along
the way so downstream modules receive uniform input.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from core.models import Document, FileType
from utils.logger import get_logger

logger = get_logger(__name__)

_EXTENSION_MAP: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".doc": FileType.DOCX,
    ".pptx": FileType.PPTX,
    ".ppt": FileType.PPTX,
    ".txt": FileType.TXT,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".png": FileType.IMAGE,
    ".jpg": FileType.IMAGE,
    ".jpeg": FileType.IMAGE,
    ".gif": FileType.IMAGE,
    ".bmp": FileType.IMAGE,
    ".tiff": FileType.IMAGE,
    ".tif": FileType.IMAGE,
    ".webp": FileType.IMAGE,
}

_RE_MULTI_SPACES = re.compile(r"[^\S\n]+")
_RE_MULTI_NEWLINES = re.compile(r"\n{3,}")
_RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ── Text cleanup ─────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Clean and normalise raw text.

    Steps:
      1. Unicode NFC normalisation.
      2. Remove stray null bytes and control characters (keep ``\\n`` and
         ``\\t``).
      3. Collapse runs of horizontal whitespace to a single space.
      4. Collapse three or more consecutive blank lines to two.
      5. Strip leading / trailing whitespace from the whole block and from
         every individual line.

    Parameters
    ----------
    text:
        Raw text produced by a parser.

    Returns
    -------
    str
        The normalised text.
    """
    if not text:
        return ""

    # 1 ─ Unicode normalisation
    text = unicodedata.normalize("NFC", text)

    # 2 ─ Remove control characters (keep \n and \t)
    text = _RE_CONTROL_CHARS.sub("", text)

    # 3 ─ Collapse horizontal whitespace to single spaces
    text = _RE_MULTI_SPACES.sub(" ", text)

    # 4 ─ Strip each line then collapse excess blank lines
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    text = "\n".join(lines)
    text = _RE_MULTI_NEWLINES.sub("\n\n", text)

    # 5 ─ Final strip
    return text.strip()


# ── Type detection ───────────────────────────────────────────────────────────

def _detect_file_type(filename: str) -> FileType:
    """Return the ``FileType`` matching *filename*'s extension."""
    suffix = Path(filename).suffix.lower()
    return _EXTENSION_MAP.get(suffix, FileType.UNKNOWN)


# ── Document normalisation ───────────────────────────────────────────────────

def _build_document(raw: dict[str, str]) -> Document:
    """Convert a single raw dict into a normalised ``Document``."""
    filename: str = raw.get("filename", "")
    text: str = raw.get("text", "")

    cleaned = normalize_text(text)
    file_type = _detect_file_type(filename)

    doc = Document(
        source_path=filename,
        file_type=file_type,
        content=cleaned,
    )

    # ── Metadata ─────────────────────────────────────────────────────────
    doc.metadata["original_filename"] = filename
    doc.metadata["char_count"] = len(cleaned)
    doc.metadata["word_count"] = len(cleaned.split()) if cleaned else 0
    doc.metadata["line_count"] = cleaned.count("\n") + (1 if cleaned else 0)

    logger.debug(
        "Normalised '%s' — %d chars, %d words",
        filename,
        doc.metadata["char_count"],
        doc.metadata["word_count"],
    )
    return doc


def normalize_documents(raw_docs: list[dict[str, str]]) -> list[Document]:
    """Normalise a batch of raw parser outputs into ``Document`` objects.

    Parameters
    ----------
    raw_docs:
        List of dicts, each containing at least ``filename`` and ``text``
        keys.

    Returns
    -------
    list[Document]
        One ``Document`` per input dict, with cleaned text and populated
        metadata.

    Example
    -------
    >>> docs = normalize_documents([
    ...     {"filename": "lezione1.pdf", "text": "  Ciao   mondo  "},
    ...     {"filename": "appunti.md",   "text": "## Titolo\\n\\nContenuto"},
    ... ])
    >>> docs[0].content
    'Ciao mondo'
    >>> docs[1].file_type
    <FileType.MARKDOWN: 'markdown'>
    """
    if not raw_docs:
        logger.info("normalize_documents called with empty list")
        return []

    documents: list[Document] = []
    for raw in raw_docs:
        try:
            doc = _build_document(raw)
            documents.append(doc)
        except Exception as exc:
            filename = raw.get("filename", "<unknown>")
            logger.warning("Failed to normalise '%s': %s", filename, exc)

    logger.info(
        "Normalisation complete — %d / %d document(s) processed",
        len(documents),
        len(raw_docs),
    )
    return documents
