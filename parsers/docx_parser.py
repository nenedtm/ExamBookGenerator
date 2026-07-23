"""DOCX parser for ExamBookGenerator.

Extracts text (paragraphs + tables) and embedded images from Word documents
using python-docx.  Images are delegated to
``pipeline.image_extractor.save_extracted_image`` for validation,
deduplication, and disk storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
from docx.opc.constants import RELATIONSHIP_TYPE as RT

from core.models import Document, ExtractedImage, FileType
from pipeline.image_extractor import save_extracted_image
from utils.logger import get_logger

logger = get_logger(__name__)

# Relationship type for images in the OPC package.
_IMAGE_REL = RT.IMAGE


def _iter_body_elements(body: Any) -> list[Any]:
    """Return body child elements in document order (paragraphs + tables)."""
    return list(body)


def _extract_paragraph_text(paragraph: Any) -> str:
    """Return the plain text of a paragraph."""
    return paragraph.text or ""


def _extract_table_text(table: Any) -> str:
    """Return the plain text of a table as Markdown-style rows."""
    rows: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_images(
    docx_doc: DocxDocument,
    source_document_id: str,
) -> list[ExtractedImage]:
    """Walk the document XML, find embedded image references, and delegate
    to ``save_extracted_image`` for each one."""
    images: list[ExtractedImage] = []
    seen_rids: set[str] = set()

    for rel in docx_doc.part.rels.values():
        if rel.reltype != _IMAGE_REL:
            continue
        rid = rel.rId
        if rid in seen_rids:
            continue
        seen_rids.add(rid)

        try:
            blob = rel.target_part.blob
        except Exception:
            logger.warning(
                "Cannot read image %s from DOCX %s — skipped",
                rid, source_document_id,
            )
            continue

        if not blob:
            continue

        # Try to find alt-text from the XML element referencing this image
        caption = _find_alt_text(docx_doc, rid)

        extracted = save_extracted_image(
            raw_bytes=blob,
            source_document_id=source_document_id,
            caption=caption,
        )
        if extracted is not None:
            images.append(extracted)

    return images


def _find_alt_text(docx_doc: DocxDocument, rid: str) -> str | None:
    """Search the document body XML for an ``<a:blip>`` referencing *rid*
    and return its ``descr`` or ``name`` attribute as caption."""
    nsmap = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    for blip in docx_doc.element.iter():
        tag = blip.tag if isinstance(blip.tag, str) else ""
        if not tag.endswith("}blip"):
            continue
        embed = blip.get(f"{{{nsmap['r']}}}embed")
        if embed != rid:
            continue
        # Walk up to the parent <a:graphic>/<a:graphicFrame> and look for
        # docPr name/descr
        parent = blip.getparent()
        while parent is not None:
            ptag = parent.tag if isinstance(parent.tag, str) else ""
            if ptag.endswith("}pic"):
                break
            parent = parent.getparent()
        if parent is not None:
            for child in parent.iter():
                ctag = child.tag if isinstance(child.tag, str) else ""
                if ctag.endswith("}docPr"):
                    descr = child.get("descr", "")
                    name = child.get("name", "")
                    return descr if descr else name if name else None
        break

    return None


def parse_docx(path: Path | str) -> tuple[Document, list[ExtractedImage]]:
    """Parse a DOCX file, extracting text and embedded images.

    Parameters
    ----------
    path:
        Path to a ``.docx`` file.

    Returns
    -------
    tuple[Document, list[ExtractedImage]]
        A ``Document`` with ``content`` and ``images`` populated, plus the
        list of ``ExtractedImage`` objects saved to disk.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    docx_path = Path(path)

    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX not found: {docx_path}")

    doc = Document(source_path=str(docx_path), file_type=FileType.DOCX)
    images: list[ExtractedImage] = []

    try:
        docx_doc = DocxDocument(docx_path)
    except Exception:
        logger.warning("Cannot open DOCX %s — corrupted or unreadable", docx_path)
        return doc, images

    # ── Metadata ────────────────────────────────────────────────────────
    core = docx_doc.core_properties
    meta_fields = {
        "title": core.title,
        "author": core.author,
        "subject": core.subject,
        "keywords": core.keywords,
        "category": core.category,
    }
    for key, val in meta_fields.items():
        if val:
            doc.metadata[key] = val

    if core.title:
        doc.title = core.title

    # ── Body content (paragraphs & tables in order) ─────────────────────
    text_parts: list[str] = []
    for element in _iter_body_elements(docx_doc.element.body):
        tag = element.tag if isinstance(element.tag, str) else ""
        if tag.endswith("}p"):
            para = element  # it's a CT_P already
            runs = [r.text for r in para.iter() if isinstance(r.text, str) and r.text]
            text = "".join(runs)
            if text.strip():
                text_parts.append(text)
        elif tag.endswith("}tbl"):
            # Build a lightweight table proxy
            from docx.table import Table as DocxTable
            table = DocxTable(element, docx_doc)
            table_text = _extract_table_text(table)
            if table_text.strip():
                text_parts.append(table_text)

    # ── Images ──────────────────────────────────────────────────────────
    images = _extract_images(docx_doc, doc.id)
    for img in images:
        doc.images.append(img.id)

    doc.content = "\n\n".join(text_parts)

    if not doc.content.strip():
        logger.info("DOCX %s yielded no extractable text", docx_path)
    else:
        logger.info(
            "Parsed DOCX %s — %d chars, %d image(s)",
            docx_path, len(doc.content), len(images),
        )

    return doc, images
