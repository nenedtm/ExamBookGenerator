"""PDF parser for ExamBookGenerator.

Extracts text and embedded images from normal PDF files using PyMuPDF.
Images are delegated to ``pipeline.image_extractor.save_extracted_image``
for validation, deduplication, and disk storage.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from core.models import Document, ExtractedImage, FileType
from pipeline.image_extractor import save_extracted_image
from utils.logger import get_logger

logger = get_logger(__name__)


def _rect_distance(a: fitz.Rect, b: fitz.Rect) -> float:
    """Euclidean distance between the centres of two rectangles."""
    ax, ay = (a.x0 + a.x1) / 2, (a.y0 + a.y1) / 2
    bx, by = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _nearby_caption(page: fitz.Page, img_rect: fitz.Rect) -> str | None:
    """Return the closest text block to *img_rect*, or *None*."""
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    best_text: str | None = None
    best_dist = float("inf")

    for block in blocks:
        if block.get("type") != 0:  # text block
            continue
        block_rect = fitz.Rect(block["bbox"])
        dist = _rect_distance(block_rect, img_rect)
        if dist < best_dist:
            best_dist = dist
            lines = [span["text"] for line in block.get("lines", [])
                     for span in line.get("spans", [])]
            text = " ".join(lines).strip()
            if text:
                best_text = text

    return best_text if best_text else None


def parse_pdf(path: Path | str) -> tuple[Document, list[ExtractedImage]]:
    """Parse a PDF file, extracting text and embedded images.

    Parameters
    ----------
    path:
        Path to a PDF file.

    Returns
    -------
    tuple[Document, list[ExtractedImage]]
        A ``Document`` with ``content`` and ``images`` populated, plus the
        list of ``ExtractedImage`` objects that were saved to disk.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = Document(source_path=str(pdf_path), file_type=FileType.PDF)
    images: list[ExtractedImage] = []
    text_parts: list[str] = []

    try:
        pdf = fitz.open(pdf_path)
    except Exception:
        logger.warning("Cannot open PDF %s — corrupted or unreadable", pdf_path)
        return doc, images

    if pdf.page_count == 0:
        logger.info("PDF %s is empty (0 pages)", pdf_path)
        pdf.close()
        return doc, images

    # ── Metadata ────────────────────────────────────────────────────────
    meta = pdf.metadata or {}
    for key in ("title", "author", "subject", "creator", "producer"):
        val = meta.get(key)
        if val:
            doc.metadata[key] = val
    doc.metadata["page_count"] = pdf.page_count

    if meta.get("title"):
        doc.title = meta["title"]

    # ── Page iteration ──────────────────────────────────────────────────
    for page_idx in range(pdf.page_count):
        page = pdf.load_page(page_idx)

        # Text
        page_text = page.get_text("text")
        if page_text.strip():
            text_parts.append(page_text)

        # Images
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            try:
                img_data = pdf.extract_image(xref)
            except Exception:
                logger.warning(
                    "Cannot extract image xref=%d from %s page %d — skipped",
                    xref, pdf_path, page_idx + 1,
                )
                continue

            raw_bytes = img_data.get("image")
            if not raw_bytes:
                continue

            # Try to find a nearby caption
            img_rects = page.get_image_rects(xref)
            caption = None
            if img_rects:
                caption = _nearby_caption(page, img_rects[0])

            extracted = save_extracted_image(
                raw_bytes=raw_bytes,
                source_document_id=doc.id,
                page_or_slide=page_idx + 1,
                caption=caption,
            )
            if extracted is not None:
                images.append(extracted)
                doc.images.append(extracted.id)

    pdf.close()

    doc.content = "\n\n".join(text_parts)

    if not doc.content.strip():
        logger.info("PDF %s yielded no extractable text", pdf_path)
    else:
        logger.info(
            "Parsed PDF %s — %d chars, %d image(s)",
            pdf_path, len(doc.content), len(images),
        )

    return doc, images
