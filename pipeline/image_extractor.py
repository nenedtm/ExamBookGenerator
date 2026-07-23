"""Centralised image extraction and deduplication for ExamBookGenerator.

Parsers (PDF, DOCX, PPTX) pass raw image bytes here instead of
implementing their own save/dedup/filter logic.

Typical parser usage::

    from pipeline.image_extractor import save_extracted_image

    img = save_extracted_image(
        raw_bytes=image_bytes,
        source_document_id=doc.id,
        page_or_slide=page_num,
        caption="Figure 3.2",
    )
    if img is not None:
        doc.images.append(img.id)
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

from core.models import ExtractedImage
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_ASSETS_DIR = Path(__file__).resolve().parent.parent / "output" / "assets" / "images"
_DEFAULT_MIN_WIDTH = 100
_DEFAULT_MIN_HEIGHT = 100


def _hash_bytes(data: bytes) -> str:
    """Return a short, deterministic hex digest of *data*."""
    return hashlib.sha256(data).hexdigest()[:16]


def save_extracted_image(
    raw_bytes: bytes,
    source_document_id: str,
    page_or_slide: int | None = None,
    caption: str | None = None,
    assets_dir: Path | str | None = None,
    min_width: int = _DEFAULT_MIN_WIDTH,
    min_height: int = _DEFAULT_MIN_HEIGHT,
) -> ExtractedImage | None:
    """Validate, deduplicate, and save an extracted image.

    Parameters
    ----------
    raw_bytes:
        Raw image data (PNG, JPEG, etc.).
    source_document_id:
        ``Document.id`` the image was extracted from.
    page_or_slide:
        Page/slide number where the image was found.
    caption:
        Original caption, if present in the source document.
    assets_dir:
        Override for the output directory.  Defaults to
        ``output/assets/images/``.
    min_width:
        Minimum width in pixels; images narrower than this are discarded.
    min_height:
        Minimum height in pixels; images shorter than this are discarded.

    Returns
    -------
    ExtractedImage or None
        A populated ``ExtractedImage`` on success, or *None* when the image
        is corrupt, too small, or otherwise invalid.
    """
    if not source_document_id:
        raise ValueError("source_document_id is required")

    out_dir = Path(assets_dir) if assets_dir is not None else _DEFAULT_ASSETS_DIR

    # ── Validate & measure ──────────────────────────────────────────────
    try:
        pil_img = Image.open(BytesIO(raw_bytes))
        pil_img.load()  # force full decode to catch truncated data
    except Exception:
        logger.warning(
            "Corrupt or unreadable image from doc %s (page/slide %s) — skipped",
            source_document_id,
            page_or_slide,
        )
        return None

    width, height = pil_img.size

    if width < min_width or height < min_height:
        logger.debug(
            "Image %dx%d from doc %s is below minimum %dx%d — skipped",
            width, height, source_document_id, min_width, min_height,
        )
        return None

    # ── Hash & save ─────────────────────────────────────────────────────
    img_hash = _hash_bytes(raw_bytes)
    suffix = pil_img.format.lower() if pil_img.format else "png"
    filename = f"{img_hash}.{suffix}"
    file_path = out_dir / filename

    out_dir.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        file_path.write_bytes(raw_bytes)
        logger.debug("Saved image %s (%dx%d)", filename, width, height)
    else:
        logger.debug("Image %s already on disk — reusing", filename)

    return ExtractedImage(
        source_document_id=source_document_id,
        file_path=str(file_path),
        page_or_slide=page_or_slide,
        caption=caption,
        width=width,
        height=height,
    )


def deduplicate_images(images: list[ExtractedImage]) -> list[ExtractedImage]:
    """Remove duplicate images keeping only the first occurrence.

    Two images are considered duplicates when their ``file_path`` is
    identical (same hash → same file on disk).

    Parameters
    ----------
    images:
        List of ``ExtractedImage`` objects, possibly containing duplicates.

    Returns
    -------
    list[ExtractedImage]
        De-duplicated list preserving original insertion order.
    """
    seen: set[str] = set()
    unique: list[ExtractedImage] = []

    for img in images:
        if img.file_path in seen:
            logger.debug("Dropping duplicate image %s (id=%s)", img.file_path, img.id)
            continue
        seen.add(img.file_path)
        unique.append(img)

    removed = len(images) - len(unique)
    if removed:
        logger.info("Deduplication: %d duplicate image(s) removed", removed)

    return unique
