"""OCR parser for ExamBookGenerator.

Extracts text from images (PNG, JPG, JPEG) using Tesseract via pytesseract.
Supports blackboard photos, scans, and text-less PDF pages converted to images.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.models import Document, FileType
from utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg"}

pytesseract_available: bool = True
try:
    import pytesseract
except ImportError:
    pytesseract_available = False
    logger.warning(
        "pytesseract not installed — OCR unavailable. "
        "Install with: pip install pytesseract && apt install tesseract-ocr"
    )


def _is_supported(path: Path) -> bool:
    """Return True if *path* has a supported image extension."""
    return path.suffix.lower() in _SUPPORTED_EXTENSIONS


def _validate_image(path: Path) -> Image.Image:
    """Open and validate an image file.

    Returns
    -------
    Image.Image
        The opened Pillow image.

    Raises
    ------
    ValueError
        If the file is not a supported image or is corrupted.
    """
    try:
        img = Image.open(path)
        img.load()  # force decode to surface corruption early
        return img
    except Exception as exc:
        raise ValueError(f"Cannot open image {path}: {exc}") from exc


def _run_ocr(img: Image.Image, lang: str = "ita+eng") -> str:
    """Run Tesseract OCR on *img* and return the extracted text.

    Parameters
    ----------
    img:
        A Pillow Image to process.
    lang:
        Tesseract language string.  Defaults to ``ita+eng`` (Italian + English).
    """
    if not pytesseract_available:
        raise RuntimeError("pytesseract is not installed")

    text = pytesseract.image_to_string(img, lang=lang)
    return text.strip()


def parse_image(
    path: Path | str,
    *,
    lang: str = "ita+eng",
) -> Document:
    """Parse an image file and extract text via OCR.

    Parameters
    ----------
    path:
        Path to a ``.png``, ``.jpg``, or ``.jpeg`` file.
    lang:
        Tesseract language code.  Defaults to ``ita+eng``.

    Returns
    -------
    Document
        A ``Document`` with OCR-extracted text and image metadata.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If pytesseract is not installed.
    """
    img_path = Path(path)

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    if not _is_supported(img_path):
        logger.warning("Unsupported image format: %s — skipping", img_path.suffix)
        return Document(
            source_path=str(img_path),
            file_type=FileType.IMAGE,
            content="",
            metadata={"error": f"unsupported format: {img_path.suffix}"},
        )

    doc = Document(source_path=str(img_path), file_type=FileType.IMAGE)

    if not pytesseract_available:
        doc.metadata["error"] = "pytesseract not installed"
        return doc

    # ── Load image ──────────────────────────────────────────────────────
    try:
        img = _validate_image(img_path)
    except ValueError as exc:
        logger.warning("Corrupt image %s — %s", img_path, exc)
        doc.metadata["error"] = str(exc)
        return doc

    # ── Image dimensions ────────────────────────────────────────────────
    doc.metadata["width"] = img.width
    doc.metadata["height"] = img.height

    # ── OCR ─────────────────────────────────────────────────────────────
    try:
        text = _run_ocr(img, lang=lang)
    except Exception as exc:
        logger.warning("OCR failed for %s — %s", img_path, exc)
        doc.metadata["error"] = str(exc)
        return doc
    finally:
        img.close()

    doc.content = text
    doc.metadata["ocr_lang"] = lang
    doc.metadata["ocr_engine"] = "tesseract"

    if not doc.content:
        logger.info("OCR produced no text for %s", img_path)
    else:
        logger.info(
            "Parsed image %s — %d chars via OCR (%s)",
            img_path, len(doc.content), lang,
        )

    return doc
