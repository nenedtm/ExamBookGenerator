"""Filesystem scanner for ExamBookGenerator.

Recursively walks a directory, detects supported file types, and produces
an inventory of ``Document`` objects ready for the parsing stage.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.models import Document, FileType
from utils.logger import get_logger

logger = get_logger(__name__)

_EXTENSION_MAP: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".pptx": FileType.PPTX,
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
    ".svg": FileType.IMAGE,
}

_INVENTORY_DIR = Path(__file__).resolve().parent.parent / "storage"
_INVENTORY_FILE = _INVENTORY_DIR / "inventory.json"


def detect_file_type(path: Path) -> FileType:
    """Determine the ``FileType`` for a given filesystem path.

    Parameters
    ----------
    path:
        Absolute or relative path to a file.

    Returns
    -------
    FileType
        The matching type, or ``FileType.UNKNOWN`` for unsupported extensions.
    """
    suffix = path.suffix.lower()
    file_type = _EXTENSION_MAP.get(suffix, FileType.UNKNOWN)
    if file_type == FileType.UNKNOWN:
        logger.debug("Unsupported extension '%s' for %s", suffix, path)
    return file_type


def scan_directory(path: Path | str) -> list[Document]:
    """Recursively scan *path* and return ``Document`` stubs for every
    supported file found.

    Parameters
    ----------
    path:
        Root directory to scan.

    Returns
    -------
    list[Document]
        One ``Document`` per discoverable file, with ``source_path`` and
        ``file_type`` populated.  ``content`` is left empty — it will be
        filled by the parsing stage.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist or is not a directory.
    """
    root = Path(path)

    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    logger.info("Scanning directory: %s", root.resolve())

    documents: list[Document] = []
    skipped = 0

    for entry in sorted(root.rglob("*")):
        if not entry.is_file():
            continue

        file_type = detect_file_type(entry)
        if file_type == FileType.UNKNOWN:
            skipped += 1
            continue

        try:
            doc = Document(
                source_path=str(entry),
                file_type=file_type,
            )
            documents.append(doc)
            logger.debug("Found: %s (%s)", entry.name, file_type.value)
        except OSError as exc:
            logger.warning("Cannot read %s: %s", entry, exc)
            skipped += 1

    logger.info(
        "Scan complete — %d supported file(s), %d skipped",
        len(documents),
        skipped,
    )
    return documents


def generate_inventory(
    documents: list[Document],
    output_path: Path | str | None = None,
) -> Path:
    """Serialise the scan results to a JSON inventory file.

    Parameters
    ----------
    documents:
        List of ``Document`` objects produced by ``scan_directory``.
    output_path:
        Override for the output file.  Defaults to
        ``storage/inventory.json``.

    Returns
    -------
    Path
        Absolute path to the written inventory file.
    """
    out = Path(output_path) if output_path is not None else _INVENTORY_FILE
    out.parent.mkdir(parents=True, exist_ok=True)

    inventory = [
        {
            "id": doc.id,
            "title": doc.title,
            "source_path": doc.source_path,
            "file_type": doc.file_type.value,
        }
        for doc in documents
    ]

    out.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Inventory written to %s (%d entries)", out, len(inventory))
    return out.resolve()
