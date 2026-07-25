"""Filesystem scanner for ExamBookGenerator.

Recursively walks a directory, detects supported file types, and produces
an inventory of ``Document`` objects ready for the parsing stage.
"""

from __future__ import annotations

import hashlib
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

_SYLLABUS_KEYWORDS: list[str] = [
    "syllabus",
    "programma",
    "program",
    "course outline",
    "piano di studi",
]


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


def detect_syllabus_candidate(
    document: Document,
    *,
    explicit_path: str | None = None,
) -> bool:
    """Check whether *document* looks like a course syllabus.

    Heuristic (case-insensitive):

    - Filename stem contains one of the known keywords, **or**
    - *explicit_path* is set and matches the document's source path.

    Parameters
    ----------
    document:
        The document to evaluate.
    explicit_path:
        An explicit syllabus path from configuration.  When provided and it
        matches ``document.source_path``, the function always returns *True*.

    Returns
    -------
    bool
    """
    if explicit_path and Path(explicit_path) == Path(document.source_path):
        return True

    stem = Path(document.source_path).stem.lower()
    for keyword in _SYLLABUS_KEYWORDS:
        if keyword in stem:
            return True

    return False


def scan_directory(
    path: Path | str,
    *,
    syllabus_enabled: bool = False,
    syllabus_path: str | None = None,
) -> list[Document]:
    """Recursively scan *path* and return ``Document`` stubs for every
    supported file found.

    Parameters
    ----------
    path:
        Root directory to scan.
    syllabus_enabled:
        When *True*, each document is tested for syllabus candidacy via
        ``detect_syllabus_candidate`` and ``Document.is_syllabus`` is set
        accordingly.
    syllabus_path:
        Explicit syllabus file path from configuration.  Passed through to
        ``detect_syllabus_candidate``.

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
    syllabus_found = False
    syllabus_doc_id: str | None = None

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

            if syllabus_enabled:
                is_syll = detect_syllabus_candidate(
                    doc,
                    explicit_path=syllabus_path,
                )
                if is_syll:
                    doc.is_syllabus = True
                    if not syllabus_found:
                        syllabus_found = True
                        syllabus_doc_id = doc.id
                    logger.info("Syllabus candidate detected: %s", entry.name)

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

    if syllabus_enabled:
        logger.info("Syllabus detected: %s", syllabus_found)
        if syllabus_found:
            logger.info("Syllabus document id: %s", syllabus_doc_id)

    return documents


def compute_file_hash(file_path: Path) -> str:
    """Compute a SHA-256 hash of the file content.

    Parameters
    ----------
    file_path:
        Path to the file.

    Returns
    -------
    str
        Hex digest of the file content.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_directory_incremental(
    path: Path | str,
    *,
    syllabus_enabled: bool = False,
    syllabus_path: str | None = None,
    force: bool = False,
) -> tuple[list[Document], set[str], set[str], set[str]]:
    """Scan directory and classify files by processing state.

    Parameters
    ----------
    path:
        Root directory to scan.
    syllabus_enabled:
        When *True*, detect syllabus candidates.
    syllabus_path:
        Explicit syllabus file path.
    force:
        When *True*, treat all files as new (ignore cache).

    Returns
    -------
    tuple
        ``(all_documents, new_hashes, modified_hashes, unchanged_hashes)``
        where each hash set contains the content_hash of files in that category.
    """
    from storage.database import get_all_processed_hashes

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    logger.info("Scanning directory (incremental): %s", root.resolve())

    previously_processed = {} if force else get_all_processed_hashes()

    documents: list[Document] = []
    new_hashes: set[str] = set()
    modified_hashes: set[str] = set()
    unchanged_hashes: set[str] = set()
    skipped = 0
    syllabus_found = False
    syllabus_doc_id: str | None = None

    for entry in sorted(root.rglob("*")):
        if not entry.is_file():
            continue

        file_type = detect_file_type(entry)
        if file_type == FileType.UNKNOWN:
            skipped += 1
            continue

        try:
            content_hash = compute_file_hash(entry)
            doc = Document(
                source_path=str(entry),
                file_type=file_type,
            )

            if syllabus_enabled:
                is_syll = detect_syllabus_candidate(
                    doc,
                    explicit_path=syllabus_path,
                )
                if is_syll:
                    doc.is_syllabus = True
                    if not syllabus_found:
                        syllabus_found = True
                        syllabus_doc_id = doc.id
                    logger.info("Syllabus candidate detected: %s", entry.name)

            documents.append(doc)

            prev_hash = previously_processed.get(str(entry))
            if prev_hash is None:
                new_hashes.add(content_hash)
                logger.debug("NEW: %s (%s)", entry.name, file_type.value)
            elif prev_hash == content_hash:
                unchanged_hashes.add(content_hash)
                logger.debug("UNCHANGED: %s (%s)", entry.name, file_type.value)
            else:
                modified_hashes.add(content_hash)
                logger.debug("MODIFIED: %s (%s)", entry.name, file_type.value)

        except OSError as exc:
            logger.warning("Cannot read %s: %s", entry, exc)
            skipped += 1

    logger.info(
        "Incremental scan complete — %d files: %d new, %d modified, %d unchanged, %d skipped",
        len(documents),
        len(new_hashes),
        len(modified_hashes),
        len(unchanged_hashes),
        skipped,
    )

    if syllabus_enabled:
        logger.info("Syllabus detected: %s", syllabus_found)
        if syllabus_found:
            logger.info("Syllabus document id: %s", syllabus_doc_id)

    return documents, new_hashes, modified_hashes, unchanged_hashes


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

    syllabus_docs = [doc for doc in documents if doc.is_syllabus]

    out.write_text(
        json.dumps(
            {
                "syllabus_detected": len(syllabus_docs) > 0,
                "syllabus_document_id": syllabus_docs[0].id if syllabus_docs else None,
                "documents": inventory,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Inventory written to %s (%d entries)", out, len(inventory))
    return out.resolve()
