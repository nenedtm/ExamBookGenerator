"""SQLite persistence for ExamBookGenerator documents.

Provides ``save_document`` / ``load_document`` backed by a single-file
SQLite database.  The database and its schema are created automatically on
first use.

Schema
------
``documents`` table:

    id            TEXT PRIMARY KEY
    title         TEXT NOT NULL
    source_path   TEXT UNIQUE NOT NULL
    file_type     TEXT NOT NULL
    content       TEXT NOT NULL
    metadata      TEXT NOT NULL          -- JSON-encoded dict
    images        TEXT NOT NULL          -- JSON-encoded list
    created_at    TEXT NOT NULL          -- ISO-8601 UTC
    updated_at    TEXT NOT NULL          -- ISO-8601 UTC
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from core.models import Document, FileType
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "output"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "exambook.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    source_path   TEXT UNIQUE NOT NULL,
    file_type     TEXT NOT NULL,
    content       TEXT NOT NULL,
    metadata      TEXT NOT NULL,
    images        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_path ON documents(source_path);

CREATE TABLE IF NOT EXISTS processing_state (
    content_hash  TEXT PRIMARY KEY,
    source_path   TEXT NOT NULL,
    file_type     TEXT NOT NULL,
    parsed        INTEGER DEFAULT 0,
    chunks_json   TEXT,
    topics_json   TEXT,
    model         TEXT,
    depth         INTEGER,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ps_source_path ON processing_state(source_path);
"""


# ── Connection helpers ───────────────────────────────────────────────────────

def _get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Return a connection to the database, creating it if needed.

    Parameters
    ----------
    db_path:
        Override for the database file.  When *None* the default
        ``output/exambook.db`` is used.
    """
    path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ── Serialisation ────────────────────────────────────────────────────────────

def _document_to_row(doc: Document, now: str) -> dict[str, str]:
    """Convert a ``Document`` to a dict suitable for INSERT/UPDATE."""
    return {
        "id": doc.id,
        "title": doc.title,
        "source_path": doc.source_path,
        "file_type": doc.file_type.value,
        "content": doc.content,
        "metadata": json.dumps(doc.metadata, ensure_ascii=False),
        "images": json.dumps(doc.images, ensure_ascii=False),
        "updated_at": now,
    }


def _row_to_document(row: sqlite3.Row) -> Document:
    """Reconstruct a ``Document`` from a database row."""
    return Document(
        id=row["id"],
        title=row["title"],
        source_path=row["source_path"],
        file_type=FileType(row["file_type"]),
        content=row["content"],
        metadata=json.loads(row["metadata"]),
        images=json.loads(row["images"]),
    )


# ── Public API ───────────────────────────────────────────────────────────────

def save_document(doc: Document, *, db_path: Path | str | None = None) -> Document:
    """Persist a ``Document`` to the local SQLite database.

    If a document with the same ``source_path`` already exists, it is
    updated in place (UPSERT).  A fresh ``created_at`` timestamp is set on
    insert only.

    Parameters
    ----------
    doc:
        The ``Document`` to save.
    db_path:
        Override for the database file path.

    Returns
    -------
    Document
        The same *doc* instance (for chaining).

    Raises
    ------
    sqlite3.Error
        On database write failures.
    """
    now = _now_iso()
    row = _document_to_row(doc, now)

    conn = _get_connection(db_path)
    try:
        conn.execute(
            """\
            INSERT INTO documents (id, title, source_path, file_type,
                                   content, metadata, images,
                                   created_at, updated_at)
            VALUES (:id, :title, :source_path, :file_type,
                    :content, :metadata, :images,
                    :ts, :updated_at)
            ON CONFLICT(source_path) DO UPDATE SET
                id         = excluded.id,
                title      = excluded.title,
                file_type  = excluded.file_type,
                content    = excluded.content,
                metadata   = excluded.metadata,
                images     = excluded.images,
                updated_at = excluded.updated_at
            """,
            {**row, "ts": now},
        )
        conn.commit()
        logger.debug("Saved document '%s' (%s)", doc.title, doc.source_path)
    finally:
        conn.close()

    return doc


def load_document(
    source_path: str,
    *,
    db_path: Path | str | None = None,
) -> Document | None:
    """Load a ``Document`` by its original source path.

    Parameters
    ----------
    source_path:
        The ``source_path`` used when the document was saved.
    db_path:
        Override for the database file path.

    Returns
    -------
    Document or None
        The matching document, or ``None`` if not found.
    """
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE source_path = ?",
            (source_path,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        logger.debug("Document not found: %s", source_path)
        return None

    doc = _row_to_document(row)
    logger.debug("Loaded document '%s'", doc.title)
    return doc


def load_document_by_id(
    doc_id: str,
    *,
    db_path: Path | str | None = None,
) -> Document | None:
    """Load a ``Document`` by its unique id.

    Parameters
    ----------
    doc_id:
        The document id.
    db_path:
        Override for the database file path.

    Returns
    -------
    Document or None
    """
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        logger.debug("Document not found by id: %s", doc_id)
        return None

    return _row_to_document(row)


def count_documents(*, db_path: Path | str | None = None) -> int:
    """Return the total number of stored documents."""
    conn = _get_connection(db_path)
    try:
        result = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return result[0]
    finally:
        conn.close()


def delete_document(
    source_path: str,
    *,
    db_path: Path | str | None = None,
) -> bool:
    """Delete a document by source path.

    Returns
    -------
    bool
        ``True`` if a row was deleted, ``False`` otherwise.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            "DELETE FROM documents WHERE source_path = ?",
            (source_path,),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("Deleted document: %s", source_path)
        return deleted
    finally:
        conn.close()


# ── Processing State API ─────────────────────────────────────────────────────

def save_processing_state(
    content_hash: str,
    source_path: str,
    file_type: str,
    *,
    parsed: bool = True,
    chunks_json: str | None = None,
    topics_json: str | None = None,
    model: str | None = None,
    depth: int | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Persist the processing state for a file.

    Parameters
    ----------
    content_hash:
        SHA-256 of the parsed file content.
    source_path:
        Original file path.
    file_type:
        File type (pdf, docx, etc.).
    parsed:
        Whether the file has been successfully parsed.
    chunks_json:
        JSON-serialized list of chunks.
    topics_json:
        JSON-serialized list of topic names.
    model:
        Model used for LLM calls on this file.
    depth:
        Depth level used for generation.
    db_path:
        Override for the database file path.
    """
    now = _now_iso()
    conn = _get_connection(db_path)
    try:
        conn.execute(
            """\
            INSERT INTO processing_state
                (content_hash, source_path, file_type, parsed,
                 chunks_json, topics_json, model, depth,
                 created_at, updated_at)
            VALUES
                (:hash, :path, :ftype, :parsed,
                 :chunks, :topics, :model, :depth,
                 :ts, :ts)
            ON CONFLICT(content_hash) DO UPDATE SET
                source_path = excluded.source_path,
                file_type   = excluded.file_type,
                parsed      = excluded.parsed,
                chunks_json = excluded.chunks_json,
                topics_json = excluded.topics_json,
                model       = excluded.model,
                depth       = excluded.depth,
                updated_at  = excluded.updated_at
            """,
            {
                "hash": content_hash,
                "path": source_path,
                "ftype": file_type,
                "parsed": int(parsed),
                "chunks": chunks_json,
                "topics": topics_json,
                "model": model,
                "depth": depth,
                "ts": now,
            },
        )
        conn.commit()
        logger.debug("Saved processing state — hash=%s, path=%s", content_hash[:12], source_path)
    finally:
        conn.close()


def get_processing_state(
    content_hash: str,
    *,
    db_path: Path | str | None = None,
) -> dict | None:
    """Retrieve processing state by content hash.

    Returns
    -------
    dict or None
        Processing state dict, or ``None`` if not found.
    """
    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM processing_state WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    return {
        "content_hash": row["content_hash"],
        "source_path": row["source_path"],
        "file_type": row["file_type"],
        "parsed": bool(row["parsed"]),
        "chunks_json": row["chunks_json"],
        "topics_json": row["topics_json"],
        "model": row["model"],
        "depth": row["depth"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_all_processed_hashes(*, db_path: Path | str | None = None) -> dict[str, str]:
    """Return a mapping of source_path → content_hash for all processed files.

    Returns
    -------
    dict
        ``{source_path: content_hash, ...}``
    """
    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT source_path, content_hash FROM processing_state"
        ).fetchall()
    finally:
        conn.close()
    return {row["source_path"]: row["content_hash"] for row in rows}


def invalidate_processing_state(
    *,
    source_path: str | None = None,
    content_hash: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Remove processing state entries.

    Parameters
    ----------
    source_path:
        If provided, invalidate all entries for this file path.
    content_hash:
        If provided, invalidate the specific hash entry.
    db_path:
        Override for the database file path.

    Returns
    -------
    int
        Number of rows deleted.
    """
    conn = _get_connection(db_path)
    try:
        if source_path:
            cursor = conn.execute(
                "DELETE FROM processing_state WHERE source_path = ?",
                (source_path,),
            )
        elif content_hash:
            cursor = conn.execute(
                "DELETE FROM processing_state WHERE content_hash = ?",
                (content_hash,),
            )
        else:
            cursor = conn.execute("DELETE FROM processing_state")
        conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.debug("Invalidated %d processing state entries", count)
        return count
    finally:
        conn.close()
