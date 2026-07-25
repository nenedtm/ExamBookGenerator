"""AI result cache for ExamBookGenerator.

Caches LLM responses keyed by a SHA-256 hash of the prompt, so repeated
queries against thousands of pages do not re-invoke the model.

Uses the same SQLite database as :mod:`storage.database` but a separate
``cache`` table.

Schema
------
``cache`` table:

    key             TEXT PRIMARY KEY     -- hex SHA-256 of prompt
    prompt_hash     TEXT NOT NULL        -- same as key, kept explicit
    prompt_preview  TEXT NOT NULL        -- first 200 chars for debugging
    response        TEXT NOT NULL        -- the cached LLM response
    model           TEXT NOT NULL        -- model name used
    created_at      TEXT NOT NULL        -- ISO-8601 UTC
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "output"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "exambook.db"

_CACHE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS cache (
    key             TEXT PRIMARY KEY,
    prompt_hash     TEXT NOT NULL,
    prompt_preview  TEXT NOT NULL,
    response        TEXT NOT NULL,
    model           TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prompt_hash ON cache(prompt_hash);
"""


# ── Connection helpers ───────────────────────────────────────────────────────

def _get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Return a connection, ensuring the cache table exists.

    Shares the same database file as :mod:`storage.database`.
    """
    path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.executescript(_CACHE_SCHEMA)
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Hashing ──────────────────────────────────────────────────────────────────

def _hash_prompt(prompt: str, model: str = "") -> str:
    """Return a SHA-256 hex digest of *prompt* + *model*.

    Including the model ensures that switching models invalidates old
    cached responses for the same prompt text.
    """
    combined = f"{model}\n{prompt}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ── Public API ───────────────────────────────────────────────────────────────

def save_cache(
    prompt: str,
    response: str,
    model: str = "",
    *,
    db_path: Path | str | None = None,
) -> str:
    """Store an LLM response in the cache.

    If an entry with the same prompt hash already exists, it is overwritten.

    Parameters
    ----------
    prompt:
        The prompt sent to the LLM.
    response:
        The LLM's response text.
    model:
        Name / identifier of the model used.
    db_path:
        Override for the database file path.

    Returns
    -------
    str
        The cache key (SHA-256 hex digest).
    """
    key = _hash_prompt(prompt, model)
    preview = prompt[:200].replace("\n", " ")
    now = _now_iso()

    conn = _get_connection(db_path)
    try:
        conn.execute(
            """\
            INSERT INTO cache (key, prompt_hash, prompt_preview,
                               response, model, created_at)
            VALUES (:key, :key, :preview, :response, :model, :ts)
            ON CONFLICT(key) DO UPDATE SET
                response   = excluded.response,
                model      = excluded.model,
                created_at = excluded.created_at
            """,
            {"key": key, "preview": preview, "response": response,
             "model": model, "ts": now},
        )
        conn.commit()
        logger.debug("Cache saved — key=%s, model=%s", key[:12], model)
    finally:
        conn.close()

    return key


def get_cache(
    prompt: str,
    *,
    model: str = "",
    db_path: Path | str | None = None,
) -> dict[str, str] | None:
    """Retrieve a cached LLM response.

    Parameters
    ----------
    prompt:
        The original prompt to look up.
    model:
        The model name to include in the cache key.
    db_path:
        Override for the database file path.

    Returns
    -------
    dict or None
        ``{"response": ..., "model": ..., "created_at": ...}`` on hit,
        ``None`` on miss.
    """
    key = _hash_prompt(prompt, model)

    conn = _get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT response, model, created_at FROM cache WHERE key = ?",
            (key,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        logger.debug("Cache miss — key=%s", key[:12])
        return None

    logger.debug("Cache hit — key=%s, model=%s", key[:12], row["model"])
    return {
        "response": row["response"],
        "model": row["model"],
        "created_at": row["created_at"],
    }


def count_cache(*, db_path: Path | str | None = None) -> int:
    """Return the total number of cached entries."""
    conn = _get_connection(db_path)
    try:
        result = conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        return result[0]
    finally:
        conn.close()


def delete_cache(
    prompt: str,
    *,
    model: str = "",
    db_path: Path | str | None = None,
) -> bool:
    """Remove a single cache entry.

    Returns
    -------
    bool
        ``True`` if a row was deleted, ``False`` otherwise.
    """
    key = _hash_prompt(prompt, model)
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def clear_cache(*, db_path: Path | str | None = None) -> int:
    """Delete all cache entries.

    Returns
    -------
    int
        Number of rows deleted.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute("DELETE FROM cache")
        conn.commit()
        count = cursor.rowcount
        logger.info("Cache cleared — %d entries removed", count)
        return count
    finally:
        conn.close()


def clear_cache_for_model(
    model: str,
    *,
    db_path: Path | str | None = None,
) -> int:
    """Delete all cache entries for a specific model.

    Useful when switching models to avoid stale responses.

    Returns
    -------
    int
        Number of rows deleted.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute("DELETE FROM cache WHERE model = ?", (model,))
        conn.commit()
        count = cursor.rowcount
        logger.info("Cache cleared for model '%s' — %d entries removed", model, count)
        return count
    finally:
        conn.close()
