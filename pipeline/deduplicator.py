"""Document deduplicator for ExamBookGenerator.

Two-phase deduplication strategy:

1. **Exact** — SHA-256 hash of normalised content.  Documents with
   identical hashes are collapsed, keeping the one with the longest
   ``source_path`` (typically the most descriptive filename).
2. **Near-duplicate** — ``difflib.SequenceMatcher`` ratio on content.
   Documents whose similarity exceeds ``similarity_threshold`` are merged,
   keeping the longest (most complete) document.

Both phases run sequentially: exact first (O(n)), then near-duplicate
(O(n²)) on the reduced set.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from difflib import SequenceMatcher

from core.models import Document
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_SIMILARITY: float = 0.92


# ── Helpers ──────────────────────────────────────────────────────────────────

def _content_hash(doc: Document) -> str:
    """Return a SHA-256 hex digest of *doc*'s content."""
    return hashlib.sha256(doc.content.encode("utf-8")).hexdigest()


def _similarity(a: Document, b: Document) -> float:
    """Return a 0-1 similarity ratio between two documents' content."""
    if not a.content and not b.content:
        return 1.0
    if not a.content or not b.content:
        return 0.0
    return SequenceMatcher(None, a.content, b.content).ratio()


def _pick_best(group: list[Document]) -> Document:
    """Return the "best" document from a group of duplicates.

    Heuristic: longest content wins.  Ties broken by longest source_path
    (more descriptive filename).
    """
    return max(
        group,
        key=lambda d: (len(d.content), len(d.source_path)),
    )


# ── Phase 1 — exact dedup ───────────────────────────────────────────────────

def _exact_dedup(documents: list[Document]) -> list[Document]:
    """Collapse documents with identical content hash."""
    buckets: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        buckets[_content_hash(doc)].append(doc)

    unique: list[Document] = []
    removed = 0
    for group in buckets.values():
        if len(group) == 1:
            unique.append(group[0])
        else:
            best = _pick_best(group)
            unique.append(best)
            removed += len(group) - 1

    if removed:
        logger.info("Exact dedup removed %d document(s)", removed)
    return unique


# ── Phase 2 — near-duplicate dedup ──────────────────────────────────────────

def _near_dedup(
    documents: list[Document],
    threshold: float,
) -> list[Document]:
    """Merge near-duplicate documents using pairwise similarity.

    Uses a Union-Find approach: each document starts in its own set.
    When two documents exceed *threshold* similarity, they are merged.
    The best document from each final set is kept.
    """
    n = len(documents)
    if n <= 1:
        return list(documents)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i + 1, n):
            if _similarity(documents[i], documents[j]) >= threshold:
                union(i, j)

    groups: dict[int, list[Document]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(documents[i])

    unique: list[Document] = []
    removed = 0
    for group in groups.values():
        best = _pick_best(group)
        unique.append(best)
        removed += len(group) - 1

    if removed:
        logger.info("Near-dedup removed %d document(s)", removed)
    return unique


# ── Public API ───────────────────────────────────────────────────────────────

def deduplicate(
    documents: list[Document],
    *,
    similarity_threshold: float = _DEFAULT_SIMILARITY,
) -> list[Document]:
    """Remove duplicate and near-duplicate documents.

    Parameters
    ----------
    documents:
        Input list of ``Document`` objects.
    similarity_threshold:
        Content similarity ratio (0-1) above which two documents are
        considered near-duplicates.  Defaults to ``0.92``.

    Returns
    -------
    list[Document]
        De-duplicated list.  Original order is not preserved.

    Example
    -------
    >>> from core.models import Document
    >>> docs = [
    ...     Document(source_path="a.pdf", content="Same text"),
    ...     Document(source_path="b.pdf", content="Same text"),
    ...     Document(source_path="c.pdf", content="Different text here"),
    ... ]
    >>> deduplicate(docs)
    [Document(..., source_path='b.pdf', ...), Document(..., source_path='c.pdf', ...)]
    """
    if not documents:
        return []

    logger.info("Deduplicating %d document(s) …", len(documents))

    # Phase 1 — exact
    unique = _exact_dedup(documents)

    # Phase 2 — near-duplicate
    unique = _near_dedup(unique, similarity_threshold)

    logger.info(
        "Deduplication complete — %d → %d document(s)",
        len(documents),
        len(unique),
    )
    return unique
