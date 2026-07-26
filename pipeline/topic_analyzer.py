"""Topic analysis using a local LLM via Ollama.

Takes a list of ``Chunk`` objects, asks the LLM to identify distinct
topics, and returns a list of ``Topic`` objects with ordering metadata
(syllabus-based or pedagogical).

Supports two modes configured through ``generation.scope``:

- **full**: analyse all chunks, produce multiple topics.
- **topic**: focus on a single user-specified topic, produce exactly one
  ``Topic`` with ``order_source="manual"``.

Usage::

    from pipeline.topic_analyzer import TopicAnalyzer, TopicNotFoundError
    from llm.ollama_client import OllamaClient

    analyzer = TopicAnalyzer(OllamaClient.from_config())
    topics, chunk_id_map = analyzer.analyze(chunks, scope="full")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from core.models import Chunk, Document, Topic
from llm.ollama_client import OllamaClient, OllamaError
from llm.prompt_manager import build_topic_prompt, build_syllabus_extraction_prompt
from utils import parse_llm_json
from utils.config import ConfigManager
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Exceptions ───────────────────────────────────────────────────────────────


class TopicAnalyzerError(OllamaError):
    """Base exception for topic analysis errors."""


class TopicNotFoundError(TopicAnalyzerError):
    """Raised in ``scope="topic"`` mode when no topic matches the focus."""


# ── Parser ───────────────────────────────────────────────────────────────────


def _parse_topics_response(raw: str) -> list[dict[str, object]]:
    """Parse the JSON response from the LLM into a list of topic dicts.

    Handles common LLM quirks (markdown fences, leading/trailing text)
    by attempting to extract the first JSON object from *raw*.
    """
    try:
        obj = parse_llm_json(raw, label="Topic analyzer")
    except ValueError as exc:
        raise TopicAnalyzerError(str(exc)) from exc

    topics = obj.get("topics")
    if not isinstance(topics, list):
        raise TopicAnalyzerError(
            f"LLM JSON is missing the 'topics' array.  Keys present: "
            f"{list(obj.keys()) if isinstance(obj, dict) else type(obj)}"
        )

    return topics  # type: ignore[no-any-return]


def _build_topic(raw: dict[str, object], order: int) -> Topic:
    """Construct a ``Topic`` from a single parsed LLM dict."""
    name = str(raw.get("name", f"Topic {order + 1}")).strip()
    # Filter out clearly invalid / garbage topic names
    if _is_invalid_topic_name(name):
        name = f"Topic {order + 1}"
    return Topic(
        name=name,
        description=str(raw.get("description", "")),
        related_documents=[str(d) for d in raw.get("related_documents", [])],
        subtopic_count=int(raw.get("subtopic_count", 0)),
        order_source=raw.get("order_source", "pedagogical"),  # type: ignore[arg-type]
        syllabus_position=raw.get("syllabus_position"),  # type: ignore[assignment]
    )


def _is_invalid_topic_name(name: str) -> bool:
    """Return ``True`` when *name* is clearly a placeholder or garbage.

    Examples of invalid names: ``topic1``, ``topic 2``, ``T1``, ``Untitled``.
    """
    import re as _re
    cleaned = name.lower().strip()
    # Matches patterns like "topic1", "topic 1", "Topic-1", "T1", etc.
    if _re.match(r"^(topic|t)\s*\d+$", cleaned):
        return True
    if cleaned in ("untitled", "no title", "n/a", "none", ""):
        return True
    # Single character or purely numeric
    if len(cleaned) <= 1 or cleaned.isdigit():
        return True
    return False


def _deduplicate_topics(topics: list[Topic]) -> list[Topic]:
    """Remove duplicate topic names, keeping the first occurrence."""
    seen: set[str] = set()
    result: list[Topic] = []
    for t in topics:
        key = t.name.lower().strip()
        if key in seen:
            logger.warning("Duplicate topic '%s' removed", t.name)
            continue
        seen.add(key)
        result.append(t)
    return result


# ── Chunk relevance helpers ──────────────────────────────────────────────────


def _chunk_ids_from_response(raw_topics: list[dict[str, object]]) -> dict[str, list[str]]:
    """Extract chunk_id mappings from the LLM response, if present.

    Returns a dict mapping topic name → list of chunk IDs.
    Only includes topics that actually have chunk IDs.
    """
    mapping: dict[str, list[str]] = {}
    for t in raw_topics:
        name = str(t.get("name", ""))
        chunk_ids = t.get("chunk_ids", [])
        if name and isinstance(chunk_ids, list) and chunk_ids:
            mapping[name] = [str(c) for c in chunk_ids]
    return mapping


def _chunks_for_topic(
    focus: str,
    chunks: list[Chunk],
    chunk_id_map: dict[str, list[str]] | None = None,
    topic_name: str | None = None,
) -> list[Chunk]:
    """Return chunks relevant to *focus*.

    If *chunk_id_map* contains an entry for *topic_name*, those IDs are
    used directly.  Otherwise falls back to keyword matching.
    """
    # Try ID-based filtering first
    if chunk_id_map and topic_name and topic_name in chunk_id_map:
        ids = set(chunk_id_map[topic_name])
        matched = [c for c in chunks if c.id in ids]
        if matched:
            return matched

    # Fallback: keyword matching on chunk content
    focus_lower = focus.lower()
    keywords = [w for w in focus_lower.split() if len(w) > 3]
    if not keywords:
        keywords = [focus_lower]

    matched: list[Chunk] = []
    for c in chunks:
        content_lower = c.content.lower()
        if any(kw in content_lower for kw in keywords):
            matched.append(c)

    return matched


# ── Syllabus parsing and coverage ────────────────────────────────────────────


def _parse_syllabus_entries_regex(text: str) -> list[str]:
    """Fast regex-based extraction of topic entries from a syllabus.

    Works well for structured lists (numbered, bulleted).  Returns an
    empty list when the text looks like free-form prose (no line-level
    structure detected).
    """
    entries: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(
            r"^[\-•*]\s+|^\d+[\.\)]\s+|^[ivxIVX]+[\.\)]\s+",
            "",
            line,
        ).strip()
        if cleaned and len(cleaned) > 2:
            entries.append(cleaned)
    return entries


def _find_syllabus_entry_for_topic(
    topic_name: str,
    syllabus_entries: list[str],
) -> int | None:
    """Return the index of the syllabus entry that best matches *topic_name*.

    Returns *None* if no match is found.
    """
    name_lower = topic_name.lower().strip()
    name_words = set(name_lower.split())

    best_idx: int | None = None
    best_score = 0.0

    for idx, entry in enumerate(syllabus_entries):
        entry_lower = entry.lower().strip()
        # Exact match
        if name_lower == entry_lower:
            return idx
        # Substring containment
        if name_lower in entry_lower or entry_lower in name_lower:
            score = min(len(name_lower), len(entry_lower)) / max(
                len(name_lower), len(entry_lower), 1,
            )
            if score > best_score:
                best_idx = idx
                best_score = score
                continue
        # Keyword overlap
        entry_words = set(entry_lower.split())
        overlap = len(name_words & entry_words)
        if overlap > 0:
            score = overlap / max(len(name_words), len(entry_words))
            if score > best_score:
                best_idx = idx
                best_score = score

    return best_idx if best_score >= 0.3 else None


def _ensure_syllabus_coverage(
    topics: list[Topic],
    syllabus_entries: list[str],
    all_chunks: list[Chunk],
) -> list[Topic]:
    """Filter and enrich topics to match the syllabus exactly.

    1. Remove topics that don't match any syllabus entry.
    2. Add fallback topics for syllabus entries not covered by LLM topics.
    3. Reorder to follow syllabus order.

    Parameters
    ----------
    topics:
        Topics already identified by the LLM.
    syllabus_entries:
        Ordered list of topic names extracted from the syllabus
        (via LLM or regex fallback).
    all_chunks:
        All source material chunks, used for keyword matching.
    """
    if not syllabus_entries:
        return topics

    # ── Step 1: Filter out topics not in syllabus ──────────────────────
    matched_topics: list[tuple[Topic, int]] = []  # (topic, syllabus_idx)
    unmatched: list[Topic] = []

    for t in topics:
        idx = _find_syllabus_entry_for_topic(t.name, syllabus_entries)
        if idx is not None:
            # Update syllabus position if not set
            if t.syllabus_position is None:
                t.syllabus_position = idx
            t.order_source = "syllabus"
            matched_topics.append((t, idx))
        else:
            unmatched.append(t)

    if unmatched:
        logger.info(
            "Filtered %d topic(s) not in syllabus: %s",
            len(unmatched),
            [t.name for t in unmatched],
        )

    # ── Step 2: Find missing syllabus entries ─────────────────────────
    covered_indices = {idx for _, idx in matched_topics}
    missing = [
        (idx, entry)
        for idx, entry in enumerate(syllabus_entries)
        if idx not in covered_indices
    ]

    # ── Step 3: Create fallback topics for missing entries ─────────────
    result: list[Topic] = [t for t, _ in matched_topics]

    for idx, entry in missing:
        entry_lower = entry.lower()
        keywords = [w for w in entry_lower.split() if len(w) > 3]
        if not keywords:
            keywords = [entry_lower]
        matched_chunks = [
            c for c in all_chunks
            if any(kw in c.content.lower() for kw in keywords)
        ]
        related_docs = list({c.document_id for c in matched_chunks})

        fallback = Topic(
            name=entry,
            description=entry,
            related_documents=related_docs,
            subtopic_count=0,
            order_source="syllabus",
            syllabus_position=idx,
        )

        result.append(fallback)
        logger.info(
            "  Fallback topic for syllabus entry %d: '%s' (%d chunks)",
            idx, entry, len(matched_chunks),
        )

    if missing:
        logger.warning(
            "Syllabus coverage gap: %d/%d entries not covered by LLM topics. "
            "Created %d fallback topics.",
            len(missing),
            len(syllabus_entries),
            len(missing),
        )

    # ── Step 4: Sort by syllabus position ──────────────────────────────
    result.sort(key=lambda t: t.syllabus_position if t.syllabus_position is not None else 999)

    return result


# ── Main class ───────────────────────────────────────────────────────────────


class TopicAnalyzer:
    """Identifies topics from source material chunks via a local LLM.

    Parameters
    ----------
    client:
        An ``OllamaClient`` instance for LLM communication.
    cfg:
        Optional ``ConfigManager`` for reading generation settings.
        When *None*, a default instance is created.
    """

    def __init__(
        self,
        client: OllamaClient,
        cfg: ConfigManager | None = None,
    ) -> None:
        self._client = client
        self._cfg = cfg or ConfigManager()

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(
        self,
        chunks: list[Chunk],
        syllabus_document: Document | None = None,
        scope: Literal["full", "topic"] | None = None,
        focus_topic: str | None = None,
        output_path: Path | str | None = None,
    ) -> tuple[list[Topic], dict[str, list[str]]]:
        """Run topic analysis on *chunks* and return ``Topic`` objects.

        Parameters
        ----------
        chunks:
            Source material chunks to analyse.
        syllabus_document:
            Optional document whose content is used as the syllabus.
            Detected when ``Document.is_syllabus == True``.
        scope:
            ``"full"`` (default) to analyse all material, or ``"topic"``
            to focus on a single topic.  Read from config when *None*.
        focus_topic:
            The topic to focus on when *scope* is ``"topic"``.
            Read from config when *None*.
        output_path:
            Where to write ``topics.json``.  Defaults to
            ``output/topics.json``.

        Returns
        -------
        tuple[list[Topic], dict[str, list[str]]]
            Identified topics in the requested order, and a mapping of
            topic name → list of chunk IDs for precise chunk association.

        Raises
        ------
        TopicNotFoundError
            If *scope* is ``"topic"`` but no topic matches *focus_topic*.
        TopicAnalyzerError
            If the LLM returns an unparseable response.
        """
        if not chunks:
            logger.warning("No chunks provided to topic analyzer")
            return [], {}

        scope = scope or self._cfg.get("generation.scope", "full")
        focus_topic = focus_topic or self._cfg.get("generation.focus_topic")

        # Resolve syllabus text
        syllabus_text = self._resolve_syllabus(syllabus_document)

        # Build prompt and query LLM
        prompt = build_topic_prompt(chunks, syllabus_text=syllabus_text)
        logger.info(
            "Sending topic analysis prompt to LLM (%d chunk(s), syllabus=%s)",
            len(chunks),
            "yes" if syllabus_text else "no",
        )

        # Topic lists can be long — start with a generous token budget
        # and retry with a larger one if the JSON is truncated.
        num_predict = 16384
        for attempt in range(3):
            raw_response = self._client.generate(
                prompt,
                options={"num_predict": num_predict},
            )
            try:
                raw_topics = _parse_topics_response(raw_response)
                break
            except TopicAnalyzerError:
                if attempt < 2:
                    num_predict *= 2
                    logger.warning(
                        "Topic analysis JSON parse failed (attempt %d), "
                        "retrying with num_predict=%d",
                        attempt + 1, num_predict,
                    )
                else:
                    raise

        # Build chunk ID map from LLM response
        chunk_id_map = _chunk_ids_from_response(raw_topics)

        # Convert to Topic objects
        topics: list[Topic] = [
            _build_topic(raw, idx) for idx, raw in enumerate(raw_topics)
        ]

        # Remove duplicate topic names (keep first occurrence)
        topics = _deduplicate_topics(topics)

        # ── Ensure all syllabus entries are covered ───────────────────
        if scope == "full" and syllabus_text:
            syllabus_entries = self._extract_syllabus_topics(syllabus_text)
            topics = _ensure_syllabus_coverage(
                topics, syllabus_entries, chunks,
            )

        # ── scope == "topic" → single-topic focus ──────────────────────
        if scope == "topic":
            topics = self._apply_topic_focus(
                topics, focus_topic, chunks, chunk_id_map,
            )

        # Persist
        out = Path(output_path) if output_path else Path("output/topics.json")
        self._save(topics, out)

        logger.info(
            "Topic analysis complete — %d topic(s) identified (scope=%s)",
            len(topics),
            scope,
        )
        return topics, chunk_id_map

    # ── Internals ───────────────────────────────────────────────────────

    def _resolve_syllabus(self, syllabus_document: Document | None) -> str | None:
        """Return the syllabus text, or *None* if no syllabus is available."""
        if syllabus_document is not None:
            logger.info("Using provided syllabus document: %s", syllabus_document.title)
            return syllabus_document.content

        # Check config for a syllabus file path
        syllabus_path = self._cfg.get("syllabus.path")
        if syllabus_path:
            path = Path(syllabus_path)
            if path.is_file():
                logger.info("Loading syllabus from config path: %s", path)
                return path.read_text(encoding="utf-8", errors="replace")
            logger.warning("Syllabus path configured but not found: %s", path)

        return None

    def _extract_syllabus_topics(self, syllabus_text: str) -> list[str]:
        """Extract an ordered list of topic names from the syllabus text.

        Uses the LLM to parse free-form prose, structured lists, or any
        mixed format.  Falls back to regex extraction if the LLM call fails.
        """
        # Fast path: if the syllabus is short and looks like a list,
        # the regex parser is sufficient and avoids an LLM call.
        regex_entries = _parse_syllabus_entries_regex(syllabus_text)
        lines = [l for l in syllabus_text.splitlines() if l.strip()]
        looks_like_list = regex_entries and len(regex_entries) >= len(lines) * 0.5

        if len(syllabus_text) < 1500 and looks_like_list:
            logger.info(
                "Syllabus looks like a structured list (%d entries) — "
                "skipping LLM extraction",
                len(regex_entries),
            )
            return regex_entries

        # Complex or prose syllabus — use the LLM
        prompt = build_syllabus_extraction_prompt(syllabus_text)
        logger.info(
            "Extracting topics from syllabus via LLM (%d chars)",
            len(syllabus_text),
        )
        try:
            raw = self._client.generate(
                prompt,
                options={"num_predict": 4096},
            )
            obj = parse_llm_json(raw, label="Syllabus extraction")
            topics_list = obj.get("topics", []) if isinstance(obj, dict) else []
            if isinstance(topics_list, list) and topics_list:
                entries = [str(t) for t in topics_list if t]
                logger.info("LLM extracted %d topic(s) from syllabus", len(entries))
                return entries
        except Exception as exc:
            logger.warning("LLM syllabus extraction failed: %s — falling back to regex", exc)

        # Fallback to regex
        if regex_entries:
            logger.info("Using regex fallback: %d entries", len(regex_entries))
        else:
            logger.warning("No syllabus entries could be extracted")
        return regex_entries

    def _apply_topic_focus(
        self,
        topics: list[Topic],
        focus_topic: str | None,
        chunks: list[Chunk],
        chunk_id_map: dict[str, list[str]],
    ) -> list[Topic]:
        """Reduce *topics* to a single topic matching *focus_topic*.

        Raises ``TopicNotFoundError`` if no match is found.
        """
        if not focus_topic:
            raise TopicAnalyzerError(
                "generation.focus_topic must be set when scope == 'topic'"
            )

        matched, best_name = self._find_closest_topic(focus_topic, topics)
        if not matched:
            raise TopicNotFoundError(
                f"No topic found matching '{focus_topic}'.  "
                f"Available topics: {[t.name for t in topics]}"
            )

        # Filter chunks to only those relevant to the focus topic
        relevant = _chunks_for_topic(
            focus_topic, chunks, chunk_id_map, topic_name=best_name,
        )

        if not relevant:
            raise TopicNotFoundError(
                f"No source chunks found that are relevant to '{focus_topic}'."
            )

        logger.info(
            "Focus mode: matched topic '%s' (%d relevant chunk(s))",
            best_name,
            len(relevant),
        )

        return [Topic(
            name=matched.name,
            description=matched.description,
            related_documents=matched.related_documents,
            subtopic_count=len(relevant),
            order_source="manual",
        )]

    @staticmethod
    def _find_closest_topic(
        focus: str, topics: list[Topic],
    ) -> tuple[Topic | None, str | None]:
        """Return the best-matching topic for *focus*.

        Tries exact (case-insensitive) match first, then substring, then
        keyword overlap.
        """
        focus_lower = focus.lower().strip()
        focus_words = set(focus_lower.split())

        best: Topic | None = None
        best_score = 0.0

        for t in topics:
            name_lower = t.name.lower().strip()

            # Exact match
            if focus_lower == name_lower:
                return t, t.name

            # Substring containment
            if focus_lower in name_lower or name_lower in focus_lower:
                score = min(len(focus_lower), len(name_lower)) / max(
                    len(focus_lower), len(name_lower), 1,
                )
                if score > best_score:
                    best = t
                    best_score = score
                    continue

            # Keyword overlap
            name_words = set(name_lower.split())
            overlap = len(focus_words & name_words)
            if overlap > 0:
                score = overlap / max(len(focus_words), len(name_words))
                if score > best_score:
                    best = t
                    best_score = score

        return best, best.name if best else None

    @staticmethod
    def _save(topics: list[Topic], path: Path) -> None:
        """Write *topics* to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "topics": [
                {
                    "name": t.name,
                    "description": t.description,
                    "related_documents": t.related_documents,
                    "order": idx,
                    "order_source": t.order_source,
                    "syllabus_position": t.syllabus_position,
                    "subtopic_count": t.subtopic_count,
                }
                for idx, t in enumerate(topics)
            ]
        }

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug("Topics saved to %s", path)
