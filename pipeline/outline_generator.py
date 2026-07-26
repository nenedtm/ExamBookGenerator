"""Outline and index-entry generation for the ExamBookGenerator pipeline.

Takes the ordered list of ``Topic`` objects produced by the topic analyzer
(STEP 18) and asks the LLM to flesh out a hierarchical chapter / section
structure.  The LLM is **not** allowed to reorder the topics — it may
only group related topics into the same chapter and add sub-sections.

Outputs
-------
* ``outline.md`` — a human-readable Markdown outline.
* ``list[IndexEntry]`` — one entry per chapter (level 1) and one per
  sub-section (level 2), with unique anchors suitable for Markdown
  internal links (``#anchor``).

Usage::

    from pipeline.outline_generator import OutlineGenerator
    from llm.ollama_client import OllamaClient

    gen = OutlineGenerator(OllamaClient.from_config())
    outline_md, entries = gen.generate(topics, scope="full")
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Literal

from core.models import IndexEntry, OutlineChapter, Topic
from llm.ollama_client import OllamaClient, OllamaError
from llm.prompt_manager import build_outline_prompt
from utils import parse_llm_json
from utils.config import ConfigManager
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Exceptions ───────────────────────────────────────────────────────────────


class OutlineGeneratorError(OllamaError):
    """Raised when outline generation or parsing fails."""


# ── Slugging ─────────────────────────────────────────────────────────────────


def slugify(text: str) -> str:
    """Convert *text* into a Markdown-friendly anchor slug.

    Normalises unicode, lowercases, strips non-alphanumeric characters
    (except hyphens/spaces), and collapses whitespace to hyphens.
    """
    # NFKD normalise, then keep only ASCII letters, digits, spaces, hyphens
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    slug = ascii_text.lower()
    # Replace non-alphanumeric (except hyphen and space) with nothing
    slug = re.sub(r"[^a-z0-9 \-]", "", slug)
    # Collapse whitespace / runs of hyphens to a single hyphen
    slug = re.sub(r"[\s\-]+", "-", slug).strip("-")
    return slug


def _unique_anchors(entries: list[IndexEntry]) -> list[IndexEntry]:
    """Ensure every anchor in *entries* is unique.

    Appends ``-2``, ``-3``, … to duplicate anchors.  Preserves the
    original list order.
    """
    seen: dict[str, int] = {}
    for entry in entries:
        base = entry.anchor
        if base in seen:
            seen[base] += 1
            entry.anchor = f"{base}-{seen[base]}"
        else:
            seen[base] = 1
    return entries


# ── Response parsing ─────────────────────────────────────────────────────────


def _parse_outline_response(raw: str) -> list[dict[str, object]]:
    """Parse the LLM's outline JSON into a list of chapter dicts.

    Each dict has keys ``"title"`` (str) and ``"sections"`` (list[str]).
    """
    try:
        obj = parse_llm_json(raw, label="Outline generator")
    except ValueError as exc:
        raise OutlineGeneratorError(str(exc)) from exc

    chapters = obj.get("chapters")
    if not isinstance(chapters, list):
        # Try case-insensitive fallback (LLMs sometimes capitalize keys)
        if isinstance(obj, dict):
            for key in obj:
                if key.lower() == "chapters" and isinstance(obj[key], list):
                    chapters = obj[key]
                    break
    if not isinstance(chapters, list):
        # Check if the LLM returned an error response
        if isinstance(obj, dict) and "error" in obj:
            error_msg = str(obj["error"])
            raise OutlineGeneratorError(
                f"LLM returned an error instead of outline: {error_msg}"
            )
        raise OutlineGeneratorError(
            f"LLM JSON is missing the 'chapters' array.  "
            f"Keys present: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}"
        )

    # Normalize sections: LLMs sometimes return [{title, content}] instead of strings
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        sections = ch.get("sections")
        if not isinstance(sections, list):
            continue
        normalized: list[str] = []
        for s in sections:
            if isinstance(s, str):
                normalized.append(s.strip())
            elif isinstance(s, dict):
                title = str(s.get("title", "")).strip()
                if title:
                    normalized.append(title)
        ch["sections"] = normalized

    return chapters  # type: ignore[no-any-return]


# ── IndexEntry builder ───────────────────────────────────────────────────────


def _build_entries(chapters: list[dict[str, object]]) -> list[IndexEntry]:
    """Convert parsed LLM chapters into a flat list of ``IndexEntry`` objects.

    Level 1 for chapter titles, level 2 for sub-sections.
    """
    entries: list[IndexEntry] = []
    order = 0

    for ch in chapters:
        title = str(ch.get("title", "")).strip()
        if not title:
            continue

        entries.append(
            IndexEntry(
                title=title,
                anchor=slugify(title),
                level=1,
                order=order,
            )
        )
        order += 1

        sections = ch.get("sections", [])
        if isinstance(sections, list):
            for sec in sections:
                sec_title = str(sec).strip()
                if not sec_title:
                    continue
                entries.append(
                    IndexEntry(
                        title=sec_title,
                        anchor=slugify(sec_title),
                        level=2,
                        order=order,
                    )
                )
                order += 1

    return entries


def _build_outline_chapters(
    raw_chapters: list[dict[str, object]],
    topics: list[Topic],
) -> list[OutlineChapter]:
    """Convert parsed LLM chapters into ``OutlineChapter`` objects.

    Each outline chapter is matched to the topics it covers by
    keyword overlap between the chapter title/sections and the
    topic names/descriptions.
    """
    result: list[OutlineChapter] = []
    for ch in raw_chapters:
        title = str(ch.get("title", "")).strip()
        if not title:
            continue

        sections_raw = ch.get("sections", [])
        sections: list[str] = []
        if isinstance(sections_raw, list):
            sections = [str(s).strip() for s in sections_raw if str(s).strip()]

        # Match this outline chapter to topics
        topic_indices = _match_chapter_to_topics(title, sections, topics)

        result.append(OutlineChapter(
            title=title,
            sections=sections,
            topic_indices=topic_indices,
        ))

    return result


def _match_chapter_to_topics(
    chapter_title: str,
    sections: list[str],
    topics: list[Topic],
) -> list[int]:
    """Find which topics are covered by an outline chapter.

    Uses keyword overlap between the chapter's title + sections and
    each topic's name + description.
    """
    import re as _re

    # Build a set of meaningful words from the chapter
    chapter_text = (chapter_title + " " + " ".join(sections)).lower()
    chapter_words = {
        w for w in _re.split(r"[^a-z0-9]+", chapter_text) if len(w) > 3
    }

    matched: list[int] = []
    for idx, topic in enumerate(topics):
        topic_text = (topic.name + " " + topic.description).lower()
        topic_words = {
            w for w in _re.split(r"[^a-z0-9]+", topic_text) if len(w) > 3
        }
        if not topic_words:
            continue
        overlap = len(chapter_words & topic_words)
        if overlap >= 2 or (overlap >= 1 and len(topic_words) <= 3):
            matched.append(idx)

    return matched


def _validate_outline(
    raw_chapters: list[dict[str, object]],
    topics: list[Topic],
    has_syllabus: bool,
) -> tuple[bool, list[str]]:
    """Validate that the outline covers all topics in correct order.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of warning messages).
    """
    warnings: list[str] = []

    # Build topic-to-chapter mapping
    topic_chapter_map: dict[int, int] = {}
    for ch_idx, ch in enumerate(raw_chapters):
        title = str(ch.get("title", "")).strip()
        sections_raw = ch.get("sections", [])
        sections: list[str] = []
        if isinstance(sections_raw, list):
            sections = [str(s).strip() for s in sections_raw if str(s).strip()]

        matched_topics = _match_chapter_to_topics(title, sections, topics)
        for topic_idx in matched_topics:
            if topic_idx not in topic_chapter_map:
                topic_chapter_map[topic_idx] = ch_idx

    # Check all topics are covered
    missing: list[int] = []
    for idx in range(len(topics)):
        if idx not in topic_chapter_map:
            missing.append(idx)

    if missing:
        missing_names = [topics[i].name for i in missing]
        warnings.append(
            f"Missing topics in outline: {', '.join(missing_names)}"
        )

    # Check order if syllabus
    if has_syllabus and len(missing) == 0:
        chapter_order = []
        for idx in range(len(topics)):
            if idx in topic_chapter_map:
                chapter_order.append(topic_chapter_map[idx])

        # Verify order is non-decreasing
        for i in range(1, len(chapter_order)):
            if chapter_order[i] < chapter_order[i - 1]:
                warnings.append(
                    f"Topic order violated: topic '{topics[i].name}' "
                    f"appears before '{topics[i - 1].name}'"
                )
                break

    return len(warnings) == 0, warnings


# ── Markdown renderer ─────────────────────────────────────────────────────────


def _render_outline_md(entries: list[IndexEntry]) -> str:
    """Render a list of ``IndexEntry`` objects into a Markdown outline."""
    lines: list[str] = []
    for entry in entries:
        prefix = "  " * (entry.level - 1)  # 2-space indent per level
        lines.append(f"{prefix}- [{entry.title}](#{entry.anchor})")
    return "\n".join(lines) + ("\n" if lines else "")


# ── Main class ───────────────────────────────────────────────────────────────


class OutlineGenerator:
    """Generates a hierarchical chapter/sub-chapter outline from topics.

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

    def generate(
        self,
        topics: list[Topic],
        *,
        scope: Literal["full", "topic"] = "full",
        output_path: Path | str | None = None,
    ) -> tuple[str, list[IndexEntry], list[OutlineChapter]]:
        """Generate a structured outline from *topics*.

        Parameters
        ----------
        topics:
            Ordered list of ``Topic`` objects (from ``TopicAnalyzer``).
            The order is preserved — the LLM is not allowed to reorder.
        scope:
            ``"full"`` for a complete manual outline, or ``"topic"`` for
            a single-topic focus.
        output_path:
            Where to write ``outline.md``.  Defaults to
            ``output/outline.md``.

        Returns
        -------
        tuple[str, list[IndexEntry], list[OutlineChapter]]
            The rendered Markdown outline string, the structured
            ``IndexEntry`` list (with unique anchors), and the list of
            ``OutlineChapter`` objects describing each planned chapter.

        Raises
        ------
        OutlineGeneratorError
            If the LLM returns an unparseable response.
        """
        if not topics:
            logger.warning("No topics provided — returning empty outline")
            return "", [], []

        # Build topic summary for the prompt
        topic_block = self._format_topics(topics)

        # Detect if topics are syllabus-ordered
        has_syllabus = any(
            t.order_source == "syllabus" for t in topics
        )

        # Query LLM
        syllabus_note = ""
        if has_syllabus:
            syllabus_note = (
                "\n\n## SYLLABUS ORDER — MANDATORY — NON-NEGOTIABLE\n\n"
                "The topics below are listed in the EXACT order of an official "
                "exam syllabus. This order is FINAL and MUST be preserved.\n\n"
                "RULES FOR SYLLABUS ORDER:\n"
                "1. Chapter 1 MUST contain Topic 1.\n"
                "2. The LAST chapter MUST contain the LAST topic.\n"
                "3. If you group topics into one chapter, they MUST be "
                "CONSECUTIVE topics from the list (e.g., topics 3, 4, 5).\n"
                "4. You MUST NEVER move a topic to a position before topics "
                "that originally preceded it.\n"
                "5. You MUST NEVER skip a topic — every numbered topic must "
                "appear in the output.\n\n"
                "VIOLATION EXAMPLES (WRONG):\n"
                "- Moving 'Calculus' before 'Algebra' if Algebra is listed first\n"
                "- Skipping topic 3 and including topics 1, 2, 4, 5\n"
                "- Putting topic 5 in chapter 2 and topic 2 in chapter 3\n\n"
                "VALID EXAMPLE (CORRECT):\n"
                "If topics are [Algebra, Calculus, Probability], valid outputs:\n"
                "- [{Algebra chapters}, {Calculus chapters}, {Probability chapters}]\n"
                "- [{Algebra+Calculus chapters}, {Probability chapters}]\n"
                "- [{Algebra chapters}, {Calculus+Probability chapters}]\n"
            )

        prompt = build_outline_prompt() + (
            f"\n\nThe topics below are already ordered (either by "
            f"{'syllabus position' if has_syllabus else 'pedagogical sequence'}). "
            f"Do NOT reorder them. You may group related topics into the same "
            f"chapter, but the relative order of topics must remain unchanged."
            f"{syllabus_note}\n\n"
            f"Topics:\n{topic_block}"
        )

        logger.info(
            "Sending outline prompt to LLM (%d topic(s), scope=%s)",
            len(topics),
            scope,
        )
        num_predict = 16384
        raw_chapters = None
        validation_warnings: list[str] = []

        for attempt in range(3):
            raw_response = self._client.generate(
                prompt,
                options={"num_predict": num_predict},
            )
            try:
                raw_chapters = _parse_outline_response(raw_response)
            except OutlineGeneratorError:
                if attempt < 2:
                    num_predict *= 2
                    logger.warning(
                        "Outline JSON parse failed (attempt %d), "
                        "retrying with num_predict=%d",
                        attempt + 1, num_predict,
                    )
                    continue
                else:
                    raise

            # Validate outline covers all topics in correct order
            is_valid, warnings = _validate_outline(raw_chapters, topics, has_syllabus)
            if is_valid:
                break

            validation_warnings = warnings
            if attempt < 2:
                # Append validation feedback to prompt for next attempt
                feedback = (
                    "\n\n## CRITICAL ERRORS IN PREVIOUS OUTLINE\n\n"
                    "Your outline had the following problems:\n"
                )
                for w in warnings:
                    feedback += f"- {w}\n"
                feedback += (
                    "\nYou MUST fix ALL issues above. "
                    "Re-output the complete corrected JSON outline."
                )
                prompt_with_feedback = prompt + feedback
                prompt = prompt_with_feedback
                num_predict *= 2
                logger.warning(
                    "Outline validation failed (attempt %d): %s, "
                    "retrying with feedback",
                    attempt + 1, "; ".join(warnings),
                )
            else:
                logger.warning(
                    "Outline validation failed after %d attempts: %s. "
                    "Proceeding with best available outline.",
                    attempt + 1, "; ".join(warnings),
                )

        # Should never happen, but satisfy type checker
        if raw_chapters is None:
            raise OutlineGeneratorError(
                "Outline generation failed: no valid response from LLM"
            )

        # Build IndexEntry list
        entries = _build_entries(raw_chapters)
        entries = _unique_anchors(entries)

        # Build OutlineChapter objects (title + sections + topic mapping)
        outline_chapters = _build_outline_chapters(raw_chapters, topics)

        # Render Markdown
        outline_md = _render_outline_md(entries)

        # Persist
        out = Path(output_path) if output_path else Path("output/outline.md")
        self._save(out, outline_md)

        logger.info(
            "Outline generated — %d chapter(s), %d total entries",
            sum(1 for e in entries if e.level == 1),
            len(entries),
        )
        return outline_md, entries, outline_chapters

    # ── Internals ───────────────────────────────────────────────────────

    @staticmethod
    def _format_topics(topics: list[Topic]) -> str:
        """Format topics into a readable block for the LLM prompt."""
        lines: list[str] = []
        for idx, t in enumerate(topics, 1):
            lines.append(f"{idx}. {t.name}")
            if t.description:
                lines.append(f"   Description: {t.description}")
            if t.subtopic_count:
                lines.append(f"   Subtopics: {t.subtopic_count}")
        return "\n".join(lines)

    @staticmethod
    def _save(path: Path, content: str) -> None:
        """Write *content* to *path*, creating parent dirs if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.debug("Outline saved to %s", path)
