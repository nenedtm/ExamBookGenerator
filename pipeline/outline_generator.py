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

    Topics with ``missing_from_notes=True`` are excluded from the
    coverage check — they will be generated as placeholder chapters
    later and don't need to appear in the LLM outline.

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

    # Check all topics are covered (skip missing_from_notes)
    missing: list[int] = []
    for idx in range(len(topics)):
        if idx in topic_chapter_map:
            continue
        # Skip topics with no notes — they'll be placeholder chapters
        if getattr(topics[idx], "missing_from_notes", False):
            continue
        missing.append(idx)

    if missing:
        missing_names = [topics[i].name for i in missing]
        warnings.append(
            f"Missing topics in outline: {', '.join(missing_names)}"
        )

    # Check order if syllabus (only for covered topics)
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
        syllabus_text: str | None = None,
    ) -> tuple[str, list[IndexEntry], list[OutlineChapter]]:
        """Generate a structured outline from *topics*.

        When topics are derived from a syllabus (``order_source == "syllabus"``),
        the outline is built directly from the topic list — one chapter per
        topic, in syllabus order — without querying the LLM.  When
        *syllabus_text* is provided, it is parsed to also extract section-level
        structure, so chapters and their sections match the syllabus exactly.

        Otherwise, the LLM is asked to group topics into chapters and suggest
        sub-sections.

        Parameters
        ----------
        topics:
            Ordered list of ``Topic`` objects (from ``TopicAnalyzer``).
        scope:
            ``"full"`` for a complete manual outline, or ``"topic`` for
            a single-topic focus.
        output_path:
            Where to write ``outline.md``.  Defaults to
            ``output/outline.md``.
        syllabus_text:
            Optional raw syllabus text.  When provided and topics are
            syllabus-derived, the outline includes sections extracted from
            the syllabus structure.

        Returns
        -------
        tuple[str, list[IndexEntry], list[OutlineChapter]]
            The rendered Markdown outline string, the structured
            ``IndexEntry`` list (with unique anchors), and the list of
            ``OutlineChapter`` objects describing each planned chapter.

        Raises
        ------
        OutlineGeneratorError
            If the LLM returns an unparseable response (non-syllabus path only).
        """
        if not topics:
            logger.warning("No topics provided — returning empty outline")
            return "", [], []

        has_syllabus = any(t.order_source == "syllabus" for t in topics)

        # ── Syllabus path: build outline directly from topics ───────────
        if has_syllabus:
            return self._generate_from_syllabus(topics, output_path, syllabus_text)

        # ── Non-syllabus path: ask LLM to organize topics into chapters ─
        topic_block = self._format_topics(topics)

        prompt = build_outline_prompt() + (
            f"\n\nThe topics below are already ordered by pedagogical "
            f"sequence. Do NOT reorder them. You may group related topics "
            f"into the same chapter, but the relative order of topics "
            f"must remain unchanged.\n\n"
            f"Topics:\n{topic_block}"
        )

        logger.info(
            "Sending outline prompt to LLM (%d topic(s), scope=%s)",
            len(topics),
            scope,
        )
        num_predict = max(16384, len(topics) * 256)
        raw_chapters = None

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

            is_valid, warnings = _validate_outline(raw_chapters, topics, has_syllabus=False)
            if is_valid:
                break

            if attempt < 2:
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
                prompt = prompt + feedback
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

    # ── Syllabus outline builder ────────────────────────────────────────

    def _generate_from_syllabus(
        self,
        topics: list[Topic],
        output_path: Path | str | None = None,
        syllabus_text: str | None = None,
    ) -> tuple[str, list[IndexEntry], list[OutlineChapter]]:
        """Build outline directly from syllabus-derived topics.

        When *syllabus_text* is provided, the section structure is extracted
        from the syllabus text so that each chapter's sub-sections match
        the syllabus exactly.  Topics corresponding to syllabus sections are
        folded into their parent chapter (not turned into separate chapters).
        No LLM call is made.
        """
        syllabus_structure: list[dict[str, object]] = []
        if syllabus_text:
            syllabus_structure = self._parse_syllabus_structure(syllabus_text)

        # (chapter_dict, list_of_topic_indices) pairs
        chapter_topic_pairs: list[tuple[dict[str, object], list[int]]] = []

        if syllabus_structure:
            # Collect all section titles from the parsed structure (lowercased)
            all_section_titles: set[str] = set()
            for ch in syllabus_structure:
                for sec in ch.get("sections", []):
                    t = sec.lower().strip()
                    if t:
                        all_section_titles.add(t)

            used_indices: set[int] = set()

            for parsed_ch in syllabus_structure:
                parsed_title = str(parsed_ch.get("title", "")).strip()
                parsed_sections = list(parsed_ch.get("sections", []))

                # Find main topic for this chapter
                ch_idx = self._find_topic_for_chapter(
                    parsed_title, topics, used_indices,
                )

                # Find topics matching each section title
                section_indices: list[int] = []
                skip = {ch_idx} if ch_idx is not None else set()
                for sec_title in parsed_sections:
                    sec_idx = self._find_topic_for_chapter(
                        sec_title, topics, used_indices | skip,
                    )
                    if sec_idx is not None:
                        section_indices.append(sec_idx)
                        skip.add(sec_idx)

                # Collect all topic indices for this chapter
                all_ch_indices: list[int] = []
                if ch_idx is not None:
                    used_indices.add(ch_idx)
                    all_ch_indices.append(ch_idx)
                for si in section_indices:
                    used_indices.add(si)
                    all_ch_indices.append(si)

                name = topics[ch_idx].name if ch_idx is not None else parsed_title
                chapter_topic_pairs.append((
                    {"title": name, "sections": parsed_sections},
                    all_ch_indices,
                ))

            # Remaining topics: skip section items (already absorbed),
            # turn everything else into individual chapters
            for i, topic in enumerate(topics):
                if i in used_indices:
                    continue
                topic_lower = topic.name.lower().strip()
                if topic_lower in all_section_titles:
                    # Find which chapter this section belongs to and add
                    # the index there instead
                    for pair in chapter_topic_pairs:
                        ch_sections = [s.lower().strip() for s in pair[0].get("sections", [])]
                        if topic_lower in ch_sections:
                            pair[1].append(i)
                            used_indices.add(i)
                            break
                    if i in used_indices:
                        continue
                # Truly leftover: individual chapter
                chapter_topic_pairs.append((
                    {"title": topic.name, "sections": []},
                    [i],
                ))
        else:
            # Flat syllabus: each topic → one chapter
            chapter_topic_pairs = [
                ({"title": topic.name, "sections": []}, [i])
                for i, topic in enumerate(topics)
            ]

        raw_chapters = [pair[0] for pair in chapter_topic_pairs]

        entries = _build_entries(raw_chapters)
        entries = _unique_anchors(entries)

        # Build OutlineChapter objects with correct topic index mapping
        outline_chapters: list[OutlineChapter] = []
        for ch_dict, topic_indices in chapter_topic_pairs:
            outline_chapters.append(OutlineChapter(
                title=str(ch_dict["title"]),
                sections=list(ch_dict.get("sections", [])),
                topic_indices=topic_indices,
            ))

        outline_md = _render_outline_md(entries)

        out = Path(output_path) if output_path else Path("output/outline.md")
        self._save(out, outline_md)

        logger.info(
            "Syllabus outline generated — %d chapter(s) from syllabus topics",
            len(outline_chapters),
        )
        return outline_md, entries, outline_chapters

    @staticmethod
    def _parse_syllabus_structure(text: str) -> list[dict[str, object]]:
        """Parse syllabus text into a hierarchical chapter + section structure.

        Handles formats like::

            ## Chapter 1: Title
            1. Section one
            2. Section two

            Chapter 2: Title
            1. Section one

            Chapter 1 - Title
            - Section one
            - Section two

            Module 1: Title
            Topic 1: Title
            Unit 1: Title
            Lezione 1: Titolo

            1. First Topic
               a. Subtopic
               b. Subtopic
            2. Second Topic

        Returns a list of dicts with keys ``"title"`` (chapter title) and
        ``"sections"`` (list of section title strings).  Returns an empty
        list if no chapter boundaries can be detected (flat syllabus).
        """
        lines = text.splitlines()
        stripped_lines = [(line, line.strip()) for line in lines]

        # ── Pass 1: detect hierarchical numbered list structure ──────────
        # Identify numbered items at minimum indent that have sub-items
        line_infos: list[dict] = []
        for raw, stripped in stripped_lines:
            if not stripped:
                line_infos.append({"type": "blank", "indent": 0})
                continue
            indent = len(raw) - len(raw.lstrip())
            num_match = re.match(r'^(\d+)[\.\)]\s+(.+)$', stripped)
            is_num = num_match is not None
            has_sub = False  # will be set after scanning ahead
            line_infos.append({
                "type": "numbered" if is_num else "other",
                "indent": indent,
                "text": stripped,
                "has_sub": False,
                "title": num_match.group(2).strip() if is_num else None,
            })

        # Mark numbered items that have sub-items (higher indent or lettered/bullet after them)
        for i, info in enumerate(line_infos):
            if info["type"] != "numbered":
                continue
            # Look ahead at non-blank lines until next numbered at same or lower indent
            for j in range(i + 1, len(line_infos)):
                next_info = line_infos[j]
                if next_info["type"] == "blank":
                    # Allow blank lines between chapter and sections
                    if j < len(line_infos) - 1:
                        continue
                    break
                if next_info["type"] == "numbered":
                    # Allow blank lines, then check indent
                    if next_info["indent"] <= info["indent"]:
                        break  # next chapter or flat item
                    else:
                        info["has_sub"] = True
                        break
                else:
                    # lettered, bullet, other indented content
                    if next_info["indent"] > info["indent"]:
                        info["has_sub"] = True
                        break
                    elif next_info["indent"] <= info["indent"]:
                        break
            if j == i + 1 and line_infos[j]["type"] == "blank":
                info["has_sub"] = True  # has content after blank

        # Collect numbered items that act as chapter headings (have sub-items)
        chapter_num_lines: set[int] = set()
        for i, info in enumerate(line_infos):
            if info["type"] == "numbered" and info["has_sub"]:
                chapter_num_lines.add(i)

        # ── Pass 2: build chapter+section structure ─────────────────────
        chapters: list[dict[str, object]] = []
        current_title: str | None = None
        current_sections: list[str] = []

        for i, (raw, stripped) in enumerate(stripped_lines):
            if not stripped:
                continue
            info = line_infos[i]
            indent = info["indent"]

            # ── Detect chapter/heading boundaries ────────────────────────
            ch_match = re.match(
                r'^\*{0,2}(?:#{1,3}\s+)?'
                r'(?:Chapter|Capitolo|Module|Modulo|Unit|Uni[tà]|Topic|Tema|'
                r'Section|Sezione|Lezione|Argomento|Parte|Part)'
                r'\s+\d+(?:\s*[:.\-]\s+|\s+)(.+?)\*{0,2}$',
                stripped, re.IGNORECASE,
            )
            if ch_match:
                if current_title is not None:
                    chapters.append({
                        "title": current_title,
                        "sections": current_sections,
                    })
                current_title = ch_match.group(1).strip()
                current_sections = []
                continue

            # Fallback: numbered item with sub-items (hierarchical list)
            if info["type"] == "numbered" and i in chapter_num_lines:
                if current_title is not None:
                    chapters.append({
                        "title": current_title,
                        "sections": current_sections,
                    })
                current_title = info["title"]
                current_sections = []
                continue

            # Markdown header with number but no keyword: ### 1. Title, ## 1. Title
            if ch_match is None:
                hdr_match = re.match(
                    r'^#{1,3}\s+\d+[\.\)]\s+(.+?)\*{0,2}$', stripped,
                )
                if hdr_match:
                    if current_title is not None:
                        chapters.append({
                            "title": current_title,
                            "sections": current_sections,
                        })
                    current_title = hdr_match.group(1).strip()
                    current_sections = []
                    continue

            if current_title is None:
                continue

            # ── Section items inside a chapter ───────────────────────────
            _section = None

            # Numbered sections: 1. Section, 1.1 Section, 1) Section
            sec_match = re.match(
                r'^\d+(?:\.\d+)*[\.\)]?\s+\*?\*?(.+?)\*?\*?\s*$',
                stripped,
            )
            if sec_match:
                _section = sec_match.group(1).strip()

            # Lettered sections: a. Section, a) Section, (a) Section
            if _section is None:
                let_match = re.match(
                    r'^\(?[a-zA-Z]\)?\.?\s+\*?\*?(.+?)\*?\*?\s*$',
                    stripped,
                )
                if let_match:
                    candidate = let_match.group(1).strip()
                    if len(candidate) > 1:
                        _section = candidate

            # Bullet sections: - Section, * Section, • Section
            if _section is None:
                bul_match = re.match(r'^[-•*]\s+\*?\*?(.+?)\*?\*?\s*$', stripped)
                if bul_match:
                    _section = bul_match.group(1).strip()

            if _section is not None:
                current_sections.append(_section)

        if current_title is not None:
            chapters.append({
                "title": current_title,
                "sections": current_sections,
            })

        return chapters if chapters else []

    @staticmethod
    def _find_topic_for_chapter(
        chapter_title: str,
        topics: list[Topic],
        used_indices: set[int],
    ) -> int | None:
        """Find the best topic index matching *chapter_title*.

        Uses multi-strategy matching: exact match (case-insensitive),
        substring containment, and word overlap (minimum 2 overlapping
        content words).  Skips already-used indices.
        """
        import re as _re

        _STOPWORDS: frozenset[str] = frozenset({
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "by", "with", "from", "as", "is", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "will", "would", "can", "could", "shall",
            "should", "may", "might", "this", "that", "these", "those",
            "it", "its", "they", "them", "their", "we", "our", "you",
            "your", "he", "she", "him", "her", "his", "not", "no",
            "nor", "all", "each", "every", "some", "any", "both",
            "few", "more", "most", "other", "into", "over", "such",
            "than", "then", "just", "about", "also", "very", "chapter",
        })

        title_lower = chapter_title.lower().strip()
        title_words = {
            w for w in _re.split(r"[^a-z0-9]+", title_lower)
            if len(w) >= 3 and w not in _STOPWORDS
        }
        best_idx: int | None = None
        best_score = 0.0

        for i, topic in enumerate(topics):
            if i in used_indices:
                continue
            name_lower = topic.name.lower().strip()

            # Exact match
            if title_lower == name_lower:
                return i

            # Substring containment
            if title_lower in name_lower or name_lower in title_lower:
                score = min(len(title_lower), len(name_lower)) / max(
                    len(title_lower), len(name_lower), 1,
                )
                if score > best_score:
                    best_idx = i
                    best_score = score

            # Word overlap (minimum 2 overlapping content words)
            name_words = {
                w for w in _re.split(r"[^a-z0-9]+", name_lower)
                if len(w) >= 3 and w not in _STOPWORDS
            }
            if title_words and name_words:
                overlap = len(title_words & name_words)
                if overlap >= 2:
                    score = overlap / max(len(title_words), len(name_words))
                    if score > best_score:
                        best_idx = i
                        best_score = score

        return best_idx if best_score >= 0.3 else None

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
