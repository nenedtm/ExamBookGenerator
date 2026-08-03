"""Chapter generator — produces one Markdown chapter per topic.

Supports two modes:

* **full** (default): generates a chapter for a ``Topic`` using
  ``build_chapter_prompt`` and returns the rendered Markdown including
  a local table of contents built from the supplied ``IndexEntry``
  list.
* **topic** (focus): generates a single stand-alone chapter for the
  user's focus topic using ``build_focus_topic_prompt``, with the
  effective ``depth_level`` taken from ``generation.focus_depth_level``.

Usage::

    from pipeline.chapter_generator import generate_chapter
    from llm.ollama_client import OllamaClient

    client = OllamaClient.from_config()
    md, title, img_ids = generate_chapter(
        topic, chunks, template, depth_level=7,
        candidate_images=images, index_entries=entries,
        scope="full", client=client,
    )
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from core.models import Chunk, Document, ExtractedImage, IndexEntry, OutlineChapter, Topic
from llm.ollama_client import OllamaClient, OllamaError
from llm.prompt_manager import (
    build_chapter_prompt,
    build_focus_topic_prompt,
)
from pipeline.image_matcher import select_images_for_chapter
from pipeline.template_engine import apply_template
from utils.config import ConfigManager
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Output token budget based on depth level ────────────────────────────────

_NUM_PREDICT_MAP: dict[int, int] = {
    1: 8192,
    2: 10240,
    3: 14336,
    4: 18432,
    5: 22528,
    6: 28672,
    7: 36864,
    8: 45056,
    9: 53248,
    10: 65536,
}


def _num_predict_for_depth(depth: int) -> int:
    """Return the Ollama ``num_predict`` token budget for *depth*."""
    clamped = max(1, min(10, depth))
    return _NUM_PREDICT_MAP[clamped]


# ── Exceptions ───────────────────────────────────────────────────────────────


class ChapterGeneratorError(OllamaError):
    """Raised when chapter generation or parsing fails."""


# ── Slug helper (mirrors outline_generator.slugify) ──────────────────────────


def _slugify(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    slug = ascii_text.lower()
    slug = re.sub(r"[^a-z0-9 \-]", "", slug)
    slug = re.sub(r"[\s\-]+", "-", slug).strip("-")
    return slug


# ── Response parsing ─────────────────────────────────────────────────────────


def _parse_chapter_response(raw: str) -> dict[str, object]:
    """Parse the LLM's chapter JSON into ``{"title", "content", "sections"}``.

    Raises ``ChapterGeneratorError`` on invalid / incomplete output.
    """
    from utils import parse_llm_json

    try:
        obj = parse_llm_json(raw, label="Chapter generator")
    except ValueError as exc:
        raise ChapterGeneratorError(str(exc)) from exc

    if not isinstance(obj, dict):
        raise ChapterGeneratorError(f"Expected JSON object, got {type(obj)}")

    # Check if the LLM returned an error response
    if "error" in obj:
        error_msg = str(obj["error"])
        raise ChapterGeneratorError(
            f"LLM returned an error instead of chapter content: {error_msg}"
        )

    for key in ("title", "content"):
        if key not in obj or not str(obj[key]).strip():
            raise ChapterGeneratorError(
                f"LLM JSON missing required key '{key}'.  "
                f"Keys present: {list(obj.keys())}"
            )

    # sections may be absent — default to empty list
    if "sections" not in obj:
        obj["sections"] = []

    return obj


def _has_meaningful_content(content: str) -> bool:
    """Check whether *content* contains body text beyond headings.

    Returns ``True`` when the content has at least a few lines of actual
    prose (not just headings, rules, or blank lines).
    """
    body_lines = 0
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^#{1,6}\s", stripped):
            continue
        if re.match(r"^---+\s*$", stripped):
            continue
        if stripped.startswith(">"):
            continue
        if stripped.startswith("[Chunk"):
            continue
        # Count lines with actual body text
        body_lines += 1
        if body_lines >= 3:
            return True
    return False


def _clean_content(content: str) -> str:
    """Remove artifact lines that should not appear in the final output.

    Strips raw chunk-metadata headers (``[Chunk ...]``, ``--- Source N ---``),
    ``[...]`` placeholder lines, and ``#stochastic-processes``-style tags.
    """
    cleaned_lines: list[str] = []
    for line in content.split("\n"):
        stripped = line.strip()
        # Remove chunk metadata headers
        if re.match(r"^\[Chunk\s", stripped):
            continue
        if re.match(r"^---\s*Source\s+\d+\s*---$", stripped):
            continue
        # Remove bare placeholder lines like "[...]"
        if stripped == "[...]":
            continue
        # Remove tag lines like "#stochastic-processes #brownian-motion ..."
        if re.match(r"^#[a-z]", stripped) and " " in stripped:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


# ── Outline-structure enforcement ────────────────────────────────────────────


def _normalize(text: str) -> str:
    """Return a lowercase alphanumeric key for fuzzy heading comparison."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_text.lower())


def _heading_texts(content: str) -> list[str]:
    """Return the text of every Markdown heading (``#``..``######``) in *content*."""
    texts: list[str] = []
    for line in content.split("\n"):
        m = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if m:
            texts.append(m.group(1).strip())
    return texts


def _missing_outline_sections(content: str, sections: list[str]) -> list[str]:
    """Return *sections* whose heading is absent from *content* (normalised)."""
    heading_keys = {_normalize(h) for h in _heading_texts(content)}
    return [s for s in sections if _normalize(s) not in heading_keys]


_HEADING_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "by", "with", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "can", "could", "shall", "should", "may", "might", "this",
    "that", "these", "those", "it", "its", "they", "them", "their", "we",
    "our", "you", "your", "he", "she", "him", "her", "his", "not", "no",
    "nor", "all", "each", "every", "some", "any", "both", "few", "more",
    "most", "other", "into", "over", "such", "than", "then", "just",
    "about", "also", "very", "chapter", "section", "part",
    "di", "del", "della", "delle", "dei", "dello", "degli", "dal", "dalla",
    "dai", "dalle", "il", "lo", "la", "le", "gli", "un", "una", "uno",
    "con", "per", "su", "tra", "fra", "in", "nel", "nella", "nei", "nelle",
    "allo", "alla", "agli", "alle", "col", "coi", "sul", "sulla", "sui",
    "che", "e", "ed", "o", "senza", "non", "come", "quale", "quali",
    "da", "dai", "di", "ai", "al", "allo", "agli",
})


def _heading_word_overlap(a: str, b: str) -> float:
    """Jaccard-like word-overlap score between two heading texts (0..1)."""
    def _words(text: str) -> set[str]:
        return {
            w for w in re.split(r"[^a-z0-9]+", text.lower())
            if len(w) > 2 and w not in _HEADING_STOPWORDS
        }

    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def _align_outline_headings(
    content: str,
    title: str,
    sections: list[str],
) -> str:
    """Rename *content* headings to match the outline *title* and *sections*.

    The first heading is rewritten to *title*; every later heading whose
    text overlaps an outline section is renamed to the exact section title
    (greedy best match, requires a word-overlap score >= 0.5).  Headings
    that match nothing are left untouched so no body text is ever lost.
    """
    lines = content.split("\n")
    first_idx = -1
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            lines[i] = f"{m.group(1)} {title}"
            first_idx = i
            break
    if first_idx < 0:
        return content

    assigned: set[int] = set()
    remaining = list(enumerate(sections))
    for i, line in enumerate(lines):
        if i == first_idx:
            continue
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not m:
            continue
        heading_text = m.group(2).strip()
        best_idx = -1
        best_score = 0.0
        for s_idx, _sec in remaining:
            if s_idx in assigned:
                continue
            score = _heading_word_overlap(heading_text, sections[s_idx])
            if score > best_score:
                best_score = score
                best_idx = s_idx
        if best_idx >= 0 and best_score >= 0.5:
            lines[i] = f"{m.group(1)} {sections[best_idx]}"
            assigned.add(best_idx)

    return "\n".join(lines)


# ── Index entries from sections ──────────────────────────────────────────────


def _build_chapter_index(
    title: str,
    sections: list[str],
) -> list[IndexEntry]:
    """Build ``IndexEntry`` list for one chapter's local TOC.

    Level 1 for the chapter title, level 2 for each section heading.
    """
    entries: list[IndexEntry] = []
    order = 0

    if title.strip():
        entries.append(
            IndexEntry(
                title=title.strip(),
                anchor=_slugify(title),
                level=1,
                order=order,
            )
        )
        order += 1

    for sec in sections:
        sec = str(sec).strip()
        if not sec:
            continue
        entries.append(
            IndexEntry(
                title=sec,
                anchor=_slugify(sec),
                level=2,
                order=order,
            )
        )
        order += 1

    return entries


# ── Image insertion ──────────────────────────────────────────────────────────


def _insert_images(
    content: str,
    matched: list[tuple[ExtractedImage, str]],
) -> str:
    """Insert image Markdown blocks into *content* at suggested placements.

    Each ``matched`` entry is ``(image, placement)`` where *placement* is
    one of ``"after introduction"``, ``"after section: <title>"``, or
    ``"before conclusion"``.

    Images are inserted in reverse order so that positional shifts caused
    by earlier insertions do not invalidate later placement targets.
    """
    if not matched:
        return content

    lines = content.split("\n")

    # Insert in reverse to keep earlier indices stable
    for img, placement in reversed(matched):
        caption = img.caption or img.ai_description or ""
        img_md = f"\n![{caption}]({img.file_path})\n"
        if caption:
            img_md += f"*{caption}*\n"

        idx = _find_insertion_index(lines, placement)
        lines = lines[:idx] + img_md.split("\n") + lines[idx:]

    return "\n".join(lines)


def _find_insertion_index(lines: list[str], placement: str) -> int:
    """Return the line index at which to insert an image for *placement*.

    recognised placements:
      * ``"after introduction"``   — after the first heading
      * ``"after section: <t>"``   — after the first ``## <t>`` or ``### <t>``
      * ``"before conclusion"``    — before the last heading, or end of file
    """
    placement = placement.strip().lower()

    if placement == "before conclusion":
        # Insert before the last heading, or at the end
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("#"):
                return i
        return len(lines)

    if placement.startswith("after section:"):
        section_title = placement[len("after section:"):].strip().lower()
        for i, line in enumerate(lines):
            if line.lower().strip().lstrip("#").strip() == section_title:
                # Insert after this heading and any immediate blank lines
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                return j
        # Fallback: section not found — insert at end
        return len(lines)

    if placement == "after introduction":
        # Insert after the first heading (the ## title line)
        for i, line in enumerate(lines):
            if line.startswith("#"):
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                return j
        return min(2, len(lines))

    # Unknown placement — append at end
    return len(lines)


# ── Source references ────────────────────────────────────────────────────────


def build_sources_block(
    chunks: list[Chunk],
    documents: list[Document] | None = None,
) -> str:
    """Build a numbered Markdown reference list for *chunks*.

    Each distinct ``Document`` backing *chunks* becomes one entry in the
    order it first appears: ``1. *Title* — path``.  Syllabus documents
    are excluded.  Returns an empty string when no documents match.
    """
    if not chunks or not documents:
        return ""

    doc_map = {d.id: d for d in documents}
    seen: set[str] = set()
    refs: list[str] = []
    for c in chunks:
        doc = doc_map.get(c.document_id)
        if doc is None or doc.is_syllabus or doc.id in seen:
            continue
        seen.add(doc.id)
        title = (doc.title or Path(doc.source_path).name).strip()
        path = doc.source_path or ""
        refs.append(f"{title} — `{path}`" if path else f"{title}")

    if not refs:
        return ""
    return "\n".join(f"{i + 1}. {ref}" for i, ref in enumerate(refs))


# ── Main public function ─────────────────────────────────────────────────────


def generate_chapter(
    topic: Topic,
    chunks: list[Chunk],
    template: str,
    *,
    depth_level: int,
    candidate_images: list[ExtractedImage] | None = None,
    index_entries: list[IndexEntry] | None = None,
    outline_chapter: OutlineChapter | None = None,
    scope: str = "full",
    client: OllamaClient | None = None,
    cfg: ConfigManager | None = None,
    documents: list[Document] | None = None,
) -> tuple[str, str, list[str]]:
    """Generate a single chapter Markdown file.

    Parameters
    ----------
    topic:
        The topic to generate a chapter for.
    chunks:
        Source material chunks relevant to this topic.
    template:
        Raw Markdown template text (with ``{{…}}`` placeholders).
    depth_level:
        Effective detail level 1-10 — the caller must resolve this
        (``focus_depth_level`` when ``scope="topic"``, otherwise
        ``depth_level``).
    candidate_images:
        All extracted images available for matching.
    index_entries:
        Pre-built entries for this chapter's local TOC.  When *None*,
        the function builds entries from the LLM's ``sections`` list.
    outline_chapter:
        The planned chapter structure from the outline generator.
        When provided, the LLM is instructed to use the given title
        and sections as the required structure.
    scope:
        ``"full"`` or ``"topic"``.
    client:
        Ollama client instance.  When *None*, a default one is created.
    cfg:
        Optional ``ConfigManager``.
    documents:
        All parsed ``Document`` objects available in this run.  Used to
        build the chapter's source-reference list (``{{sources}}``) from
        the ``Document`` objects backing *chunks*, and to let the LLM
        cite those sources inline.

    Returns
    -------
    tuple[str, str, list[str]]
        ``(chapter_md, title, inserted_image_ids)``

    Raises
    ------
    ChapterGeneratorError
        If the LLM returns an unparseable response.
    """
    if client is None:
        client = OllamaClient.from_config()
    if cfg is None:
        cfg = ConfigManager()

    candidate_images = candidate_images or []

    # ── Placeholder for topics missing from notes ──────────────────────
    if topic.missing_from_notes:
        logger.info(
            "Topic '%s' has no source material — generating placeholder chapter",
            topic.name,
        )
        title = topic.name
        content = (
            f"## {title}\n\n"
            f"> **Note:** This topic is part of the exam syllabus but no "
            f"matching content was found in the provided notes. "
            f"Please refer to the official course material or textbook "
            f"for this topic.\n\n"
            f"### Topics to study\n\n"
            f"{topic.description or title}\n"
        )
        sections = ["Topics to study"]

        chapter_entries = index_entries or _build_chapter_index(title, sections)
        chapter_md = apply_template(
            template,
            title=title,
            content=content,
            index_entries=chapter_entries,
            images=[],
            sources=build_sources_block(chunks, documents),
            include_toc=True,
        )
        logger.info(
            "Placeholder chapter '%s' generated (no source material)",
            title,
        )
        return chapter_md, title, []

    # ── Build prompt ───────────────────────────────────────────────────
    sources_block = build_sources_block(chunks, documents)
    if scope == "topic":
        prompt = build_focus_topic_prompt(
            topic.name, chunks, depth_level, sources=sources_block,
        )
    else:
        prompt = build_chapter_prompt(
            topic, chunks, depth_level,
            outline_chapter=outline_chapter,
            sources=sources_block,
        )

    # ── Query LLM ─────────────────────────────────────────────────────
    num_predict = _num_predict_for_depth(depth_level)
    logger.info(
        "Generating chapter for topic '%s' (scope=%s, depth=%d, %d chunks, num_predict=%d)",
        topic.name, scope, depth_level, len(chunks), num_predict,
    )
    parsed = None
    for attempt in range(5):
        raw_response = client.generate(
            prompt,
            options={"num_predict": num_predict},
        )
        problems: list[str] = []
        try:
            candidate = _parse_chapter_response(raw_response)
            # Verify the content has actual body text, not just headings
            content_str = str(candidate.get("content", "")).strip()
            if not _has_meaningful_content(content_str):
                problems.append(
                    "The content field contained only headings without "
                    "body paragraphs."
                )
            # Verify the chapter follows the planned outline structure
            if outline_chapter is not None:
                missing = _missing_outline_sections(
                    content_str, outline_chapter.sections,
                )
                if missing:
                    problems.append(
                        "The chapter is missing the following REQUIRED "
                        "section headings: " + "; ".join(missing) + "."
                    )
            if not problems:
                parsed = candidate
                break
            logger.warning(
                "Chapter for '%s' failed validation (attempt %d): %s",
                topic.name, attempt + 1, "; ".join(problems),
            )
        except ChapterGeneratorError:
            problems.append("The JSON was invalid or missing required keys.")

        if attempt < 4:
            num_predict = int(num_predict * 1.5)
            # Build feedback about what went wrong
            feedback = (
                "\n\n## CRITICAL ERROR IN PREVIOUS ATTEMPT\n\n"
                + " ".join(problems) + "\n"
                "You MUST write 4-5 SUBSTANTIAL PARAGRAPHS of body text "
                "under EVERY section heading. Each paragraph must be at "
                "least 3-4 sentences long. Do NOT output empty sections "
                "or sections with only headings.\n"
                "When a REQUIRED chapter structure is given, you MUST use "
                "EXACTLY those chapter title and section headings — do NOT "
                "rename or omit them.\n"
                "Re-output the COMPLETE corrected JSON with full content."
            )
            prompt_with_feedback = prompt + feedback
            prompt = prompt_with_feedback
            logger.warning(
                "Chapter generation failed for '%s' (attempt %d), "
                "retrying with num_predict=%d",
                topic.name, attempt + 1, num_predict,
            )

    if parsed is None:
        raise ChapterGeneratorError(
            f"Failed to generate meaningful content for topic '{topic.name}' "
            f"after 5 attempts"
        )

    title = str(parsed["title"]).strip()
    content = str(parsed["content"]).strip()
    content = _clean_content(content)
    sections: list[str] = parsed.get("sections", [])  # type: ignore[assignment]

    # ── Enforce the planned outline structure ────────────────────────────
    # The LLM is asked to respect it, but may still drift; forcing the title
    # and aligning the section headings guarantees the generated chapter
    # follows the same index shown to the user in ``indice.md``.
    if outline_chapter is not None:
        title = outline_chapter.title
        content = _align_outline_headings(content, title, outline_chapter.sections)
        sections = list(outline_chapter.sections)

    # ── Extra-in-notes banner ──────────────────────────────────────────
    if topic.extra_in_notes:
        extra_note = (
            "> **Note:** This topic was found in the notes but is not part "
            "of the official exam syllabus. It is included here for "
            "completeness.\n\n"
        )
        content = extra_note + content

    # ── Index entries ──────────────────────────────────────────────────
    if index_entries is None:
        index_entries = _build_chapter_index(title, sections)

    # ── Image matching ─────────────────────────────────────────────────
    inserted_ids: list[str] = []
    image_dicts: list[dict[str, str]] = []

    if candidate_images and content:
        try:
            matched = select_images_for_chapter(
                topic, content, candidate_images,
                client=client, max_images=3,
            )
            if matched:
                content = _insert_images(content, matched)
                for img, _placement in matched:
                    inserted_ids.append(img.id)
                    caption = img.caption or img.ai_description or ""
                    image_dicts.append({"caption": caption, "path": img.file_path})
        except Exception as exc:
            logger.warning(
                "Image matching failed for topic '%s': %s — continuing without images",
                topic.name, exc,
            )

    # ── Template rendering ─────────────────────────────────────────────
    chapter_md = apply_template(
        template,
        title=title,
        content=content,
        index_entries=index_entries,
        images=image_dicts,
        sources=sources_block,
        include_toc=True,
    )

    logger.info(
        "Chapter '%s' generated — %d chars, %d images inserted",
        title, len(chapter_md), len(inserted_ids),
    )
    return chapter_md, title, inserted_ids
