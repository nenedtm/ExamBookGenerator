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

import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

from core.models import IndexEntry, Topic
from llm.ollama_client import OllamaClient, OllamaError
from llm.prompt_manager import build_outline_prompt
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
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Try direct parse
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise OutlineGeneratorError(
                "LLM returned unrecognisable outline — could not extract "
                f"JSON.  Raw output (first 500 chars): {text[:500]!r}"
            )
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise OutlineGeneratorError(
                f"LLM returned invalid JSON: {exc}.  "
                f"Raw output (first 500 chars): {text[:500]!r}"
            ) from exc

    chapters = obj.get("chapters")
    if not isinstance(chapters, list):
        raise OutlineGeneratorError(
            f"LLM JSON is missing the 'chapters' array.  "
            f"Keys present: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}"
        )

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
    ) -> tuple[str, list[IndexEntry]]:
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
        tuple[str, list[IndexEntry]]
            The rendered Markdown outline string and the structured
            ``IndexEntry`` list (with unique anchors).

        Raises
        ------
        OutlineGeneratorError
            If the LLM returns an unparseable response.
        """
        if not topics:
            logger.warning("No topics provided — returning empty outline")
            return "", []

        # Build topic summary for the prompt
        topic_block = self._format_topics(topics)

        # Query LLM
        prompt = build_outline_prompt() + (
            "\n\nIMPORTANT: The topics below are already ordered (either by "
            "syllabus position or pedagogical sequence).  Do NOT reorder "
            "them.  You may group related topics into the same chapter, but "
            "the relative order of topics must remain unchanged.\n\n"
            f"Topics:\n{topic_block}"
        )

        logger.info(
            "Sending outline prompt to LLM (%d topic(s), scope=%s)",
            len(topics),
            scope,
        )
        raw_response = self._client.generate(prompt)
        raw_chapters = _parse_outline_response(raw_response)

        # Build IndexEntry list
        entries = _build_entries(raw_chapters)
        entries = _unique_anchors(entries)

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
        return outline_md, entries

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
