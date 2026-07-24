"""Centralized prompt builder for the ExamBookGenerator pipeline.

Every LLM interaction in the project goes through a builder function
defined here.  This keeps all prompt logic in a single, auditable
module and guarantees consistent instructions across the pipeline.

Usage::

    from llm.prompt_manager import build_chapter_prompt

    prompt = build_chapter_prompt(topic, chunks, depth_level=7)
"""

from __future__ import annotations

import json
from typing import Any

from core.models import Chunk, Topic
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_ALWAYS_ENGLISH = (
    "Always write the output in English, regardless of the language of "
    "the source material."
)

_DEPTH_MAP: dict[tuple[int, int], str] = {
    (1, 2): (
        "Produce a concise, tightly written summary. Cover every topic and "
        "subtopic, but keep explanations brief (1-3 sentences each). "
        "No topic may be omitted. The result should read like a dense "
        "revision card, not a list of bullet points."
    ),
    (3, 5): (
        "Write a clear, moderately detailed university-level explanation. "
        "Each topic should flow naturally into the next. Use prose, not "
        "bullet lists. Include short examples where they aid understanding. "
        "The text must read as a unified chapter, not as a collection of "
        "independent notes."
    ),
    (6, 8): (
        "Write a thorough, well-structured university-level chapter. "
        "Integrate all material into continuous, flowing prose with proper "
        "academic style. Include derivations, worked examples, and "
        "discussion of common exam pitfalls. The text must read as a "
        "cohesive textbook chapter, not as rearranged lecture notes."
    ),
    (9, 10): (
        "Write an exhaustive, highly detailed university-level chapter. "
        "Do not omit any nuance, edge case, formula derivation, or example "
        "present in the source material. Maintain a formal academic tone "
        "throughout. Every paragraph must be original prose that synthesizes "
        "the source material — never copy passages verbatim."
    ),
}


def _depth_instruction(level: int) -> str:
    """Translate a numeric depth level (1-10) to a textual instruction."""
    clamped = max(1, min(10, level))
    for (lo, hi), text in _DEPTH_MAP.items():
        if lo <= clamped <= hi:
            return text
    # Fallback (should never happen with clamping above)
    return _DEPTH_MAP[(3, 5)]


def _chunks_text(
    chunks: list[Chunk],
    *,
    max_chars: int = 80_000,
    depth_level: int = 5,
) -> str:
    """Serialize a list of chunks into a readable text block for the prompt.

    The *max_chars* limit scales with *depth_level*: at depth 1 the cap
    is the base ``max_chars``; at depth 10 it is 3× that value, so the
    LLM receives more source material for higher detail levels.
    """
    depth_factor = 1.0 + (max(1, min(10, depth_level)) - 1) * 0.22
    effective_limit = int(max_chars * depth_factor)
    parts: list[str] = []
    total = 0
    for c in chunks:
        header = f"[Chunk {c.id} | document={c.document_id} | pos={c.position}]"
        body = c.content
        if total + len(header) + len(body) > effective_limit:
            remaining = effective_limit - total
            if remaining > 200:
                body = body[: remaining - 20] + "\n...[truncated]"
                parts.append(f"{header}\n{body}")
            break
        parts.append(f"{header}\n{body}")
        total += len(header) + len(body)
    return "\n\n".join(parts)


# ── Prompt builders ──────────────────────────────────────────────────────────


def build_topic_prompt(
    chunks: list[Chunk],
    syllabus_text: str | None,
) -> str:
    """Build a prompt that asks the LLM to identify topics from *chunks*.

    When *syllabus_text* is provided the model is instructed to align
    topics to the syllabus entries and set ``order_source="syllabus"``
    with a ``syllabus_position``.  When *syllabus_text* is ``None`` the
    model must determine a pedagogical ordering
    (``order_source="pedagogical"``): prerequisites first, then advanced
    topics.

    Returns
    -------
    str
        The full prompt string ready to be sent to ``OllamaClient.generate``.
    """
    chunks_block = _chunks_text(chunks)

    if syllabus_text:
        order_section = (
            "A syllabus is provided below.  You MUST align the identified "
            "topics to the syllabus entries as closely as possible:\n\n"
            "<syllabus>\n"
            f"{syllabus_text}\n"
            "</syllabus>\n\n"
            "For each topic, determine:\n"
            "- 'order_source': always set to \"syllabus\"\n"
            "- 'syllabus_position': the 0-based index of the syllabus entry "
            "this topic best matches.  If a topic does not match any "
            "syllabus entry, set syllabus_position to null."
        )
    else:
        order_section = (
            "No syllabus has been provided.  You must determine a "
            "pedagogically coherent ordering for the topics:\n\n"
            "- Prerequisite topics must appear before topics that depend on "
            "them.\n"
            "- Foundational / introductory concepts must come before advanced "
            "treatments.\n\n"
            "For each topic, determine:\n"
            "- 'order_source': always set to \"pedagogical\"\n"
            "- 'syllabus_position': always set to null"
        )

    return (
        f"You are an expert academic planner.  Analyze the following "
        f"source material chunks and identify all distinct topics "
        f"(subjects / themes) they cover.\n\n"
        f"{_ALWAYS_ENGLISH}\n\n"
        f"{order_section}\n\n"
        f"Source material chunks:\n\n"
        f"{chunks_block}\n\n"
        f"Return a JSON object with the key \"topics\" containing a list "
        f"of objects, each with exactly these fields:\n"
        f"- \"name\" (string): short, human-readable topic name.\n"
        f"- \"description\" (string): brief summary of the topic.\n"
        f"- \"related_documents\" (list of strings): document IDs that "
        f"discuss this topic.\n"
        f"- \"chunk_ids\" (list of strings): the Chunk IDs from the "
        f"source material above that are relevant to this topic.\n"
        f"- \"subtopic_count\" (integer >= 0): number of distinct "
        f"sub-topics or key ideas within this topic.\n"
        f"- \"order_source\" (string): either \"syllabus\" or "
        f"\"pedagogical\".\n"
        f"- \"syllabus_position\" (integer or null): position in the "
        f"syllabus, or null.\n\n"
        f"Output ONLY the JSON object, no markdown fences, no commentary."
    )


def build_outline_prompt() -> str:
    """Build a prompt that asks the LLM to produce a chapter/subchapter outline.

    The topics passed to the model already carry their ordering (from
    ``build_topic_prompt``).  This prompt must **not** re-order them —
    it should respect the existing sequence and flesh out a hierarchical
    outline (chapters → sections → sub-sections).

    Returns
    -------
    str
        The full prompt string.
    """
    return (
        "You are an expert academic editor.  You are given a list of "
        "topics, each already ordered (either by syllabus position or "
        "pedagogical sequence).  Your task is to produce a structured "
        "chapter/sub-chapter outline for a study manual.\n\n"
        f"{_ALWAYS_ENGLISH}\n\n"
        "IMPORTANT: Do NOT reorder the topics.  Respect the sequence as "
        "given.  You may group related topics into the same chapter if "
        "that makes pedagogical sense, but the relative order of topics "
        "must remain unchanged.\n\n"
        "For each chapter, list its title and the titles of its "
        "sub-sections (if any).\n\n"
        "Return a JSON object with the key \"chapters\" containing a list "
        "of objects, each with:\n"
        "- \"title\" (string): the chapter heading.\n"
        "- \"sections\" (list of strings): the sub-section headings "
        "within this chapter (may be an empty list).\n\n"
        "Output ONLY the JSON object, no markdown fences, no commentary."
    )


def build_chapter_prompt(
    topic: Topic,
    chunks: list[Chunk],
    depth_level: int,
) -> str:
    """Build a prompt to generate the Markdown content for a single chapter.

    Parameters
    ----------
    topic:
        The ``Topic`` this chapter covers.
    chunks:
        Source material chunks relevant to this topic.
    depth_level:
        Detail level 1-10.  Translated to a textual instruction via
        ``_depth_instruction``.

    Returns
    -------
    str
        The full prompt string.
    """
    depth_text = _depth_instruction(depth_level)
    chunks_block = _chunks_text(chunks, depth_level=depth_level)

    return (
        f"You are an expert academic author writing a university-level "
        f"study manual. Write a comprehensive chapter on the following topic.\n\n"
        f"{_ALWAYS_ENGLISH}\n\n"
        f"## CRITICAL WRITING RULES\n\n"
        f"1. **Do NOT copy or paste** from the source material below. You "
        f"must **synthesize** the information into original, flowing prose.\n"
        f"2. **Never start a paragraph with a fragment or a bullet list** "
        f"from the source. Rewrite everything in complete, grammatically "
        f"correct sentences.\n"
        f"3. **Connect ideas logically.** Use transitional phrases (e.g. "
        f"'Consequently,', 'This implies that', 'Building on the previous "
        f"concept,'). Each section must flow naturally into the next.\n"
        f"4. **Adopt a formal academic tone.** Write as if this were a "
        f"textbook chapter, not a set of lecture notes. Avoid colloquial "
        f"language, parenthetical asides, and informal abbreviations.\n"
        f"5. **Integrate all sources.** The chunks below come from multiple "
        f"documents. Merge overlapping explanations, resolve contradictions, "
        f"and present a single authoritative version.\n"
        f"6. **Use the third person or impersonal constructions** ('It can "
        f"be shown that', 'The theory states') rather than addressing the "
        f"reader directly.\n\n"
        f"## Topic\n\n"
        f"**{topic.name}** — {topic.description}\n\n"
        f"## Detail level\n\n"
        f"{depth_text}\n\n"
        f"The chapter length must scale naturally with the number and size "
        f"of the source chunks — more material means a longer chapter. "
        f"Do NOT impose a fixed word count.\n\n"
        f"## Required structure\n\n"
        f"The chapter must include:\n"
        f"- A clear introduction motivating the topic.\n"
        f"- Core theoretical explanations written in continuous prose.\n"
        f"- Rigorous treatment of sub-topics, each building on the previous.\n"
        f"- Worked examples or concrete applications where appropriate.\n"
        f"- Discussion of common exam pitfalls or frequently tested points.\n"
        f"- A brief concluding paragraph summarizing the key takeaways.\n\n"
        f"## Source material\n\n"
        f"The following chunks contain the raw material you must synthesize. "
        f"Use them as factual input — but the output must be entirely your "
        f"own prose, restructured for clarity and coherence.\n\n"
        f"{chunks_block}\n\n"
        f"Write the chapter in Markdown. Use ## for the chapter title and "
        f"### for sub-sections. Do NOT include a table of contents or "
        f"index — these are built programmatically elsewhere.\n\n"
        f"Return a JSON object with:\n"
        f"- \"title\" (string): the chapter title.\n"
        f"- \"content\" (string): the full Markdown body of the chapter.\n"
        f"- \"sections\" (list of strings): the titles of each ## or ### "
        f"heading in the content, in order.\n\n"
        f"Output ONLY the JSON object, no markdown fences, no commentary."
    )


def build_focus_topic_prompt(
    focus_topic: str,
    chunks: list[Chunk],
    depth_level: int,
) -> str:
    """Build a prompt for ``scope="topic"`` mode.

    When the user has chosen to focus on a single topic, this prompt
    instructs the LLM to concentrate exclusively on *focus_topic*,
    using only the relevant chunks (which may come from multiple
    documents/topics that were automatically detected as pertinent),
    and explicitly ignoring all other material.

    Parameters
    ----------
    focus_topic:
        The topic the user wants to study.
    chunks:
        Pre-filtered chunks relevant to *focus_topic*.
    depth_level:
        Detail level 1-10 (typically ``generation.focus_depth_level``).

    Returns
    -------
    str
        The full prompt string.
    """
    depth_text = _depth_instruction(depth_level)
    chunks_block = _chunks_text(chunks, depth_level=depth_level)

    return (
        f"You are an expert academic author. The user is studying a "
        f"single specific topic and wants a focused, in-depth chapter "
        f"written at university level.\n\n"
        f"{_ALWAYS_ENGLISH}\n\n"
        f"## CRITICAL WRITING RULES\n\n"
        f"1. **Do NOT copy or paste** from the source material below. "
        f"**Synthesize** all information into original, flowing prose.\n"
        f"2. **Never start a paragraph with a fragment or a bullet list** "
        f"from the source. Rewrite everything in complete sentences.\n"
        f"3. **Connect ideas logically.** Use transitional phrases. Each "
        f"section must flow naturally into the next.\n"
        f"4. **Adopt a formal academic tone** — textbook style, not "
        f"lecture notes.\n"
        f"5. **Integrate all sources.** Merge overlapping explanations and "
        f"present a single authoritative account.\n\n"
        f"## Focus topic\n\n"
        f"**{focus_topic}**\n\n"
        f"## Detail level\n\n"
        f"{depth_text}\n\n"
        f"IMPORTANT: Concentrate ONLY on the focus topic above. The "
        f"source chunks may contain mentions of other, unrelated topics — "
        f"you must IGNORE those entirely. Every paragraph must be directly "
        f"relevant to the focus topic.\n\n"
        f"The chapter length must scale naturally with the number and size "
        f"of the source chunks — more material means a longer chapter. "
        f"Do NOT impose a fixed word count.\n\n"
        f"## Required structure\n\n"
        f"The chapter must include:\n"
        f"- A clear introduction motivating the topic.\n"
        f"- Core theoretical explanations written in continuous prose.\n"
        f"- Rigorous treatment of sub-topics.\n"
        f"- Worked examples or concrete applications where appropriate.\n"
        f"- Discussion of common exam pitfalls.\n"
        f"- A brief concluding paragraph.\n\n"
        f"## Source material\n\n"
        f"The following chunks are pre-filtered for the focus topic. "
        f"Use them as factual input — but the output must be entirely your "
        f"own prose.\n\n"
        f"{chunks_block}\n\n"
        f"Write the chapter in Markdown. Use ## for the chapter title and "
        f"### for sub-sections. Do NOT include a table of contents or "
        f"index — these are built programmatically elsewhere.\n\n"
        f"Return a JSON object with:\n"
        f"- \"title\" (string): the chapter title (must match the focus "
        f"topic).\n"
        f"- \"content\" (string): the full Markdown body of the chapter.\n"
        f"- \"sections\" (list of strings): the titles of each ## or ### "
        f"heading in the content, in order.\n\n"
        f"Output ONLY the JSON object, no markdown fences, no commentary."
    )


def build_image_caption_prompt(image_context: str) -> str:
    """Build a prompt asking the vision model to describe an image.

    Parameters
    ----------
    image_context:
        Optional surrounding text (e.g. page text or slide notes) to
        give the model context about where the image came from.

    Returns
    -------
    str
        The full prompt string for ``OllamaClient.generate_with_image``.
    """
    ctx = (
        f"\n\nSurrounding context from the source document:\n"
        f"{image_context}"
        if image_context
        else ""
    )
    return (
        "Describe this image in 1-2 concise sentences in English.  "
        "Explain what it shows and why it might be educationally "
        "relevant for a university study manual."
        f"{ctx}"
    )


def build_image_relevance_prompt(
    chapter_content: str,
    image_description: str,
) -> str:
    """Build a prompt that asks whether an image is relevant to a chapter.

    Parameters
    ----------
    chapter_content:
        The Markdown body of the chapter to evaluate against.
    image_description:
        The vision model's description of the image.

    Returns
    -------
    str
        The full prompt string.
    """
    return (
        "You are an academic editor deciding whether to include an "
        "image in a chapter of a study manual.\n\n"
        f"{_ALWAYS_ENGLISH}\n\n"
        f"Chapter content (excerpt):\n"
        f"---\n{chapter_content}\n---\n\n"
        f"Image description:\n{image_description}\n\n"
        "Decide:\n"
        "1. Is this image relevant to the chapter?  (yes / no)\n"
        "2. If yes, where in the chapter should it be placed?  Use one "
        "of: \"after introduction\", \"after section: <title>\", or "
        "\"before conclusion\".\n\n"
        'Return a JSON object with keys "relevant" (boolean) and '
        '"placement" (string or null).\n'
        "Output ONLY the JSON object, no markdown fences, no commentary."
    )
