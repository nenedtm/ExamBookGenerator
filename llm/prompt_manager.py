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

from core.models import Chunk, OutlineChapter, Topic
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_ALWAYS_ENGLISH = (
    "Always write the output in English, regardless of the language of "
    "the source material."
)

_DEPTH_MAP: dict[tuple[int, int], str] = {
    (1, 2): (
        "Write a concise, high-yield revision summary. Cover every topic "
        "and sub-topic, but keep each explanation to 1-3 sentences. Use "
        "precise definitions and key results only. No derivations, no "
        "examples. The output should be a dense, scannable reference — "
        "every line must carry information."
    ),
    (3, 4): (
        "Write a clear, moderately detailed explanation at university level. "
        "Each concept should be defined, followed by a brief justification "
        "or intuition. Include one short example per sub-topic where it "
        "aids understanding. Use prose, not bullet lists. The chapter "
        "should read as unified notes, not as a collection of fragments."
    ),
    (5, 6): (
        "Write a thorough, well-structured university-level chapter. For "
        "every concept: provide a formal definition, state the relevant "
        "theorem or principle, give at least one worked example, and "
        "discuss when or why the result applies. Use continuous prose "
        "with proper academic style. Include derivations where they "
        "clarify the logic. Mention at least two common exam pitfalls or "
        "frequently tested traps per major topic."
    ),
    (7, 8): (
        "Write an exhaustive, textbook-quality university chapter. Every "
        "concept must receive: (1) a precise definition, (2) the full "
        "derivation or proof, (3) at least one worked example, (4) "
        "connections to related topics, and (5) discussion of edge cases "
        "or common misconceptions. Anticipate exam questions and address "
        "them proactively. Use formal academic tone throughout. Cross-"
        "reference concepts within the chapter. Every paragraph must "
        "synthesize the source material into original prose — never copy "
        "verbatim."
    ),
    (9, 10): (
        "Write a comprehensive, publication-quality university treatise. "
        "Cover every nuance, every proof variant, every edge case present "
        "in the source material. For each major result: state it formally, "
        "prove it, provide multiple examples (at least one elementary, one "
        "advanced), discuss its limitations, and explain how it connects "
        "to the broader theory. Include historical context where relevant. "
        "Anticipate and preemptively address every common student "
        "misconception. Use precise technical language. Every section "
        "must be dense with information — no filler, no padding, no "
        "vague transitional paragraphs."
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
    max_chars: int = 120_000,
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
    for idx, c in enumerate(chunks, 1):
        body = c.content.strip()
        # Skip chunks that are too short or look like garbage
        if len(body) < 20:
            continue
        header = f"--- Source {idx} ---"
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
    chunks_block = _chunks_text(chunks, max_chars=100_000)

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
        f"You are an expert academic planner.  Your task is to perform a "
        f"thorough analysis of the source material below and identify "
        f"EVERY distinct, self-contained topic it covers.\n\n"
        f"{_ALWAYS_ENGLISH}\n\n"
        f"## RULES FOR TOPIC IDENTIFICATION\n\n"
        f"1. **Be exhaustive.** Every major concept, theorem, method, or "
        f"theme that appears in the source material MUST become its own "
        f"topic entry. If a document covers three different subjects, "
        f"that should produce at least three topic entries.\n"
        f"2. **Be granular.** Do NOT merge distinct subjects into a single "
        f"vague entry like \"Mathematics\" or \"General topics\". Each "
        f"topic should be specific enough to fill one chapter in a study "
        f"manual (e.g. \"Brownian Motion and Its Properties\", not just "
        f"'Stochastic Processes').\n"
        f"3. **Avoid redundancy.** Two entries must not cover the same "
        f"material. If two concepts are deeply intertwined, merge them "
        f"into one entry. But if they are merely mentioned in the same "
        f"document while being conceptually distinct, keep them separate.\n"
        f"4. **Each topic must be self-contained.** A reader studying only "
        f"that topic should gain a complete understanding of it, without "
        f"needing to consult other chapters.\n\n"
        f"{order_section}\n\n"
        f"Source material chunks:\n\n"
        f"{chunks_block}\n\n"
        f"Return a JSON object with the key \"topics\" containing a list "
        f"of objects, each with exactly these fields:\n"
        f"- \"name\" (string): a precise, descriptive topic name "
        f"(3-8 words, no abbreviations).\n"
        f"- \"description\" (string): a 1-2 sentence summary of what the "
        f"topic covers and why it matters.\n"
        f"- \"related_documents\" (list of strings): document IDs that "
        f"discuss this topic.\n"
        f"- \"chunk_ids\" (list of strings): the Chunk IDs from the "
        f"source material above that are relevant to this topic.\n"
        f"- \"subtopic_count\" (integer >= 1): the number of distinct "
        f"sub-topics or key ideas within this topic. This MUST be >= 1 "
        f"for every topic (a topic with 0 subtopics is invalid).\n"
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
        "You are an expert academic editor designing a university study "
        "manual.  You are given a list of topics, each already ordered "
        "(either by syllabus position or pedagogical sequence).  Your "
        "task is to produce a DETAILED, HIERARCHICAL chapter outline.\n\n"
        f"{_ALWAYS_ENGLISH}\n\n"
        "## RULES\n\n"
        "1. **Do NOT reorder the topics.** Respect the sequence as given. "
        "You may group related topics into the same chapter if that makes "
        "pedagogical sense, but the relative order must remain unchanged.\n"
        "2. **Every topic MUST appear in the outline.** No topic may be "
        "silently dropped.\n"
        "3. **Every chapter must have at least 3 sub-sections.** A chapter "
        "with only a title and no sections is not acceptable.\n"
        "4. **Sub-sections must be specific and descriptive.** Instead of "
        "\"Overview\", write \"Definition and Fundamental Properties\". "
        "Instead of \"Details\", write \"Derivation of the Master Equation\".\n"
        "5. **Group logically.** If two topics are deeply intertwined, "
        "combine them into one chapter with distinct sections. If a topic "
        "is large enough, it can be its own chapter.\n"
        "6. **Aim for completeness.** The outline should be detailed enough "
        "that a reader could understand the scope of the manual from the "
        "TOC alone, without reading the text.\n\n"
        "Return a JSON object with the key \"chapters\" containing a list "
        "of objects, each with:\n"
        "- \"title\" (string): the chapter heading (concise but descriptive).\n"
        "- \"sections\" (list of strings): the sub-section headings within "
        "this chapter. Each entry is a string. Minimum 3 entries per "
        "chapter.\n\n"
        "Output ONLY the JSON object, no markdown fences, no commentary."
    )


def build_chapter_prompt(
    topic: Topic,
    chunks: list[Chunk],
    depth_level: int,
    outline_chapter: OutlineChapter | None = None,
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
    outline_chapter:
        When provided, the LLM must use the given title and section
        structure exactly.  This ensures the generated chapter matches
        the pre-planned outline.

    Returns
    -------
    str
        The full prompt string.
    """
    depth_text = _depth_instruction(depth_level)
    chunks_block = _chunks_text(chunks, depth_level=depth_level)

    # ── Outline structure section ──────────────────────────────────────
    outline_section = ""
    if outline_chapter:
        section_list = "\n".join(
            f"  {i+1}. {s}" for i, s in enumerate(outline_chapter.sections)
        )
        outline_section = (
            f"## REQUIRED CHAPTER STRUCTURE\n\n"
            f"You MUST write the chapter with EXACTLY this title and "
            f"these section headings.  Do NOT invent new headings, "
            f"do NOT rename them, do NOT omit any.  Each section must "
            f"contain substantive prose (minimum 3-4 paragraphs per "
            f"section).\n\n"
            f"**Chapter title:** {outline_chapter.title}\n\n"
            f"**Sections (in order):**\n{section_list}\n\n"
        )

    return (
        f"You are an expert academic author writing a university-level "
        f"study manual. Write a comprehensive, DETAILED chapter on the "
        f"following topic.\n\n"
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
        f"reader directly.\n"
        f"7. **Be exhaustive.** Every section must contain substantial "
        f"content — at least 3-4 substantial paragraphs. A section with "
        f"only 1-2 sentences is unacceptable. Elaborate, give examples, "
        f"provide context.\n"
        f"8. **Include worked examples.** For every major concept or "
        f"formula, provide at least one concrete, fully worked example "
        f"with step-by-step reasoning.\n\n"
        f"## Topic\n\n"
        f"**{topic.name}** — {topic.description}\n\n"
        f"{outline_section}"
        f"## Detail level\n\n"
        f"{depth_text}\n\n"
        f"The chapter length must scale naturally with the number and size "
        f"of the source chunks — more material means a longer chapter. "
        f"Do NOT impose a fixed word count. A typical chapter at depth 7+ "
        f"should be 3000-8000 words. Shorter chapters are acceptable only "
        f"if the source material is genuinely limited.\n\n"
        f"## Required structure\n\n"
        f"The chapter must include:\n"
        f"- A clear introduction motivating the topic and outlining what "
        f"will be covered.\n"
        f"- Core theoretical explanations written in continuous prose, "
        f"with formal definitions stated precisely.\n"
        f"- Rigorous treatment of sub-topics, each building on the previous "
        f"one. Every sub-topic must receive its own section.\n"
        f"- At least one worked example per major concept.\n"
        f"- Discussion of common exam pitfalls or frequently tested points.\n"
        f"- Connections to related topics where appropriate.\n"
        f"- A brief concluding paragraph summarizing the key takeaways.\n\n"
        f"## Source material\n\n"
        f"The following chunks contain the raw material you must synthesize. "
        f"Use them as factual input — but the output must be entirely your "
        f"own prose, restructured for clarity and coherence.\n\n"
        f"{chunks_block}\n\n"
        f"Write the chapter in Markdown. Use ## for the chapter title and "
        f"### for sub-sections. Do NOT include a table of contents or "
        f"index — these are built programmatically elsewhere.\n\n"
        f"CRITICAL: Every section heading (## and ###) MUST be followed by "
        f"at least 3-4 substantial paragraphs of body text. Never leave a "
        f"heading without content below it. If you include a heading, you "
        f"must write substantive, detailed text for that section.\n\n"
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
        f"present a single authoritative account.\n"
        f"6. **Be exhaustive.** Every section must contain at least 3-4 "
        f"substantial paragraphs. Elaborate, give examples, provide context.\n"
        f"7. **Include worked examples.** For every major concept, provide "
        f"at least one concrete, fully worked example.\n\n"
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
        f"Do NOT impose a fixed word count. A typical chapter at depth 7+ "
        f"should be 3000-8000 words.\n\n"
        f"## Required structure\n\n"
        f"The chapter must include:\n"
        f"- A clear introduction motivating the topic.\n"
        f"- Core theoretical explanations written in continuous prose, "
        f"with formal definitions.\n"
        f"- Rigorous treatment of sub-topics, each with its own section.\n"
        f"- At least one worked example per major concept.\n"
        f"- Discussion of common exam pitfalls.\n"
        f"- Connections to related topics where appropriate.\n"
        f"- A brief concluding paragraph.\n\n"
        f"## Source material\n\n"
        f"The following chunks are pre-filtered for the focus topic. "
        f"Use them as factual input — but the output must be entirely your "
        f"own prose.\n\n"
        f"{chunks_block}\n\n"
        f"Write the chapter in Markdown. Use ## for the chapter title and "
        f"### for sub-sections. Do NOT include a table of contents or "
        f"index — these are built programmatically elsewhere.\n\n"
        f"CRITICAL: Every section heading (## and ###) MUST be followed by "
        f"at least 3-4 substantial paragraphs of body text. Never leave a "
        f"heading without content below it.\n\n"
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


def build_syllabus_extraction_prompt(syllabus_text: str) -> str:
    """Build a prompt that extracts a structured topic list from syllabus prose.

    The syllabus may be a bullet-point list, a numbered outline, or a
    free-form prose description of the course content.  This prompt
    asks the LLM to extract every distinct teachable topic in the order
    it appears in the text.

    Parameters
    ----------
    syllabus_text:
        The raw text of the syllabus / programma document.

    Returns
    -------
    str
        The full prompt string.
    """
    return (
        "You are an academic planner.  The text below is a course "
        "syllabus (programma).  It may be a structured list, a numbered "
        "outline, or a free-form prose description of everything covered "
        "in the course.\n\n"
        f"{_ALWAYS_ENGLISH}\n\n"
        "Your task: extract every distinct teachable topic from this "
        "syllabus.  A topic is a self-contained subject that could "
        "occupy one chapter in a study manual.\n\n"
        "Rules:\n"
        "- Preserve the original order from the syllabus.\n"
        "- If a paragraph describes multiple sub-topics, split them into "
        "separate entries.\n"
        "- Do NOT merge distinct topics into one entry.\n"
        "- Do NOT omit any topic, no matter how briefly mentioned.\n"
        "- Each entry should be a short topic name (2-8 words).\n\n"
        "Syllabus text:\n\n"
        f"---\n{syllabus_text}\n---\n\n"
        "Return a JSON object with the key \"topics\" containing a list "
        "of strings, each being one topic name.\n\n"
        "Output ONLY the JSON object, no markdown fences, no commentary."
    )
