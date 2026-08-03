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

_LECTURE_FORMATTING = (
    "## LECTURE-STYLE FORMATTING\n\n"
    "Write the chapter in the style of readable Obsidian lecture notes, "
    "structuring the content with callout blocks. Use EXACTLY these "
    "callout types:\n\n"
    "> [!formula] Formula Name — every formula, definition, or result "
    "worth memorising, with LaTeX ($$...$$) inside the callout.\n"
    "> [!theorem] Theorem Name — every theorem, including its hypotheses.\n"
    "> [!proof] Proof — Name — every proof, with no skipped steps.\n"
    "> [!exercise] Title — worked exercises, step by step.\n"
    "> [!question] Title — exam-style questions.\n"
    "> [!review] Note — common pitfalls, misconceptions, and review "
    "reminders.\n"
    "> [!recap] 🟡 Lecture Recap — chapter-level summary / key takeaways.\n\n"
    "Rules:\n"
    "- Wrap EVERY formula or definition in a `> [!formula]` callout.\n"
    "- Wrap EVERY theorem in a `> [!theorem]` callout and its proof in a "
    "`> [!proof]` callout.\n"
    "- Present each worked example inside a `> [!exercise]` callout.\n"
    "- Present each exam question inside a `> [!question]` callout, "
    "immediately followed by its answer in normal prose.\n"
    "- Use `> [!review]` for exam traps, common errors, and misconceptions.\n"
    "- End the chapter with a `> [!recap]` callout summarising the key "
    "takeaways.\n"
    "- Do NOT nest callouts and NEVER leave a callout empty — always fill "
    "it with substantive content.\n"
    "- Keep flowing prose between callouts; callouts highlight key elements, "
    "they do not replace the text.\n\n"
)

_DEPTH_MAP: dict[tuple[int, int], str] = {
    (1, 2): (
        "Write a concise, high-yield revision summary suitable for last-minute "
        "exam preparation. For EVERY topic and sub-topic:\n"
        "- Provide a precise, textbook-quality definition in 1-2 sentences.\n"
        "- State the key formula, theorem, or result in its exact form.\n"
        "- List the essential properties or conditions in compact bullet points.\n"
        "- Include ONE minimal worked example per major concept (3-5 sentences max).\n"
        "- Highlight the single most common exam trap per topic.\n"
        "- Use cross-references: 'See also [Topic Name]' for related concepts.\n"
        "The output should be a dense, scannable reference sheet — every line "
        "must carry information. No filler, no narrative, no motivation. "
        "Format: use ## for topic headings, ### for sub-topics, and bold for "
        "key terms. Minimum 2-3 paragraphs per ## section even at this depth."
    ),
    (3, 4): (
        "Write a clear, well-organized university-level chapter. For EVERY "
        "concept:\n"
        "- Provide a formal definition with precise terminology.\n"
        "- Explain the intuition behind the definition in 2-3 sentences.\n"
        "- State the relevant theorem or principle with its hypotheses.\n"
        "- Give ONE fully worked example with step-by-step reasoning.\n"
        "- Discuss when the result applies and when it fails (brief limitations).\n"
        "- Mention at least one common misconception students have.\n"
        "Write in continuous prose with proper academic style — not bullet "
        "lists. Use transitional phrases between paragraphs. Each section "
        "should read as unified notes, building logically from definitions "
        "to applications. Include brief connections to other topics where "
        "relevant. Minimum 3-4 paragraphs per ### subsection."
    ),
    (5, 6): (
        "Write a thorough, comprehensive university-level chapter with full "
        "academic rigor. For EVERY concept:\n"
        "- Provide a precise formal definition.\n"
        "- State the complete theorem with ALL hypotheses and conclusions.\n"
        "- Provide a FULL derivation or proof — do not skip steps.\n"
        "- Give at least TWO worked examples: one elementary, one intermediate.\n"
        "- Discuss edge cases, boundary conditions, and degenerate cases.\n"
        "- Explain common pitfalls and frequently tested exam traps.\n"
        "- Connect to at least two other concepts in the manual.\n"
        "Write in formal academic prose with proper mathematical notation. "
        "Every paragraph must synthesize source material into original writing. "
        "Include motivation before each major result. Discuss historical context "
        "where it aids understanding. Address potential counterexamples and "
        "explain why they don't invalidate the result. Minimum 4-5 paragraphs "
        "per ### subsection. Each major theorem deserves its own paragraph."
    ),
    (7, 8): (
        "Write an exhaustive, textbook-quality university chapter of the "
        "highest academic standard. For EVERY concept, theorem, and method:\n"
        "- Provide a precise, rigorous definition with all necessary conditions.\n"
        "- State the COMPLETE theorem or principle with full hypotheses.\n"
        "- Provide the FULL derivation or proof with every step explained.\n"
        "- Give at least TWO fully worked examples (one elementary, one advanced).\n"
        "- Discuss ALL edge cases, degenerate cases, and boundary conditions.\n"
        "- Anticipate exam questions and answer them proactively.\n"
        "- Address common student misconceptions and explain why they are wrong.\n"
        "- Cross-reference at least 3 related concepts within the chapter.\n"
        "- Discuss limitations, open problems, or active research directions.\n"
        "- Provide intuitive explanations alongside formal proofs.\n"
        "Write in formal academic prose with precise technical language. Every "
        "paragraph must synthesize source material into original, authoritative "
        "prose — never copy verbatim. Include motivation before each result. "
        "Use examples to illustrate abstract concepts. Discuss when and why "
        "results fail. Compare alternative approaches where relevant. "
        "Minimum 5-6 paragraphs per ### subsection. Major theorems deserve "
        "dedicated sections with full proofs."
    ),
    (9, 10): (
        "Write an exhaustive, exam-preparation-grade university manual "
        "chapter that serves as the definitive study reference for this "
        "topic. The chapter MUST combine two complementary formats:\n\n"
        "=== PART A: DETAILED EXPOSITION (50-60% of the chapter) ===\n"
        "For EVERY concept, theorem, method, and result:\n"
        "- Provide a precise, rigorous definition with ALL necessary and "
        "sufficient conditions.\n"
        "- Mark important terms with **Key Word:** bold markers, followed "
        "by a concise definition.\n"
        "- State COMPLETE theorems/principles with full hypotheses.\n"
        "- Provide FULL derivations or proofs with every step justified.\n"
        "- Give at least THREE worked examples: elementary, intermediate, "
        "and advanced/application-level.\n"
        "- Use **Example** callout blocks for concrete illustrations.\n"
        "- Use **Note** callout blocks for caveats, clarifications, and "
        "non-obvious implications.\n"
        "- Discuss ALL edge cases, boundary conditions, and degenerate "
        "limits.\n"
        "- Anticipate and preemptively address common student "
        "misconceptions.\n"
        "- When comparing two or more concepts, use markdown comparison "
        "tables with columns for each concept and rows for each feature.\n"
        "- Number all step-by-step procedures (e.g., '1. First, ... 2. "
        "Then, ...').\n"
        "- Discuss limitations, practical implementation concerns, and "
        "when/why results fail.\n"
        "- Cross-reference at least 3 related concepts within the manual.\n\n"
        "=== PART B: TASK-BASED EXAM Q&A (30-40% of the chapter) ===\n"
        "After the detailed exposition, include a dedicated exam-preparation "
        "section structured as follows:\n"
        "- Group questions into logical Tasks (e.g., 'Task 1 - [Subtopic]', "
        "'Task 2 - [Subtopic]').\n"
        "- Each Task contains multiple numbered questions (Q1.1, Q1.2, ... "
        "Q2.1, Q2.2, ...).\n"
        "- Each question is a specific, exam-style question that tests "
        "deep understanding.\n"
        "- Each question is immediately followed by a comprehensive, "
        "detailed answer in prose (not just bullet points).\n"
        "- Answers should reference concepts from the exposition above "
        "and synthesize across multiple sub-topics.\n"
        "- Include at least 6-10 questions per Task, depending on the "
        "breadth of the subtopic.\n"
        "- Include at least one Task dedicated to similarities, "
        "differences, or comparisons between related concepts.\n\n"
        "=== PART C: SUM-UP (10% of the chapter) ===\n"
        "End the chapter with a structured Sum-up section:\n"
        "- Organize by sub-topic (use subheadings for each major area).\n"
        "- Each sub-topic contains a comprehensive bullet-point review.\n"
        "- Every bullet point should be a complete, self-contained fact "
        "that could appear on an exam.\n"
        "- Include cross-references and connections between sub-topics.\n"
        "- If applicable, end with a comparison table summarizing key "
        "differences between related systems/concepts.\n\n"
        "=== FORMATTING RULES ===\n"
        "- Use ## for the chapter title, ### for major sections, #### "
        "for sub-sections.\n"
        "- Use bold **Key Word:** markers for defining terminology.\n"
        "- Use > [!exercise] callouts for illustrative examples.\n"
        "- Use > [!review] callouts for important caveats.\n"
        "- Use markdown tables for structured comparisons.\n"
        "- Use numbered lists for step-by-step procedures.\n"
        "- Use bullet lists for enumerating properties, features, or "
        "characteristics.\n"
        "- Write in formal academic prose with precise technical language.\n"
        "- Minimum 4-5 paragraphs per detailed exposition section.\n"
        "- Minimum 2-3 paragraphs per exam Q&A answer.\n"
        "- Every section must be dense with information — no filler, "
        "no padding."
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
    max_chars: int = 150_000,
    depth_level: int = 5,
) -> str:
    """Serialize a list of chunks into a readable text block for the prompt.

    The *max_chars* limit scales with *depth_level*: at depth 1 the cap
    is the base ``max_chars``; at depth 10 it is 4× that value, so the
    LLM receives more source material for higher detail levels.
    """
    depth_factor = 1.0 + (max(1, min(10, depth_level)) - 1) * 0.33
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
            "A syllabus is provided below.  You MUST:\n"
            "1. Align the identified topics to the syllabus entries as "
            "closely as possible — each syllabus entry should produce "
            "at least one topic.\n"
            "2. **Strictly preserve the order from the syllabus.** The "
            "topics you identify MUST appear in the EXACT SAME ORDER as "
            "the corresponding syllabus entries. Do NOT reorder, sort, "
            "or rearrange topics based on your own pedagogical judgment.\n"
            "3. If a single syllabus entry covers multiple distinct "
            "sub-topics, split them into separate topic entries, but "
            "maintain the relative order of the parent syllabus entry.\n\n"
            "<syllabus>\n"
            f"{syllabus_text}\n"
            "</syllabus>\n\n"
            "For each topic, determine:\n"
            "- 'order_source': always set to \"syllabus\"\n"
            "- 'syllabus_position': the 0-based index of the syllabus entry "
            "this topic best matches.  If a topic does not match any "
            "syllabus entry, set syllabus_position to null.\n\n"
            "CRITICAL: The output order of topics MUST match the syllabus "
            "order. This is the most important rule — the generated manual "
            "will follow this exact sequence."
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
        "## CRITICAL RULES — VIOLATION = FAILURE\n\n"
        "1. **DO NOT REORDER THE TOPICS.** The topics are listed below in "
        "their EXACT required order. You MUST preserve this order. Topics "
        "must appear in chapters in the same sequence as the numbered list. "
        "You may group CONSECUTIVE related topics into one chapter, but you "
        "must NEVER move a topic before or after topics it was not adjacent to.\n"
        "2. **EVERY TOPIC MUST APPEAR.** Check each topic off as you include it. "
        "If a topic is missing, the output is invalid.\n"
        "3. **EVERY CHAPTER MUST HAVE 4-6 SUB-SECTIONS.** A chapter with fewer "
        "than 4 sections is invalid.\n"
        "4. **Sub-sections must be specific and descriptive.** Instead of "
        "\"Overview\", write \"Definition and Fundamental Properties\". "
        "Instead of \"Details\", write \"Derivation of the Master Equation\". "
        "Instead of \"Examples\", write \"Worked Example: Solving the Heat "
        "Equation with Boundary Conditions\".\n"
        "5. **Group logically.** If two topics are deeply intertwined, "
        "combine them into one chapter with distinct sections. If a topic "
        "is large enough, it can be its own chapter.\n"
        "6. **Aim for completeness.** The outline should be detailed enough "
        "that a reader could understand the scope of the manual from the "
        "TOC alone, without reading the text.\n"
        "7. **Each section should be exam-relevant.** Sections should "
        "address concepts, theorems, methods, or applications that are "
        "commonly tested. Avoid vague or generic section titles.\n"
        "8. **Include practical sections.** Each chapter should have at "
        "least one section dedicated to worked examples, and at least one "
        "section addressing common pitfalls or exam strategies.\n\n"
        "## OUTPUT FORMAT\n\n"
        "Return a JSON object with the key \"chapters\" containing a list "
        "of objects, each with:\n"
        "- \"title\" (string): the chapter heading (concise but descriptive).\n"
        "- \"sections\" (list of strings): the sub-section headings within "
        "this chapter. Each entry is a string. Minimum 4 entries per "
        "chapter.\n\n"
        "Example: if topics are [\"Linear Algebra\", \"Calculus\", \"Probability\"], "
        "a valid output is:\n"
        '{"chapters": [{"title": "Linear Algebra", '
        '"sections": ["Vector Spaces", "Matrix Operations", '
        '"Eigenvalues and Eigenvectors", "Linear Transformations"]}, '
        '{"title": "Calculus", '
        '"sections": ["Limits and Continuity", "Differentiation", '
        '"Integration", "Series and Sequences"]}, '
        '{"title": "Probability", '
        '"sections": ["Sample Spaces", "Random Variables", '
        '"Probability Distributions", "Expectation and Variance"]}]}\n\n'
        "Output ONLY the JSON object, no markdown fences, no commentary."
    )


def _sources_citation_section(sources: str) -> str:
    """Build a prompt section asking the LLM to cite *sources* inline.

    *sources* is a numbered Markdown list (from
    ``chapter_generator.build_sources_block``); the same numbering is
    rendered into the chapter's ``{{sources}}`` section, so inline
    ``[n]`` citations stay consistent with the final reference list.
    Returns an empty string when no sources are provided.
    """
    if not sources.strip():
        return ""
    return (
        "## SOURCES TO CITE\n\n"
        "The information in this chapter comes from the following numbered "
        "sources. Cite them inline in the text as [1], [2], ... whenever "
        "you present a definition, theorem, example, or fact taken from "
        "them (place the citation right after the relevant sentence). "
        "The chapter's reference list is generated automatically from "
        "these same sources — do NOT add your own references section at "
        "the end of the content, and do NOT invent sources beyond the "
        "ones listed below.\n\n"
        f"{sources}\n\n"
    )


def build_chapter_prompt(
    topic: Topic,
    chunks: list[Chunk],
    depth_level: int,
    outline_chapter: OutlineChapter | None = None,
    sources: str = "",
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
    sources:
        Numbered Markdown source list (one per line).  When non-empty,
        the model is told to cite these sources inline as ``[n]``.

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

    # ── Depth-specific structure template ────────────────────────────────
    depth_template = ""
    if depth_level >= 9:
        depth_template = (
            "## CHAPTER TEMPLATE (depth 9-10)\n\n"
            "Follow this EXACT structural template for the chapter:\n\n"
            "```\n"
            "## [Chapter Title]\n\n"
            "[Brief introduction: 2-3 paragraphs motivating the topic, "
            "explaining its importance, and outlining the chapter structure.]\n\n"
            "### [Subtopic 1 - Detailed Exposition]\n"
            "[4-6 paragraphs of rigorous explanation with definitions, "
            "derivations, examples.]\n"
            "[Use **Key Word:** markers for definitions.]\n"
            "[Use > [!exercise] callouts for illustrations.]\n"
            "[Use > [!review] callouts for caveats.]\n\n"
            "### [Subtopic 2 - Detailed Exposition]\n"
            "[Continue for each major subtopic...]\n\n"
            "[If comparing concepts, use a markdown table:]\n"
            "| Feature | Concept A | Concept B |\n"
            "| --- | --- | --- |\n"
            "| ... | ... | ... |\n\n"
            "### [Comparative Analysis]\n"
            "[Task N - Topic: side-by-side comparison.]\n\n"
            "---\n\n"
            "### Task 1 - [Subtopic Group A]\n\n"
            "#### Q1.1 [Exam question]\n"
            "[Detailed, comprehensive answer in 3-5 paragraphs.]\n\n"
            "#### Q1.2 [Exam question]\n"
            "[Detailed answer...]\n\n"
            "[Continue with Q1.3, Q1.4, ...]\n\n"
            "### Task 2 - [Subtopic Group B]\n\n"
            "#### Q2.1 [Exam question]\n"
            "[Detailed answer...]\n\n"
            "[Continue with more Tasks as needed...]\n\n"
            "### Task N - [Comparative Task]\n"
            "[Questions comparing and contrasting concepts from different "
            "parts of the chapter.]\n\n"
            "---\n\n"
            "### Sum-up\n\n"
            "#### [Subtopic 1 Name]\n"
            "- [Complete, self-contained bullet-point fact]\n"
            "- [Complete, self-contained bullet-point fact]\n"
            "- [Continue...]\n\n"
            "#### [Subtopic 2 Name]\n"
            "- [Complete, self-contained bullet-point fact]\n"
            "- [Continue...]\n\n"
            "[If applicable, end with a comparison table summarizing key "
            "differences.]\n"
            "```\n\n"
            "The Tasks and Q&A section should contain 30-40% of the total "
            "chapter content. The Sum-up should be 10% of the total. "
            "The remaining 50-60% is detailed exposition.\n\n"
        )
    elif depth_level >= 7:
        depth_template = (
            "## REQUIRED STRUCTURE\n\n"
            "The chapter must include:\n"
            "- A clear introduction motivating the topic and explaining "
            "its importance.\n"
            "- Core theoretical explanations with formal definitions.\n"
            "- Rigorous treatment of sub-topics, each with its own section.\n"
            "- Full derivations or proofs for major theorems.\n"
            "- At least one worked example per major concept.\n"
            "- Discussion of common exam pitfalls.\n"
            "- Connections to related topics.\n"
            "- Edge cases and limitations.\n"
            "- A concluding paragraph summarizing key takeaways.\n\n"
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
        f"content — at least 4-5 substantial paragraphs. A section with "
        f"only 1-2 sentences is unacceptable. Elaborate, give examples, "
        f"provide context, discuss implications.\n"
        f"8. **Include worked examples.** For every major concept or "
        f"formula, provide at least one concrete, fully worked example "
        f"with step-by-step reasoning. At depth 7+, provide two examples "
        f"per major concept: one elementary, one advanced.\n"
        f"9. **Anticipate exam questions.** For each major concept, identify "
        f"what an examiner would ask and address it proactively in the text.\n"
        f"10. **Cross-reference.** When discussing a concept that relates to "
        f"another topic in the manual, mention the connection explicitly.\n\n"
        f"## Topic\n\n"
        f"**{topic.name}** — {topic.description}\n\n"
        f"{outline_section}"
        f"## Detail level\n\n"
        f"{depth_text}\n\n"
        f"{depth_template}"
        f"{_LECTURE_FORMATTING}"
        f"{_sources_citation_section(sources)}"
        f"The chapter length must scale naturally with the number and size "
        f"of the source chunks — more material means a longer chapter. "
        f"Do NOT impose a fixed word count. A typical chapter at depth 5+ "
        f"should be 4000-8000 words. At depth 7+, aim for 6000-12000 words. "
        f"At depth 9-10, aim for 10000-20000 words. Shorter chapters are "
        f"acceptable only if the source material is genuinely limited.\n\n"
        f"## Source material\n\n"
        f"The following chunks contain the raw material you must synthesize. "
        f"Use them as factual input — but the output must be entirely your "
        f"own prose, restructured for clarity and coherence.\n\n"
        f"{chunks_block}\n\n"
        f"Write the chapter in Markdown. Use ## for the chapter title and "
        f"### for sub-sections. Do NOT include a table of contents or "
        f"index — these are built programmatically elsewhere.\n\n"
        f"CRITICAL: Every section heading (## and ###) MUST be followed by "
        f"at least 4-5 substantial paragraphs of body text. Never leave a "
        f"heading without content below it. If you include a heading, you "
        f"must write substantive, detailed text for that section. A section "
        f"with fewer than 4 paragraphs is incomplete.\n\n"
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
    sources: str = "",
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
    sources:
        Numbered Markdown source list.  When non-empty, the model is
        told to cite these sources inline as ``[n]``.

    Returns
    -------
    str
        The full prompt string.
    """
    depth_text = _depth_instruction(depth_level)
    chunks_block = _chunks_text(chunks, depth_level=depth_level)

    # ── Depth-specific structure template ────────────────────────────────
    focus_depth_template = ""
    if depth_level >= 9:
        focus_depth_template = (
            "## CHAPTER TEMPLATE (depth 9-10)\n\n"
            "Follow this EXACT structural template:\n\n"
            "```\n"
            "## [Chapter Title]\n\n"
            "[Brief introduction: 2-3 paragraphs motivating the topic.]\n\n"
            "### [Subtopic 1 - Detailed Exposition]\n"
            "[4-6 paragraphs with definitions, derivations, examples.]\n"
            "[Use **Key Word:** markers for definitions.]\n"
            "[Use > [!exercise] callouts for illustrations.]\n"
            "[Use > [!review] callouts for caveats.]\n\n"
            "### [Subtopic 2 - Detailed Exposition]\n"
            "[Continue for each major subtopic...]\n\n"
            "---\n\n"
            "### Task 1 - [Subtopic Group A]\n\n"
            "#### Q1.1 [Exam question]\n"
            "[Detailed answer in 3-5 paragraphs.]\n\n"
            "#### Q1.2 [Exam question]\n"
            "[Detailed answer...]\n\n"
            "### Task 2 - [Comparative Task]\n\n"
            "#### Q2.1 [Comparison question]\n"
            "[Detailed answer...]\n\n"
            "---\n\n"
            "### Sum-up\n\n"
            "#### [Subtopic 1 Name]\n"
            "- [Self-contained bullet-point fact]\n"
            "- [Self-contained bullet-point fact]\n\n"
            "#### [Subtopic 2 Name]\n"
            "- [Self-contained bullet-point fact]\n"
            "```\n\n"
            "The Tasks and Q&A section should contain 30-40% of the "
            "total chapter content. The Sum-up should be 10%. The "
            "remaining 50-60% is detailed exposition.\n\n"
        )

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
        f"6. **Be exhaustive.** Every section must contain at least 4-5 "
        f"substantial paragraphs. Elaborate, give examples, provide context, "
        f"discuss implications and limitations.\n"
        f"7. **Include worked examples.** For every major concept, provide "
        f"at least one concrete, fully worked example with step-by-step "
        f"reasoning. At depth 7+, provide two examples per concept.\n"
        f"8. **Anticipate exam questions.** Identify what an examiner would "
        f"ask and address it proactively.\n"
        f"9. **Discuss edge cases and limitations.** For every theorem or "
        f"method, explain when it fails and what the boundary conditions are.\n\n"
        f"## Focus topic\n\n"
        f"**{focus_topic}**\n\n"
        f"## Detail level\n\n"
        f"{depth_text}\n\n"
        f"{focus_depth_template}"
        f"{_LECTURE_FORMATTING}"
        f"{_sources_citation_section(sources)}"
        f"IMPORTANT: Concentrate ONLY on the focus topic above. The "
        f"source chunks may contain mentions of other, unrelated topics — "
        f"you must IGNORE those entirely. Every paragraph must be directly "
        f"relevant to the focus topic.\n\n"
        f"The chapter length must scale naturally with the number and size "
        f"of the source chunks — more material means a longer chapter. "
        f"Do NOT impose a fixed word count. A typical chapter at depth 5+ "
        f"should be 4000-8000 words. At depth 7+, aim for 6000-12000 words. "
        f"At depth 9-10, aim for 10000-20000 words.\n\n"
        f"## Required structure\n\n"
        f"The chapter must include:\n"
        f"- A clear introduction motivating the topic and explaining its "
        f"importance in the broader field.\n"
        f"- Core theoretical explanations written in continuous prose, "
        f"with formal definitions stated precisely.\n"
        f"- Full derivations or proofs for major results — never skip steps.\n"
        f"- Rigorous treatment of sub-topics, each with its own section.\n"
        f"- At least one worked example per major concept, with step-by-step "
        f"reasoning.\n"
        f"- Discussion of common exam pitfalls and frequently tested points.\n"
        f"- Edge cases, limitations, and conditions under which results fail.\n"
        f"- Connections to related topics where appropriate.\n"
        f"- A concluding paragraph summarizing key takeaways.\n\n"
        f"## Source material\n\n"
        f"The following chunks are pre-filtered for the focus topic. "
        f"Use them as factual input — but the output must be entirely your "
        f"own prose.\n\n"
        f"{chunks_block}\n\n"
        f"Write the chapter in Markdown. Use ## for the chapter title and "
        f"### for sub-sections. Do NOT include a table of contents or "
        f"index — these are built programmatically elsewhere.\n\n"
        f"CRITICAL: Every section heading (## and ###) MUST be followed by "
        f"at least 4-5 substantial paragraphs of body text. Never leave a "
        f"heading without content below it. A section with fewer than 4 "
        f"paragraphs is incomplete.\n\n"
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
