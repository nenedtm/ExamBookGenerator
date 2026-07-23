#!/usr/bin/env python3
"""Example: using prompt_manager for both 'full' and 'topic' modes.

Demonstrates every builder function and prints the resulting prompts
so you can inspect them before wiring them to OllamaClient.

Run::

    python -m llm.example_prompt_manager
"""

from __future__ import annotations

import json
import textwrap

from core.models import Chunk, Topic
from llm.prompt_manager import (
    build_chapter_prompt,
    build_focus_topic_prompt,
    build_image_caption_prompt,
    build_image_relevance_prompt,
    build_outline_prompt,
    build_topic_prompt,
)

# ── Fake data ────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    Chunk(
        id="c001", document_id="doc_a", position=0,
        content="Linear algebra deals with vector spaces and linear mappings. "
                "Key concepts include bases, dimension, eigenvalues.",
    ),
    Chunk(
        id="c002", document_id="doc_a", position=1,
        content="A matrix is a rectangular array of numbers. Matrix "
                "multiplication is associative but not commutative.",
    ),
    Chunk(
        id="c003", document_id="doc_b", position=0,
        content="Calculus: derivatives measure instantaneous rate of change. "
                "The chain rule is essential for composite functions.",
    ),
]

SAMPLE_SYLLABUS = (
    "1. Linear Algebra — vector spaces, bases, eigenvalues\n"
    "2. Calculus — limits, derivatives, integrals\n"
    "3. Probability & Statistics — distributions, hypothesis testing"
)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ── Mode 1: full ────────────────────────────────────────────────────────────

def demo_full_mode() -> None:
    """Walk through the full-material pipeline (syllabus provided)."""
    _print_section("MODE: full  (syllabus provided)")

    # Step 1 — Topic analysis
    prompt = build_topic_prompt(SAMPLE_CHUNKS, syllabus_text=SAMPLE_SYLLABUS)
    print("[build_topic_prompt — with syllabus]")
    print(textwrap.indent(prompt, "  "))
    print()

    # Step 2 — Outline
    prompt = build_outline_prompt()
    print("[build_outline_prompt]")
    print(textwrap.indent(prompt, "  "))
    print()

    # Step 3 — Chapter generation
    topic = Topic(
        name="Linear Algebra",
        description="Vector spaces, bases, eigenvalues, and matrix operations.",
        related_documents=["doc_a"],
        subtopic_count=4,
    )
    prompt = build_chapter_prompt(topic, SAMPLE_CHUNKS[:2], depth_level=6)
    print("[build_chapter_prompt  depth_level=6]")
    print(textwrap.indent(prompt, "  "))
    print()

    # Step 4 — Image prompts
    prompt = build_image_caption_prompt("Figure 1: Eigenvalue decomposition")
    print("[build_image_caption_prompt]")
    print(textwrap.indent(prompt, "  "))
    print()

    chapter_md = "## Linear Algebra\n\nEigenvalues are scalars..."
    prompt = build_image_relevance_prompt(chapter_md, "A matrix diagram")
    print("[build_image_relevance_prompt]")
    print(textwrap.indent(prompt, "  "))


# ── Mode 2: topic (focus) ───────────────────────────────────────────────────

def demo_topic_mode() -> None:
    """Walk through the focus-topic pipeline (no syllabus)."""
    _print_section("MODE: topic  (focus on a single topic, no syllabus)")

    # Step 1 — Topic analysis (no syllabus → pedagogical order)
    prompt = build_topic_prompt(SAMPLE_CHUNKS, syllabus_text=None)
    print("[build_topic_prompt — without syllabus]")
    print(textwrap.indent(prompt, "  "))
    print()

    # Step 2 — Focus chapter
    prompt = build_focus_topic_prompt(
        focus_topic="Eigenvalues and Eigenvectors",
        chunks=SAMPLE_CHUNKS[:1],
        depth_level=8,
    )
    print("[build_focus_topic_prompt  depth_level=8]")
    print(textwrap.indent(prompt, "  "))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    demo_full_mode()
    _print_section("")
    demo_topic_mode()
    print("\nDone.")


if __name__ == "__main__":
    main()
