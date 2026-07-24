"""ExamBookGenerator — CLI entry point and pipeline orchestrator.

Usage::

    python main.py --input ./my_materials
    python main.py --input ./material --scope topic --topic "Linear Algebra"
    python main.py --input ./material --no-interactive --depth 7
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

from core.models import Document, ExtractedImage, FileType, Topic
from llm.ollama_client import OllamaClient
from pipeline.chapter_generator import generate_chapter
from pipeline.chunker import create_chunks
from pipeline.deduplicator import deduplicate
from pipeline.image_matcher import select_images_for_chapter
from pipeline.merge import merge_chapters
from pipeline.outline_generator import OutlineGenerator
from pipeline.scanner import scan_directory
from pipeline.template_engine import load_template
from pipeline.topic_analyzer import TopicAnalyzer, TopicNotFoundError, _chunks_for_topic
from pipeline.validator import validate_manual
from utils.config import ConfigManager, ConfigValidationError, _DEFAULT_CONFIG_PATH
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


# ── Logging bootstrap ────────────────────────────────────────────────────────


def _read_log_level() -> int:
    try:
        with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        level_str = raw.get("logging", {}).get("level", "INFO").upper()
        return _LEVEL_MAP.get(level_str, logging.INFO)
    except (FileNotFoundError, yaml.YAMLError):
        return logging.INFO


# ── CLI argument parsing ─────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ExamBookGenerator",
        description="Generate a university exam study manual from source material.",
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to the directory containing source material files. "
             "If omitted, the GUI is launched.",
    )
    parser.add_argument(
        "--template", default="template.md",
        help="Path to the Markdown template file (default: template.md).",
    )
    parser.add_argument(
        "--model",
        help="Override the LLM model name from config.",
    )
    parser.add_argument(
        "--output",
        help="Override the output directory (default: output/).",
    )
    parser.add_argument(
        "--depth", type=int, choices=range(1, 11), metavar="1-10",
        help="Detail level 1-10 (default: from config).",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Disable image extraction and matching.",
    )
    parser.add_argument(
        "--syllabus",
        help="Explicit path to a syllabus file (bypasses auto-detection).",
    )
    parser.add_argument(
        "--scope", choices=["full", "topic"], default=None,
        help="Generation scope: 'full' manual or single 'topic' (default: full).",
    )
    parser.add_argument(
        "--topic",
        help="Focus topic name (required when --scope topic).",
    )
    parser.add_argument(
        "--focus-depth", type=int, choices=range(1, 11), metavar="1-10",
        help="Detail level for topic-focus mode (default: --depth or config).",
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Disable all interactive prompts (for scripts/CI).",
    )
    return parser


# ── Interactive prompts ──────────────────────────────────────────────────────


def _ask_syllabus() -> str | None:
    print("\n--- Syllabus ---")
    print("Do you have a course syllabus / programma?")
    print("If yes, enter the file path.  Press Enter to skip.\n")
    path = input("Syllabus path (Enter to skip): ").strip()
    return path if path else None


def _ask_scope() -> tuple[str, str | None, int]:
    print("\n--- Scope ---")
    print("What would you like to generate?")
    print("  [1] Full manual (all topics)")
    print("  [2] Single topic focus\n")
    choice = input("Select [1/2] (default: 1): ").strip() or "1"

    if choice == "2":
        topic_name = input("Topic name: ").strip()
        if not topic_name:
            print("ERROR: topic name is required for focus mode.")
            sys.exit(1)
        depth_str = input(
            "Detail level for this topic (1-10, Enter to use default): "
        ).strip()
        focus_depth = int(depth_str) if depth_str.isdigit() else 5
        return "topic", topic_name, focus_depth

    return "full", None, 5


# ── Document parsing ─────────────────────────────────────────────────────────


def _parse_document(doc: Document) -> tuple[Document, list[ExtractedImage]]:
    """Dispatch to the appropriate parser based on file type."""
    path = Path(doc.source_path)

    try:
        if doc.file_type == FileType.PDF:
            from parsers.pdf_parser import parse_pdf
            return parse_pdf(path)

        if doc.file_type == FileType.DOCX:
            from parsers.docx_parser import parse_docx
            return parse_docx(path)

        if doc.file_type == FileType.PPTX:
            from parsers.pptx_parser import parse_pptx
            return parse_pptx(path)

        if doc.file_type in (FileType.TXT, FileType.MARKDOWN):
            content = path.read_text(encoding="utf-8", errors="replace")
            doc.content = content
            return doc, []

        # Images and unknown types — skip content parsing
        return doc, []

    except Exception as exc:
        logger.warning("Failed to parse %s: %s", path.name, exc)
        return doc, []


def _parse_all(documents: list[Document]) -> tuple[list[Document], list[ExtractedImage]]:
    """Parse all documents and collect extracted images."""
    all_images: list[ExtractedImage] = []
    parsed: list[Document] = []

    for doc in documents:
        if doc.file_type == FileType.IMAGE:
            continue  # standalone images are not parsed as documents
        parsed_doc, images = _parse_document(doc)
        parsed.append(parsed_doc)
        all_images.extend(images)

    logger.info("Parsed %d document(s), extracted %d image(s)", len(parsed), len(all_images))
    return parsed, all_images


# ── Main pipeline ────────────────────────────────────────────────────────────


def run_pipeline(
    args: argparse.Namespace,
    progress_callback: callable | None = None,
) -> tuple[Path, dict]:
    """Execute the full generation pipeline and return the output path.

    Parameters
    ----------
    args:
        Namespace from CLI or programmatically built.
    progress_callback:
        Optional ``(step: int, message: str)`` callback for progress updates.
        *step* is 1-based; total is 8 steps.
    """
    def _progress(step: int, msg: str) -> None:
        logger.info("Step %d/8: %s", step, msg)
        if progress_callback:
            progress_callback(step, msg)

    start_time = time.monotonic()

    # ── 1. Config ──────────────────────────────────────────────────────
    cfg = ConfigManager()

    if args.model:
        cfg.set("llm.model", args.model)
    if args.output:
        cfg.set("output.language", cfg.get("output.language", "en"))
    if args.depth:
        cfg.set("generation.depth_level", args.depth)

    # ── 2. Resolve scope ──────────────────────────────────────────────
    scope = args.scope or "full"
    focus_topic = args.topic
    focus_depth = args.focus_depth or cfg.get("generation.depth_level", 5)
    depth_level = cfg.get("generation.depth_level", 5)

    if scope == "topic":
        if not focus_topic:
            if args.no_interactive:
                print("ERROR: --topic is required when --scope topic is used.", file=sys.stderr)
                sys.exit(1)
            focus_topic = input("\nEnter the topic to focus on: ").strip()
            if not focus_topic:
                print("ERROR: topic name cannot be empty.", file=sys.stderr)
                sys.exit(1)
            depth_str = input(
                "Detail level for this topic (1-10, Enter to use default): "
            ).strip()
            focus_depth = int(depth_str) if depth_str.isdigit() else focus_depth

        cfg.set("generation.scope", "topic")
        cfg.set("generation.focus_topic", focus_topic)
        cfg.set("generation.focus_depth_level", focus_depth)
        effective_depth = focus_depth
    else:
        cfg.set("generation.scope", "full")
        effective_depth = depth_level

    # ── 3. Syllabus resolution ────────────────────────────────────────
    syllabus_path = args.syllabus
    if not syllabus_path and not args.no_interactive:
        syllabus_path = _ask_syllabus()

    if syllabus_path:
        cfg.set("syllabus.enabled", "true")
        cfg.set("syllabus.path", syllabus_path)
    else:
        cfg.set("syllabus.enabled", "auto")
        cfg.set("syllabus.path", None)

    # ── 4. Interactive scope selection (if not set via CLI) ────────────
    if args.scope is None and not args.no_interactive:
        scope, focus_topic, focus_depth = _ask_scope()
        if scope == "topic":
            cfg.set("generation.scope", "topic")
            cfg.set("generation.focus_topic", focus_topic)
            cfg.set("generation.focus_depth_level", focus_depth)
            effective_depth = focus_depth
        else:
            cfg.set("generation.scope", "full")
            effective_depth = depth_level

    # ── 5. Validate inputs ────────────────────────────────────────────
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"ERROR: not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"ERROR: template not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    # ── 6. Setup output directory ─────────────────────────────────────
    output_dir = Path(args.output) if args.output else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("ExamBookGenerator pipeline starting")
    logger.info(
        "Input: %s | Scope: %s | Depth: %d | Images: %s",
        input_dir.resolve(), scope, effective_depth,
        "disabled" if args.no_images else "enabled",
    )

    # ── 7. Scan ───────────────────────────────────────────────────────
    _progress(1, "Scanning input directory...")
    syllabus_enabled = cfg.get("syllabus.enabled", "auto") != "false"
    documents = scan_directory(
        input_dir,
        syllabus_enabled=syllabus_enabled,
        syllabus_path=cfg.get("syllabus.path"),
    )
    if not documents:
        print("ERROR: no supported files found in the input directory.", file=sys.stderr)
        sys.exit(1)
    logger.info("Found %d document(s)", len(documents))

    # ── 8. Parse ──────────────────────────────────────────────────────
    _progress(2, "Parsing documents...")
    parsed_docs, extracted_images = _parse_all(documents)
    non_empty = [d for d in parsed_docs if d.content.strip()]
    if not non_empty:
        print("ERROR: no text content could be extracted from the documents.", file=sys.stderr)
        sys.exit(1)

    # ── 9. Deduplicate ────────────────────────────────────────────────
    _progress(3, "Deduplicating...")
    unique_docs = deduplicate(non_empty)
    logger.info("After dedup: %d document(s)", len(unique_docs))

    # ── 10. Chunk ─────────────────────────────────────────────────────
    _progress(4, "Chunking documents...")
    all_chunks = []
    for doc in unique_docs:
        chunks = create_chunks(doc)
        all_chunks.extend(chunks)
    logger.info("Total chunks: %d", len(all_chunks))

    # ── 11. Topic analysis ────────────────────────────────────────────
    _progress(5, "Analysing topics...")
    client = OllamaClient.from_config(cfg)
    analyzer = TopicAnalyzer(client, cfg)

    # Find syllabus document
    syllabus_doc = None
    if syllabus_enabled:
        for d in parsed_docs:
            if d.is_syllabus:
                syllabus_doc = d
                break

    topics = analyzer.analyze(
        all_chunks,
        syllabus_document=syllabus_doc,
        scope=scope,
        focus_topic=focus_topic,
    )
    logger.info("Identified %d topic(s)", len(topics))

    # ── 12. Outline ───────────────────────────────────────────────────
    _progress(6, "Generating outline...")
    outline_gen = OutlineGenerator(client, cfg)
    _, index_entries = outline_gen.generate(topics, scope=scope)

    # ── 13. Chapter generation ────────────────────────────────────────
    _progress(7, "Generating chapters...")
    template = load_template(template_path)
    chapters_md: list[str] = []

    for topic in topics:
        logger.info("  Generating chapter: %s", topic.name)

        # Collect chunks for this topic
        topic_chunks = [
            c for c in all_chunks
            if c.document_id in set(topic.related_documents)
        ]

        # Fallback: if no chunks matched by document ID, try keyword matching
        if not topic_chunks:
            topic_chunks = _chunks_for_topic(topic.name, all_chunks)
            if topic_chunks:
                logger.info(
                    "  Fallback: found %d chunk(s) for '%s' via keyword matching",
                    len(topic_chunks), topic.name,
                )
            else:
                logger.warning(
                    "  No source material found for topic '%s' — "
                    "generating from topic description only",
                    topic.name,
                )

        # Filter images for this topic's documents
        topic_doc_ids = set(topic.related_documents)
        topic_images = [
            img for img in extracted_images
            if img.source_document_id in topic_doc_ids
        ]

        chapter_md, _, _ = generate_chapter(
            topic,
            topic_chunks,
            template,
            depth_level=effective_depth,
            candidate_images=topic_images if not args.no_images else [],
            scope=scope,
            client=client,
            cfg=cfg,
        )
        chapters_md.append(chapter_md)

    # ── 14. Merge ─────────────────────────────────────────────────────
    _progress(8, "Assembling final output...")
    output_path = merge_chapters(
        chapters_md,
        index_entries,
        cfg,
        topics=topics,
        focus_topic=focus_topic if scope == "topic" else None,
    )

    # ── 15. Validate ──────────────────────────────────────────────────
    _progress(8, "Running validation...")
    manual_text = output_path.read_text(encoding="utf-8", errors="replace")
    chapters_meta = [
        {"title": t.name, "order": i, "syllabus_position": getattr(t, "syllabus_position", None)}
        for i, t in enumerate(topics)
    ]
    validation = validate_manual(
        manual_text, topics, cfg,
        images=extracted_images if not args.no_images else None,
        chapters_meta=chapters_meta,
    )

    elapsed = time.monotonic() - start_time
    logger.info("Pipeline complete in %.1fs", elapsed)
    logger.info("Output: %s", output_path.resolve())
    logger.info("Validation: %s", validation["overall"])

    return output_path, validation


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    log_level = _read_log_level()
    setup_logging(level=log_level)

    parser = build_parser()
    args = parser.parse_args()

    # If --input is omitted, launch the GUI instead of the CLI.
    if args.input is None:
        try:
            from gui.app import launch
        except ImportError as exc:
            print(
                "GUI requires PySide6.  Install it with:  pip install PySide6",
                file=sys.stderr,
            )
            sys.exit(1)
        launch()
        return

    try:
        output_path, validation = run_pipeline(args)

        print(f"\n{'='*60}")
        print(f"  ExamBookGenerator — Complete!")
        print(f"{'='*60}")
        print(f"  Output:  {output_path.resolve()}")
        print(f"  Status:  {validation['overall'].upper()}")
        s = validation["summary"]
        print(f"  Checks:  {s['passed']} passed, {s['warnings']} warnings, {s['failed']} failed")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except ConfigValidationError as exc:
        logger.critical("Configuration error: %s", exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.critical("File not found: %s", exc)
        sys.exit(1)
    except TopicNotFoundError as exc:
        logger.critical("Topic not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("Unexpected error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
