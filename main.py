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

from core.models import Document, ExtractedImage, FileType, OutlineChapter, Topic
from llm.ollama_client import OllamaClient
from pipeline.chapter_generator import generate_chapter
from pipeline.chunker import create_chunks
from pipeline.deduplicator import deduplicate
from pipeline.image_matcher import select_images_for_chapter
from pipeline.merge import merge_chapters
from pipeline.outline_generator import OutlineGenerator
from pipeline.scanner import scan_directory, scan_directory_incremental, compute_file_hash
from pipeline.template_engine import load_template
from pipeline.topic_analyzer import TopicAnalyzer, TopicNotFoundError, _chunks_for_topic
from pipeline.validator import validate_manual
from storage.database import (
    get_processing_state,
    save_processing_state,
    invalidate_processing_state,
    _get_connection,
)
from storage.cache import clear_cache_for_model
from utils.config import ConfigManager, ConfigValidationError, _DEFAULT_CONFIG_PATH
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def _chunks_from_cache(chunks_json: str) -> list:
    """Deserialize chunks from JSON cache."""
    import json
    from core.models import Chunk
    raw = json.loads(chunks_json)
    return [Chunk(**c) for c in raw]


def _chunks_to_cache(chunks: list) -> str:
    """Serialize chunks to JSON for caching."""
    import json
    from core.models import Chunk
    serializable = []
    for c in chunks:
        if isinstance(c, Chunk):
            serializable.append({
                "id": c.id, "document_id": c.document_id,
                "content": c.content, "position": c.position,
            })
        elif hasattr(c, "id") and hasattr(c, "content"):
            serializable.append({
                "id": str(c.id), "document_id": str(getattr(c, "document_id", "")),
                "content": str(c.content), "position": int(getattr(c, "position", 0)),
            })
    return json.dumps(serializable, ensure_ascii=False)


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
    parser.add_argument(
        "--chapters",
        help="Comma-separated chapter numbers to generate (e.g. '1,3,5'). "
             "When omitted in non-interactive mode, all chapters are generated.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force full reprocessing, ignoring all caches.",
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
        focus_depth = int(depth_str) if depth_str.isdigit() else 7
        return "topic", topic_name, focus_depth

    return "full", None, 7


# ── Chapter selection and incremental generation ────────────────────────────


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    slug = ascii_text.lower()
    slug = slug.replace("'", "").replace("'", "")
    slug = __import__("re").sub(r"[^a-z0-9 \-]", "", slug)
    slug = __import__("re").sub(r"[\s\-]+", "-", slug).strip("-")
    return slug


def _chapter_filename(idx: int, title: str) -> str:
    """Build filename for an individual chapter file: ``cap_N_slug.md``."""
    slug = _slugify(title)
    return f"cap_{idx + 1}_{slug}.md"


def _save_chapter_file(chapters_dir: Path, idx: int, title: str, chapter_md: str) -> Path:
    """Write a single chapter to ``chapters_dir/cap_N_slug.md``."""
    chapters_dir.mkdir(parents=True, exist_ok=True)
    filename = _chapter_filename(idx, title)
    path = chapters_dir / filename
    path.write_text(chapter_md, encoding="utf-8")
    logger.info("Chapter %d saved to %s", idx + 1, path)
    return path


def _load_existing_chapters(chapters_dir: Path, total: int) -> set[int]:
    """Scan *chapters_dir* and return set of 0-based indices already generated.

    Only counts files whose index is within ``[0, total)``.
    """
    existing: set[int] = set()
    if not chapters_dir.exists():
        return existing
    for f in chapters_dir.iterdir():
        if not f.is_file() or not f.name.startswith("cap_") or not f.name.endswith(".md"):
            continue
        try:
            # cap_N_slug.md → extract N
            num_str = f.name.split("_")[1]
            num = int(num_str)
            if 1 <= num <= total:
                existing.add(num - 1)
        except (IndexError, ValueError):
            continue
    return existing


def _write_indice(
    output_dir: Path,
    outline_chapters: list,
    selected: set[int] | None,
    existing: set[int],
) -> Path:
    """Write ``output/indice.md`` with numbered chapters, sections and status.

    ``selected`` is *None* meaning all, or a set of 0-based indices.
    ``existing`` is the set of 0-based indices already on disk.
    """
    lines = [
        "# Indice del Manuale",
        "",
        "Capitoli generati sono contrassegnati con [X].",
        "",
    ]
    for i, ch in enumerate(outline_chapters):
        mark = "[X]" if i in existing else "[ ]"
        lines.append(f"- {mark} {i + 1}. {ch.title}")
        for sec in ch.sections:
            lines.append(f"    - {sec}")
    lines.append("")

    out_path = output_dir / "indice.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Indice saved to %s", out_path)
    return out_path


def _ask_chapter_selection(outline_chapters: list, existing: set[int]) -> set[int] | None:
    """Interactive prompt: show full indice and ask user to select chapters.

    Returns None for "all", or a set of 0-based indices.
    """
    print("\n" + "=" * 60)
    print("  INDICE DEL MANUALE")
    print("=" * 60)
    for i, ch in enumerate(outline_chapters):
        status = "  [gia' generato]" if i in existing else ""
        print(f"\n  [{i + 1}] {ch.title}{status}")
        for sec in ch.sections:
            print(f"       - {sec}")
    print("\n" + "=" * 60)

    raw = input(
        "\nSeleziona i capitoli da generare (numeri separati da virgola, "
        "o Invio per generare tutti): "
    ).strip()

    if not raw:
        return None  # all

    selected: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            num = int(part)
            if 1 <= num <= len(outline_chapters):
                selected.add(num - 1)
    return selected


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


def _collect_chunks_for_topic(
    topic: Topic,
    all_chunks: list,
    chunk_id_map: dict[str, list[str]] | None = None,
) -> list:
    """Collect all chunks relevant to a topic, with keyword fallback.

    Uses chunk_id_map for precise matching when available, falls back to
    document_id matching, then keyword matching.
    """
    # Try chunk ID-based filtering first (most precise)
    if chunk_id_map and topic.name in chunk_id_map:
        ids = set(chunk_id_map[topic.name])
        matched = [c for c in all_chunks if c.id in ids]
        if matched:
            return matched

    # Fallback: match by document_id (all chunks from related docs)
    topic_chunks = [
        c for c in all_chunks
        if c.document_id in set(topic.related_documents)
    ]
    if not topic_chunks:
        topic_chunks = _chunks_for_topic(topic.name, all_chunks)
    return topic_chunks


def _collect_images_for_topic(topic: Topic, all_images: list, no_images: bool) -> list:
    """Collect images for a topic's documents."""
    if no_images:
        return []
    topic_doc_ids = set(topic.related_documents)
    return [img for img in all_images if img.source_document_id in topic_doc_ids]


def _find_closest_topic_for_chapter(
    chapter_title: str,
    topics: list,
    exclude_indices: set[int],
) -> object | None:
    """Find the best matching uncovered topic for an outline chapter title."""
    import re as _re
    title_words = {
        w for w in _re.split(r"[^a-z0-9]+", chapter_title.lower()) if len(w) > 3
    }
    best = None
    best_score = 0
    for idx, t in enumerate(topics):
        if idx in exclude_indices:
            continue
        topic_words = {
            w for w in _re.split(r"[^a-z0-9]+", t.name.lower()) if len(w) > 3
        }
        overlap = len(title_words & topic_words)
        if overlap > best_score:
            best_score = overlap
            best = t
    return best


def _build_chapter_local_entries(outline_chapter) -> list:
    """Build IndexEntry list for a chapter's local TOC from outline data."""
    from core.models import IndexEntry
    from pipeline.outline_generator import slugify

    entries = []
    entries.append(IndexEntry(
        title=outline_chapter.title,
        anchor=slugify(outline_chapter.title),
        level=1, order=0,
    ))
    for i, sec in enumerate(outline_chapter.sections, 1):
        entries.append(IndexEntry(
            title=sec,
            anchor=slugify(sec),
            level=2, order=i,
        ))
    return entries


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

    # Handle model change — invalidate chapter-level cache
    if args.model:
        # Check if model changed from previous run
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT DISTINCT model FROM processing_state WHERE model IS NOT NULL LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if row and row["model"] != args.model:
            logger.info("Model changed from '%s' to '%s' — invalidating chapter cache", row["model"], args.model)
            clear_cache_for_model(row["model"])

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

    # ── 7. Scan (incremental) ──────────────────────────────────────────
    _progress(1, "Scanning input directory...")
    syllabus_enabled = cfg.get("syllabus.enabled", "auto") != "false"
    force = getattr(args, "force", False)

    if force:
        documents = scan_directory(
            input_dir,
            syllabus_enabled=syllabus_enabled,
            syllabus_path=cfg.get("syllabus.path"),
        )
        new_hashes = {d.source_path for d in documents}
        modified_hashes: set[str] = set()
        unchanged_hashes: set[str] = set()
    else:
        documents, new_hashes, modified_hashes, unchanged_hashes = scan_directory_incremental(
            input_dir,
            syllabus_enabled=syllabus_enabled,
            syllabus_path=cfg.get("syllabus.path"),
        )

    if not documents:
        print("ERROR: no supported files found in the input directory.", file=sys.stderr)
        sys.exit(1)
    logger.info(
        "Found %d document(s) — %d new, %d modified, %d unchanged",
        len(documents), len(new_hashes), len(modified_hashes), len(unchanged_hashes),
    )

    # ── 8. Parse (incremental) ────────────────────────────────────────
    _progress(2, "Parsing documents...")

    docs_to_parse = []
    cached_chunks_map: dict[str, list] = {}  # source_path → chunks
    file_hashes: dict[str, str] = {}  # source_path → content_hash (from scan)

    # Recompute hashes for all documents (same as scan does)
    for doc in documents:
        try:
            fhash = compute_file_hash(Path(doc.source_path))
            file_hashes[doc.source_path] = fhash
        except (OSError, ValueError):
            file_hashes[doc.source_path] = ""

    for doc in documents:
        fhash = file_hashes.get(doc.source_path, "")
        if fhash in unchanged_hashes and not force:
            # Load cached chunks from DB
            state = get_processing_state(fhash)
            if state and state["chunks_json"]:
                cached_chunks_map[doc.source_path] = _chunks_from_cache(state["chunks_json"])
                logger.debug("Loaded cached chunks for %s", doc.source_path)
                continue
        docs_to_parse.append(doc)

    parsed_docs, extracted_images = _parse_all(docs_to_parse)

    non_empty = [d for d in parsed_docs if d.content.strip()]

    if not non_empty and not cached_chunks_map:
        print("ERROR: no text content could be extracted from the documents.", file=sys.stderr)
        sys.exit(1)

    # ── 9. Deduplicate ────────────────────────────────────────────────
    _progress(3, "Deduplicating...")
    unique_docs = deduplicate(non_empty)
    logger.info("After dedup: %d document(s)", len(unique_docs))

    # ── 10. Chunk (incremental) ───────────────────────────────────────
    _progress(4, "Chunking documents...")
    all_chunks = []

    # Add cached chunks from unchanged files
    for src_path, chunks in cached_chunks_map.items():
        all_chunks.extend(chunks)

    # Chunk newly parsed documents
    for doc in parsed_docs:
        if not doc.content.strip():
            continue
        chunks = create_chunks(doc)
        all_chunks.extend(chunks)

        # Save chunks to processing state using pre-computed hash
        fhash = file_hashes.get(doc.source_path, "")
        if fhash:
            save_processing_state(
                fhash,
                doc.source_path,
                doc.file_type.value,
                parsed=True,
                chunks_json=_chunks_to_cache(chunks),
            )

    logger.info("Total chunks: %d", len(all_chunks))

    # ── 11. Topic analysis (with caching) ─────────────────────────────
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

    topics_changed = len(new_hashes) > 0 or len(modified_hashes) > 0
    cached_topics = None

    if not topics_changed and not force:
        # Try to load cached topics from the most recent processing state
        import json
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT topics_json FROM processing_state WHERE topics_json IS NOT NULL ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if row and row["topics_json"]:
            try:
                raw_topics = json.loads(row["topics_json"])
                cached_topics = [Topic(**t) for t in raw_topics]
                logger.info("Loaded %d cached topic(s)", len(cached_topics))
            except Exception as exc:
                logger.warning("Failed to load cached topics: %s", exc)
                cached_topics = None

    if cached_topics and not topics_changed:
        topics = cached_topics
        chunk_id_map: dict[str, list[str]] = {}
    else:
        topics, chunk_id_map = analyzer.analyze(
            all_chunks,
            syllabus_document=syllabus_doc,
            scope=scope,
            focus_topic=focus_topic,
        )
        # Save topics to all processing states
        import json
        try:
            topics_json = json.dumps(
                [{"name": t.name, "description": t.description,
                  "related_documents": t.related_documents,
                  "subtopic_count": t.subtopic_count,
                  "order_source": t.order_source,
                  "syllabus_position": t.syllabus_position,
                  "missing_from_notes": t.missing_from_notes,
                  "extra_in_notes": t.extra_in_notes}
                 for t in topics],
                ensure_ascii=False,
            )
            conn = _get_connection()
            try:
                conn.execute(
                    "UPDATE processing_state SET topics_json = ?",
                    (topics_json,),
                )
                conn.commit()
            finally:
                conn.close()
        except (TypeError, AttributeError) as exc:
            logger.debug("Could not cache topics: %s", exc)

    logger.info("Identified %d topic(s)", len(topics))

    # ── 12. Outline ───────────────────────────────────────────────────
    _progress(6, "Generating outline...")
    outline_gen = OutlineGenerator(client, cfg)

    syllabus_text: str | None = syllabus_doc.content if syllabus_doc else None

    outline_cached = False
    if not topics_changed and not force:
        outline_path = Path("output/outline.md")
        if outline_path.exists():
            try:
                # Try to load existing outline
                existing_outline = outline_path.read_text(encoding="utf-8")
                if existing_outline.strip():
                    logger.info("Reusing existing outline (topics unchanged)")
                    outline_cached = True
                    # Parse existing outline for index_entries and outline_chapters
                    _, index_entries, outline_chapters = outline_gen.generate(
                        topics, scope=scope, syllabus_text=syllabus_text,
                    )
            except Exception:
                pass

    if not outline_cached:
        _, index_entries, outline_chapters = outline_gen.generate(
            topics, scope=scope, syllabus_text=syllabus_text,
        )

    # ── 12b. Indice and chapter selection ──────────────────────────────
    chapters_dir = output_dir / "chapters"

    if scope == "topic":
        # Topic focus mode — no selection, single chapter
        selected_chapters: set[int] | None = None
    else:
        # Write indice file
        _write_indice(output_dir, outline_chapters, None, set())

        # Check which chapters already exist on disk
        existing_chapters = _load_existing_chapters(chapters_dir, len(outline_chapters))
        if existing_chapters:
            logger.info(
                "Found %d already-generated chapter(s) in %s",
                len(existing_chapters), chapters_dir,
            )

        # Interactive selection
        if not args.no_interactive:
            selected_chapters = _ask_chapter_selection(outline_chapters, existing_chapters)
        else:
            # Non-interactive: use --chapters flag or generate all
            chapters_flag = getattr(args, "chapters", None)
            if chapters_flag:
                selected_chapters = set()
                for part in chapters_flag.split(","):
                    part = part.strip()
                    if part.isdigit():
                        num = int(part)
                        if 1 <= num <= len(outline_chapters):
                            selected_chapters.add(num - 1)
            else:
                selected_chapters = None  # all

    # ── 13. Chapter generation (incremental) ──────────────────────────
    _progress(7, "Generating chapters...")
    template = load_template(template_path)
    chapters_md: list[str] = []
    covered_topic_indices: set[int] = set()

    # Track which topics have changed (have new/modified chunks)
    changed_topic_names: set[str] = set()
    if topics_changed:
        # All topics are considered changed if any file changed
        changed_topic_names = {t.name for t in topics}

    # Load cached chapters if available and topics haven't changed
    cached_chapters: dict[str, str] = {}  # topic_name → chapter_md
    if not changed_topic_names and not force:
        chapters_path = Path("output/Exam_Manual.md")
        if chapters_path.exists():
            # We can't easily split the merged manual back into chapters,
            # so we regenerate if chapters are needed. The LLM cache handles this.
            pass

    # Tracks existing chapters across the loop (resume support)
    if scope == "topic":
        existing_chapters_local: set[int] = set()
        selected_chapters_local: set[int] | None = None
    else:
        existing_chapters_local = _load_existing_chapters(chapters_dir, len(outline_chapters))
        selected_chapters_local = selected_chapters

    if scope == "topic":
        # Focus mode: single chapter, ignore outline structure
        for topic in topics:
            logger.info("  Generating chapter: %s", topic.name)
            topic_chunks = _collect_chunks_for_topic(topic, all_chunks, chunk_id_map)
            topic_images = _collect_images_for_topic(topic, extracted_images, args.no_images)
            chapter_md, _, _ = generate_chapter(
                topic, topic_chunks, template,
                depth_level=effective_depth,
                candidate_images=topic_images,
                scope=scope, client=client, cfg=cfg,
            )
            chapters_md.append(chapter_md)
    else:
        # Full mode: iterate over outline chapters, matching topics to each
        for ch_idx, outline_ch in enumerate(outline_chapters):
            # Skip chapters not selected by the user
            if selected_chapters_local is not None and ch_idx not in selected_chapters_local:
                logger.info("  Skipping chapter %d: %s (not selected)", ch_idx + 1, outline_ch.title)
                continue

            # Skip chapters already generated on disk (resume support)
            if ch_idx in existing_chapters_local:
                logger.info(
                    "  Chapter %d: %s — already generated, loading from disk",
                    ch_idx + 1, outline_ch.title,
                )
                chapter_path = chapters_dir / _chapter_filename(ch_idx, outline_ch.title)
                if chapter_path.exists():
                    chapters_md.append(chapter_path.read_text(encoding="utf-8"))
                continue

            logger.info("  Generating chapter %d: %s", ch_idx + 1, outline_ch.title)

            # Collect topics matched to this outline chapter
            matched_topics = [
                topics[i] for i in outline_ch.topic_indices
                if i < len(topics)
            ]
            # Track which topics are covered
            for i in outline_ch.topic_indices:
                if i < len(topics):
                    covered_topic_indices.add(i)

            # Collect all chunks from matched topics
            all_topic_chunks = []
            all_topic_images = []
            for t in matched_topics:
                all_topic_chunks.extend(_collect_chunks_for_topic(t, all_chunks, chunk_id_map))
                all_topic_images.extend(_collect_images_for_topic(t, extracted_images, args.no_images))

            # If no topics matched, find the closest topic by name
            if not matched_topics and topics:
                best_topic = _find_closest_topic_for_chapter(outline_ch.title, topics, covered_topic_indices)
                if best_topic:
                    idx = topics.index(best_topic)
                    covered_topic_indices.add(idx)
                    all_topic_chunks.extend(_collect_chunks_for_topic(best_topic, all_chunks, chunk_id_map))
                    all_topic_images.extend(_collect_images_for_topic(best_topic, extracted_images, args.no_images))
                    matched_topics.append(best_topic)

            # Deduplicate chunks by content
            seen_content: set[str] = set()
            unique_chunks = []
            for c in all_topic_chunks:
                if c.content not in seen_content:
                    seen_content.add(c.content)
                    unique_chunks.append(c)

            # Deduplicate images by id
            seen_img_ids: set[str] = set()
            unique_images = []
            for img in all_topic_images:
                if img.id not in seen_img_ids:
                    seen_img_ids.add(img.id)
                    unique_images.append(img)

            # Use the first matched topic as the primary topic for the prompt
            primary_topic = matched_topics[0] if matched_topics else Topic(
                name=outline_ch.title, description=outline_ch.title,
            )

            # Build index entries for this chapter's local TOC
            chapter_entries = _build_chapter_local_entries(outline_ch)

            chapter_md, _, _ = generate_chapter(
                primary_topic, unique_chunks, template,
                depth_level=effective_depth,
                candidate_images=unique_images if not args.no_images else [],
                index_entries=chapter_entries,
                outline_chapter=outline_ch,
                scope="full", client=client, cfg=cfg,
            )
            chapters_md.append(chapter_md)

            # Save chapter to disk immediately
            _save_chapter_file(chapters_dir, ch_idx, outline_ch.title, chapter_md)
            existing_chapters_local.add(ch_idx)

            # Update indice with current progress
            _write_indice(output_dir, outline_chapters, selected_chapters, existing_chapters_local)

        # Generate chapters for any uncovered topics
        for idx, topic in enumerate(topics):
            if idx not in covered_topic_indices:
                # For uncovered topics we don't have an outline_ch index,
                # so we always generate them (they weren't in the outline)
                logger.info("  Generating chapter for uncovered topic: %s", topic.name)
                topic_chunks = _collect_chunks_for_topic(topic, all_chunks, chunk_id_map)
                topic_images = _collect_images_for_topic(topic, extracted_images, args.no_images)
                chapter_md, _, _ = generate_chapter(
                    topic, topic_chunks, template,
                    depth_level=effective_depth,
                    candidate_images=topic_images if not args.no_images else [],
                    scope=scope, client=client, cfg=cfg,
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


# ── Split pipeline phases (for Streamlit two-phase flow) ────────────────────


def run_outline_phase(
    args: argparse.Namespace,
    progress_callback: callable | None = None,
) -> dict:
    """Run steps 1-6: scan, parse, chunk, topic analysis, outline generation.

    Returns a state dict with all intermediate data needed by
    :func:`run_chapters_phase`.
    """
    def _progress(step: int, msg: str) -> None:
        logger.info("Step %d/8: %s", step, msg)
        if progress_callback:
            progress_callback(step, msg)

    # ── 1. Config ──────────────────────────────────────────────────────
    cfg = ConfigManager()
    if args.model:
        cfg.set("llm.model", args.model)
    if args.depth:
        cfg.set("generation.depth_level", args.depth)

    if args.model:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT DISTINCT model FROM processing_state WHERE model IS NOT NULL LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if row and row["model"] != args.model:
            clear_cache_for_model(row["model"])

    # ── 2. Resolve scope ──────────────────────────────────────────────
    scope = args.scope or "full"
    focus_topic = args.topic
    focus_depth = args.focus_depth or cfg.get("generation.depth_level", 5)
    depth_level = cfg.get("generation.depth_level", 5)

    if scope == "topic":
        cfg.set("generation.scope", "topic")
        cfg.set("generation.focus_topic", focus_topic or "")
        cfg.set("generation.focus_depth_level", focus_depth)
        effective_depth = focus_depth
    else:
        cfg.set("generation.scope", "full")
        effective_depth = depth_level

    # ── 3. Syllabus ──────────────────────────────────────────────────
    syllabus_path = args.syllabus
    if syllabus_path:
        cfg.set("syllabus.enabled", "true")
        cfg.set("syllabus.path", syllabus_path)
    else:
        cfg.set("syllabus.enabled", "auto")
        cfg.set("syllabus.path", None)

    # ── 4. Validate inputs ────────────────────────────────────────────
    input_dir = Path(args.input)
    template_path = Path(args.template)
    output_dir = Path(args.output) if args.output else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 5. Scan ───────────────────────────────────────────────────────
    _progress(1, "Scanning input directory...")
    syllabus_enabled = cfg.get("syllabus.enabled", "auto") != "false"
    force = getattr(args, "force", False)

    if force:
        documents = scan_directory(
            input_dir,
            syllabus_enabled=syllabus_enabled,
            syllabus_path=cfg.get("syllabus.path"),
        )
        new_hashes = {d.source_path for d in documents}
        modified_hashes: set[str] = set()
        unchanged_hashes: set[str] = set()
    else:
        documents, new_hashes, modified_hashes, unchanged_hashes = scan_directory_incremental(
            input_dir,
            syllabus_enabled=syllabus_enabled,
            syllabus_path=cfg.get("syllabus.path"),
        )

    # ── 6. Parse ──────────────────────────────────────────────────────
    _progress(2, "Parsing documents...")
    docs_to_parse = []
    cached_chunks_map: dict[str, list] = {}
    file_hashes: dict[str, str] = {}

    for doc in documents:
        try:
            fhash = compute_file_hash(Path(doc.source_path))
            file_hashes[doc.source_path] = fhash
        except (OSError, ValueError):
            file_hashes[doc.source_path] = ""

    for doc in documents:
        fhash = file_hashes.get(doc.source_path, "")
        if fhash in unchanged_hashes and not force:
            state = get_processing_state(fhash)
            if state and state["chunks_json"]:
                cached_chunks_map[doc.source_path] = _chunks_from_cache(state["chunks_json"])
                continue
        docs_to_parse.append(doc)

    parsed_docs, extracted_images = _parse_all(docs_to_parse)
    non_empty = [d for d in parsed_docs if d.content.strip()]

    # ── 7. Deduplicate + Chunk ────────────────────────────────────────
    _progress(3, "Deduplicating...")
    unique_docs = deduplicate(non_empty)

    _progress(4, "Chunking documents...")
    all_chunks = []
    for src_path, chunks in cached_chunks_map.items():
        all_chunks.extend(chunks)
    for doc in parsed_docs:
        if not doc.content.strip():
            continue
        chunks = create_chunks(doc)
        all_chunks.extend(chunks)
        fhash = file_hashes.get(doc.source_path, "")
        if fhash:
            save_processing_state(
                fhash, doc.source_path, doc.file_type.value,
                parsed=True, chunks_json=_chunks_to_cache(chunks),
            )

    # ── 8. Topic analysis ─────────────────────────────────────────────
    _progress(5, "Analysing topics...")
    client = OllamaClient.from_config(cfg)
    analyzer = TopicAnalyzer(client, cfg)

    syllabus_doc = None
    if syllabus_enabled:
        for d in parsed_docs:
            if d.is_syllabus:
                syllabus_doc = d
                break

    topics_changed = len(new_hashes) > 0 or len(modified_hashes) > 0
    cached_topics = None

    if not topics_changed and not force:
        import json
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT topics_json FROM processing_state WHERE topics_json IS NOT NULL "
                "ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if row and row["topics_json"]:
            try:
                raw_topics = json.loads(row["topics_json"])
                cached_topics = [Topic(**t) for t in raw_topics]
            except Exception:
                cached_topics = None

    if cached_topics and not topics_changed:
        topics = cached_topics
        chunk_id_map: dict[str, list[str]] = {}
    else:
        topics, chunk_id_map = analyzer.analyze(
            all_chunks, syllabus_document=syllabus_doc,
            scope=scope, focus_topic=focus_topic,
        )
        import json
        try:
            topics_json = json.dumps(
                [{"name": t.name, "description": t.description,
                  "related_documents": t.related_documents,
                  "subtopic_count": t.subtopic_count,
                  "order_source": t.order_source,
                  "syllabus_position": t.syllabus_position,
                  "missing_from_notes": t.missing_from_notes,
                  "extra_in_notes": t.extra_in_notes}
                 for t in topics],
                ensure_ascii=False,
            )
            conn = _get_connection()
            try:
                conn.execute("UPDATE processing_state SET topics_json = ?", (topics_json,))
                conn.commit()
            finally:
                conn.close()
        except (TypeError, AttributeError):
            pass

    # ── 9. Outline ────────────────────────────────────────────────────
    _progress(6, "Generating outline...")
    outline_gen = OutlineGenerator(client, cfg)

    syllabus_text: str | None = syllabus_doc.content if syllabus_doc else None

    outline_cached = False
    if not topics_changed and not force:
        outline_path = output_dir / "outline.md"
        if outline_path.exists():
            try:
                existing_outline = outline_path.read_text(encoding="utf-8")
                if existing_outline.strip():
                    outline_cached = True
                    _, index_entries, outline_chapters = outline_gen.generate(
                        topics, scope=scope, syllabus_text=syllabus_text,
                    )
            except Exception:
                pass

    if not outline_cached:
        _, index_entries, outline_chapters = outline_gen.generate(
            topics, scope=scope, syllabus_text=syllabus_text,
        )

    # ── 10. Write indice ──────────────────────────────────────────────
    chapters_dir = output_dir / "chapters"
    existing_chapters = _load_existing_chapters(chapters_dir, len(outline_chapters))
    _write_indice(output_dir, outline_chapters, None, existing_chapters)

    template = load_template(template_path)

    return {
        "topics": topics,
        "all_chunks": all_chunks,
        "extracted_images": extracted_images,
        "outline_chapters": outline_chapters,
        "index_entries": index_entries,
        "client": client,
        "cfg": cfg,
        "template": template,
        "scope": scope,
        "effective_depth": effective_depth,
        "focus_topic": focus_topic,
        "args": args,
        "chunk_id_map": chunk_id_map,
        "output_dir": output_dir,
        "chapters_dir": chapters_dir,
        "existing_chapters": existing_chapters,
    }


def run_chapters_phase(
    state: dict,
    selected_indices: set[int] | None = None,
    progress_callback: callable | None = None,
) -> tuple[Path, dict]:
    """Run steps 7-8: chapter generation + merge.

    Parameters
    ----------
    state:
        The state dict returned by :func:`run_outline_phase`.
    selected_indices:
        0-based indices of chapters to generate. ``None`` means all.
    progress_callback:
        Optional ``(step: int, message: str)`` callback.

    Returns
    -------
    tuple[Path, dict]
        ``(output_path, validation)`` — same as :func:`run_pipeline`.
    """
    def _progress(step: int, msg: str) -> None:
        logger.info("Step %d/8: %s", step, msg)
        if progress_callback:
            progress_callback(step, msg)

    topics = state["topics"]
    all_chunks = state["all_chunks"]
    extracted_images = state["extracted_images"]
    outline_chapters = state["outline_chapters"]
    index_entries = state["index_entries"]
    client = state["client"]
    cfg = state["cfg"]
    template = state["template"]
    scope = state["scope"]
    effective_depth = state["effective_depth"]
    focus_topic = state["focus_topic"]
    args = state["args"]
    chunk_id_map = state["chunk_id_map"]
    output_dir = state["output_dir"]
    chapters_dir = state["chapters_dir"]

    _progress(7, "Generating chapters...")
    chapters_md: list[str] = []
    covered_topic_indices: set[int] = set()

    existing_chapters_local = _load_existing_chapters(chapters_dir, len(outline_chapters))
    selected_chapters_local = selected_indices

    if scope == "topic":
        for topic in topics:
            logger.info("  Generating chapter: %s", topic.name)
            topic_chunks = _collect_chunks_for_topic(topic, all_chunks, chunk_id_map)
            topic_images = _collect_images_for_topic(topic, extracted_images, args.no_images)
            chapter_md, _, _ = generate_chapter(
                topic, topic_chunks, template,
                depth_level=effective_depth,
                candidate_images=topic_images,
                scope=scope, client=client, cfg=cfg,
            )
            chapters_md.append(chapter_md)
    else:
        for ch_idx, outline_ch in enumerate(outline_chapters):
            if selected_chapters_local is not None and ch_idx not in selected_chapters_local:
                logger.info("  Skipping chapter %d: %s (not selected)", ch_idx + 1, outline_ch.title)
                continue

            if ch_idx in existing_chapters_local:
                logger.info("  Chapter %d: %s — already on disk", ch_idx + 1, outline_ch.title)
                chapter_path = chapters_dir / _chapter_filename(ch_idx, outline_ch.title)
                if chapter_path.exists():
                    chapters_md.append(chapter_path.read_text(encoding="utf-8"))
                continue

            logger.info("  Generating chapter %d: %s", ch_idx + 1, outline_ch.title)

            matched_topics = [
                topics[i] for i in outline_ch.topic_indices if i < len(topics)
            ]
            for i in outline_ch.topic_indices:
                if i < len(topics):
                    covered_topic_indices.add(i)

            all_topic_chunks = []
            all_topic_images = []
            for t in matched_topics:
                all_topic_chunks.extend(_collect_chunks_for_topic(t, all_chunks, chunk_id_map))
                all_topic_images.extend(_collect_images_for_topic(t, extracted_images, args.no_images))

            if not matched_topics and topics:
                best_topic = _find_closest_topic_for_chapter(outline_ch.title, topics, covered_topic_indices)
                if best_topic:
                    idx = topics.index(best_topic)
                    covered_topic_indices.add(idx)
                    all_topic_chunks.extend(_collect_chunks_for_topic(best_topic, all_chunks, chunk_id_map))
                    all_topic_images.extend(_collect_images_for_topic(best_topic, extracted_images, args.no_images))
                    matched_topics.append(best_topic)

            seen_content: set[str] = set()
            unique_chunks = []
            for c in all_topic_chunks:
                if c.content not in seen_content:
                    seen_content.add(c.content)
                    unique_chunks.append(c)

            seen_img_ids: set[str] = set()
            unique_images = []
            for img in all_topic_images:
                if img.id not in seen_img_ids:
                    seen_img_ids.add(img.id)
                    unique_images.append(img)

            primary_topic = matched_topics[0] if matched_topics else Topic(
                name=outline_ch.title, description=outline_ch.title,
            )
            chapter_entries = _build_chapter_local_entries(outline_ch)

            chapter_md, _, _ = generate_chapter(
                primary_topic, unique_chunks, template,
                depth_level=effective_depth,
                candidate_images=unique_images if not args.no_images else [],
                index_entries=chapter_entries,
                outline_chapter=outline_ch,
                scope="full", client=client, cfg=cfg,
            )
            chapters_md.append(chapter_md)
            _save_chapter_file(chapters_dir, ch_idx, outline_ch.title, chapter_md)
            existing_chapters_local.add(ch_idx)
            _write_indice(output_dir, outline_chapters, selected_chapters_local, existing_chapters_local)

        for idx, topic in enumerate(topics):
            if idx not in covered_topic_indices:
                logger.info("  Generating chapter for uncovered topic: %s", topic.name)
                topic_chunks = _collect_chunks_for_topic(topic, all_chunks, chunk_id_map)
                topic_images = _collect_images_for_topic(topic, extracted_images, args.no_images)
                chapter_md, _, _ = generate_chapter(
                    topic, topic_chunks, template,
                    depth_level=effective_depth,
                    candidate_images=topic_images if not args.no_images else [],
                    scope=scope, client=client, cfg=cfg,
                )
                chapters_md.append(chapter_md)

    # ── Merge ──────────────────────────────────────────────────────────
    _progress(8, "Assembling final output...")
    output_path = merge_chapters(
        chapters_md, index_entries, cfg,
        topics=topics,
        focus_topic=focus_topic if scope == "topic" else None,
    )

    # ── Validate ───────────────────────────────────────────────────────
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

    logger.info("Output: %s", output_path.resolve())
    logger.info("Validation: %s", validation["overall"])
    return output_path, validation


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
