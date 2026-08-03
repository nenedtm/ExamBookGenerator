"""Template engine for the ExamBookGenerator pipeline.

Reads a user-defined ``template.md`` and renders the final Markdown output
by replacing the supported placeholders with the generated content.

Supported variables
-------------------
``{{title}}``
    Book or chapter heading.

``{{toc}}``
    **Programmatically** built from ``list[IndexEntry]`` — never generated
    by the LLM.  Rendered as a Markdown list with ``- [Title](#anchor)``
    syntax, indented one level for sub-sections (``level >= 2``).  When the
    placeholder sits inside a blockquote line (e.g. ``> {{toc}}`` inside an
    Obsidian callout), every rendered line keeps the ``>`` prefix so the
    whole index stays inside the callout.  When ``include_toc`` is *False*,
    the placeholder is replaced with an empty string (and any trailing blank
    line produced by the template is collapsed).

``{{content}}``
    The LLM-generated Markdown body.

``{{sources}}``
    Bibliography / reference list.

``{{images}}``
    Optional — only rendered when images are provided.  Each entry is
    ``![caption](path)`` with the caption in italics.

Usage::

    from pipeline.template_engine import load_template, apply_template
    from core.models import IndexEntry

    tpl = load_template("template.md")
    out = apply_template(
        tpl,
        title="Linear Algebra",
        content="## Vector Spaces\\n...",
        index_entries=[
            IndexEntry(title="Intro", anchor="intro", level=1, order=0),
        ],
        sources="1. ...\n2. ...",
        images=[{"caption": "Matrix diagram", "path": "images/matrix.png"}],
        include_toc=True,
    )
"""

from __future__ import annotations

import re
from pathlib import Path

from core.models import IndexEntry
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Exceptions ───────────────────────────────────────────────────────────────


class TemplateError(Exception):
    """Raised when template loading or rendering fails."""


class TemplateNotFoundError(TemplateError):
    """Raised when the template file does not exist."""


class TemplateRenderError(TemplateError):
    """Raised when a required variable is missing or rendering fails."""


# ── Loading ──────────────────────────────────────────────────────────────────


def load_template(path: str | Path) -> str:
    """Read and return the raw template content from *path*.

    Parameters
    ----------
    path:
        Filesystem path to a Markdown template file.

    Returns
    -------
    str
        The template text with ``{{…}}`` placeholders intact.

    Raises
    ------
    TemplateNotFoundError
        If *path* does not exist.
    TemplateError
        If the file cannot be read.
    """
    p = Path(path)
    if not p.exists():
        raise TemplateNotFoundError(f"Template not found: {p}")
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise TemplateError(f"Cannot read template {p}: {exc}") from exc

    logger.debug("Loaded template from %s (%d chars)", p, len(content))
    return content


# ── TOC rendering ────────────────────────────────────────────────────────────


def render_toc(index_entries: list[IndexEntry]) -> str:
    """Render *index_entries* into a Markdown table-of-contents string.

    Level 1 entries are top-level list items; level 2+ entries are
    indented with two spaces per level.

    Returns an empty string if *index_entries* is empty.
    """
    if not index_entries:
        return ""

    lines: list[str] = []
    for entry in index_entries:
        indent = "  " * (entry.level - 1)
        lines.append(f"{indent}- [{entry.title}](#{entry.anchor})")
    return "\n".join(lines)


# ── Images block ─────────────────────────────────────────────────────────────


def render_images(images: list[dict[str, str]]) -> str:
    """Render a list of image descriptors into a Markdown block.

    Each dict is expected to have keys ``caption`` and ``path``.
    If no images are provided, returns an empty string.
    """
    if not images:
        return ""

    lines: list[str] = []
    for img in images:
        caption = img.get("caption", "")
        path = img.get("path", "")
        lines.append(f"![{caption}]({path})")
        if caption:
            lines.append(f"*{caption}*")
        lines.append("")  # blank line between images

    # Remove trailing blank line
    return "\n".join(lines).rstrip()


# ── Template rendering ───────────────────────────────────────────────────────


def _render_toc_placeholder(template: str, toc_block: str) -> str:
    """Replace the ``{{toc}}`` placeholder in *template* with *toc_block*.

    When the placeholder line starts with a Markdown blockquote marker
    (``> {{toc}}`` — typical inside an Obsidian callout), every rendered
    TOC line is prefixed with the same marker so the index stays inside
    the blockquote.  If *toc_block* is empty, the whole placeholder line
    (including its blockquote marker) is removed.
    """
    match = re.search(r"^(\s*>+\s*)\{\{toc\}\}\s*$", template, re.MULTILINE)
    if match:
        if not toc_block:
            return template.replace(match.group(0), "")
        prefix = match.group(1)
        indented = "\n".join(prefix + line for line in toc_block.split("\n"))
        return template.replace(match.group(0), indented)
    return template.replace("{{toc}}", toc_block)


def apply_template(
    template: str,
    *,
    title: str = "",
    content: str = "",
    index_entries: list[IndexEntry] | None = None,
    sources: str = "",
    images: list[dict[str, str]] | None = None,
    include_toc: bool = True,
) -> str:
    """Replace placeholders in *template* with the provided values.

    Parameters
    ----------
    template:
        Raw Markdown template text (from :func:`load_template`).
    title:
        Value for ``{{title}}``.
    content:
        Value for ``{{content}}``.
    index_entries:
        Entries used to build ``{{toc}}``.  When *None* or empty and
        *include_toc* is *True*, the ``{{toc}}`` placeholder is
        replaced with an empty string.  When the placeholder line in
        the template carries a blockquote prefix (``> {{toc}}``), the
        rendered list is kept inside the blockquote.
    sources:
        Value for ``{{sources}}``.
    images:
        List of dicts with ``caption`` and ``path`` keys, rendered into
        ``{{images}}``.  When *None* or empty, the block is empty.
    include_toc:
        When *False*, ``{{toc}}`` is always replaced with an empty
        string regardless of *index_entries*.

    Returns
    -------
    str
        The fully-rendered Markdown text.

    Raises
    ------
    TemplateRenderError
        If ``{{title}}`` or ``{{content}}`` placeholders are absent from
        the template.
    """
    # Validate required placeholders
    for var in ("title", "content"):
        if f"{{{{{var}}}}}" not in template:
            raise TemplateRenderError(
                f"Template is missing required variable {{{{{var}}}}}"
            )

    # Build TOC block
    toc_block = ""
    if include_toc and index_entries:
        toc_block = render_toc(index_entries)

    # Build images block
    images_block = render_images(images or [])

    # Build sources block (raw passthrough)
    sources_block = sources

    # ── Substitution order matters: TOC must land between title and content ──
    # Replace in reverse of desired visual order so that earlier placeholders
    # are not mangled by later ones.
    out = template
    out = out.replace("{{images}}", images_block)
    out = out.replace("{{sources}}", sources_block)
    out = out.replace("{{content}}", content)
    out = _render_toc_placeholder(out, toc_block)
    out = out.replace("{{title}}", title)

    # ── Post-processing ───────────────────────────────────────────────────
    # When TOC is empty, collapse the placeholder line so we don't leave
    # a blank line between title and content.
    out = _collapse_empty_line_block(out, "{{toc}}")

    logger.debug("Template rendered (%d chars)", len(out))
    return out


def _collapse_empty_line_block(text: str, marker: str) -> str:
    """Collapse a block where *marker* was replaced by an empty string.

    If the replacement leaves a blank line (or two) between the
    surrounding paragraphs, compress them to at most one blank line.
    """
    # Pattern: title line + blank line(s) + next non-blank line
    pattern = re.compile(
        r"([^\n]+)\n\n+([^\n])",
    )
    return pattern.sub(r"\1\n\n\2", text)


# ── Anchor verification helper ──────────────────────────────────────────────


def verify_anchors(
    content: str,
    index_entries: list[IndexEntry],
) -> list[str]:
    """Return a list of anchor IDs from *index_entries* whose ``#anchor``
    is **not** found as a heading slug in *content*.

    Useful for detecting mismatches between the LLM-generated content
    headings and the programme-generated TOC anchors.

    A heading ``## Vector Spaces`` produces the slug ``vector-spaces``
    which matches ``#vector-spaces`` in the TOC.
    """
    # Extract all heading slugs from the content
    heading_pattern = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
    found_slugs: set[str] = set()
    for m in heading_pattern.finditer(content):
        heading_text = m.group(1).strip()
        slug = _heading_to_slug(heading_text)
        found_slugs.add(slug)

    missing: list[str] = []
    for entry in index_entries:
        if entry.anchor not in found_slugs:
            missing.append(entry.anchor)
    return missing


def _heading_to_slug(text: str) -> str:
    """Convert a Markdown heading text to the slug GitHub would generate.

    This mirrors the simple slugify used for TOC anchors:
    lowercase, strip non-alphanumeric, collapse spaces to hyphens.
    """
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    slug = ascii_text.lower()
    slug = re.sub(r"[^a-z0-9 \-]", "", slug)
    slug = re.sub(r"[\s\-]+", "-", slug).strip("-")
    return slug
