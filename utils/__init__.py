import json
import re


_VALID_ESCAPES = set("\"\\/\bfnrtu")


def sanitize_llm_json(text: str) -> str:
    """Fix invalid JSON escape sequences produced by LLMs.

    LLMs often emit backslash escapes like ``\\s``, ``\\c``, ``\\p`` inside
    JSON strings.  These are not valid JSON escapes.  This function removes
    the backslash from each invalid ``\\X``, preserving valid ones.
    """
    def _fix(m: re.Match) -> str:
        ch = m.group(1)
        if ch in _VALID_ESCAPES:
            return '\\' + ch
        return ch

    return re.sub(r'\\(.)', _fix, text)


def _try_parse_markdown_outline(text: str) -> dict | None:
    """Attempt to parse a markdown-formatted outline into JSON.

    Handles the common pattern where LLMs return markdown instead of JSON:
        ### Chapter Title
        Sections:
        1. Section one
        2. Section two

    Returns a dict with key "chapters" or None if parsing fails.
    """
    chapters: list[dict[str, object]] = []
    current_title: str | None = None
    current_sections: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()

        # Match markdown headers: ### Chapter Title or ## Chapter Title
        header_match = re.match(r'^#{2,3}\s+(?:Chapter\s+\d+:\s*)?(.+)', stripped, re.IGNORECASE)
        if header_match:
            # Save previous chapter if exists
            if current_title and current_sections:
                chapters.append({"title": current_title, "sections": current_sections})
            current_title = header_match.group(1).strip().rstrip(":")
            current_sections = []
            continue

        # Match numbered list items: 1. Section title or - Section title
        if current_title is not None:
            item_match = re.match(r'^\d+\.\s+\*?\*?(.+?)\*?\*?\s*$', stripped)
            if item_match:
                section = item_match.group(1).strip()
                if section.lower() not in ("sections:", "sections"):
                    current_sections.append(section)
                    continue

    # Save last chapter
    if current_title and current_sections:
        chapters.append({"title": current_title, "sections": current_sections})

    if chapters:
        return {"chapters": chapters}
    return None


def parse_llm_json(text: str, *, label: str = "LLM") -> dict:
    """Parse JSON from LLM output, stripping fences and fixing escapes.

    Falls back to parsing markdown-formatted outlines if JSON parsing fails.
    Raises ``ValueError`` with a descriptive message on failure.
    """
    text = text.strip()

    # Strip code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    text = sanitize_llm_json(text)

    # Try standard JSON parsing
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from surrounding text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1], strict=False)
        except json.JSONDecodeError:
            pass

    # Fallback: try parsing as markdown outline
    md_result = _try_parse_markdown_outline(text)
    if md_result is not None:
        return md_result

    # All attempts failed
    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"{label} returned unrecognisable JSON.  "
            f"Raw (first 500 chars): {text[:500]!r}"
        )
    raise ValueError(
        f"{label} returned invalid JSON.  "
        f"Raw (first 500 chars): {text[:500]!r}"
    )
