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

    Also handles bold markers, dash separators, bullet points, etc.

    Returns a dict with key "chapters" or None if parsing fails.
    """
    chapters: list[dict[str, object]] = []
    current_title: str | None = None
    current_sections: list[str] = []

    # Strip leading/trailing whitespace and remove empty leading lines
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    for line in lines:
        # Match chapter headings:
        #   **Chapter 1: Title**   (bold markdown)
        #   ### Chapter 1: Title   (markdown)
        #   ## Chapter 1: Title    (markdown)
        #   Chapter 1: Title       (plain)
        #   Chapter 1 - Title      (dash separator)
        #   Chapter 1. Title       (dot separator)
        header_match = re.match(
            r'^\*{0,2}(?:#{2,3}\s+)?Chapter\s+\d+\s*[:.\-]\s*(.+?)\*{0,2}$',
            line, re.IGNORECASE,
        )
        if header_match:
            if current_title and current_sections:
                chapters.append({"title": current_title, "sections": current_sections})
            current_title = header_match.group(1).strip()
            current_sections = []
            continue

        if current_title is None:
            continue

        # Match numbered list items: 1. Section title
        # Also handles bold: **1. Section** or *1. Section*
        # Also handles sub-numbering: 1.1 Section, 1.1. Section
        item_match = re.match(
            r'^\*{0,2}\d+(?:\.\d+)*\.?\s+\*?\*?(.+?)\*?\*?\s*$',
            line,
        )
        if item_match:
            section = item_match.group(1).strip()
            if section.lower() not in ("sections:", "sections"):
                current_sections.append(section)
                continue

        # Match bullet points: - Section or * Section
        bullet_match = re.match(r'^[-*]\s+\*?\*?(.+?)\*?\*?\s*$', line)
        if bullet_match:
            section = bullet_match.group(1).strip()
            if section.lower() not in ("sections:", "sections"):
                current_sections.append(section)
                continue

    # Save last chapter
    if current_title and current_sections:
        chapters.append({"title": current_title, "sections": current_sections})

    if chapters:
        return {"chapters": chapters}
    return None


def _repair_truncated_json(text: str) -> str | None:
    """Attempt to repair JSON truncated by token limits.

    When the LLM output is cut off mid-object, this function tries to
    find the last complete entry and close all open structures in the
    correct reverse-nesting order.

    Returns the repaired JSON string, or *None* if repair fails.
    """
    # Find the last complete array entry: "},\n" or "]\n" or "}" or "]"
    # Work backwards from the end to find the last complete item
    last_complete = -1

    for pattern in (r'\},\s*$', r'\],\s*$', r'\}', r'\]'):
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            candidate = m.end()
            if candidate > last_complete:
                last_complete = candidate

    if last_complete <= 0:
        return None

    # Take everything up to the last complete entry
    repaired = text[:last_complete].rstrip()

    # Remove trailing comma if present (before closing)
    repaired = re.sub(r',\s*$', '', repaired)

    # Track the actual stack of open structures in order
    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in repaired:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            stack.append('}')
        elif ch == '[':
            stack.append(']')
        elif ch in ('}', ']'):
            if stack and stack[-1] == ch:
                stack.pop()

    # Close in reverse nesting order
    suffix = ''.join(reversed(stack))
    result = repaired + suffix

    # Verify it parses
    try:
        obj = json.loads(result, strict=False)
        if isinstance(obj, dict) and "chapters" in obj:
            return result
    except json.JSONDecodeError:
        pass

    # Try wrapping in {"chapters": ...} if the result is just a list
    try:
        obj = json.loads(result, strict=False)
        if isinstance(obj, list):
            wrapped = json.dumps({"chapters": obj}, ensure_ascii=False)
            return wrapped
    except json.JSONDecodeError:
        pass

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

    # Try repairing truncated JSON (token limit cut-off)
    repaired = _repair_truncated_json(text)
    if repaired is not None:
        try:
            return json.loads(repaired, strict=False)
        except json.JSONDecodeError:
            pass
        # Also try extracting from the repaired text
        start_r = repaired.find("{")
        end_r = repaired.rfind("}")
        if start_r != -1 and end_r > start_r:
            try:
                return json.loads(repaired[start_r : end_r + 1], strict=False)
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
