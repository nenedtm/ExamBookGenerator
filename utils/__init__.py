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


def _repair_truncated_json(text: str) -> str | None:
    """Attempt to repair JSON truncated by token limits.

    When the LLM output is cut off mid-object, this function tries to
    find the last complete entry and close all open structures.

    Returns the repaired JSON string, or *None* if repair fails.
    """
    # Find the last complete array entry: ",\n" or "},\n" or "]\n"
    # Work backwards from the end to find the last complete item
    last_complete = -1

    # Look for last complete object in an array: },
    for pattern in (r'\},\s*$', r'\}', r'\],\s*$', r'\]'):
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

    # Count open brackets/braces to determine what to close
    open_braces = 0
    open_brackets = 0
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
            open_braces += 1
        elif ch == '}':
            open_braces -= 1
        elif ch == '[':
            open_brackets += 1
        elif ch == ']':
            open_brackets -= 1

    # Close in reverse order: first close arrays, then objects
    suffix = ']' * max(0, open_brackets) + '}' * max(0, open_braces)

    # Add the "chapters" key wrapper if it seems to be missing
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
