import json
import re


def sanitize_llm_json(text: str) -> str:
    """Fix invalid JSON escape sequences produced by LLMs.

    LLMs often emit backslash escapes like ``\\s``, ``\\c``, ``\\p`` inside
    JSON strings.  These are not valid JSON escapes.  This function replaces
    each invalid ``\\X`` with just ``X``, preserving the intended character.
    """
    _INVALID_ESCAPE_RE = re.compile(r'\\(?!["\\\/bfnrtuu])')

    def _fix(m: re.Match) -> str:
        return m.group(0)[1]

    return _INVALID_ESCAPE_RE.sub(_fix, text)


def parse_llm_json(text: str, *, label: str = "LLM") -> dict:
    """Parse JSON from LLM output, stripping fences and fixing escapes.

    Raises ``ValueError`` with a descriptive message on failure.
    """
    text = text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    text = sanitize_llm_json(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"{label} returned unrecognisable JSON.  "
                f"Raw (first 500 chars): {text[:500]!r}"
            )
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label} returned invalid JSON: {exc}.  "
                f"Raw (first 500 chars): {text[:500]!r}"
            ) from exc
