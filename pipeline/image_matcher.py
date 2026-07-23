"""Image matcher — selects and positions relevant images per chapter.

Uses the Ollama vision model (``llava`` by default) to describe
extracted images and judge their relevance to a chapter draft.
When the vision model is unavailable, a lightweight keyword-based
heuristic is used instead.

Usage::

    from pipeline.image_matcher import select_images_for_chapter
    from llm.ollama_client import OllamaClient

    client = OllamaClient.from_config()
    matched = select_images_for_chapter(
        topic, chapter_draft, candidate_images, client=client,
    )
    for img, placement in matched:
        print(f"Insert {img.file_path} {placement}")
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Sequence

from core.models import ExtractedImage, Topic
from llm.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaTimeoutError,
    VisionModelUnavailableError,
)
from llm.prompt_manager import build_image_caption_prompt, build_image_relevance_prompt
from utils import parse_llm_json
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Skip images whose file is missing or unreadable.
_MIN_IMAGE_BYTES = 100

# Maximum number of candidates to evaluate with the vision model per call.
_MAX_CANDIDATES_FOR_VISION = 15


# ── describe_image ────────────────────────────────────────────────────────────


def describe_image(
    image: ExtractedImage,
    *,
    client: OllamaClient | None = None,
    context: str = "",
    timeout: float = 60.0,
) -> str:
    """Send *image* to the vision model and return a short English description.

    The result is also stored in ``image.ai_description`` for downstream
    reuse (avoiding duplicate calls).

    Parameters
    ----------
    image:
        The extracted image to describe.
    client:
        Ollama client instance.  When *None*, a default one is created.
    context:
        Optional surrounding text from the source document to help the
        model understand the image's origin.
    timeout:
        Per-image timeout in seconds.  If the vision model does not
        respond within this window the image is skipped.

    Returns
    -------
    str
        The description, or an empty string if the call failed.

    Raises
    ------
    (Never raises — failures are logged and return ``""``.)
    """
    if image.ai_description:
        return image.ai_description

    if not image.file_path:
        logger.warning("Image %s has no file_path — skipping description", image.id)
        return ""

    img_path = Path(image.file_path)
    if not img_path.is_file():
        logger.warning("Image file not found: %s — skipping", img_path)
        return ""

    if client is None:
        client = OllamaClient.from_config()

    prompt = build_image_caption_prompt(context)

    try:
        start = time.monotonic()
        description = client.generate_with_image(prompt, img_path)
        elapsed = time.monotonic() - start

        if elapsed > timeout:
            logger.warning(
                "Vision model took %.1fs (> %.1fs threshold) for image %s — "
                "description may be incomplete",
                elapsed, timeout, image.id,
            )

        description = description.strip()
        image.ai_description = description
        logger.debug("Image %s described (%d chars)", image.id, len(description))
        return description

    except VisionModelUnavailableError as exc:
        logger.warning(
            "Vision model unavailable for image %s: %s", image.id, exc
        )
        return ""
    except (OllamaTimeoutError, OllamaConnectionError) as exc:
        logger.warning(
            "Vision model error for image %s: %s", image.id, exc
        )
        return ""
    except OllamaError as exc:
        logger.warning(
            "Unexpected Ollama error for image %s: %s", image.id, exc
        )
        return ""
    except FileNotFoundError as exc:
        logger.warning("Image file disappeared: %s", exc)
        return ""
    except Exception as exc:
        logger.warning("Unexpected error describing image %s: %s", image.id, exc)
        return ""


# ── select_images_for_chapter ────────────────────────────────────────────────


def select_images_for_chapter(
    topic: Topic,
    chapter_draft: str,
    candidate_images: Sequence[ExtractedImage],
    *,
    client: OllamaClient | None = None,
    max_images: int = 3,
) -> list[tuple[ExtractedImage, str]]:
    """Select the most relevant images for a chapter and suggest placements.

    Parameters
    ----------
    topic:
        The topic the chapter covers (used to filter candidates by
        ``related_documents``).
    chapter_draft:
        The Markdown body of the chapter (already written or in draft form).
    candidate_images:
        All images extracted from the source documents.
    client:
        Ollama client instance.  When *None*, a default one is created.
    max_images:
        Maximum number of images to return (default 3).

    Returns
    -------
    list[tuple[ExtractedImage, str]]
        Each tuple is ``(image, placement)`` where *placement* is one of
        ``"after introduction"``, ``"after section: <title>"``, or
        ``"before conclusion"``.

    Notes
    -----
    * If the vision model is not available, falls back to
      :func:`fallback_heuristic_match` (no placement suggestions — the
      caller receives the images with a generic placement).
    * Individual image timeouts do not block the whole call — the image
      is simply skipped.
    """
    # ── 1. Filter candidates to documents related to this topic ────────
    related_ids = set(topic.related_documents)
    filtered = [
        img for img in candidate_images
        if img.source_document_id in related_ids
    ]

    if not filtered:
        logger.info(
            "No candidate images for topic '%s' (no matching document IDs)", topic.name
        )
        return []

    logger.info(
        "Topic '%s': %d candidate images after document filtering",
        topic.name, len(filtered),
    )

    # ── 2. Cap candidates for vision inference ─────────────────────────
    # Prioritise images that already have an ai_description (already analysed).
    described = [img for img in filtered if img.ai_description]
    undescribed = [img for img in filtered if not img.ai_description]
    ordered = described + undescribed[: max(0, _MAX_CANDIDATES_FOR_VISION - len(described))]
    candidates = ordered[:_MAX_CANDIDATES_FOR_VISION]

    # ── 3. Try AI-based matching ──────────────────────────────────────
    try:
        return _ai_select(topic, chapter_draft, candidates, client, max_images)
    except (VisionModelUnavailableError, OllamaConnectionError, OllamaTimeoutError):
        logger.warning(
            "Vision model unavailable — falling back to heuristic matching "
            "for topic '%s'",
            topic.name,
        )
        return _heuristic_select(topic, candidates, max_images)
    except OllamaError as exc:
        logger.warning(
            "Ollama error during image selection for topic '%s': %s — "
            "falling back to heuristic",
            topic.name, exc,
        )
        return _heuristic_select(topic, candidates, max_images)


def _ai_select(
    topic: Topic,
    chapter_draft: str,
    candidates: list[ExtractedImage],
    client: OllamaClient | None,
    max_images: int,
) -> list[tuple[ExtractedImage, str]]:
    """AI-powered image selection using the vision model."""
    if client is None:
        client = OllamaClient.from_config()

    # Quick connectivity check — if it fails, propagate the exception
    # so the caller can fall back to heuristics.
    if not client.check_connection():
        raise OllamaConnectionError("Ollama not reachable")

    matched: list[tuple[ExtractedImage, str]] = []

    for img in candidates:
        if len(matched) >= max_images:
            break

        # Step A: describe if needed
        description = describe_image(img, client=client)
        if not description:
            logger.debug("Skipping image %s — no description available", img.id)
            continue

        # Step B: relevance check
        prompt = build_image_relevance_prompt(chapter_draft, description)
        try:
            raw = client.generate(prompt)
        except OllamaError as exc:
            logger.warning("Relevance check failed for image %s: %s", img.id, exc)
            continue

        parsed = _parse_relevance_response(raw)
        if parsed is None or not parsed.get("relevant"):
            logger.debug("Image %s deemed not relevant", img.id)
            continue

        placement = parsed.get("placement", "before conclusion") or "before conclusion"
        matched.append((img, placement))
        logger.debug(
            "Image %s selected → %s", img.id, placement
        )

    return matched


def _parse_relevance_response(raw: str) -> dict | None:
    """Parse the JSON relevance response from the LLM.

    Returns a dict with ``"relevant"`` (bool) and ``"placement"`` (str | None),
    or *None* if parsing fails.
    """
    try:
        obj = parse_llm_json(raw, label="Image matcher")
    except ValueError:
        return None

    if not isinstance(obj, dict):
        return None

    relevant = obj.get("relevant", False)
    placement = obj.get("placement")

    return {"relevant": bool(relevant), "placement": placement}


# ── Heuristic fallback ───────────────────────────────────────────────────────


def _heuristic_select(
    topic: Topic,
    candidates: list[ExtractedImage],
    max_images: int,
) -> list[tuple[ExtractedImage, str]]:
    """Keyword-based heuristic selection — no LLM involved."""
    selected = fallback_heuristic_match(topic, candidates)
    # Assign generic placement for heuristic matches
    return [(img, "before conclusion") for img in selected[:max_images]]


def fallback_heuristic_match(
    topic: Topic,
    candidate_images: Sequence[ExtractedImage],
) -> list[ExtractedImage]:
    """Select images using simple keyword matching — no AI.

    This is the safe fallback when the vision model is unavailable.
    It compares words from the topic name and description against each
    image's original caption and file path.

    Returns an empty list if no strong signal is found — better no image
    than a wrong one.

    Parameters
    ----------
    topic:
        The topic to match against.
    candidate_images:
        Candidate images (should already be filtered by document ID).

    Returns
    -------
    list[ExtractedImage]
        Images that matched, in descending relevance order.
    """
    if not candidate_images:
        return []

    topic_words = _extract_keywords(
        f"{topic.name} {topic.description}"
    )

    if not topic_words:
        return []

    scored: list[tuple[int, ExtractedImage]] = []

    for img in candidate_images:
        caption_text = img.caption or ""
        path_text = Path(img.file_path).stem if img.file_path else ""
        img_text = f"{caption_text} {path_text}"
        img_words = _extract_keywords(img_text)

        if not img_words:
            continue

        overlap = topic_words & img_words
        score = len(overlap)

        if score > 0:
            scored.append((score, img))

    if not scored:
        logger.debug(
            "Heuristic match: no keyword overlap for topic '%s'", topic.name
        )
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    matched = [img for _, img in scored]

    logger.info(
        "Heuristic match: %d/%d images matched for topic '%s'",
        len(matched), len(candidate_images), topic.name,
    )
    return matched


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase keywords from *text*.

    Strips punctuation, lowercases, and filters out very short words.
    """
    if not text:
        return set()

    # Lowercase, keep only alphanumeric and spaces
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    words = set(cleaned.split())

    # Filter very short words (likely noise)
    return {w for w in words if len(w) > 3}
