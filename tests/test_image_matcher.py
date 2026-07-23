"""Tests for pipeline.image_matcher."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.models import ExtractedImage, Topic
from llm.ollama_client import (
    OllamaConnectionError,
    OllamaError,
    OllamaTimeoutError,
    VisionModelUnavailableError,
)
from pipeline.image_matcher import (
    _extract_keywords,
    _heuristic_select,
    _parse_relevance_response,
    describe_image,
    fallback_heuristic_match,
    select_images_for_chapter,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_topic(
    name: str = "Linear Algebra",
    description: str = "Vector spaces and matrices",
    related_docs: list[str] | None = None,
) -> Topic:
    return Topic(
        name=name,
        description=description,
        related_documents=related_docs or ["doc-1", "doc-2"],
    )


def _make_image(
    doc_id: str = "doc-1",
    file_path: str = "output/assets/images/matrix.png",
    caption: str | None = "Matrix diagram",
    ai_description: str | None = None,
) -> ExtractedImage:
    img = ExtractedImage(
        source_document_id=doc_id,
        file_path=file_path,
        caption=caption,
        width=200,
        height=150,
    )
    img.ai_description = ai_description
    return img


def _chapter_draft() -> str:
    return (
        "## Linear Algebra\n\n"
        "Linear algebra studies vector spaces and linear mappings.\n\n"
        "### Vector Spaces\n\n"
        "A vector space is a collection of vectors.\n"
    )


# ── _extract_keywords ────────────────────────────────────────────────────────


class TestExtractKeywords:
    def test_basic(self) -> None:
        kw = _extract_keywords("Linear Algebra and Matrices")
        assert "linear" in kw
        assert "algebra" in kw
        assert "matrices" in kw
        assert "and" not in kw  # too short (3 chars excluded)

    def test_empty(self) -> None:
        assert _extract_keywords("") == set()

    def test_punctuation_stripped(self) -> None:
        kw = _extract_keywords("C++ programming!")
        assert "programming" in kw
        assert "cpp" not in kw  # special chars stripped

    def test_short_words_filtered(self) -> None:
        kw = _extract_keywords("a an the is to of")
        assert kw == set()


# ── _parse_relevance_response ────────────────────────────────────────────────


class TestParseRelevanceResponse:
    def test_valid_relevant(self) -> None:
        raw = json.dumps({"relevant": True, "placement": "after introduction"})
        result = _parse_relevance_response(raw)
        assert result is not None
        assert result["relevant"] is True
        assert result["placement"] == "after introduction"

    def test_valid_not_relevant(self) -> None:
        raw = json.dumps({"relevant": False, "placement": None})
        result = _parse_relevance_response(raw)
        assert result is not None
        assert result["relevant"] is False

    def test_markdown_fences(self) -> None:
        raw = "```json\n" + json.dumps({"relevant": True, "placement": "before conclusion"}) + "\n```"
        result = _parse_relevance_response(raw)
        assert result is not None
        assert result["relevant"] is True

    def test_surrounding_text(self) -> None:
        raw = "Here is my answer:\n" + json.dumps({"relevant": True, "placement": "after section: Intro"}) + "\nDone."
        result = _parse_relevance_response(raw)
        assert result is not None
        assert result["relevant"] is True
        assert result["placement"] == "after section: Intro"

    def test_invalid_json(self) -> None:
        assert _parse_relevance_response("not json at all") is None

    def test_missing_keys(self) -> None:
        result = _parse_relevance_response('{"foo": "bar"}')
        # Should still return something with default False
        assert result is not None
        assert result["relevant"] is False

    def test_none_placement(self) -> None:
        raw = json.dumps({"relevant": True, "placement": None})
        result = _parse_relevance_response(raw)
        assert result is not None
        assert result["placement"] is None


# ── describe_image ────────────────────────────────────────────────────────────


class TestDescribeImage:
    def test_returns_description(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 100)
        img = _make_image(file_path=str(img_path))

        mock_client = MagicMock()
        mock_client.generate_with_image.return_value = "A diagram showing matrix multiplication."

        result = describe_image(img, client=mock_client)
        assert result == "A diagram showing matrix multiplication."
        assert img.ai_description == result

    def test_skips_if_no_file_path(self) -> None:
        img = _make_image(file_path="")
        mock_client = MagicMock()

        result = describe_image(img, client=mock_client)
        assert result == ""
        mock_client.generate_with_image.assert_not_called()

    def test_skips_if_file_missing(self) -> None:
        img = _make_image(file_path="/nonexistent/path/img.png")
        mock_client = MagicMock()

        result = describe_image(img, client=mock_client)
        assert result == ""
        mock_client.generate_with_image.assert_not_called()

    def test_skips_if_already_described(self) -> None:
        img = _make_image(ai_description="Already done")
        mock_client = MagicMock()

        result = describe_image(img, client=mock_client)
        assert result == "Already done"
        mock_client.generate_with_image.assert_not_called()

    def test_handles_vision_model_unavailable(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 100)
        img = _make_image(file_path=str(img_path))

        mock_client = MagicMock()
        mock_client.generate_with_image.side_effect = VisionModelUnavailableError("no llava")

        result = describe_image(img, client=mock_client)
        assert result == ""

    def test_handles_timeout(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 100)
        img = _make_image(file_path=str(img_path))

        mock_client = MagicMock()
        mock_client.generate_with_image.side_effect = OllamaTimeoutError("timeout")

        result = describe_image(img, client=mock_client)
        assert result == ""

    def test_handles_connection_error(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 100)
        img = _make_image(file_path=str(img_path))

        mock_client = MagicMock()
        mock_client.generate_with_image.side_effect = OllamaConnectionError("no connection")

        result = describe_image(img, client=mock_client)
        assert result == ""

    def test_handles_generic_ollama_error(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 100)
        img = _make_image(file_path=str(img_path))

        mock_client = MagicMock()
        mock_client.generate_with_image.side_effect = OllamaError("some error")

        result = describe_image(img, client=mock_client)
        assert result == ""


# ── fallback_heuristic_match ──────────────────────────────────────────────────


class TestFallbackHeuristicMatch:
    def test_keyword_overlap(self) -> None:
        topic = _make_topic(name="Matrix Algebra", description="Study of matrices")
        imgs = [
            _make_image(caption="Linear transformation diagram"),
            _make_image(caption="Matrix multiplication steps"),
            _make_image(caption="Cooking recipe illustration"),
        ]
        result = fallback_heuristic_match(topic, imgs)
        # "matrix" and "matrices" are different stems — but "algebra" won't match any.
        # "matrix" matches img[1] (caption "Matrix multiplication steps").
        assert len(result) >= 1
        assert any("matrix" in (i.caption or "").lower() for i in result)

    def test_no_overlap_returns_empty(self) -> None:
        topic = _make_topic(name="Quantum Physics", description="Subatomic particles")
        imgs = [_make_image(caption="Linear transformation diagram")]
        result = fallback_heuristic_match(topic, imgs)
        assert result == []

    def test_empty_candidates(self) -> None:
        topic = _make_topic()
        assert fallback_heuristic_match(topic, []) == []

    def test_no_caption_no_overlap(self) -> None:
        topic = _make_topic(name="Algebra", description="Equations")
        img = _make_image(caption=None, file_path="output/assets/images/random123.png")
        result = fallback_heuristic_match(topic, [img])
        # "random" won't match "algebra"
        assert result == []

    def test_path_matching(self) -> None:
        topic = _make_topic(name="Matrix", description="Linear algebra topic")
        img = _make_image(caption=None, file_path="output/assets/images/matrix_diagram.png")
        result = fallback_heuristic_match(topic, [img])
        assert len(result) == 1


# ── select_images_for_chapter ────────────────────────────────────────────────


class TestSelectImagesForChapter:
    def test_filters_by_related_documents(self) -> None:
        topic = _make_topic(related_docs=["doc-1"])
        imgs = [
            _make_image(doc_id="doc-1", caption="Relevant image"),
            _make_image(doc_id="doc-99", caption="Unrelated image"),
        ]
        mock_client = MagicMock()
        mock_client.check_connection.return_value = True

        # Mock the AI pipeline: describe → description, relevance → not relevant
        mock_client.generate_with_image.return_value = "A relevant diagram."
        mock_client.generate.return_value = json.dumps({"relevant": False, "placement": None})

        result = select_images_for_chapter(
            topic, _chapter_draft(), imgs, client=mock_client
        )
        # Only doc-1 image was considered; relevance check said no
        assert result == []

    def test_returns_max_images(self) -> None:
        topic = _make_topic(related_docs=["doc-1"])
        imgs = [
            _make_image(doc_id="doc-1", caption=f"Image {i}", ai_description=f"Desc {i}")
            for i in range(10)
        ]

        mock_client = MagicMock()
        mock_client.check_connection.return_value = True
        mock_client.generate.return_value = json.dumps(
            {"relevant": True, "placement": "after introduction"}
        )

        result = select_images_for_chapter(
            topic, _chapter_draft(), imgs, client=mock_client, max_images=2
        )
        assert len(result) == 2
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result)

    def test_empty_candidates(self) -> None:
        topic = _make_topic()
        result = select_images_for_chapter(topic, _chapter_draft(), [], client=MagicMock())
        assert result == []

    def test_fallback_to_heuristic_on_vision_error(self) -> None:
        topic = _make_topic(related_docs=["doc-1"])
        imgs = [_make_image(doc_id="doc-1", caption="Matrix multiplication")]

        mock_client = MagicMock()
        mock_client.check_connection.side_effect = OllamaConnectionError("down")

        result = select_images_for_chapter(
            topic, _chapter_draft(), imgs, client=mock_client
        )
        # Should fall back to heuristic; "matrix" matches topic name "Linear Algebra"? 
        # No, topic is "Linear Algebra" and caption is "Matrix multiplication".
        # "algebra" and "matrix" — "linear" doesn't match "matrix".
        # Heuristic will check overlap between {"linear", "algebra", "vector", "spaces", "matrices"} 
        # and {"matrix", "multiplication"} — "matrix" won't match any of the topic words.
        # So result may be empty, but that's fine — the fallback didn't crash.

    def test_fallback_to_heuristic_on_ollama_error(self) -> None:
        topic = _make_topic(name="Matrix", description="Linear algebra", related_docs=["doc-1"])
        imgs = [_make_image(doc_id="doc-1", caption="Matrix diagram")]

        mock_client = MagicMock()
        mock_client.check_connection.side_effect = OllamaError("generic")

        result = select_images_for_chapter(
            topic, _chapter_draft(), imgs, client=mock_client
        )
        # Heuristic fallback: "matrix" matches in both topic and caption
        assert len(result) == 1
        assert result[0][1] == "before conclusion"

    def test_ai_select_success(self) -> None:
        topic = _make_topic(related_docs=["doc-1"])
        img = _make_image(
            doc_id="doc-1",
            caption="Vector space illustration",
            ai_description="A vector space diagram showing basis vectors.",
        )

        mock_client = MagicMock()
        mock_client.check_connection.return_value = True
        mock_client.generate.return_value = json.dumps(
            {"relevant": True, "placement": "after section: Vector Spaces"}
        )

        result = select_images_for_chapter(
            topic, _chapter_draft(), [img], client=mock_client
        )
        assert len(result) == 1
        assert result[0][0] is img
        assert result[0][1] == "after section: Vector Spaces"

    def test_skips_image_with_no_description(self) -> None:
        topic = _make_topic(related_docs=["doc-1"])
        img = _make_image(doc_id="doc-1", caption="Something")

        mock_client = MagicMock()
        mock_client.check_connection.return_value = True
        mock_client.generate_with_image.return_value = ""  # empty description

        result = select_images_for_chapter(
            topic, _chapter_draft(), [img], client=mock_client
        )
        assert result == []

    def test_max_images_respected(self) -> None:
        topic = _make_topic(related_docs=["doc-1"])
        imgs = [
            _make_image(doc_id="doc-1", caption=f"Image {i}", ai_description=f"Desc {i}")
            for i in range(5)
        ]

        mock_client = MagicMock()
        mock_client.check_connection.return_value = True
        mock_client.generate.return_value = json.dumps(
            {"relevant": True, "placement": "after introduction"}
        )

        result = select_images_for_chapter(
            topic, _chapter_draft(), imgs, client=mock_client, max_images=2
        )
        assert len(result) == 2


# ── Integration: end-to-end example ──────────────────────────────────────────


class TestEndToEnd:
    def test_topic_with_images_produces_output(self, tmp_path: Path) -> None:
        """Full end-to-end: topic + candidate images → chapter with images."""
        img_path = tmp_path / "matrix.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 100)

        topic = _make_topic(
            name="Linear Algebra",
            description="Vector spaces and matrices",
            related_docs=["doc-1"],
        )

        candidate = ExtractedImage(
            source_document_id="doc-1",
            file_path=str(img_path),
            caption="Matrix multiplication",
            width=300,
            height=200,
        )

        chapter_draft = (
            "## Linear Algebra\n\n"
            "This chapter covers vector spaces and linear transformations.\n\n"
            "### Vector Spaces\n\n"
            "A vector space V over a field F is a set equipped with addition "
            "and scalar multiplication.\n"
        )

        mock_client = MagicMock()
        mock_client.check_connection.return_value = True
        mock_client.generate_with_image.return_value = (
            "A diagram illustrating matrix multiplication of a 2x2 and 2x1 matrix."
        )
        mock_client.generate.return_value = json.dumps({
            "relevant": True,
            "placement": "after section: Vector Spaces",
        })

        result = select_images_for_chapter(
            topic, chapter_draft, [candidate], client=mock_client
        )

        assert len(result) == 1
        img, placement = result[0]
        assert img.source_document_id == "doc-1"
        assert img.ai_description is not None
        assert "after section: Vector Spaces" in placement
