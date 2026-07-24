"""Tests for pipeline.topic_analyzer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.models import Chunk, Document, Topic
from pipeline.topic_analyzer import (
    TopicAnalyzer,
    TopicAnalyzerError,
    TopicNotFoundError,
    _build_topic,
    _chunk_ids_from_response,
    _chunks_for_topic,
    _parse_topics_response,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _chunks() -> list[Chunk]:
    return [
        Chunk(id="c1", document_id="doc_a", content="Vector spaces and bases", position=0),
        Chunk(id="c2", document_id="doc_a", content="Eigenvalues and eigenvectors", position=1),
        Chunk(id="c3", document_id="doc_b", content="Derivatives and integrals", position=0),
        Chunk(id="c4", document_id="doc_b", content="Chain rule and product rule", position=1),
    ]


def _syllabus_doc() -> Document:
    return Document(
        id="syl1",
        title="Course Syllabus",
        content="1. Linear Algebra\n2. Calculus",
        is_syllabus=True,
    )


def _llm_topics_response() -> str:
    return json.dumps({
        "topics": [
            {
                "name": "Linear Algebra",
                "description": "Vector spaces and linear transformations",
                "related_documents": ["doc_a"],
                "chunk_ids": ["c1", "c2"],
                "subtopic_count": 2,
                "order_source": "syllabus",
                "syllabus_position": 0,
            },
            {
                "name": "Calculus",
                "description": "Derivatives, integrals, and limits",
                "related_documents": ["doc_b"],
                "chunk_ids": ["c3", "c4"],
                "subtopic_count": 2,
                "order_source": "syllabus",
                "syllabus_position": 1,
            },
        ]
    })


def _llm_single_topic_response() -> str:
    return json.dumps({
        "topics": [
            {
                "name": "Linear Algebra",
                "description": "Vector spaces and eigenvalues",
                "related_documents": ["doc_a"],
                "chunk_ids": ["c1", "c2"],
                "subtopic_count": 2,
                "order_source": "pedagogical",
                "syllabus_position": None,
            },
        ]
    })


# ── Response parsing ─────────────────────────────────────────────────────────

class TestParseTopicsResponse:
    def test_valid_json(self) -> None:
        raw = _llm_topics_response()
        result = _parse_topics_response(raw)
        assert len(result) == 2
        assert result[0]["name"] == "Linear Algebra"

    def test_markdown_fences_stripped(self) -> None:
        raw = "```json\n" + _llm_topics_response() + "\n```"
        result = _parse_topics_response(raw)
        assert len(result) == 2

    def test_surrounding_text(self) -> None:
        raw = "Here are the topics:\n" + _llm_topics_response() + "\nHope that helps!"
        result = _parse_topics_response(raw)
        assert len(result) == 2

    def test_missing_topics_key_raises(self) -> None:
        with pytest.raises(TopicAnalyzerError, match="missing the 'topics' array"):
            _parse_topics_response('{"other": []}')

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(TopicAnalyzerError, match="unrecognisable JSON"):
            _parse_topics_response("not json at all {{{")

    def test_empty_topics_list(self) -> None:
        result = _parse_topics_response('{"topics": []}')
        assert result == []


# ── Topic building ───────────────────────────────────────────────────────────

class TestBuildTopic:
    def test_basic(self) -> None:
        raw = {
            "name": "Algebra",
            "description": "Study of structures",
            "related_documents": ["d1"],
            "subtopic_count": 3,
            "order_source": "syllabus",
            "syllabus_position": 0,
        }
        t = _build_topic(raw, order=0)
        assert t.name == "Algebra"
        assert t.description == "Study of structures"
        assert t.related_documents == ["d1"]
        assert t.subtopic_count == 3
        assert t.order_source == "syllabus"
        assert t.syllabus_position == 0

    def test_defaults_on_missing_fields(self) -> None:
        t = _build_topic({}, order=5)
        assert t.name == "Topic 6"
        assert t.subtopic_count == 0


# ── Chunk filtering ──────────────────────────────────────────────────────────

class TestChunksForTopic:
    def test_keyword_matching(self) -> None:
        chunks = _chunks()
        result = _chunks_for_topic("eigenvalues", chunks)
        assert any(c.id == "c2" for c in result)

    def test_id_based_matching(self) -> None:
        chunks = _chunks()
        id_map = {"Linear Algebra": ["c1", "c2"]}
        result = _chunks_for_topic("Linear Algebra", chunks, id_map, "Linear Algebra")
        ids = {c.id for c in result}
        assert ids == {"c1", "c2"}

    def test_id_map_empty_falls_back_to_keywords(self) -> None:
        chunks = _chunks()
        id_map: dict[str, list[str]] = {"Linear Algebra": []}
        result = _chunks_for_topic("derivative", chunks, id_map, "Linear Algebra")
        assert any("derivative" in c.content.lower() for c in result)

    def test_no_match_returns_empty(self) -> None:
        chunks = _chunks()
        result = _chunks_for_topic("quantum physics", chunks)
        assert result == []


class TestChunkIdsFromResponse:
    def test_extracts_ids(self) -> None:
        raw = json.loads(_llm_topics_response())
        result = _chunk_ids_from_response(raw["topics"])
        assert result["Linear Algebra"] == ["c1", "c2"]

    def test_missing_chunk_ids(self) -> None:
        result = _chunk_ids_from_response([{"name": "X"}])
        assert result == {}


# ── TopicAnalyzer — full mode ────────────────────────────────────────────────

class TestTopicAnalyzerFull:
    def test_with_syllabus(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_topics_response()

        analyzer = TopicAnalyzer(mock_client)
        topics = analyzer.analyze(
            _chunks(),
            syllabus_document=_syllabus_doc(),
            output_path=tmp_path / "topics.json",
        )

        assert len(topics) == 2
        assert topics[0].name == "Linear Algebra"
        assert topics[1].name == "Calculus"
        assert topics[0].order_source == "syllabus"
        assert topics[0].syllabus_position == 0

    def test_without_syllabus(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_topics_response()

        analyzer = TopicAnalyzer(mock_client)
        topics = analyzer.analyze(
            _chunks(),
            syllabus_document=None,
            output_path=tmp_path / "topics.json",
        )

        assert len(topics) == 2
        mock_client.generate.assert_called_once()

    def test_saves_json(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_topics_response()
        out = tmp_path / "topics.json"

        analyzer = TopicAnalyzer(mock_client)
        analyzer.analyze(_chunks(), output_path=out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert "topics" in data
        assert len(data["topics"]) == 2
        assert data["topics"][0]["name"] == "Linear Algebra"

    def test_empty_chunks_returns_empty(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        analyzer = TopicAnalyzer(mock_client)
        topics = analyzer.analyze([], output_path=tmp_path / "topics.json")
        assert topics == []
        mock_client.generate.assert_not_called()


# ── TopicAnalyzer — topic (focus) mode ──────────────────────────────────────

class TestTopicAnalyzerFocus:
    def test_focus_exact_match(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_topics_response()

        analyzer = TopicAnalyzer(mock_client)
        topics = analyzer.analyze(
            _chunks(),
            scope="topic",
            focus_topic="Linear Algebra",
            output_path=tmp_path / "topics.json",
        )

        assert len(topics) == 1
        assert topics[0].name == "Linear Algebra"
        assert topics[0].order_source == "manual"
        assert topics[0].subtopic_count == 2  # 2 relevant chunks

    def test_focus_fuzzy_match(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_topics_response()

        analyzer = TopicAnalyzer(mock_client)
        topics = analyzer.analyze(
            _chunks(),
            scope="topic",
            focus_topic="Calculus",  # substring of "Calculus"
            output_path=tmp_path / "topics.json",
        )

        assert len(topics) == 1
        assert topics[0].name == "Calculus"
        assert topics[0].order_source == "manual"

    def test_focus_not_found_raises(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_topics_response()

        analyzer = TopicAnalyzer(mock_client)
        with pytest.raises(TopicNotFoundError, match="No topic found matching"):
            analyzer.analyze(
                _chunks(),
                scope="topic",
                focus_topic="Quantum Physics",
                output_path=tmp_path / "topics.json",
            )

    def test_focus_no_focus_topic_raises(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_topics_response()

        analyzer = TopicAnalyzer(mock_client)
        with pytest.raises(TopicAnalyzerError, match="focus_topic must be set"):
            analyzer.analyze(
                _chunks(),
                scope="topic",
                focus_topic=None,
                output_path=tmp_path / "topics.json",
            )

    def test_focus_no_relevant_chunks_raises(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = json.dumps({
            "topics": [
                {
                    "name": "Quantum Physics",
                    "description": "Quantum mechanics",
                    "related_documents": ["doc_c"],
                    "chunk_ids": [],
                    "subtopic_count": 0,
                    "order_source": "pedagogical",
                    "syllabus_position": None,
                },
            ]
        })

        analyzer = TopicAnalyzer(mock_client)
        with pytest.raises(TopicNotFoundError, match="No source chunks"):
            analyzer.analyze(
                _chunks(),
                scope="topic",
                focus_topic="Quantum Physics",
                output_path=tmp_path / "topics.json",
            )


# ── TopicAnalyzer — config integration ───────────────────────────────────────

class TestTopicAnalyzerConfig:
    def test_reads_scope_from_config(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_topics_response()

        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = lambda key, default=None: {
            "generation.scope": "topic",
            "generation.focus_topic": "Linear Algebra",
        }.get(key, default)

        analyzer = TopicAnalyzer(mock_client, cfg=mock_cfg)
        topics = analyzer.analyze(_chunks(), output_path=tmp_path / "topics.json")

        assert len(topics) == 1
        assert topics[0].name == "Linear Algebra"

    def test_reads_syllabus_from_config_path(self, tmp_path: Path) -> None:
        syllabus_file = tmp_path / "syllabus.txt"
        syllabus_file.write_text("1. Algebra\n2. Geometry")

        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_topics_response()

        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = lambda key, default=None: {
            "generation.scope": "full",
            "syllabus.path": str(syllabus_file),
        }.get(key, default)

        analyzer = TopicAnalyzer(mock_client, cfg=mock_cfg)
        analyzer.analyze(_chunks(), output_path=tmp_path / "topics.json")

        # Verify prompt was called (syllabus loaded from file)
        mock_client.generate.assert_called_once()
        call_args = mock_client.generate.call_args[0][0]
        assert "Algebra" in call_args  # syllabus content in prompt

    def test_syllabus_document_takes_precedence(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = _llm_topics_response()

        mock_cfg = MagicMock()
        mock_cfg.get.side_effect = lambda key, default=None: {
            "generation.scope": "full",
            "syllabus.path": "/nonexistent/file.txt",
        }.get(key, default)

        analyzer = TopicAnalyzer(mock_client, cfg=mock_cfg)
        analyzer.analyze(
            _chunks(),
            syllabus_document=_syllabus_doc(),
            output_path=tmp_path / "topics.json",
        )

        call_args = mock_client.generate.call_args[0][0]
        assert "1. Linear Algebra" in call_args  # from syllabus doc content


# ── TopicAnalyzer — error handling ───────────────────────────────────────────

class TestTopicAnalyzerErrors:
    def test_invalid_llm_response_raises(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = "I don't understand."

        analyzer = TopicAnalyzer(mock_client)
        with pytest.raises(TopicAnalyzerError, match="unrecognisable JSON"):
            analyzer.analyze(_chunks(), output_path=tmp_path / "topics.json")

    def test_llm_topics_array_wrong_type(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.generate.return_value = '{"topics": "not a list"}'

        analyzer = TopicAnalyzer(mock_client)
        with pytest.raises(TopicAnalyzerError, match="missing the 'topics' array"):
            analyzer.analyze(_chunks(), output_path=tmp_path / "topics.json")
