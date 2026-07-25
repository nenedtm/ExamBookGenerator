"""Tests for storage.cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from storage.cache import clear_cache, count_cache, delete_cache, get_cache, save_cache


# ── Helpers ─────────────────────────────────────────────────────────────────

def _db(tmp_path: Path) -> str:
    return str(tmp_path / "test.db")


# ── save_cache / get_cache ──────────────────────────────────────────────────

class TestSaveCache:
    def test_returns_key(self, tmp_path: Path) -> None:
        key = save_cache("prompt", "response", db_path=_db(tmp_path))
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest

    def test_retrievable(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_cache("test prompt", "test response", model="gpt-4", db_path=db)
        result = get_cache("test prompt", model="gpt-4", db_path=db)
        assert result is not None
        assert result["response"] == "test response"

    def test_model_preserved(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_cache("p", "r", model="claude-3", db_path=db)
        result = get_cache("p", model="claude-3", db_path=db)
        assert result["model"] == "claude-3"

    def test_created_at_set(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_cache("p", "r", db_path=db)
        result = get_cache("p", db_path=db)
        assert result["created_at"]


class TestCacheHit:
    def test_exact_match(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_cache("exact prompt", "the answer", db_path=db)
        result = get_cache("exact prompt", db_path=db)
        assert result["response"] == "the answer"

    def test_different_prompts_different_keys(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_cache("prompt A", "response A", db_path=db)
        save_cache("prompt B", "response B", db_path=db)
        assert get_cache("prompt A", db_path=db)["response"] == "response A"
        assert get_cache("prompt B", db_path=db)["response"] == "response B"


class TestCacheMiss:
    def test_returns_none(self, tmp_path: Path) -> None:
        result = get_cache("nonexistent prompt", db_path=_db(tmp_path))
        assert result is None

    def test_empty_db(self, tmp_path: Path) -> None:
        result = get_cache("anything", db_path=_db(tmp_path))
        assert result is None


class TestOverwrite:
    def test_new_response_wins(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_cache("prompt", "old", model="v1", db_path=db)
        save_cache("prompt", "new", model="v2", db_path=db)
        # Different models → different keys → no overwrite
        assert get_cache("prompt", model="v1", db_path=db)["response"] == "old"
        assert get_cache("prompt", model="v2", db_path=db)["response"] == "new"

    def test_same_model_overwrites(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_cache("prompt", "old", model="v1", db_path=db)
        save_cache("prompt", "new", model="v1", db_path=db)
        result = get_cache("prompt", model="v1", db_path=db)
        assert result["response"] == "new"
        assert result["model"] == "v1"


class TestCountCache:
    def test_empty(self, tmp_path: Path) -> None:
        assert count_cache(db_path=_db(tmp_path)) == 0

    def test_after_inserts(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_cache("a", "ra", db_path=db)
        save_cache("b", "rb", db_path=db)
        assert count_cache(db_path=db) == 2


class TestDeleteCache:
    def test_existing(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_cache("prompt", "response", db_path=db)
        assert delete_cache("prompt", db_path=db) is True
        assert get_cache("prompt", db_path=db) is None

    def test_nonexistent(self, tmp_path: Path) -> None:
        assert delete_cache("missing", db_path=_db(tmp_path)) is False


class TestClearCache:
    def test_removes_all(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_cache("a", "ra", db_path=db)
        save_cache("b", "rb", db_path=db)
        removed = clear_cache(db_path=db)
        assert removed == 2
        assert count_cache(db_path=db) == 0

    def test_empty_cache(self, tmp_path: Path) -> None:
        assert clear_cache(db_path=_db(tmp_path)) == 0


class TestEdgeCases:
    def test_unicode_prompt(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        prompt = "Spiegami l'algebra lineare in italiano"
        save_cache(prompt, "risposta", db_path=db)
        result = get_cache(prompt, db_path=db)
        assert result["response"] == "risposta"

    def test_empty_response(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        save_cache("p", "", db_path=db)
        result = get_cache("p", db_path=db)
        assert result["response"] == ""

    def test_long_prompt(self, tmp_path: Path) -> None:
        db = _db(tmp_path)
        long = "x" * 100_000
        save_cache(long, "ok", db_path=db)
        result = get_cache(long, db_path=db)
        assert result["response"] == "ok"

    def test_different_models_same_prompt(self, tmp_path: Path) -> None:
        """Same prompt with different models produces different cache entries."""
        db = _db(tmp_path)
        save_cache("prompt", "response-a", model="model-a", db_path=db)
        save_cache("prompt", "response-b", model="model-b", db_path=db)
        assert get_cache("prompt", model="model-a", db_path=db)["response"] == "response-a"
        assert get_cache("prompt", model="model-b", db_path=db)["response"] == "response-b"

    def test_database_auto_created(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "sub" / "dir" / "cache.db")
        save_cache("p", "r", db_path=db_path)
        assert Path(db_path).exists()

    def test_independent_of_database_module(self, tmp_path: Path) -> None:
        """Cache and document tables coexist without interference."""
        from storage.database import count_documents, save_document
        from core.models import Document

        db = _db(tmp_path)
        save_cache("prompt", "response", db_path=db)
        save_document(Document(source_path="/a.pdf"), db_path=db)
        assert count_cache(db_path=db) == 1
        assert count_documents(db_path=db) == 1
