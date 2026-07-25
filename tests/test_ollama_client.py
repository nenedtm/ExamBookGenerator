"""Tests for llm.ollama_client."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaResponseError,
    OllamaTimeoutError,
    VisionModelUnavailableError,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _client(**kwargs: Any) -> OllamaClient:
    return OllamaClient(host="http://localhost:11434", model="test-model", cache_enabled=False, **kwargs)


def _mock_response(data: dict[str, Any], status: int = 200) -> MagicMock:
    """Create a mock urllib response object."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _tags_response(*model_names: str) -> dict[str, Any]:
    return {"models": [{"name": n} for n in model_names]}


# ── Exceptions ──────────────────────────────────────────────────────────────

class TestExceptions:
    def test_ollama_error_is_base(self) -> None:
        assert issubclass(OllamaConnectionError, OllamaError)
        assert issubclass(OllamaTimeoutError, OllamaError)
        assert issubclass(OllamaResponseError, OllamaError)
        assert issubclass(VisionModelUnavailableError, OllamaError)

    def test_vision_error_is_catchable_separately(self) -> None:
        with pytest.raises(VisionModelUnavailableError):
            raise VisionModelUnavailableError("model not found")


# ── Constructor / repr ──────────────────────────────────────────────────────

class TestConstructor:
    def test_defaults(self) -> None:
        c = OllamaClient()
        assert c._host == "http://127.0.0.1:11434"
        assert c._model == "llama3"
        assert c._timeout == 120
        assert c._max_retries == 3

    def test_custom(self) -> None:
        c = OllamaClient(host="http://myhost:9999", model="mistral", timeout=10, max_retries=1)
        assert c._host == "http://myhost:9999"
        assert c._model == "mistral"
        assert c._timeout == 10
        assert c._max_retries == 1

    def test_trailing_slash_stripped(self) -> None:
        c = OllamaClient(host="http://localhost:11434/")
        assert c._host == "http://localhost:11434"

    def test_repr(self) -> None:
        c = _client()
        r = repr(c)
        assert "localhost:11434" in r
        assert "test-model" in r


# ── check_connection ────────────────────────────────────────────────────────

class TestCheckConnection:
    def test_connected(self) -> None:
        c = _client(max_retries=1)
        resp = _mock_response(_tags_response("llama3"))
        with patch("llm.ollama_client.urlopen", return_value=resp):
            assert c.check_connection() is True

    def test_not_connected(self) -> None:
        c = _client(max_retries=1)
        with patch("llm.ollama_client.urlopen", side_effect=OSError("refused")):
            assert c.check_connection() is False


# ── generate ────────────────────────────────────────────────────────────────

class TestGenerate:
    def test_basic(self) -> None:
        c = _client(max_retries=1)
        resp = _mock_response({"response": "Hello world"})
        with patch("llm.ollama_client.urlopen", return_value=resp):
            result = c.generate("Say hello")
        assert result == "Hello world"

    def test_uses_default_model(self) -> None:
        c = _client(max_retries=1)
        captured: dict[str, Any] = {}

        def capture(req: Any, **kwargs: Any) -> MagicMock:
            if hasattr(req, "data") and req.data:
                captured["body"] = json.loads(req.data)
            return _mock_response({"response": "ok"})

        with patch("llm.ollama_client.urlopen", side_effect=capture):
            c.generate("test")
        assert captured["body"]["model"] == "test-model"

    def test_override_model(self) -> None:
        c = _client(max_retries=1)
        captured: dict[str, Any] = {}

        def capture(req: Any, **kwargs: Any) -> MagicMock:
            if hasattr(req, "data") and req.data:
                captured["body"] = json.loads(req.data)
            return _mock_response({"response": "ok"})

        with patch("llm.ollama_client.urlopen", side_effect=capture):
            c.generate("test", model="mistral")
        assert captured["body"]["model"] == "mistral"

    def test_system_prompt(self) -> None:
        c = _client(max_retries=1)
        captured: dict[str, Any] = {}

        def capture(req: Any, **kwargs: Any) -> MagicMock:
            if hasattr(req, "data") and req.data:
                captured["body"] = json.loads(req.data)
            return _mock_response({"response": "ok"})

        with patch("llm.ollama_client.urlopen", side_effect=capture):
            c.generate("test", system="You are helpful")
        assert captured["body"]["system"] == "You are helpful"

    def test_empty_response_raises(self) -> None:
        c = _client(max_retries=1)
        resp = _mock_response({"response": ""})
        with patch("llm.ollama_client.urlopen", return_value=resp):
            with pytest.raises(OllamaResponseError, match="empty"):
                c.generate("test")

    def test_missing_response_key_raises(self) -> None:
        c = _client(max_retries=1)
        resp = _mock_response({})
        with patch("llm.ollama_client.urlopen", return_value=resp):
            with pytest.raises(OllamaResponseError, match="empty"):
                c.generate("test")


# ── chat ────────────────────────────────────────────────────────────────────

class TestChat:
    def test_basic(self) -> None:
        c = _client(max_retries=1)
        resp = _mock_response({"message": {"content": "Hi there!"}})
        with patch("llm.ollama_client.urlopen", return_value=resp):
            result = c.chat([{"role": "user", "content": "Hello"}])
        assert result == "Hi there!"

    def test_uses_default_model(self) -> None:
        c = _client(max_retries=1)
        captured: dict[str, Any] = {}

        def capture(req: Any, **kwargs: Any) -> MagicMock:
            if hasattr(req, "data") and req.data:
                captured["body"] = json.loads(req.data)
            return _mock_response({"message": {"content": "ok"}})

        with patch("llm.ollama_client.urlopen", side_effect=capture):
            c.chat([{"role": "user", "content": "test"}])
        assert captured["body"]["model"] == "test-model"

    def test_empty_content_raises(self) -> None:
        c = _client(max_retries=1)
        resp = _mock_response({"message": {"content": ""}})
        with patch("llm.ollama_client.urlopen", return_value=resp):
            with pytest.raises(OllamaResponseError, match="empty"):
                c.chat([{"role": "user", "content": "test"}])


# ── generate_with_image ─────────────────────────────────────────────────────

class TestGenerateWithImage:
    def test_basic(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        c = _client(max_retries=1)
        resp = _mock_response({"response": "A photo of a cat"})
        with patch("llm.ollama_client.urlopen", return_value=resp):
            with patch.object(c, "_is_model_available", return_value=True):
                result = c.generate_with_image("Describe this", str(img))
        assert result == "A photo of a cat"

    def test_file_not_found(self) -> None:
        c = _client(max_retries=1)
        with pytest.raises(FileNotFoundError, match="not found"):
            c.generate_with_image("Describe", "/nonexistent/img.png")

    def test_vision_model_not_configured(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 10)

        c = _client(max_retries=1)
        with patch("llm.ollama_client.ConfigManager") as MockCfg:
            instance = MockCfg.return_value
            instance.get.return_value = None
            with pytest.raises(VisionModelUnavailableError, match="No vision model"):
                c.generate_with_image("Describe", str(img))

    def test_model_not_available(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 10)

        c = _client(max_retries=1)
        with patch.object(c, "_is_model_available", return_value=False):
            with pytest.raises(VisionModelUnavailableError, match="not available"):
                c.generate_with_image("Describe", str(img), model="llava")

    def test_vision_request_failure_wraps_error(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 10)

        c = _client(max_retries=1)
        with patch.object(c, "_is_model_available", return_value=True):
            with patch.object(c, "_request", side_effect=OllamaConnectionError("down")):
                with pytest.raises(VisionModelUnavailableError, match="failed"):
                    c.generate_with_image("Describe", str(img), model="llava")

    def test_explicit_model_overrides_config(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 10)

        c = _client(max_retries=1)
        captured: dict[str, Any] = {}

        def capture(req: Any, **kwargs: Any) -> MagicMock:
            if hasattr(req, "data") and req.data:
                captured["body"] = json.loads(req.data)
            return _mock_response({"response": "ok"})

        with patch("llm.ollama_client.urlopen", side_effect=capture):
            with patch.object(c, "_is_model_available", return_value=True):
                c.generate_with_image("Describe", str(img), model="custom-vision")
        assert captured["body"]["model"] == "custom-vision"


# ── Retry / error handling ──────────────────────────────────────────────────

class TestRetryLogic:
    def test_retries_on_connection_error(self) -> None:
        c = _client(max_retries=2)
        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("refused")
            return _mock_response({"response": "ok"})

        with patch("llm.ollama_client.urlopen", side_effect=side_effect):
            with patch("llm.ollama_client.time.sleep"):
                result = c.generate("test")
        assert result == "ok"
        assert call_count == 2

    def test_exhausts_retries(self) -> None:
        c = _client(max_retries=2)
        with patch("llm.ollama_client.urlopen", side_effect=OSError("refused")):
            with patch("llm.ollama_client.time.sleep"):
                with pytest.raises(OllamaConnectionError, match="Cannot reach"):
                    c.generate("test")

    def test_timeout_raises_timeout_error(self) -> None:
        c = _client(max_retries=2)
        with patch("llm.ollama_client.urlopen", side_effect=TimeoutError("timed out")):
            with patch("llm.ollama_client.time.sleep"):
                with pytest.raises(OllamaTimeoutError, match="timed out"):
                    c.generate("test")

    def test_http_error_raises_response_error(self) -> None:
        from urllib.error import HTTPError

        c = _client(max_retries=1)
        exc = HTTPError(
            url="http://localhost:11434/api/generate",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=BytesIO(b"error body"),
        )
        with patch("llm.ollama_client.urlopen", side_effect=exc):
            with pytest.raises(OllamaResponseError, match="HTTP 500"):
                c.generate("test")

    def test_invalid_json_raises_response_error(self) -> None:
        c = _client(max_retries=1)
        resp = MagicMock()
        resp.read.return_value = b"not json"
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        with patch("llm.ollama_client.urlopen", return_value=resp):
            with pytest.raises(OllamaResponseError, match="invalid JSON"):
                c.generate("test")


# ── _is_model_available ─────────────────────────────────────────────────────

class TestIsModelAvailable:
    def test_model_found(self) -> None:
        c = _client(max_retries=1)
        resp = _mock_response(_tags_response("llama3", "mistral"))
        with patch("llm.ollama_client.urlopen", return_value=resp):
            assert c._is_model_available("llama3") is True

    def test_model_not_found(self) -> None:
        c = _client(max_retries=1)
        resp = _mock_response(_tags_response("llama3"))
        with patch("llm.ollama_client.urlopen", return_value=resp):
            assert c._is_model_available("llava") is False

    def test_connection_error_returns_false(self) -> None:
        c = _client(max_retries=1)
        with patch("llm.ollama_client.urlopen", side_effect=OSError("refused")):
            assert c._is_model_available("llava") is False


# ── _encode_image ───────────────────────────────────────────────────────────

class TestEncodeImage:
    def test_returns_base64(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        import base64
        expected = base64.b64encode(img.read_bytes()).decode("utf-8")
        result = OllamaClient._encode_image(img)
        assert result == expected

    def test_empty_file(self, tmp_path: Path) -> None:
        img = tmp_path / "empty.png"
        img.write_bytes(b"")
        import base64
        result = OllamaClient._encode_image(img)
        assert result == base64.b64encode(b"").decode("utf-8")


# ── from_config ─────────────────────────────────────────────────────────────

class TestFromConfig:
    def test_creates_from_config(self) -> None:
        with patch("llm.ollama_client.ConfigManager") as MockCfg:
            instance = MockCfg.return_value
            instance.get.side_effect = lambda key, default=None: {
                "llm.host": "http://custom:9999",
                "llm.model": "mistral",
                "llm.timeout": 30,
                "llm.max_retries": 5,
            }.get(key, default)
            c = OllamaClient.from_config()
        assert c._host == "http://custom:9999"
        assert c._model == "mistral"
        assert c._timeout == 30
        assert c._max_retries == 5
