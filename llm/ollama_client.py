"""Ollama client for local LLM interaction.

Provides ``OllamaClient`` — a thin wrapper around the Ollama HTTP API
that handles connection, retries, timeouts, and both text-only and
text+image (vision) modes.

All calls go through the local Ollama instance (no cloud APIs).

Usage::

    from llm.ollama_client import OllamaClient

    client = OllamaClient.from_config()
    reply = client.generate("Summarise this topic in 3 sentences.")
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from utils.config import ConfigManager
from utils.logger import get_logger
from storage.cache import save_cache, get_cache

logger = get_logger(__name__)


# ── Exceptions ──────────────────────────────────────────────────────────────

class OllamaError(Exception):
    """Base exception for Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Cannot reach the Ollama server."""


class OllamaTimeoutError(OllamaError):
    """Ollama request timed out."""


class OllamaResponseError(OllamaError):
    """Ollama returned an unexpected or error response."""


class VisionModelUnavailableError(OllamaError):
    """The configured vision model is not available in Ollama.

    Callers should catch this specifically to activate fallback behaviour
    (e.g. skip image processing) without breaking the text-only pipeline.
    """


# ── Client ──────────────────────────────────────────────────────────────────

class OllamaClient:
    """Thin HTTP client for the Ollama API.

    Parameters
    ----------
    host:
        Base URL of the Ollama server (default ``http://127.0.0.1:11434``).
    model:
        Default model name for text generation / chat.
    timeout:
        HTTP timeout in seconds for each request.
    max_retries:
        Number of retry attempts on connection or timeout errors.
    """

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen3",
        timeout: int = 120,
        max_retries: int = 3,
        cache_enabled: bool = True,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._cache_enabled = cache_enabled

    @classmethod
    def from_config(cls, cfg: ConfigManager | None = None) -> OllamaClient:
        """Create a client from the project configuration.

        Parameters
        ----------
        cfg:
            A ``ConfigManager`` instance.  When *None*, a new one is
            created from the default ``config.yaml``.
        """
        if cfg is None:
            cfg = ConfigManager()
        return cls(
            host=cfg.get("llm.host", "http://127.0.0.1:11434"),
            model=cfg.get("llm.model", "qwen3"),
            timeout=int(cfg.get("llm.timeout", 120)),
            max_retries=int(cfg.get("llm.max_retries", 3)),
        )

    # ── Public API ──────────────────────────────────────────────────────

    def check_connection(self) -> bool:
        """Return *True* if the Ollama server is reachable and responsive."""
        try:
            resp = self._request("/api/tags", method="GET")
            return resp is not None
        except OllamaError:
            return False

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Send a prompt to Ollama and return the generated text.

        Results are cached: identical prompts with the same model return
        the cached response without invoking Ollama.

        Parameters
        ----------
        prompt:
            The user prompt.
        model:
            Override the default model for this call.
        system:
            Optional system message to prepend.
        options:
            Optional Ollama-specific options (temperature, top_p, etc.).
        """
        effective_model = model or self._model
        cache_key_prompt = prompt
        if system:
            cache_key_prompt = f"SYSTEM: {system}\n\nUSER: {prompt}"

        if self._cache_enabled:
            try:
                cached = get_cache(cache_key_prompt, model=effective_model)
                if cached is not None:
                    logger.debug("LLM cache hit for generate() — model=%s", effective_model)
                    return cached["response"]
            except Exception:
                pass

        body: dict[str, Any] = {
            "model": effective_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            body["system"] = system
        if options:
            body["options"] = options

        resp = self._request("/api/generate", body=body)
        result = self._extract_response_text(resp, endpoint="generate")

        if self._cache_enabled:
            try:
                save_cache(cache_key_prompt, result, model=effective_model)
            except Exception:
                pass
        return result

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Send a multi-turn conversation to Ollama and return the reply.

        Results are cached: identical messages with the same model return
        the cached response without invoking Ollama.

        Parameters
        ----------
        messages:
            List of ``{"role": "user"|"assistant"|"system", "content": "..."}``.
        model:
            Override the default model for this call.
        options:
            Optional Ollama-specific options.
        """
        effective_model = model or self._model

        cache_key_prompt = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if self._cache_enabled:
            try:
                cached = get_cache(cache_key_prompt, model=effective_model)
                if cached is not None:
                    logger.debug("LLM cache hit for chat() — model=%s", effective_model)
                    return cached["response"]
            except Exception:
                pass

        body: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "stream": False,
        }
        if options:
            body["options"] = options

        resp = self._request("/api/chat", body=body)
        result = self._extract_chat_text(resp)

        if self._cache_enabled:
            try:
                save_cache(cache_key_prompt, result, model=effective_model)
            except Exception:
                pass
        return result

    def generate_with_image(
        self,
        prompt: str,
        image_path: str | Path,
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Send a text prompt together with an image to a vision model.

        Parameters
        ----------
        prompt:
            The text prompt describing what to do with the image.
        image_path:
            Path to a local image file (PNG, JPEG, etc.).
        model:
            Override the default model.  When *None*, the value from
            ``images.vision_model`` in the config is used.
        options:
            Optional Ollama-specific options.

        Raises
        ------
        VisionModelUnavailableError
            If the vision model cannot be found or the request fails due
            to a model-related issue.  Callers can catch this to activate
            a fallback (e.g. skip image processing).
        FileNotFoundError
            If *image_path* does not exist.
        """
        img_path = Path(image_path)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image_b64 = self._encode_image(img_path)

        # Determine which vision model to use
        if model is None:
            cfg = ConfigManager()
            model = cfg.get("images.vision_model", "llava")

        if model is None:
            raise VisionModelUnavailableError(
                "No vision model configured (images.vision_model is null)"
            )

        # Verify model availability before sending
        if not self._is_model_available(model):
            raise VisionModelUnavailableError(
                f"Vision model '{model}' is not available in Ollama. "
                f"Pull it with: ollama pull {model}"
            )

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        if options:
            body["options"] = options

        try:
            resp = self._request("/api/generate", body=body)
        except OllamaError as exc:
            raise VisionModelUnavailableError(
                f"Vision model '{model}' failed: {exc}"
            ) from exc

        return self._extract_response_text(resp, endpoint="generate")

    # ── Internals ───────────────────────────────────────────────────────

    def _request(
        self,
        endpoint: str,
        *,
        body: dict[str, Any] | None = None,
        method: str = "POST",
    ) -> dict[str, Any]:
        """Send an HTTP request to Ollama with retry logic.

        Returns the parsed JSON response body.

        Raises
        ------
        OllamaConnectionError
            If Ollama is unreachable after all retries.
        OllamaTimeoutError
            If the request times out after all retries.
        OllamaResponseError
            If the server returns a non-200 status or invalid JSON.
        """
        url = f"{self._host}{endpoint}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                req = Request(url, data=data, headers=headers, method=method)  # noqa: S310
                with urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
                    raw = resp.read()
                    return json.loads(raw)
            except HTTPError as exc:
                body_text = ""
                try:
                    body_text = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                raise OllamaResponseError(
                    f"Ollama returned HTTP {exc.code}: {body_text}"
                ) from exc
            except URLError as exc:
                last_error = exc
                logger.warning(
                    "Ollama connection error (attempt %d/%d): %s",
                    attempt, self._max_retries, exc.reason,
                )
            except TimeoutError as exc:
                last_error = exc
                logger.warning(
                    "Ollama timeout (attempt %d/%d): %s",
                    attempt, self._max_retries, exc,
                )
            except OSError as exc:
                last_error = exc
                logger.warning(
                    "Ollama OS error (attempt %d/%d): %s",
                    attempt, self._max_retries, exc,
                )
            except json.JSONDecodeError as exc:
                raise OllamaResponseError(
                    f"Ollama returned invalid JSON: {exc}"
                ) from exc

            if attempt < self._max_retries:
                backoff = min(2 ** attempt, 30)
                logger.info("Retrying in %ds...", backoff)
                time.sleep(backoff)

        # All retries exhausted
        if isinstance(last_error, TimeoutError):
            raise OllamaTimeoutError(
                f"Ollama request timed out after {self._max_retries} attempts"
            ) from last_error
        raise OllamaConnectionError(
            f"Cannot reach Ollama at {self._host} after {self._max_retries} attempts"
        ) from last_error

    def _is_model_available(self, model_name: str) -> bool:
        """Return *True* if *model_name* is pulled in the local Ollama."""
        try:
            resp = self._request("/api/tags", method="GET")
            names = [m.get("name", "") for m in resp.get("models", [])]
            return any(model_name in n for n in names)
        except OllamaError:
            return False

    @staticmethod
    def _encode_image(path: Path) -> str:
        """Read an image file and return its base64-encoded content."""
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    @staticmethod
    def _extract_response_text(resp: dict[str, Any], *, endpoint: str) -> str:
        """Extract the text response from an Ollama generate/chat response."""
        text = resp.get("response", "")
        if not text:
            raise OllamaResponseError(
                f"Ollama {endpoint} returned empty response"
            )
        return text

    @staticmethod
    def _extract_chat_text(resp: dict[str, Any]) -> str:
        """Extract the message content from an Ollama chat response."""
        message = resp.get("message", {})
        text = message.get("content", "")
        if not text:
            raise OllamaResponseError(
                "Ollama chat returned empty message content"
            )
        return text

    def __repr__(self) -> str:
        return (
            f"OllamaClient(host={self._host!r}, model={self._model!r}, "
            f"timeout={self._timeout}, max_retries={self._max_retries})"
        )
