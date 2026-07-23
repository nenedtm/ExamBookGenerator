"""Centralised configuration management for ExamBookGenerator.

Provides ``ConfigManager`` — the single entry-point that every other module
should use to read configuration values.  No module should open ``config.yaml``
directly.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from urllib.error import URLError

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# ── Defaults ────────────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "output": {
        "language": "en",
        "filename": "Exam_Manual.md",
    },
    "llm": {
        "host": "http://127.0.0.1:11434",
        "model": "llama3",
        "timeout": 120,
        "max_retries": 3,
    },
    "structure": {
        "include_toc": True,
    },
    "syllabus": {
        "enabled": "auto",
        "path": None,
    },
    "generation": {
        "depth_level": 5,
        "length_mode": "topic_driven",
        "scope": "full",
        "focus_topic": None,
        "focus_depth_level": None,
    },
    "images": {
        "extract": True,
        "match_to_chapters": True,
        "vision_model": "llava",
        "assets_dir": "output/assets/images",
        "min_width": 100,
        "min_height": 100,
    },
    "logging": {
        "level": "INFO",
        "file": "output/logs/exam_book_generator.log",
    },
}

_DEPTH_MIN = 1
_DEPTH_MAX = 10


# ── Exceptions ──────────────────────────────────────────────────────────────

class ConfigValidationError(Exception):
    """Raised when a configuration value fails validation."""


# ── Helpers ─────────────────────────────────────────────────────────────────

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a copy of *base*."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve(data: dict[str, Any], dotpath: str, default: Any = None) -> Any:
    """Traverse *data* using a dotted path like ``'a.b.c'``."""
    keys = dotpath.split(".")
    current: Any = data
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current


def _set_path(data: dict[str, Any], dotpath: str, value: Any) -> None:
    """Set a value inside a nested dict using a dotted path."""
    keys = dotpath.split(".")
    target = data
    for k in keys[:-1]:
        if k not in target or not isinstance(target[k], dict):
            target[k] = {}
        target = target[k]
    target[keys[-1]] = value


def _is_ollama_model_available(model_name: str, host: str = "http://127.0.0.1:11434") -> bool:
    """Return *True* if *model_name* is pulled in the local Ollama instance."""
    try:
        with urlopen(f"{host}/api/tags", timeout=3) as resp:  # noqa: S310
            import json
            body = json.loads(resp.read())
            names = [m.get("name", "") for m in body.get("models", [])]
            return any(model_name in n for n in names)
    except (URLError, OSError, ValueError):
        return False


# ── ConfigManager ───────────────────────────────────────────────────────────

class ConfigManager:
    """Typed, validated, and centralised configuration store.

    Usage::

        cfg = ConfigManager()
        depth = cfg.get("generation.depth_level")   # int
        lang  = cfg.get("output.language")           # str

        cfg.set("generation.depth_level", 8)         # runtime override

    Parameters
    ----------
    path:
        Filesystem path to a YAML config file.  When *None* the default
        ``config.yaml`` at the project root is used.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
        self._data: dict[str, Any] = {}
        self._load()
        self._validate()

    # ── Public API ──────────────────────────────────────────────────────

    def get(self, dotpath: str, default: Any = None) -> Any:
        """Return a configuration value using dot-notation.

        Parameters
        ----------
        dotpath:
            Dotted key path, e.g. ``"generation.depth_level"``.
        default:
            Value returned when the key does not exist.
        """
        return _resolve(self._data, dotpath, default)

    def set(self, dotpath: str, value: Any) -> None:
        """Override a configuration value at runtime.

        Parameters
        ----------
        dotpath:
            Dotted key path, e.g. ``"output.language"``.
        value:
            New value to assign.
        """
        _set_path(self._data, dotpath, value)
        logger.info("Config override: %s = %r", dotpath, value)

    @property
    def raw(self) -> dict[str, Any]:
        """Return a deep copy of the underlying configuration dict."""
        return deepcopy(self._data)

    def __repr__(self) -> str:
        return f"ConfigManager(path={self._path!s})"

    # ── Internals ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self._path}")

        logger.info("Loading configuration from %s", self._path)

        with open(self._path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        self._data = _deep_merge(_DEFAULTS, raw)

    def _validate(self) -> None:
        self._validate_depth_level()
        self._validate_language()
        self._validate_vision_model()
        self._validate_scope()
        self._validate_syllabus_enabled()
        self._validate_focus_topic()
        self._validate_focus_depth_level()
        self._validate_llm()

    # ── v3 helpers ──────────────────────────────────────────────────────

    def get_effective_depth_level(self) -> int:
        """Return the active depth level for the current scope.

        When ``generation.scope == "topic"`` and ``focus_depth_level`` is
        set, that value is returned.  Otherwise falls back to the global
        ``generation.depth_level``.
        """
        scope = self.get("generation.scope", "full")
        if scope == "topic":
            focus = self.get("generation.focus_depth_level")
            if focus is not None:
                return focus
            base = self.get("generation.depth_level")
            logger.warning(
                "generation.focus_depth_level not set in topic mode — "
                "falling back to generation.depth_level (%d)",
                base,
            )
            return base
        return self.get("generation.depth_level")

    def _validate_depth_level(self) -> None:
        raw_value = self.get("generation.depth_level")
        if not isinstance(raw_value, int):
            raise ConfigValidationError(
                f"generation.depth_level must be an integer, got {type(raw_value).__name__}"
            )
        if raw_value < _DEPTH_MIN:
            logger.warning(
                "generation.depth_level=%d is below minimum %d — clamping to %d",
                raw_value, _DEPTH_MIN, _DEPTH_MIN,
            )
            self.set("generation.depth_level", _DEPTH_MIN)
        elif raw_value > _DEPTH_MAX:
            logger.warning(
                "generation.depth_level=%d is above maximum %d — clamping to %d",
                raw_value, _DEPTH_MAX, _DEPTH_MAX,
            )
            self.set("generation.depth_level", _DEPTH_MAX)

    def _validate_language(self) -> None:
        lang = self.get("output.language")
        if lang is None or (isinstance(lang, str) and lang.strip() == ""):
            logger.warning(
                "output.language not set or empty — defaulting to 'en'"
            )
            self.set("output.language", "en")

    def _validate_vision_model(self) -> None:
        model = self.get("images.vision_model")
        if model is None:
            logger.info("images.vision_model is not set — vision features disabled")
            return
        if not _is_ollama_model_available(model):
            logger.warning(
                "Ollama model '%s' not found locally — vision features will be "
                "disabled (fallback: images skipped). Pull it with: ollama pull %s",
                model, model,
            )

    def _validate_scope(self) -> None:
        scope = self.get("generation.scope")
        if scope not in ("full", "topic"):
            raise ConfigValidationError(
                f"generation.scope must be 'full' or 'topic', got {scope!r}"
            )

    def _validate_syllabus_enabled(self) -> None:
        enabled = self.get("syllabus.enabled")
        if enabled not in ("auto", True, False):
            raise ConfigValidationError(
                f"syllabus.enabled must be 'auto', true, or false, got {enabled!r}"
            )

    def _validate_focus_topic(self) -> None:
        scope = self.get("generation.scope")
        if scope == "topic":
            topic = self.get("generation.focus_topic")
            if not topic:
                raise ConfigValidationError(
                    "generation.focus_topic is required when generation.scope == 'topic'"
                )

    def _validate_focus_depth_level(self) -> None:
        raw = self.get("generation.focus_depth_level")
        if raw is None:
            return
        if not isinstance(raw, int):
            raise ConfigValidationError(
                f"generation.focus_depth_level must be an integer, got {type(raw).__name__}"
            )
        if raw < _DEPTH_MIN or raw > _DEPTH_MAX:
            logger.warning(
                "generation.focus_depth_level=%d is outside range %d-%d — clamping",
                raw, _DEPTH_MIN, _DEPTH_MAX,
            )
            clamped = max(_DEPTH_MIN, min(_DEPTH_MAX, raw))
            self.set("generation.focus_depth_level", clamped)

    def _validate_llm(self) -> None:
        host = self.get("llm.host")
        if not host or not isinstance(host, str):
            logger.warning("llm.host not set — defaulting to http://127.0.0.1:11434")
            self.set("llm.host", "http://127.0.0.1:11434")

        model = self.get("llm.model")
        if not model or not isinstance(model, str):
            logger.warning("llm.model not set — defaulting to 'llama3'")
            self.set("llm.model", "llama3")

        timeout = self.get("llm.timeout")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            logger.warning("llm.timeout must be a positive number — defaulting to 120")
            self.set("llm.timeout", 120)

        max_retries = self.get("llm.max_retries")
        if not isinstance(max_retries, int) or max_retries < 0:
            logger.warning("llm.max_retries must be a non-negative integer — defaulting to 3")
            self.set("llm.max_retries", 3)
