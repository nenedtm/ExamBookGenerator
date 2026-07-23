"""Tests for utils.config — ConfigManager."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from utils.config import ConfigManager, ConfigValidationError, _deep_merge, _resolve, _set_path


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def cfg() -> ConfigManager:
    """Load the real project config.yaml."""
    return ConfigManager()


@pytest.fixture()
def tmp_cfg(tmp_path: Path) -> ConfigManager:
    """Return a factory that writes a YAML snippet and returns a ConfigManager."""

    def _make(content: str) -> ConfigManager:
        p = tmp_path / "test_config.yaml"
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return ConfigManager(p)

    return _make


# ── Defaults & merge ───────────────────────────────────────────────────────

class TestDefaults:
    def test_defaults_are_populated(self) -> None:
        merged = _deep_merge({"a": 1}, {"b": 2})
        assert merged == {"a": 1, "b": 2}

    def test_deep_merge_nested(self) -> None:
        base = {"a": {"b": 1, "c": 2}}
        over = {"a": {"c": 99, "d": 3}}
        result = _deep_merge(base, over)
        assert result == {"a": {"b": 1, "c": 99, "d": 3}}

    def test_deep_merge_does_not_mutate_base(self) -> None:
        base = {"a": {"b": 1}}
        over = {"a": {"b": 2}}
        _deep_merge(base, over)
        assert base == {"a": {"b": 1}}


# ── Dot-notation access ────────────────────────────────────────────────────

class TestResolve:
    def test_get_existing_key(self, cfg: ConfigManager) -> None:
        assert cfg.get("generation.depth_level") == 5

    def test_get_nested_key(self, cfg: ConfigManager) -> None:
        assert cfg.get("output.language") == "en"

    def test_get_missing_key_returns_default(self, cfg: ConfigManager) -> None:
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_get_missing_key_returns_none(self, cfg: ConfigManager) -> None:
        assert cfg.get("nonexistent.key") is None

    def test_resolve_helper(self) -> None:
        data = {"a": {"b": {"c": 42}}}
        assert _resolve(data, "a.b.c") == 42

    def test_resolve_helper_missing(self) -> None:
        data = {"a": 1}
        assert _resolve(data, "a.b.c", default="nope") == "nope"


# ── Set / override ─────────────────────────────────────────────────────────

class TestSet:
    def test_set_creates_path(self, cfg: ConfigManager) -> None:
        cfg.set("new.nested.key", "hello")
        assert cfg.get("new.nested.key") == "hello"

    def test_set_overwrites_existing(self, cfg: ConfigManager) -> None:
        old = cfg.get("generation.depth_level")
        cfg.set("generation.depth_level", 3)
        assert cfg.get("generation.depth_level") == 3
        cfg.set("generation.depth_level", old)  # restore

    def test_raw_is_defensive_copy(self, cfg: ConfigManager) -> None:
        r = cfg.raw
        r["generation"]["depth_level"] = 999
        assert cfg.get("generation.depth_level") != 999


# ── Validation: depth_level ────────────────────────────────────────────────

class TestDepthLevelValidation:
    def test_valid_depth(self, tmp_cfg: callable) -> None:
        cfg = tmp_cfg("generation:\n  depth_level: 7\n")
        assert cfg.get("generation.depth_level") == 7

    def test_depth_below_min_clamps(self, tmp_cfg: callable) -> None:
        cfg = tmp_cfg("generation:\n  depth_level: -3\n")
        assert cfg.get("generation.depth_level") == 1

    def test_depth_above_max_clamps(self, tmp_cfg: callable) -> None:
        cfg = tmp_cfg("generation:\n  depth_level: 99\n")
        assert cfg.get("generation.depth_level") == 10

    def test_depth_zero_clamps(self, tmp_cfg: callable) -> None:
        cfg = tmp_cfg("generation:\n  depth_level: 0\n")
        assert cfg.get("generation.depth_level") == 1

    def test_depth_string_raises(self, tmp_cfg: callable) -> None:
        with pytest.raises(ConfigValidationError, match="integer"):
            tmp_cfg("generation:\n  depth_level: high\n")

    def test_depth_float_raises(self, tmp_cfg: callable) -> None:
        with pytest.raises(ConfigValidationError, match="integer"):
            tmp_cfg("generation:\n  depth_level: 5.5\n")


# ── Validation: language ───────────────────────────────────────────────────

class TestLanguageValidation:
    def test_valid_language(self, cfg: ConfigManager) -> None:
        assert cfg.get("output.language") == "en"

    def test_missing_language_defaults_to_en(self, tmp_cfg: callable) -> None:
        cfg = tmp_cfg("output:\n  filename: test.md\n")
        assert cfg.get("output.language") == "en"

    def test_empty_string_defaults_to_en(self, tmp_cfg: callable) -> None:
        cfg = tmp_cfg("output:\n  language: ''\n")
        assert cfg.get("output.language") == "en"


# ── File not found ─────────────────────────────────────────────────────────

class TestFileNotFound:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            ConfigManager(Path("/definitely/does/not/exist.yaml"))
