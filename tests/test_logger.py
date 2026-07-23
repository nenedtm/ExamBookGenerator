"""Tests for utils.logger."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import utils.logger as logger_mod
from utils.logger import get_logger, setup_logging


@pytest.fixture(autouse=True)
def _reset_logger_state() -> None:
    """Reset the initialised flag and strip handlers between tests."""
    logger_mod._INITIALIZED = False
    root = logging.getLogger()
    root.handlers.clear()
    yield
    logger_mod._INITIALIZED = False
    root.handlers.clear()


# ── get_logger ──────────────────────────────────────────────────────────────

class TestGetLogger:
    def test_returns_standard_logger(self) -> None:
        log = get_logger("my.module")
        assert isinstance(log, logging.Logger)

    def test_logger_name_matches(self) -> None:
        log = get_logger("parsers.pdf")
        assert log.name == "parsers.pdf"

    def test_dunder_name(self) -> None:
        log = get_logger(__name__)
        assert log.name.startswith("tests.test_logger")


# ── setup_logging ───────────────────────────────────────────────────────────

class TestSetupLogging:

    def _ours(self) -> list[logging.Handler]:
        """Return only the handlers added by our setup_logging."""
        return [h for h in logging.getLogger().handlers
                if type(h).__name__ in ("StreamHandler", "FileHandler")]

    def test_adds_two_handlers(self, tmp_path: Path) -> None:
        setup_logging(log_file=tmp_path / "test.log")
        assert len(self._ours()) == 2

    def test_handler_types(self, tmp_path: Path) -> None:
        setup_logging(log_file=tmp_path / "test.log")
        types = {type(h) for h in self._ours()}
        assert logging.StreamHandler in types
        assert logging.FileHandler in types

    def test_log_file_is_created(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        logging.getLogger("test").info("write test")
        for h in logging.getLogger().handlers:
            h.flush()
        assert log_file.exists()
        assert log_file.stat().st_size > 0

    def test_deduplication(self, tmp_path: Path) -> None:
        setup_logging(log_file=tmp_path / "test.log")
        count_first = len(self._ours())
        setup_logging(log_file=tmp_path / "test.log")
        count_second = len(self._ours())
        assert count_first == count_second == 2

    def test_custom_level(self, tmp_path: Path) -> None:
        setup_logging(level=logging.DEBUG, log_file=tmp_path / "test.log")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_message_appears_in_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        msg = "unique marker 42xyz"
        logging.getLogger("test_module").info(msg)
        for h in logging.getLogger().handlers:
            h.flush()
        content = log_file.read_text(encoding="utf-8")
        assert msg in content

    def test_message_appears_in_console(self, capsys: pytest.CaptureFixture[str]) -> None:
        setup_logging(log_file="/dev/null")
        msg = "console marker 99abc"
        logging.getLogger("test_console").info(msg)
        for h in logging.getLogger().handlers:
            h.flush()
        captured = capsys.readouterr()
        assert msg in captured.out
