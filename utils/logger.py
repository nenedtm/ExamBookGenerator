"""Centralised logging for ExamBookGenerator.

Every module should obtain its logger through ``get_logger``::

    from utils.logger import get_logger
    logger = get_logger(__name__)

Initialisation happens exactly once via ``setup_logging``, typically called
from ``main.py`` before anything else.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_INITIALIZED: bool = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_DIR = Path(__file__).resolve().parent.parent / "output" / "logs"


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure the root logger exactly once.

    Subsequent calls are silently ignored (handler deduplication).

    Parameters
    ----------
    level:
        Minimum severity propagated to handlers.  Defaults to ``INFO``.
    log_file:
        Override for the log file path.  When *None* the default
        ``output/logs/exam_book_generator.log`` is used.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        log_file = _LOG_DIR / "exam_book_generator.log"
    else:
        log_file = Path(log_file)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(file_handler)

    _INITIALIZED = True

    logging.getLogger(__name__).info(
        "Logging initialised — level=%s, file=%s",
        logging.getLevelName(level),
        log_file,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    """
    return logging.getLogger(name)
