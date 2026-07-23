"""ExamBookGenerator — entry point."""

from __future__ import annotations

import logging
import sys

import yaml

from utils.config import ConfigManager, ConfigValidationError, _DEFAULT_CONFIG_PATH
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def _read_log_level() -> int:
    """Extract the desired log level from config.yaml before ConfigManager is ready."""
    try:
        with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        level_str = raw.get("logging", {}).get("level", "INFO").upper()
        return _LEVEL_MAP.get(level_str, logging.INFO)
    except (FileNotFoundError, yaml.YAMLError):
        return logging.INFO


def main() -> None:
    """Load config, initialise logging, and start the system."""
    log_level = _read_log_level()
    setup_logging(level=log_level)

    try:
        config = ConfigManager()
    except FileNotFoundError as exc:
        logger.critical("Configuration file missing: %s", exc)
        sys.exit(1)
    except ConfigValidationError as exc:
        logger.critical("Configuration validation failed: %s", exc)
        sys.exit(1)

    logger.info("ExamBookGenerator started")
    logger.info(
        "Config loaded — depth_level=%s, language=%s, output=%s",
        config.get("generation.depth_level"),
        config.get("output.language"),
        config.get("output.filename"),
    )

    logger.info("System ready. Press Ctrl+C to exit.")


if __name__ == "__main__":
    main()
