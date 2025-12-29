"""Logging helpers for benchmark scripts.

Provides a consistent console + file logging setup so runs can be traced
and reproduced from logs without modifying each script's logging boilerplate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union


def _resolve_level(level: str) -> int:
    """Map a string level to a logging level constant."""
    return getattr(logging, str(level).upper(), logging.INFO)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
) -> logging.Logger:
    """Configure root logging with optional file and console handlers."""
    logger = logging.getLogger()
    resolved_level = _resolve_level(level)

    # Clear existing handlers to avoid duplicate logs in repeated runs.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    logger.setLevel(resolved_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        handler = logging.StreamHandler()
        handler.setLevel(resolved_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(resolved_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
