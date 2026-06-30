"""Logging configuration for the orc_bo package.

All package modules obtain a logger via :func:`logging.getLogger(__name__)` and emit
through it instead of calling ``print``. Applications (CLI, benchmarks, tests) call
:func:`configure_logging` once at start-up to attach a handler and set the level.

The default level is ``INFO`` and can be overridden with the ``ORC_BO_LOGLEVEL``
environment variable (e.g. ``ORC_BO_LOGLEVEL=DEBUG``).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# Root logger name for the package; child loggers inherit its handler/level.
ROOT_LOGGER_NAME = "orc_bo"


def configure_logging(level: Optional[str | int] = None, *, force: bool = False) -> logging.Logger:
    """Configure and return the package root logger.

    Parameters
    ----------
    level:
        Logging level as a name (``"INFO"``, ``"DEBUG"``, ...) or numeric value. When
        ``None``, the ``ORC_BO_LOGLEVEL`` environment variable is used, falling back to
        ``"INFO"``.
    force:
        When ``True``, existing handlers on the package root logger are removed before a
        fresh one is attached. Useful in tests or repeated CLI invocations.

    Returns
    -------
    logging.Logger
        The configured ``orc_bo`` root logger.
    """
    if level is None:
        level = os.environ.get("ORC_BO_LOGLEVEL", "INFO")
    if isinstance(level, str):
        level = level.upper()

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)

    # Do not propagate to the root logger to avoid duplicate records.
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the package root namespace."""
    if not name.startswith(ROOT_LOGGER_NAME):
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)
