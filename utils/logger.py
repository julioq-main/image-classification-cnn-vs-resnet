"""
Utilities for configuring the root logger with console and optional file output.
"""

import logging
import sys
from pathlib import Path

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def set_logger(log_level: str, log_file: str | None = None) -> None:
    """
    Configure the root logger with a console handler and an optional file handler.

    Parameters
    ----------
    log_level : str
        Logging level. Valid values are ``DEBUG``, ``INFO``, ``WARNING``,
        ``ERROR``, and ``CRITICAL`` (case-insensitive). Falls back to
        ``INFO`` with a warning if an unrecognized value is provided.
    log_file : str, optional
        Path to the output log file. If None, output is directed to stdout
        only. The parent directory is created automatically if it does not
        exist. Default is None.

    Notes
    -----
    Configures the root logger, so all module-level loggers created with
    ``logging.getLogger(__name__)`` inherit its settings automatically.

    Clears existing handlers before applying new ones, so this function
    can be called more than once without handlers accumulating. Should
    still be called once at program startup, before any other module
    initializes a logger.

    The log format applied to all handlers is::

        YYYY-MM-DD HH:MM:SS | LEVEL | logger_name : message

    Examples
    --------
    >>> set_logger("DEBUG")
    >>> set_logger("INFO", log_file="experiments/testing/testing.log")
    """

    if log_level not in _VALID_LEVELS:
        logging.warning(f"Unrecognized log level '{log_level}', defaulting to INFO.")
        log_level = "INFO"

    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8", mode="w"))

    # Remove existing handlers to ensure the call is never silently ignored
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )