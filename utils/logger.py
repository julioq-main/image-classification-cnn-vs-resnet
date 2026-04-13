"""
utils/logger.py

Provides utilities for configuring the root logger with console and optional
file output.

Functions:
- set_logger: Configures the root logger with a specified level and format.

Notes:
- Configures the root logger, so all module-level loggers created with
  logging.getLogger(__name__) inherit its settings automatically.
- Should be called once at the start of the program, before any other module
  initializes a logger.
- If a log file path is provided, the parent directory is created automatically
  if it does not exist.
"""

import logging
import sys
from pathlib import Path

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

def set_logger(log_level: str, log_file: str = None) -> None:
    """
    Configures the root logger with a console handler and an optional file
    handler.

    Args:
        log_level (str): Logging level as a string. Valid values are "DEBUG",
            "INFO", "WARNING", "ERROR", and "CRITICAL". Case-insensitive.
            Defaults to INFO if an unrecognized value is provided.
        log_file (str, optional): Path to the output log file. If None, logging
            is only directed to stdout. Defaults to None.

    Returns:
        None

    Behavior:
        - Validates log_level and warns if unrecognized, falling back to INFO.
        - Clears any existing handlers on the root logger before applying new
          ones, ensuring basicConfig is not silently ignored on repeat calls.
        - Always attaches a StreamHandler directed to stdout.
        - If log_file is provided, attaches a UTF-8 FileHandler and creates
          the parent directory if it does not exist.
        - Applies a unified format to all handlers:
          "YYYY-MM-DD HH:MM:SS | LEVEL | logger_name : message"
    """ 
    if log_level not in _VALID_LEVELS:
        logging.warning(f"Unrecognized log level '{log_level}', defaulting to INFO.")
        log_level = "INFO"

    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    # Remove existing handlers to ensure the call is never silently ignored
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-5s | %(name)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )

    