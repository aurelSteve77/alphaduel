"""Central logging setup: pretty console output + rotating log files.

Call :func:`setup_logging` once at process start (the CLI does this). Everywhere
else just do ``log = get_logger(__name__)`` and use it like the stdlib logger.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_ROOT = "alphaduel"
_CONFIGURED = False

_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_PLAIN_CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_logging(
    level: str | int = "INFO",
    log_dir: str | Path | None = "logs",
    log_file: str = "alphaduel.log",
    *,
    use_rich: bool = True,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``alphaduel`` logger with a console and (optional) file sink.

    Args:
        level: Root level for the package logger (name or numeric).
        log_dir: Directory for log files; ``None`` disables file logging.
        log_file: File name written under ``log_dir`` (rotated).
        use_rich: Use Rich's colourised handler for stdout when available.
        max_bytes / backup_count: Rotation policy for the file handler.
        force: Re-configure even if already set up (clears existing handlers).

    Returns:
        The configured package logger.
    """
    global _CONFIGURED

    logger = logging.getLogger(_ROOT)
    if _CONFIGURED and not force:
        return logger

    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    logger.addHandler(_console_handler(use_rich, level))

    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path / log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger


def _console_handler(use_rich: bool, level: str | int) -> logging.Handler:
    if use_rich:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(
                rich_tracebacks=True, show_path=False, markup=True
            )
            handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        except ImportError:
            handler = _plain_console()
    else:
        handler = _plain_console()
    handler.setLevel(level)
    return handler


def _plain_console() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_PLAIN_CONSOLE_FORMAT))
    return handler


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the ``alphaduel`` logger (e.g. ``get_logger(__name__)``)."""
    if not name or name == _ROOT:
        return logging.getLogger(_ROOT)
    suffix = name[len(_ROOT) + 1 :] if name.startswith(_ROOT + ".") else name
    return logging.getLogger(_ROOT).getChild(suffix)
