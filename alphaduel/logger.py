"""Central logging setup for alphaduel.

Configuration is read from ``configs/project.yaml`` under the ``logging`` key.
Call ``setup_logging()`` once (done automatically on ``import alphaduel``), then
use ``get_logger(__name__)`` in any module.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from threading import Lock

from alphaduel.configuration import Configuration

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_LOGGER = "alphaduel"

_setup_lock = Lock()
_configured = False


def _resolve_log_dir(log_dir: str | Path) -> Path:
    path = Path(log_dir)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_formatter(cfg: Configuration) -> logging.Formatter:
    return logging.Formatter(
        fmt=cfg.get("logging.format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
        datefmt=cfg.get("logging.datefmt", "%Y-%m-%d %H:%M:%S"),
    )


def setup_logging(*, force: bool = False) -> None:
    """Configure the package logger from ``project.yaml``.

    Idempotent unless ``force=True``. Attaches stdout and/or file handlers
    (plain, rotating, and/or timed-rotating) according to config.
    """
    global _configured
    with _setup_lock:
        if _configured and not force:
            return

        cfg = Configuration()
        level_name = str(cfg.get("logging.level", "INFO")).upper()
        level = getattr(logging, level_name, logging.INFO)
        formatter = _build_formatter(cfg)

        logger = logging.getLogger(_PACKAGE_LOGGER)
        logger.setLevel(level)
        logger.propagate = False
        logger.handlers.clear()

        if cfg.get("logging.to_stdout", True):
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        if cfg.get("logging.to_file", True):
            log_dir = _resolve_log_dir(cfg.get("logging.log_dir", "logs"))
            filename = cfg.get("logging.filename", "alphaduel.log")
            rotation_enabled = bool(cfg.get("logging.rotation.enabled", True))
            timed_enabled = bool(cfg.get("logging.timed_rotation.enabled", False))

            if rotation_enabled:
                rotating = RotatingFileHandler(
                    log_dir / filename,
                    maxBytes=int(cfg.get("logging.rotation.max_bytes", 10 * 1024 * 1024)),
                    backupCount=int(cfg.get("logging.rotation.backup_count", 5)),
                    encoding="utf-8",
                )
                rotating.setLevel(level)
                rotating.setFormatter(formatter)
                logger.addHandler(rotating)
            elif not timed_enabled:
                file_handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            if timed_enabled:
                timed_name = cfg.get("logging.timed_rotation.filename", "alphaduel.timed.log")
                timed = TimedRotatingFileHandler(
                    log_dir / timed_name,
                    when=str(cfg.get("logging.timed_rotation.when", "midnight")),
                    interval=int(cfg.get("logging.timed_rotation.interval", 1)),
                    backupCount=int(cfg.get("logging.timed_rotation.backup_count", 14)),
                    encoding="utf-8",
                )
                timed.setLevel(level)
                timed.setFormatter(formatter)
                logger.addHandler(timed)

        _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the package logger.

    Prefer ``get_logger(__name__)`` so records show the calling module.
    """
    if not _configured:
        setup_logging()

    if name is None or name == _PACKAGE_LOGGER:
        return logging.getLogger(_PACKAGE_LOGGER)
    if name.startswith(f"{_PACKAGE_LOGGER}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_PACKAGE_LOGGER}.{name}")
