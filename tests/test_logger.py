"""Tests for the logging setup and get_logger factory."""

from __future__ import annotations

import logging

from alphaduel.logger import get_logger, setup_logging


def test_get_logger_prefixes_package_name():
    log = get_logger("data.download")
    assert log.name == "alphaduel.data.download"


def test_get_logger_none_returns_root_package_logger():
    assert get_logger().name == "alphaduel"


def test_already_prefixed_name_not_doubled():
    assert get_logger("alphaduel.foo").name == "alphaduel.foo"


def test_setup_is_idempotent():
    setup_logging()
    before = len(logging.getLogger("alphaduel").handlers)
    setup_logging()
    after = len(logging.getLogger("alphaduel").handlers)
    assert before == after


def test_handlers_configured():
    setup_logging(force=True)
    handlers = logging.getLogger("alphaduel").handlers
    assert handlers, "expected at least one handler"
