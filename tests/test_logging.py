from __future__ import annotations

import logging

from home_optimizer.app.logging import configure_logging
from home_optimizer.app.settings import AppSettings


def test_settings_normalize_log_level() -> None:
    settings = AppSettings.from_options(
        {
            "database_path": "/tmp/home-optimizer-test.db",
            "log_level": "debug",
        }
    )

    assert settings.log_level == "DEBUG"


def test_configure_logging_quiets_scheduler_at_info() -> None:
    configure_logging("INFO")

    assert logging.getLogger("apscheduler.executors.default").level == logging.WARNING


def test_configure_logging_keeps_scheduler_verbose_at_debug() -> None:
    configure_logging("DEBUG")

    assert logging.getLogger("apscheduler.executors.default").level == logging.DEBUG

