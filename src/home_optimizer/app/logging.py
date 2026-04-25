from __future__ import annotations

import logging


def configure_logging(level: int | str = logging.INFO) -> None:
    resolved_level = _resolve_log_level(level)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    _configure_third_party_loggers(resolved_level)


def _resolve_log_level(level: int | str) -> int:
    if isinstance(level, int):
        return level

    resolved_level = logging.getLevelName(level.upper())
    if not isinstance(resolved_level, int):
        raise ValueError(f"Invalid log level: {level}")
    return resolved_level


def _configure_third_party_loggers(level: int) -> None:
    scheduler_level = logging.DEBUG if level <= logging.DEBUG else logging.WARNING
    network_level = logging.DEBUG if level <= logging.DEBUG else logging.WARNING
    logging.getLogger("apscheduler").setLevel(scheduler_level)
    logging.getLogger("apscheduler.executors.default").setLevel(scheduler_level)
    logging.getLogger("httpx").setLevel(network_level)
    logging.getLogger("httpcore").setLevel(network_level)
