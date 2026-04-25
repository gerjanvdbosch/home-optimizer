from __future__ import annotations

import logging
from typing import Protocol

from apscheduler.schedulers.background import BackgroundScheduler

LOGGER = logging.getLogger(__name__)


class ForecastRunner(Protocol):
    enabled: bool

    def refresh_forecast(self) -> int: ...


class ForecastScheduler:
    def __init__(
        self,
        runner: ForecastRunner,
        interval_seconds: int = 3600,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")

        self.runner = runner
        self.interval_seconds = interval_seconds
        self.scheduler = BackgroundScheduler()

    def start(self) -> None:
        if not self.runner.enabled:
            LOGGER.info("Forecast scheduler not started: Open-Meteo configuration incomplete")
            return

        if self.scheduler.running:
            return

        self.runner.refresh_forecast()
        self.scheduler.add_job(
            self.runner.refresh_forecast,
            "interval",
            seconds=self.interval_seconds,
            id="forecast:refresh",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        self.scheduler.start()
        LOGGER.info("Forecast scheduler started")

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        LOGGER.info("Forecast scheduler stopped")
