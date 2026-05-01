from __future__ import annotations

import logging
from typing import Protocol

from apscheduler.schedulers.background import BackgroundScheduler

LOGGER = logging.getLogger(__name__)


class HistoricalWeatherRunner(Protocol):
    def import_historical_weather(self) -> int: ...


class HistoricalWeatherScheduler:
    def __init__(
        self,
        runner: HistoricalWeatherRunner,
        interval_seconds: int = 86400,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")

        self.runner = runner
        self.interval_seconds = interval_seconds
        self.scheduler = BackgroundScheduler()

    def start(self) -> None:
        if self.scheduler.running:
            return

        self.runner.import_historical_weather()
        self.scheduler.add_job(
            self.runner.import_historical_weather,
            "interval",
            seconds=self.interval_seconds,
            id="historical-weather:import",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        self.scheduler.start()
        LOGGER.info("Historical weather scheduler started")

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        LOGGER.info("Historical weather scheduler stopped")
