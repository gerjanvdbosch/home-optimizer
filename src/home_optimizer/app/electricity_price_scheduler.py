from __future__ import annotations

import logging
from typing import Protocol

from apscheduler.schedulers.background import BackgroundScheduler

LOGGER = logging.getLogger(__name__)


class ElectricityPriceRunner(Protocol):
    @property
    def enabled(self) -> bool: ...

    def refresh_prices(self) -> int: ...


class ElectricityPriceScheduler:
    def __init__(
        self,
        runner: ElectricityPriceRunner,
        interval_seconds: int,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")

        self.runner = runner
        self.interval_seconds = interval_seconds
        self.scheduler = BackgroundScheduler()

    def start(self) -> None:
        if not self.runner.enabled:
            LOGGER.info("Electricity price scheduler not started: pricing configuration incomplete")
            return

        if self.scheduler.running:
            return

        self.runner.refresh_prices()
        self.scheduler.add_job(
            self.runner.refresh_prices,
            "interval",
            seconds=self.interval_seconds,
            id="electricity-prices:refresh",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        self.scheduler.start()
        LOGGER.info("Electricity price scheduler started")

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        LOGGER.info("Electricity price scheduler stopped")

