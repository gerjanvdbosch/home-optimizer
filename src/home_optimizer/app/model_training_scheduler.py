from __future__ import annotations

import logging
from typing import Protocol

from apscheduler.schedulers.background import BackgroundScheduler

LOGGER = logging.getLogger(__name__)


class ModelTrainingRunner(Protocol):
    def train_full_dataset_model(self) -> object | None: ...


class ModelTrainingScheduler:
    def __init__(
        self,
        runner: ModelTrainingRunner,
        *,
        sync_hour: int = 2,
        sync_minute: int = 0,
    ) -> None:
        if not 0 <= sync_hour <= 23:
            raise ValueError("sync_hour must be between 0 and 23")
        if not 0 <= sync_minute <= 59:
            raise ValueError("sync_minute must be between 0 and 59")

        self.runner = runner
        self.sync_hour = sync_hour
        self.sync_minute = sync_minute
        self.scheduler = BackgroundScheduler()

    def start(self) -> None:
        if self.scheduler.running:
            return

        self.scheduler.add_job(
            self.runner.train_full_dataset_model,
            "cron",
            hour=self.sync_hour,
            minute=self.sync_minute,
            id="identified-model:train",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        self.scheduler.start()
        LOGGER.info("Model training scheduler started")

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        LOGGER.info("Model training scheduler stopped")
