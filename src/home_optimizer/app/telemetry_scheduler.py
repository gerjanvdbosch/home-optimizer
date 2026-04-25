from __future__ import annotations

import logging
from typing import Protocol

from apscheduler.schedulers.background import BackgroundScheduler

from home_optimizer.domain.sensors import SensorSpec

LOGGER = logging.getLogger(__name__)


class TelemetryRunner(Protocol):
    specs: list[SensorSpec]

    def collect_sensor(self, spec: SensorSpec) -> bool: ...

    def flush_complete_minutes(self) -> int: ...

    def flush_all(self) -> int: ...


class TelemetryScheduler:
    def __init__(
        self,
        collector: TelemetryRunner,
        flush_interval_seconds: int = 300,
    ) -> None:
        if flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be greater than zero")

        self.collector = collector
        self.flush_interval_seconds = flush_interval_seconds
        self.scheduler = BackgroundScheduler()

    def start(self) -> None:
        if not self.collector.specs:
            LOGGER.info("Telemetry scheduler not started: no sensors configured")
            return

        if self.scheduler.running:
            return

        for spec in self.collector.specs:
            self.scheduler.add_job(
                self.collector.collect_sensor,
                "interval",
                seconds=spec.poll_interval_seconds,
                args=[spec],
                id=f"telemetry:collect:{spec.name}",
                max_instances=1,
                coalesce=True,
                replace_existing=True,
            )

        self.scheduler.add_job(
            self.collector.flush_complete_minutes,
            "interval",
            seconds=self.flush_interval_seconds,
            id="telemetry:flush",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        self.scheduler.start()
        LOGGER.info(
            "Telemetry scheduler started for %s sensors",
            len(self.collector.specs),
        )

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

        self.collector.flush_all()
        LOGGER.info("Telemetry scheduler stopped")

