from __future__ import annotations

import logging
from datetime import datetime, timedelta
from threading import Event, Lock, Thread

from home_optimizer.domain.clock import utc_now
from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.domain.time import ensure_utc
from home_optimizer.domain.units import parse_sensor_value
from home_optimizer.features.live_collection.buffer import LiveMinuteBuffer
from home_optimizer.features.live_collection.ports import LiveSampleRepository, LiveStateGateway

LOGGER = logging.getLogger(__name__)


class LiveCollectionService:
    def __init__(
        self,
        gateway: LiveStateGateway,
        repository: LiveSampleRepository,
        specs: list[SensorSpec],
        flush_interval_seconds: int = 300,
        idle_sleep_seconds: float = 1.0,
    ) -> None:
        if flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be greater than zero")
        if idle_sleep_seconds <= 0:
            raise ValueError("idle_sleep_seconds must be greater than zero")

        self.gateway = gateway
        self.repository = repository
        self.specs = specs
        self.flush_interval = timedelta(seconds=flush_interval_seconds)
        self.idle_sleep_seconds = idle_sleep_seconds
        self.buffer = LiveMinuteBuffer(repository.source)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._last_poll: dict[str, datetime] = {}
        self._last_flush: datetime | None = None

    def start(self) -> None:
        if not self.specs:
            LOGGER.info("Live collection not started: no sensors configured")
            return

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="live-collection", daemon=True)
        self._thread.start()
        LOGGER.info("Live collection started for %s sensors", len(self.specs))

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.flush(include_current_minute=True)
        LOGGER.info("Live collection stopped")

    def collect_due(self, now: datetime | None = None) -> int:
        now_utc = ensure_utc(now or utc_now())
        collected = 0

        for spec in self.specs:
            last_poll = self._last_poll.get(spec.name)
            if last_poll and now_utc - last_poll < timedelta(seconds=spec.poll_interval_seconds):
                continue

            if self.collect_sensor(spec, now_utc):
                collected += 1
            self._last_poll[spec.name] = now_utc

        if self._last_flush is None:
            self._last_flush = now_utc
        elif now_utc - self._last_flush >= self.flush_interval:
            self.flush(now_utc)
            self._last_flush = now_utc

        return collected

    def collect_sensor(self, spec: SensorSpec, now: datetime | None = None) -> bool:
        try:
            state = self.gateway.get_state(spec.entity_id)
        except Exception:
            LOGGER.exception("Live collection failed for %s", spec.name)
            return False

        value = parse_sensor_value(state.get("state"), spec.unit)
        if value is None:
            return False

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value *= spec.conversion_factor

        timestamp = ensure_utc(now or utc_now())

        with self._lock:
            self.buffer.add(spec, timestamp, value)

        return True

    def flush(
        self,
        now: datetime | None = None,
        *,
        include_current_minute: bool = False,
    ) -> int:
        cutoff = (
            None
            if include_current_minute
            else self._floor_minute(ensure_utc(now or utc_now()))
        )

        with self._lock:
            samples = self.buffer.pop_samples_before(cutoff)

        self.repository.write_samples(samples)
        if samples:
            LOGGER.info("Live collection flushed %s minute samples", len(samples))
        return len(samples)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.collect_due()
            self._stop_event.wait(self.idle_sleep_seconds)

    @staticmethod
    def _floor_minute(value: datetime) -> datetime:
        return value.replace(second=0, microsecond=0)
