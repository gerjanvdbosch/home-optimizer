from __future__ import annotations

import logging
from datetime import datetime
from threading import Lock

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
    ) -> None:
        self.gateway = gateway
        self.repository = repository
        self.specs = specs
        self.buffer = LiveMinuteBuffer(repository.source)
        self._lock = Lock()

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

    def flush_complete_minutes(self, now: datetime | None = None) -> int:
        cutoff = self._floor_minute(ensure_utc(now or utc_now()))
        return self._flush(cutoff)

    def flush_all(self) -> int:
        return self._flush(cutoff=None)

    def _flush(self, cutoff: datetime | None) -> int:
        with self._lock:
            samples = self.buffer.pop_samples_before(cutoff)
        self.repository.write_samples(samples)

        if samples:
            LOGGER.info("Live collection flushed %s minute samples", len(samples))
        return len(samples)

    @staticmethod
    def _floor_minute(value: datetime) -> datetime:
        return value.replace(second=0, microsecond=0)
