from __future__ import annotations

import logging
import httpx
from datetime import datetime
from threading import Lock

from home_optimizer.domain.clock import utc_now
from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.domain.time import ensure_utc
from home_optimizer.domain.units import parse_sensor_value
from home_optimizer.features.telemetry.buffer import TelemetryMinuteBuffer
from home_optimizer.features.telemetry.ports import TelemetrySampleRepository, TelemetryStateGateway

LOGGER = logging.getLogger(__name__)


class TelemetryService:
    def __init__(
        self,
        gateway: TelemetryStateGateway,
        repository: TelemetrySampleRepository,
        specs: list[SensorSpec],
    ) -> None:
        self.gateway = gateway
        self.repository = repository
        self.specs = specs
        self.buffer = TelemetryMinuteBuffer(repository.source)
        self._lock = Lock()

    def collect_sensor(self, spec: SensorSpec, now: datetime | None = None) -> bool:
        try:
            state = self.gateway.get_state(spec.entity_id)
        except (httpx.HTTPStatusError, httpx.RequestError) as err:
            LOGGER.warning("Telemetry skipped for %s: %s", spec.name, err)
            return False
        except Exception:
            LOGGER.exception("Telemetry failed for %s", spec.name)
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
            LOGGER.info("Telemetry flushed %s minute samples", len(samples))
        return len(samples)

    @staticmethod
    def _floor_minute(value: datetime) -> datetime:
        return value.replace(second=0, microsecond=0)
