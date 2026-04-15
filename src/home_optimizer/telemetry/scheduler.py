"""APScheduler-driven telemetry sampling and aggregation.

The collector samples live sensors at a relatively high cadence and writes one
aggregated SQL row per flush window.  This preserves signal quality without
storing every minute-level sample permanently.
"""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock
from typing import Any

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler

from ..sensors.base import LiveReadings, SensorBackend
from .models import TelemetryCollectorSettings
from .repository import TelemetryRepository

NUMERIC_READING_FIELD_NAMES: tuple[str, ...] = (
    "room_temperature_c",
    "outdoor_temperature_c",
    "hp_supply_temperature_c",
    "hp_return_temperature_c",
    "hp_flow_lpm",
    "hp_electric_power_kw",
    "grid_import_kw",
    "grid_export_kw",
    "pv_output_kw",
    "thermostat_setpoint_c",
    "dhw_top_temperature_c",
    "dhw_bottom_temperature_c",
    "shutter_living_room_pct",
    "boiler_ambient_temp_c",
    "refrigerant_condensation_temp_c",
    "refrigerant_temp_c",
)

#: Boolean fields stored as (fraction-active, last-value) pairs.
#: The fraction captures transient events that ended before the flush window;
#: the last value initialises the next Kalman step.
BOOL_READING_FIELD_NAMES: tuple[str, ...] = (
    "defrost_active",
    "booster_heater_active",
)

_UNIT_SUFFIXES: tuple[str, ...] = ("_c", "_kw", "_lpm", "_pct")


def _stat_column_name(field_name: str, statistic: str) -> str:
    """Translate a :class:`LiveReadings` field name into an aggregate column name."""
    for suffix in _UNIT_SUFFIXES:
        if field_name.endswith(suffix):
            return f"{field_name[:-len(suffix)]}_{statistic}{suffix}"
    raise ValueError(f"Unsupported telemetry field suffix for {field_name!r}.")


def aggregate_readings(samples: Sequence[LiveReadings]) -> dict[str, Any]:
    """Aggregate buffered live readings into one SQL row payload.

    Parameters
    ----------
    samples:
        Ordered or unordered sequence of live sensor snapshots.  The function
        sorts on timestamp so out-of-order inserts still produce a correct bucket.

    Returns
    -------
    dict[str, Any]
        Keyword arguments ready for :class:`home_optimizer.telemetry.models.TelemetryAggregate`.

    Notes
    -----
    Numeric fields (temperature, power, flow, shutter position) are aggregated
    as arithmetic mean + last observed value.

    Boolean fields (``defrost_active``, ``booster_heater_active``) are aggregated
    as fraction-of-samples-True (captures transient events) + last observed value
    (initialises the next Kalman step).
    """
    if not samples:
        raise ValueError("samples must not be empty.")

    ordered_samples = sorted(samples, key=lambda sample: sample.timestamp)
    first_sample = ordered_samples[0]
    last_sample = ordered_samples[-1]
    distinct_modes = {sample.hp_mode for sample in ordered_samples}
    if len(distinct_modes) != 1:
        raise ValueError(
            "Telemetry buckets must contain exactly one hp_mode. "
            f"Received modes: {sorted(distinct_modes)!r}."
        )

    aggregate: dict[str, Any] = {
        "bucket_start_utc": first_sample.timestamp,
        "bucket_end_utc": last_sample.timestamp,
        "sample_count": len(ordered_samples),
        "hp_mode_last": last_sample.hp_mode,
    }

    # Numeric fields: persist arithmetic mean and last value.
    for field_name in NUMERIC_READING_FIELD_NAMES:
        values = np.asarray(
            [getattr(sample, field_name) for sample in ordered_samples],
            dtype=float,
        )
        aggregate[_stat_column_name(field_name, "mean")] = float(np.mean(values))
        aggregate[_stat_column_name(field_name, "last")] = float(values[-1])

    # Boolean fields: persist fraction-active (mean of 0/1 array) and last state.
    for field_name in BOOL_READING_FIELD_NAMES:
        values = np.asarray(
            [float(getattr(sample, field_name)) for sample in ordered_samples],
            dtype=float,
        )
        aggregate[f"{field_name}_fraction"] = float(np.mean(values))
        aggregate[f"{field_name}_last"] = bool(ordered_samples[-1].__getattribute__(field_name))

    return aggregate


class BufferedTelemetryCollector:
    """Sample live telemetry frequently and persist aggregated SQL buckets.

    Architecture
    ------------
    * ``sample_once`` reads a complete :class:`LiveReadings` snapshot from the
      configured :class:`SensorBackend` and stores it in an in-memory buffer.
    * ``flush_once`` aggregates the buffered snapshots into one SQL row and
      clears the buffer.
    * ``start`` wires both actions into APScheduler interval jobs.

    This separation keeps the physical sensor read path testable and lets the
    caller choose whether jobs are driven by APScheduler or invoked manually.
    """

    def __init__(
        self,
        backend: SensorBackend,
        repository: TelemetryRepository,
        settings: TelemetryCollectorSettings,
        scheduler: BackgroundScheduler | None = None,
    ) -> None:
        self._backend = backend
        self._repository = repository
        self._settings = settings
        self._scheduler = scheduler or BackgroundScheduler(timezone=settings.timezone_name)
        self._buffer: list[LiveReadings] = []
        self._lock = Lock()
        self._sample_job_id = f"{settings.job_id_prefix}-sample"
        self._flush_job_id = f"{settings.job_id_prefix}-flush"

    @property
    def scheduler(self) -> BackgroundScheduler:
        """Expose the scheduler for observability and integration tests."""
        return self._scheduler

    @property
    def pending_sample_count(self) -> int:
        """Current number of in-memory samples waiting to be flushed."""
        with self._lock:
            return len(self._buffer)

    def _persist_samples(self, samples: Sequence[LiveReadings]) -> dict[str, Any] | None:
        """Aggregate and persist a completed telemetry bucket."""
        if not samples:
            return None
        aggregate = aggregate_readings(samples)
        self._repository.add_aggregate(aggregate)
        return aggregate

    def sample_once(self) -> LiveReadings:
        """Read one live telemetry sample and append it to the current buffer.

        If ``hp_mode`` changes relative to the buffered samples, the current
        bucket is flushed immediately before the new-mode sample is appended.
        This prevents UFH and DHW hydraulic operating points from being averaged
        into a single physically meaningless SQL row.
        """
        reading = self._backend.read_all()
        samples_to_flush: list[LiveReadings] = []
        with self._lock:
            if self._buffer and self._buffer[-1].hp_mode != reading.hp_mode:
                samples_to_flush = list(self._buffer)
                self._buffer.clear()
            self._buffer.append(reading)

        self._persist_samples(samples_to_flush)
        return reading

    def flush_once(self) -> dict[str, Any] | None:
        """Persist the current buffer as one aggregated database row.

        Returns ``None`` when no samples are buffered.
        """
        with self._lock:
            if not self._buffer:
                return None
            samples = list(self._buffer)
            self._buffer.clear()

        return self._persist_samples(samples)

    def start(self) -> None:
        """Create tables, register APScheduler jobs, and start the scheduler."""
        self._repository.create_schema()
        self._scheduler.add_job(
            self.sample_once,
            trigger="interval",
            seconds=self._settings.sampling_interval_seconds,
            id=self._sample_job_id,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        self._scheduler.add_job(
            self.flush_once,
            trigger="interval",
            seconds=self._settings.flush_interval_seconds,
            id=self._flush_job_id,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        if not self._scheduler.running:
            self._scheduler.start()

    def shutdown(self, *, flush: bool = True, wait: bool = False) -> None:
        """Stop APScheduler and optionally persist the final partial bucket."""
        if flush:
            self.flush_once()
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
        self._backend.close()

    def job_ids(self) -> tuple[str, str]:
        """Return ``(sample_job_id, flush_job_id)`` for diagnostics."""
        return self._sample_job_id, self._flush_job_id

