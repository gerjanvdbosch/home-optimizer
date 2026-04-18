"""Dataset builders for offline thermal-parameter calibration.

The first calibration stage uses only low-solar ``off`` windows to estimate an
*effective* building envelope model.  This is intentionally conservative:
without active UFH excitation the full 2-state UFH model is not structurally
identifiable, but the aggregate envelope loss and effective thermal capacity are.
"""

from __future__ import annotations

from bisect import bisect_left
from datetime import datetime
from typing import Sequence

from .models import (
    UFHActiveCalibrationDataset,
    UFHActiveCalibrationSample,
    UFHActiveCalibrationSettings,
    UFHCalibrationDataset,
    UFHCalibrationSample,
    UFHOffCalibrationSettings,
)
from ..telemetry.models import ForecastSnapshot, TelemetryAggregate

_SECONDS_PER_HOUR: float = 3600.0


class _ForecastLookup:
    """Nearest-neighbour lookup from UTC timestamps to persisted forecast steps."""

    def __init__(self, forecast_rows: Sequence[ForecastSnapshot]) -> None:
        rows = sorted(forecast_rows, key=lambda row: row.valid_at_utc)
        self._rows = rows
        self._times = [row.valid_at_utc.timestamp() for row in rows]

    def nearest(
        self,
        target_utc: datetime,
        *,
        tolerance_hours: float,
    ) -> ForecastSnapshot | None:
        """Return the closest forecast step within the requested tolerance window."""
        if not self._rows:
            return None
        target_seconds = target_utc.timestamp()
        index = bisect_left(self._times, target_seconds)
        candidates: list[ForecastSnapshot] = []
        if index < len(self._rows):
            candidates.append(self._rows[index])
        if index > 0:
            candidates.append(self._rows[index - 1])
        if not candidates:
            return None
        closest = min(candidates, key=lambda row: abs(row.valid_at_utc.timestamp() - target_seconds))
        max_delta_seconds = tolerance_hours * _SECONDS_PER_HOUR
        if abs(closest.valid_at_utc.timestamp() - target_seconds) > max_delta_seconds:
            return None
        return closest


def build_ufh_off_calibration_dataset(
    aggregates: Sequence[TelemetryAggregate],
    forecast_rows: Sequence[ForecastSnapshot],
    settings: UFHOffCalibrationSettings,
) -> UFHCalibrationDataset:
    """Build low-solar off-mode transition samples for first-stage UFH calibration.

    The fitted transition sample uses the last room temperature of one bucket as
    ``T[k]`` and the last room temperature of the following bucket as ``T[k+1]``.
    Mean disturbances from the *second* bucket approximate the forcing over the
    interval ``[k, k+1]``.

    Args:
        aggregates: Historical telemetry buckets ordered arbitrarily.
        forecast_rows: Persisted forecast snapshots used to filter out solar-active windows.
        settings: Validated calibration thresholds and parameter bounds.

    Returns:
        Dataset containing validated off-mode transition samples.

    Raises:
        ValueError: If fewer than ``settings.min_sample_count`` usable samples remain.
    """
    rows = sorted(aggregates, key=lambda row: row.bucket_end_utc)
    forecast_lookup = _ForecastLookup(forecast_rows)
    samples: list[UFHCalibrationSample] = []

    for previous_row, next_row in zip(rows, rows[1:]):
        if previous_row.hp_mode_last != settings.off_mode_name:
            continue
        if next_row.hp_mode_last != settings.off_mode_name:
            continue
        if previous_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            continue
        if next_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            continue
        if previous_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            continue
        if next_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            continue

        dt_hours = (next_row.bucket_end_utc - previous_row.bucket_end_utc).total_seconds() / _SECONDS_PER_HOUR
        if dt_hours <= 0.0 or dt_hours > settings.max_pair_dt_hours:
            continue

        forecast = forecast_lookup.nearest(
            next_row.bucket_end_utc,
            tolerance_hours=settings.forecast_alignment_tolerance_hours,
        )
        gti_w_per_m2 = float(forecast.gti_w_per_m2) if forecast is not None else 0.0
        if gti_w_per_m2 > settings.max_gti_w_per_m2:
            continue

        samples.append(
            UFHCalibrationSample(
                interval_start_utc=previous_row.bucket_end_utc,
                interval_end_utc=next_row.bucket_end_utc,
                dt_hours=dt_hours,
                room_temperature_start_c=float(previous_row.room_temperature_last_c),
                room_temperature_end_c=float(next_row.room_temperature_last_c),
                outdoor_temperature_mean_c=float(next_row.outdoor_temperature_mean_c),
                gti_w_per_m2=gti_w_per_m2,
                household_elec_power_mean_kw=float(next_row.household_elec_power_mean_kw),
            )
        )

    if len(samples) < settings.min_sample_count:
        raise ValueError(
            "Not enough low-solar off-mode samples for UFH calibration: "
            f"required >= {settings.min_sample_count}, found {len(samples)}."
        )
    return UFHCalibrationDataset(samples=tuple(samples))


def build_ufh_active_calibration_dataset(
    aggregates: Sequence[TelemetryAggregate],
    forecast_rows: Sequence[ForecastSnapshot],
    settings: UFHActiveCalibrationSettings,
) -> UFHActiveCalibrationDataset:
    """Build replay samples for active UFH RC calibration.

    Each sample replays one transition ``x[k] -> x[k+1]`` using the mean UFH power
    and disturbances from the second bucket of a consecutive pair, while the room
    temperature at the pair endpoints supplies the measured scalar output.

    Args:
        aggregates: Historical telemetry buckets ordered arbitrarily.
        forecast_rows: Persisted forecast rows used to reconstruct GTI [W/m²].
        settings: Validated active-calibration configuration and reference parameters.

    Returns:
        Dataset containing validated replay samples for the active UFH fitter.

    Raises:
        ValueError: If too few usable active samples remain after filtering.
    """
    rows = sorted(aggregates, key=lambda row: row.bucket_end_utc)
    forecast_lookup = _ForecastLookup(forecast_rows)
    samples: list[UFHActiveCalibrationSample] = []
    dt_reference_hours = settings.reference_parameters.dt_hours

    for previous_row, next_row in zip(rows, rows[1:]):
        if previous_row.hp_mode_last != settings.active_mode_name:
            continue
        if next_row.hp_mode_last != settings.active_mode_name:
            continue
        if previous_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            continue
        if next_row.defrost_active_fraction > settings.max_defrost_active_fraction:
            continue
        if previous_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            continue
        if next_row.booster_heater_active_fraction > settings.max_booster_active_fraction:
            continue

        dt_hours = (next_row.bucket_end_utc - previous_row.bucket_end_utc).total_seconds() / _SECONDS_PER_HOUR
        if dt_hours <= 0.0 or dt_hours > settings.max_pair_dt_hours:
            continue
        if abs(dt_hours - dt_reference_hours) > settings.dt_compatibility_tolerance_hours:
            continue
        if float(next_row.hp_thermal_power_mean_kw) < settings.min_ufh_power_kw:
            continue

        forecast = forecast_lookup.nearest(
            next_row.bucket_end_utc,
            tolerance_hours=settings.forecast_alignment_tolerance_hours,
        )
        if forecast is None:
            continue
        if float(forecast.gti_w_per_m2) > settings.max_gti_w_per_m2:
            continue

        samples.append(
            UFHActiveCalibrationSample(
                interval_start_utc=previous_row.bucket_end_utc,
                interval_end_utc=next_row.bucket_end_utc,
                dt_hours=dt_hours,
                room_temperature_start_c=float(previous_row.room_temperature_last_c),
                room_temperature_end_c=float(next_row.room_temperature_last_c),
                outdoor_temperature_mean_c=float(next_row.outdoor_temperature_mean_c),
                gti_w_per_m2=float(forecast.gti_w_per_m2),
                internal_gain_proxy_kw=float(next_row.household_elec_power_mean_kw),
                ufh_power_mean_kw=float(next_row.hp_thermal_power_mean_kw),
            )
        )

    if len(samples) < settings.min_sample_count:
        raise ValueError(
            "Not enough active UFH samples for RC calibration: "
            f"required >= {settings.min_sample_count}, found {len(samples)}."
        )
    return UFHActiveCalibrationDataset(samples=tuple(samples))


