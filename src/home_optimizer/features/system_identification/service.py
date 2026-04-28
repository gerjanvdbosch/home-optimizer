from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from home_optimizer.domain.charts import ChartPoint, ChartSeries
from home_optimizer.features.system_identification.models import (
    IdentificationMetrics,
    ThermalModelCoefficients,
    ThermalModelIdentificationResult,
)


class SystemIdentificationError(ValueError):
    pass


@dataclass(frozen=True)
class _TrainingSample:
    room_temperature: float
    outdoor_temperature: float
    heatpump_power: float
    solar_gain: float
    next_room_temperature: float


@dataclass(frozen=True)
class _TimedValue:
    timestamp: datetime
    value: float


def identify_room_temperature_model(
    room_temperature: ChartSeries,
    outdoor_temperature: ChartSeries,
    heatpump_power: ChartSeries,
    solar_gain: ChartSeries | None = None,
    *,
    sample_interval_minutes: int = 15,
    max_input_age_minutes: int = 20,
) -> ThermalModelIdentificationResult:
    """Fit a linear one-step room-temperature model for MPC.

    Model form:
        T_room[k+1] = c + a*T_room[k] + b*T_out[k] + d*P_hp[k] + e*solar[k]
    """
    if sample_interval_minutes <= 0:
        raise SystemIdentificationError("sample_interval_minutes must be positive")

    samples = _build_training_samples(
        room_temperature=room_temperature,
        outdoor_temperature=outdoor_temperature,
        heatpump_power=heatpump_power,
        solar_gain=solar_gain,
        sample_interval=timedelta(minutes=sample_interval_minutes),
        max_input_age=timedelta(minutes=max_input_age_minutes),
    )
    if len(samples) < 6:
        raise SystemIdentificationError(
            "not enough aligned samples to identify a room-temperature model"
        )

    x = np.array(
        [
            [
                1.0,
                sample.room_temperature,
                sample.outdoor_temperature,
                sample.heatpump_power,
                sample.solar_gain,
            ]
            for sample in samples
        ],
        dtype=float,
    )
    y = np.array([sample.next_room_temperature for sample in samples], dtype=float)

    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    predictions = x @ coefficients
    residuals = y - predictions

    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    total_variance = float(np.sum((y - np.mean(y)) ** 2))
    if total_variance == 0.0:
        r_squared = 1.0
    else:
        r_squared = 1.0 - float(np.sum(residuals**2)) / total_variance

    return ThermalModelIdentificationResult(
        target_name=f"{room_temperature.name}_next",
        input_names=[
            room_temperature.name,
            outdoor_temperature.name,
            heatpump_power.name,
            solar_gain.name if solar_gain else "solar_gain",
        ],
        sample_interval_minutes=sample_interval_minutes,
        coefficients=ThermalModelCoefficients(
            intercept=float(coefficients[0]),
            room_temperature=float(coefficients[1]),
            outdoor_temperature=float(coefficients[2]),
            heatpump_power=float(coefficients[3]),
            solar_gain=float(coefficients[4]),
        ),
        metrics=IdentificationMetrics(
            sample_count=len(samples),
            rmse=rmse,
            mae=mae,
            r_squared=r_squared,
        ),
    )


def _build_training_samples(
    *,
    room_temperature: ChartSeries,
    outdoor_temperature: ChartSeries,
    heatpump_power: ChartSeries,
    solar_gain: ChartSeries | None,
    sample_interval: timedelta,
    max_input_age: timedelta,
) -> list[_TrainingSample]:
    room_points = _timed_values(room_temperature.points)
    next_room_by_timestamp = {point.timestamp: point.value for point in room_points}
    outdoor_points = _timed_values(outdoor_temperature.points)
    heatpump_points = _timed_values(heatpump_power.points)
    solar_points = _timed_values(solar_gain.points) if solar_gain else []

    samples: list[_TrainingSample] = []
    outdoor_cursor = _LatestValueCursor(outdoor_points)
    heatpump_cursor = _LatestValueCursor(heatpump_points)
    solar_cursor = _LatestValueCursor(solar_points)
    for point in room_points:
        timestamp = point.timestamp
        next_room = next_room_by_timestamp.get(timestamp + sample_interval)
        if next_room is None:
            continue

        outdoor = outdoor_cursor.latest_at(timestamp, max_input_age)
        heatpump = heatpump_cursor.latest_at(timestamp, max_input_age)
        if outdoor is None or heatpump is None:
            continue

        solar = solar_cursor.latest_at(timestamp, max_input_age)
        samples.append(
            _TrainingSample(
                room_temperature=point.value,
                outdoor_temperature=outdoor,
                heatpump_power=heatpump,
                solar_gain=solar or 0.0,
                next_room_temperature=next_room,
            )
        )

    return samples


def _timed_values(points: list[ChartPoint]) -> list[_TimedValue]:
    return sorted(
        [
            _TimedValue(
                timestamp=_parse_timestamp(point.timestamp),
                value=point.value,
            )
            for point in points
        ],
        key=lambda point: point.timestamp,
    )


class _LatestValueCursor:
    def __init__(self, points: list[_TimedValue]) -> None:
        self.points = points
        self.index = 0
        self.latest: _TimedValue | None = None

    def latest_at(self, timestamp: datetime, max_age: timedelta) -> float | None:
        while (
            self.index < len(self.points)
            and self.points[self.index].timestamp <= timestamp
        ):
            self.latest = self.points[self.index]
            self.index += 1

        if self.latest is None or timestamp - self.latest.timestamp > max_age:
            return None
        return self.latest.value


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
