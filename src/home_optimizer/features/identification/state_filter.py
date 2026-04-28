from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timedelta

from home_optimizer.domain import NumericSeries, TextSeries
from home_optimizer.domain.time import parse_datetime

EXCLUDED_HEATPUMP_MODES = {
    "dhw",
    "legionella",
}


class IdentificationStateFilter:
    def __init__(
        self,
        *,
        defrost_active: NumericSeries | None = None,
        booster_heater_active: NumericSeries | None = None,
        hp_mode: TextSeries | None = None,
        max_state_age: timedelta = timedelta(minutes=20),
    ) -> None:
        self.defrost = _numeric_values(defrost_active)
        self.booster = _numeric_values(booster_heater_active)
        self.hp_mode = _text_values(hp_mode)
        self.max_state_age = max_state_age

    def is_valid(self, timestamp: datetime) -> bool:
        if _is_active(self.defrost, timestamp, self.max_state_age):
            return False
        if _is_active(self.booster, timestamp, self.max_state_age):
            return False

        mode = _latest_text_value(self.hp_mode, timestamp, self.max_state_age)
        if mode is None:
            return True
        return _normalized_mode(mode) not in EXCLUDED_HEATPUMP_MODES


def _is_active(
    points: list[tuple[datetime, float]],
    timestamp: datetime,
    max_age: timedelta,
) -> bool:
    value = _latest_numeric_value(points, timestamp, max_age)
    return value is not None and value > 0.0


def _numeric_values(series: NumericSeries | None) -> list[tuple[datetime, float]]:
    if series is None:
        return []
    return sorted(
        [
            (parse_datetime(point.timestamp), point.value)
            for point in series.points
        ],
        key=lambda point: point[0],
    )


def _text_values(series: TextSeries | None) -> list[tuple[datetime, str]]:
    if series is None:
        return []
    return sorted(
        [
            (parse_datetime(point.timestamp), point.value)
            for point in series.points
        ],
        key=lambda point: point[0],
    )


def _normalized_mode(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _latest_numeric_value(
    points: list[tuple[datetime, float]],
    timestamp: datetime,
    max_age: timedelta,
) -> float | None:
    if not points:
        return None

    timestamps = [point[0] for point in points]
    index = bisect_right(timestamps, timestamp) - 1
    if index < 0:
        return None

    point_timestamp, value = points[index]
    if timestamp - point_timestamp > max_age:
        return None
    return value


def _latest_text_value(
    points: list[tuple[datetime, str]],
    timestamp: datetime,
    max_age: timedelta,
) -> str | None:
    if not points:
        return None

    timestamps = [point[0] for point in points]
    index = bisect_right(timestamps, timestamp) - 1
    if index < 0:
        return None

    point_timestamp, value = points[index]
    if timestamp - point_timestamp > max_age:
        return None
    return value
