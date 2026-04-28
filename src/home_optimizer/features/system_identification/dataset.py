from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta

from home_optimizer.features.system_identification.schemas import NumericPoint, NumericSeries


@dataclass(frozen=True)
class TimedValue:
    timestamp: datetime
    value: float


@dataclass(frozen=True)
class IdentificationRow:
    timestamp: datetime
    features: dict[str, float]
    target: float


class SeriesCursor:
    def __init__(self, points: list[TimedValue]) -> None:
        self.points = points
        self.index = 0
        self.latest: TimedValue | None = None

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


class SeriesLookup:
    def __init__(self, points: list[TimedValue]) -> None:
        self.points = points
        self.timestamps = [point.timestamp for point in points]

    def latest_at(self, timestamp: datetime, max_age: timedelta) -> float | None:
        index = bisect_right(self.timestamps, timestamp) - 1
        if index < 0:
            return None

        point = self.points[index]
        if timestamp - point.timestamp > max_age:
            return None
        return point.value


def timed_values(series: NumericSeries) -> list[TimedValue]:
    return sorted(
        [
            TimedValue(
                timestamp=parse_timestamp(point.timestamp),
                value=point.value,
            )
            for point in series.points
        ],
        key=lambda point: point.timestamp,
    )


def points_by_timestamp(series: NumericSeries) -> dict[datetime, float]:
    return {point.timestamp: point.value for point in timed_values(series)}


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def numeric_series(
    name: str,
    unit: str | None,
    rows: list[tuple[datetime, float]],
) -> NumericSeries:
    return NumericSeries(
        name=name,
        unit=unit,
        points=[
            NumericPoint(
                timestamp=timestamp.isoformat(),
                value=value,
            )
            for timestamp, value in rows
        ],
    )
