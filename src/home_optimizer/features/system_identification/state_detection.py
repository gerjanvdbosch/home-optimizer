from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from home_optimizer.features.system_identification.dataset import (
    SeriesCursor,
    parse_timestamp,
    timed_values,
)
from home_optimizer.features.system_identification.schemas import NumericSeries, TextSeries

EXCLUDED_HEATPUMP_MODES = {
    "dhw",
    "legionella",
}


class StateMask:
    def __init__(
        self,
        *,
        defrost_active: NumericSeries | None = None,
        booster_heater_active: NumericSeries | None = None,
        hp_mode: TextSeries | None = None,
        max_state_age: timedelta = timedelta(minutes=20),
    ) -> None:
        self.defrost = SeriesCursor(timed_values(defrost_active)) if defrost_active else None
        self.booster = (
            SeriesCursor(timed_values(booster_heater_active))
            if booster_heater_active
            else None
        )
        self.hp_mode = _TextSeriesCursor(_text_values(hp_mode)) if hp_mode else None
        self.max_state_age = max_state_age

    def is_valid_room_model_state(self, timestamp: datetime) -> bool:
        if _is_active(self.defrost, timestamp, self.max_state_age):
            return False
        if _is_active(self.booster, timestamp, self.max_state_age):
            return False

        mode = self.hp_mode.latest_at(timestamp, self.max_state_age) if self.hp_mode else None
        if mode is None:
            return True
        return _normalized_mode(mode) not in EXCLUDED_HEATPUMP_MODES


@dataclass(frozen=True)
class _TimedText:
    timestamp: datetime
    value: str


class _TextSeriesCursor:
    def __init__(self, points: list[_TimedText]) -> None:
        self.points = points
        self.index = 0
        self.latest: _TimedText | None = None

    def latest_at(self, timestamp: datetime, max_age: timedelta) -> str | None:
        while (
            self.index < len(self.points)
            and self.points[self.index].timestamp <= timestamp
        ):
            self.latest = self.points[self.index]
            self.index += 1

        if self.latest is None or timestamp - self.latest.timestamp > max_age:
            return None
        return self.latest.value


def _is_active(cursor: SeriesCursor | None, timestamp: datetime, max_age: timedelta) -> bool:
    value = cursor.latest_at(timestamp, max_age) if cursor else None
    return value is not None and value > 0.0


def _text_values(series: TextSeries | None) -> list[_TimedText]:
    if series is None:
        return []
    return sorted(
        [
            _TimedText(
                timestamp=parse_timestamp(point.timestamp),
                value=point.value,
            )
            for point in series.points
        ],
        key=lambda point: point.timestamp,
    )


def _normalized_mode(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")
