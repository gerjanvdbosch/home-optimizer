from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.features.history_import.models import MinuteSample, SensorPoint


class MinuteResampler:
    def __init__(self, source: str) -> None:
        self.source = source

    def resample(
        self,
        points: list[SensorPoint],
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime,
        initial_value: Any | None = None,
    ) -> list[MinuteSample]:
        if spec.method == "ffill":
            return self._forward_fill_rows(points, spec, start_time, end_time, initial_value)
        if spec.method == "interpolate":
            return self._interpolated_rows(points, spec, start_time, end_time)
        if spec.method == "time_weighted_mean":
            return self._time_weighted_mean_rows(points, spec, start_time, end_time, initial_value)

        return self._mean_rows(points, spec)

    def _mean_rows(
        self,
        points: list[SensorPoint],
        spec: SensorSpec,
    ) -> list[MinuteSample]:
        grouped: dict[datetime, list[Any]] = defaultdict(list)
        for point in points:
            grouped[self._floor_minute(point.timestamp)].append(point.value)

        rows: list[MinuteSample] = []
        for minute, values in sorted(grouped.items()):
            row = self._build_row(minute, values, spec)
            if row:
                rows.append(row)
        return rows

    def _forward_fill_rows(
        self,
        points: list[SensorPoint],
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime,
        initial_value: Any | None = None,
    ) -> list[MinuteSample]:
        rows: list[MinuteSample] = []
        point_index = 0
        last_value: Any | None = initial_value
        minute = self._floor_minute(start_time)

        while minute < end_time:
            next_minute = minute + timedelta(minutes=1)

            while point_index < len(points) and points[point_index].timestamp < next_minute:
                last_value = points[point_index].value
                point_index += 1

            if last_value is not None and minute >= start_time:
                row = self._build_row(minute, [last_value], spec)
                if row:
                    rows.append(row)

            minute = next_minute

        return rows

    def _interpolated_rows(
        self,
        points: list[SensorPoint],
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime,
    ) -> list[MinuteSample]:
        numeric_points = [
            (point.timestamp, float(point.value))
            for point in points
            if isinstance(point.value, (int, float)) and not isinstance(point.value, bool)
        ]

        if len(numeric_points) < 2:
            return self._forward_fill_rows(points, spec, start_time, end_time)

        rows: list[MinuteSample] = []
        point_index = 0
        minute = self._floor_minute(start_time)

        while minute < end_time:
            while (
                point_index + 1 < len(numeric_points)
                and numeric_points[point_index + 1][0] <= minute
            ):
                point_index += 1

            value: float | None = None
            if point_index + 1 < len(numeric_points):
                left_ts, left_value = numeric_points[point_index]
                right_ts, right_value = numeric_points[point_index + 1]
                if left_ts <= minute <= right_ts:
                    span = (right_ts - left_ts).total_seconds()
                    fraction = 0.0 if span <= 0 else (minute - left_ts).total_seconds() / span
                    value = left_value + ((right_value - left_value) * fraction)
            elif numeric_points[point_index][0] <= minute:
                value = numeric_points[point_index][1]

            if value is not None and minute >= start_time:
                row = self._build_row(minute, [value], spec)
                if row:
                    rows.append(row)

            minute += timedelta(minutes=1)

        return rows

    def _time_weighted_mean_rows(
        self,
        points: list[SensorPoint],
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime,
        initial_value: Any | None = None,
    ) -> list[MinuteSample]:
        numeric_points = [
            (point.timestamp, float(point.value))
            for point in points
            if isinstance(point.value, (int, float)) and not isinstance(point.value, bool)
        ]
        if not numeric_points:
            return []

        rows: list[MinuteSample] = []
        point_index = 0
        current_value: float | None = (
            float(initial_value) if isinstance(initial_value, (int, float)) else None
        )
        minute = self._floor_minute(start_time)

        while minute < end_time:
            next_minute = min(minute + timedelta(minutes=1), end_time)

            while point_index < len(numeric_points) and numeric_points[point_index][0] <= minute:
                current_value = numeric_points[point_index][1]
                point_index += 1

            cursor = max(minute, start_time)
            local_index = point_index
            local_value = current_value
            weighted_total = 0.0
            covered_seconds = 0.0
            segment_values: list[float] = []

            while cursor < next_minute:
                segment_end = next_minute
                if (
                    local_index < len(numeric_points)
                    and numeric_points[local_index][0] < next_minute
                ):
                    segment_end = max(cursor, numeric_points[local_index][0])

                if local_value is not None and segment_end > cursor:
                    seconds = (segment_end - cursor).total_seconds()
                    weighted_total += local_value * seconds
                    covered_seconds += seconds
                    segment_values.append(local_value)

                cursor = segment_end

                while (
                    local_index < len(numeric_points)
                    and numeric_points[local_index][0] <= cursor
                ):
                    local_value = numeric_points[local_index][1]
                    local_index += 1

            point_index = local_index
            current_value = local_value

            if covered_seconds > 0:
                rows.append(
                    self._build_numeric_row(
                        minute=minute,
                        mean_value=weighted_total / covered_seconds,
                        min_value=min(segment_values),
                        max_value=max(segment_values),
                        last_value=segment_values[-1],
                        sample_count=1,
                        spec=spec,
                    )
                )

            minute = next_minute

        return rows

    def _build_row(
        self,
        minute: datetime,
        values: list[Any],
        spec: SensorSpec,
    ) -> MinuteSample | None:
        if not values:
            return None

        numeric_values = [
            value
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        last_value = values[-1]

        mean_real = None
        min_real = None
        max_real = None
        last_real = None
        last_text = None
        last_bool = None

        if numeric_values:
            mean_real = sum(numeric_values) / len(numeric_values)
            min_real = min(numeric_values)
            max_real = max(numeric_values)

        if isinstance(last_value, bool):
            last_bool = int(last_value)
        elif isinstance(last_value, (int, float)):
            last_real = float(last_value)
        else:
            last_text = str(last_value)

        return MinuteSample(
            timestamp_minute=minute,
            name=spec.name,
            source=self.source,
            entity_id=spec.entity_id,
            category=spec.category,
            unit=spec.unit,
            mean_real=mean_real,
            min_real=min_real,
            max_real=max_real,
            last_real=last_real,
            last_text=last_text,
            last_bool=last_bool,
            sample_count=len(values),
        )

    def _build_numeric_row(
        self,
        minute: datetime,
        mean_value: float,
        min_value: float,
        max_value: float,
        last_value: float,
        sample_count: int,
        spec: SensorSpec,
    ) -> MinuteSample:
        return MinuteSample(
            timestamp_minute=minute,
            name=spec.name,
            source=self.source,
            entity_id=spec.entity_id,
            category=spec.category,
            unit=spec.unit,
            mean_real=mean_value,
            min_real=min_value,
            max_real=max_value,
            last_real=last_value,
            last_text=None,
            last_bool=None,
            sample_count=sample_count,
        )

    @staticmethod
    def _floor_minute(value: datetime) -> datetime:
        return value.replace(second=0, microsecond=0)
