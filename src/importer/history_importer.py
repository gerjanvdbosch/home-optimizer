from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from client.homeassistant import HomeAssistantClient
from config.sensor_definitions import SensorSpec
from database.models import ImportChunk, Sample1m
from database.session import Database

LOGGER = logging.getLogger(__name__)


class HomeAssistantHistoryImporter:
    def __init__(
        self,
        ha_client: HomeAssistantClient,
        database: Database,
        chunk_days: int = 3,
        source: str = "home_assistant_history",
    ) -> None:
        if chunk_days <= 0:
            raise ValueError("chunk_days must be greater than zero")

        self.ha = ha_client
        self.database = database
        self.chunk_days = chunk_days
        self.source = source

    def import_sensor(
        self,
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime | None = None,
    ) -> int:
        start = self._to_utc(start_time)
        end = self._to_utc(end_time or datetime.now(timezone.utc))

        total_written = 0
        cursor = start

        while cursor < end:
            chunk_end = min(
                cursor + timedelta(days=self.chunk_days),
                end,
            )

            if self._chunk_already_imported(
                spec=spec,
                start_time=cursor,
                end_time=chunk_end,
            ):
                LOGGER.info(
                    "Skip %s: %s to %s already imported",
                    spec.name,
                    cursor.isoformat(),
                    chunk_end.isoformat(),
                )
                cursor = chunk_end
                continue

            history = self.ha.get_history(
                entity_id=spec.entity_id,
                start_time=cursor,
                end_time=chunk_end,
                minimal_response=True,
            )

            rows = self._aggregate_history_to_minutes(
                history=history,
                spec=spec,
                start_time=cursor,
                end_time=chunk_end,
            )

            self._write_rows(rows)
            self._mark_chunk_imported(
                spec=spec,
                start_time=cursor,
                end_time=chunk_end,
                row_count=len(rows),
            )
            total_written += len(rows)

            cursor = chunk_end

        return total_written

    def import_many(
        self,
        specs: list[SensorSpec],
        start_time: datetime,
        end_time: datetime | None = None,
    ) -> dict[str, int]:
        results: dict[str, int] = {}

        for spec in specs:
            written = self.import_sensor(
                spec=spec,
                start_time=start_time,
                end_time=end_time,
            )
            results[spec.name] = written

        return results

    def _aggregate_history_to_minutes(
        self,
        history: list[dict[str, Any]],
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Sample1m]:
        points = self._history_points(history, spec)

        if spec.method == "ffill":
            return self._forward_fill_rows(points, spec, start_time, end_time)

        if spec.method == "interpolate":
            return self._interpolated_rows(points, spec, start_time, end_time)

        return self._mean_rows(points, spec)

    def _history_points(
        self,
        history: list[dict[str, Any]],
        spec: SensorSpec,
    ) -> list[tuple[datetime, Any]]:
        points: list[tuple[datetime, Any]] = []

        for item in history:
            parsed = self._parse_value(item.get("state"))

            if parsed is None:
                continue

            if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
                parsed *= spec.conversion_factor

            ts_raw = item.get("last_changed") or item.get("last_updated")

            if not ts_raw:
                continue

            points.append((self._to_utc(ts_raw), parsed))

        return sorted(points, key=lambda point: point[0])

    def _mean_rows(
        self,
        points: list[tuple[datetime, Any]],
        spec: SensorSpec,
    ) -> list[Sample1m]:
        grouped: dict[datetime, list[Any]] = defaultdict(list)

        for ts, value in points:
            minute = self._floor_minute(ts)
            grouped[minute].append(value)

        rows: list[Sample1m] = []

        for minute, values in sorted(grouped.items()):
            row = self._build_row(
                minute=minute,
                values=values,
                spec=spec,
            )

            if row:
                rows.append(row)

        return rows

    def _forward_fill_rows(
        self,
        points: list[tuple[datetime, Any]],
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Sample1m]:
        rows: list[Sample1m] = []
        point_index = 0
        last_value: Any | None = None
        minute = self._floor_minute(start_time)

        while minute < end_time:
            next_minute = minute + timedelta(minutes=1)

            while point_index < len(points) and points[point_index][0] < next_minute:
                last_value = points[point_index][1]
                point_index += 1

            if last_value is not None and minute >= start_time:
                row = self._build_row(minute, [last_value], spec)
                if row:
                    rows.append(row)

            minute = next_minute

        return rows

    def _interpolated_rows(
        self,
        points: list[tuple[datetime, Any]],
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Sample1m]:
        numeric_points = [
            (ts, float(value))
            for ts, value in points
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]

        if len(numeric_points) < 2:
            return self._forward_fill_rows(points, spec, start_time, end_time)

        rows: list[Sample1m] = []
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

    def _chunk_already_imported(
        self,
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime,
    ) -> bool:
        with self.database.session() as session:
            existing = session.get(
                ImportChunk,
                {
                    "source": self.source,
                    "name": spec.name,
                    "start_time_utc": start_time.isoformat(),
                },
            )

        return existing is not None and existing.end_time_utc == end_time.isoformat()

    def _mark_chunk_imported(
        self,
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime,
        row_count: int,
    ) -> None:
        marker = ImportChunk(
            source=self.source,
            name=spec.name,
            start_time_utc=start_time.isoformat(),
            end_time_utc=end_time.isoformat(),
            row_count=row_count,
            imported_at_utc=datetime.now(timezone.utc).isoformat(),
        )

        with self.database.session() as session:
            session.merge(marker)
            session.commit()

    def _build_row(
        self,
        minute: datetime,
        values: list[Any],
        spec: SensorSpec,
    ) -> Sample1m | None:
        if not values:
            return None

        numeric_values = [
            v for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool)
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

        return Sample1m(
            timestamp_minute_utc=minute.isoformat(),
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

    def _write_rows(
        self,
        rows: list[Sample1m],
    ) -> None:
        if not rows:
            return

        with self.database.session() as session:
            for row in rows:
                session.merge(row)

            session.commit()

    @staticmethod
    def _parse_value(value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()

        if value in (
            None,
            "",
            "unknown",
            "unavailable",
            "none",
        ):
            return None

        if value == "on":
            return True

        if value == "off":
            return False

        try:
            return float(value)
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _to_utc(
        value: datetime | str,
    ) -> datetime:
        if isinstance(value, str):
            dt = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        else:
            dt = value

        if dt.tzinfo is None:
            raise ValueError(
                "datetime must be timezone-aware"
            )

        return dt.astimezone(timezone.utc)

    @staticmethod
    def _floor_minute(value: datetime) -> datetime:
        return value.replace(second=0, microsecond=0)
