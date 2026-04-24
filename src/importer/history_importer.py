from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from client.homeassistant import HomeAssistantClient
from config.sensor_definitions import SensorSpec
from database.models import Sample1m
from database.session import Database


class HomeAssistantHistoryImporter:
    def __init__(
        self,
        ha_client: HomeAssistantClient,
        database: Database,
        chunk_days: int = 3,
        source: str = "home_assistant_history",
    ) -> None:
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
                print(
                    f"Skip {spec.name}: "
                    f"{cursor.isoformat()} → {chunk_end.isoformat()} already imported"
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
            )

            self._write_rows(rows)
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
    ) -> list[Sample1m]:
        grouped: dict[datetime, list[Any]] = defaultdict(list)

        for item in history:
            raw_value = item.get("state")
            parsed = self._parse_value(raw_value)

            if parsed is None:
                continue

            ts_raw = (
                item.get("last_changed")
                or item.get("last_updated")
            )

            if not ts_raw:
                continue

            ts = self._to_utc(ts_raw)

            minute = ts.replace(
                second=0,
                microsecond=0,
            )

            grouped[minute].append(parsed)

        rows: list[Sample1m] = []

        for minute, values in grouped.items():
            row = self._build_row(
                minute=minute,
                values=values,
                spec=spec,
            )

            if row:
                rows.append(row)

        return rows

    def _chunk_already_imported(
        self,
        spec: SensorSpec,
        start_time: datetime,
        end_time: datetime,
    ) -> bool:
        with self.database.session() as session:
            existing_count = (
                session.query(Sample1m)
                .filter(Sample1m.name == spec.name)
                .filter(Sample1m.source == self.source)
                .filter(Sample1m.timestamp_minute_utc >= start_time.isoformat())
                .filter(Sample1m.timestamp_minute_utc < end_time.isoformat())
                .count()
            )

        expected_minutes = int((end_time - start_time).total_seconds() // 60)

        return existing_count >= expected_minutes

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
            if isinstance(v, (int, float))
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
