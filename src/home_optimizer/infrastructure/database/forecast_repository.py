from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert

from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.time import normalize_utc_timestamp, parse_datetime
from home_optimizer.infrastructure.database.orm_models import ForecastValue
from home_optimizer.infrastructure.database.session import Database

SQLITE_MAX_VARIABLES = 999
FORECAST_VALUE_COLUMN_COUNT = 6
FORECAST_INSERT_BATCH_SIZE = SQLITE_MAX_VARIABLES // FORECAST_VALUE_COLUMN_COUNT


class ForecastRepository:
    def __init__(
        self,
        database: Database,
        source: str = "openmeteo",
    ) -> None:
        self.database = database
        self.source = source

    def latest_created_at(self) -> datetime | None:
        with self.database.session() as session:
            timestamp = session.execute(
                select(func.max(ForecastValue.created_at_utc)).where(
                    ForecastValue.source == self.source,
                )
            ).scalar_one()

        return parse_datetime(timestamp) if timestamp else None

    def write_entries(self, entries: list[ForecastEntry]) -> None:
        if not entries:
            return

        with self.database.session() as session:
            for entry in entries:
                session.merge(self._to_orm_value(entry))
            session.commit()

    def write_new_entries(self, entries: list[ForecastEntry]) -> int:
        if not entries:
            return 0

        rows = [
            {
                "created_at_utc": normalize_utc_timestamp(entry.created_at_utc),
                "forecast_time_utc": normalize_utc_timestamp(entry.forecast_time_utc),
                "name": entry.name,
                "value": entry.value,
                "unit": entry.unit,
                "source": entry.source,
            }
            for entry in entries
        ]
        inserted_rows = 0
        with self.database.session() as session:
            for start_index in range(0, len(rows), FORECAST_INSERT_BATCH_SIZE):
                batch = rows[start_index : start_index + FORECAST_INSERT_BATCH_SIZE]
                result = session.execute(
                    insert(ForecastValue).values(batch).prefix_with("OR IGNORE")
                )
                inserted_rows += int(result.rowcount or 0)
            session.commit()

        return inserted_rows

    def _to_orm_value(self, entry: ForecastEntry) -> ForecastValue:
        return ForecastValue(
            created_at_utc=normalize_utc_timestamp(entry.created_at_utc),
            forecast_time_utc=normalize_utc_timestamp(entry.forecast_time_utc),
            name=entry.name,
            value=entry.value,
            unit=entry.unit,
            source=entry.source,
        )
