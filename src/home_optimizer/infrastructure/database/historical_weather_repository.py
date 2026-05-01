from __future__ import annotations

from sqlalchemy.dialects.sqlite import insert

from home_optimizer.domain.historical_weather import HistoricalWeatherEntry
from home_optimizer.domain.time import normalize_utc_timestamp
from home_optimizer.infrastructure.database.orm_models import HistoricalWeatherValue
from home_optimizer.infrastructure.database.session import Database

SQLITE_MAX_VARIABLES = 999
HISTORICAL_WEATHER_COLUMN_COUNT = 5
HISTORICAL_WEATHER_INSERT_BATCH_SIZE = SQLITE_MAX_VARIABLES // HISTORICAL_WEATHER_COLUMN_COUNT


class HistoricalWeatherRepository:
    def __init__(
        self,
        database: Database,
        source: str = "openmeteo_archive",
    ) -> None:
        self.database = database
        self.source = source

    def write_new_entries(self, entries: list[HistoricalWeatherEntry]) -> int:
        if not entries:
            return 0

        rows = [
            {
                "timestamp_utc": normalize_utc_timestamp(entry.timestamp_utc),
                "name": entry.name,
                "value": entry.value,
                "unit": entry.unit,
                "source": entry.source,
            }
            for entry in entries
        ]
        inserted_rows = 0
        with self.database.session() as session:
            for start_index in range(0, len(rows), HISTORICAL_WEATHER_INSERT_BATCH_SIZE):
                batch = rows[start_index : start_index + HISTORICAL_WEATHER_INSERT_BATCH_SIZE]
                result = session.execute(
                    insert(HistoricalWeatherValue).values(batch).prefix_with("OR IGNORE")
                )
                inserted_rows += int(result.rowcount or 0)
            session.commit()

        return inserted_rows
