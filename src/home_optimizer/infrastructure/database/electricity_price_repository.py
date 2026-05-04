from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.sqlite import insert

from home_optimizer.domain.pricing import PriceInterval
from home_optimizer.domain.time import normalize_utc_timestamp, parse_datetime
from home_optimizer.infrastructure.database.orm_models import ElectricityPriceIntervalValue
from home_optimizer.infrastructure.database.session import Database

SQLITE_MAX_VARIABLES = 999
ELECTRICITY_PRICE_INTERVAL_COLUMN_COUNT = 6
ELECTRICITY_PRICE_INTERVAL_INSERT_BATCH_SIZE = (
    SQLITE_MAX_VARIABLES // ELECTRICITY_PRICE_INTERVAL_COLUMN_COUNT
)


class ElectricityPriceRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def latest_interval_end(self, source: str) -> datetime | None:
        with self.database.session() as session:
            timestamp = session.execute(
                select(func.max(ElectricityPriceIntervalValue.end_time_utc)).where(
                    ElectricityPriceIntervalValue.source == source,
                )
            ).scalar_one()

        return parse_datetime(timestamp) if timestamp else None

    def upsert_intervals(self, intervals: list[PriceInterval]) -> int:
        return self._write_intervals(intervals)

    def replace_future_intervals(
        self,
        *,
        source: str,
        from_time: datetime,
        intervals: list[PriceInterval],
    ) -> int:
        with self.database.session() as session:
            session.execute(
                delete(ElectricityPriceIntervalValue).where(
                    ElectricityPriceIntervalValue.source == source,
                    ElectricityPriceIntervalValue.end_time_utc > normalize_utc_timestamp(from_time),
                )
            )
            written_rows = self._write_intervals(intervals, session=session)
            session.commit()

        return written_rows

    def _write_intervals(self, intervals: list[PriceInterval], *, session=None) -> int:
        if not intervals:
            return 0

        owns_session = session is None
        active_session = session or self.database.session()
        written_rows = 0
        try:
            rows = [
                {
                    "name": interval.name,
                    "start_time_utc": normalize_utc_timestamp(interval.start_time_utc),
                    "end_time_utc": normalize_utc_timestamp(interval.end_time_utc),
                    "source": interval.source,
                    "unit": interval.unit,
                    "value": interval.value,
                }
                for interval in intervals
            ]
            for start_index in range(0, len(rows), ELECTRICITY_PRICE_INTERVAL_INSERT_BATCH_SIZE):
                batch = rows[start_index : start_index + ELECTRICITY_PRICE_INTERVAL_INSERT_BATCH_SIZE]
                statement = insert(ElectricityPriceIntervalValue).values(batch)
                result = active_session.execute(
                    statement.on_conflict_do_update(
                        index_elements=[
                            ElectricityPriceIntervalValue.name,
                            ElectricityPriceIntervalValue.start_time_utc,
                            ElectricityPriceIntervalValue.end_time_utc,
                            ElectricityPriceIntervalValue.source,
                        ],
                        set_={
                            "unit": statement.excluded.unit,
                            "value": statement.excluded.value,
                        },
                    )
                )
                written_rows += len(batch)
            if owns_session:
                active_session.commit()
        finally:
            if owns_session:
                active_session.close()

        return written_rows

