from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select

from home_optimizer.shared.db.orm_models import ImportChunk, Sample1m
from home_optimizer.shared.db.session import Database
from home_optimizer.shared.sensors.definitions import SensorSpec
from home_optimizer.shared.time.clock import utc_now


class HistoryImportRepository:
    def __init__(
        self,
        database: Database,
        source: str = "home_assistant_history",
    ) -> None:
        self.database = database
        self.source = source

    def chunk_already_imported(
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

    def mark_chunk_imported(
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
            imported_at_utc=utc_now().isoformat(),
        )

        with self.database.session() as session:
            session.merge(marker)
            session.commit()

    def write_rows(self, rows: list[Sample1m]) -> None:
        if not rows:
            return

        with self.database.session() as session:
            for row in rows:
                session.merge(row)
            session.commit()

    def last_stored_value_before(
        self,
        spec: SensorSpec,
        before_time: datetime,
    ) -> Any | None:
        with self.database.session() as session:
            stmt = (
                select(Sample1m)
                .where(
                    Sample1m.name == spec.name,
                    Sample1m.source == self.source,
                    Sample1m.timestamp_minute_utc < before_time.isoformat(),
                )
                .order_by(Sample1m.timestamp_minute_utc.desc())
                .limit(1)
            )
            row = session.execute(stmt).scalar_one_or_none()

        if row is None:
            return None
        if row.last_bool is not None:
            return bool(row.last_bool)
        if row.last_real is not None:
            return row.last_real
        if row.last_text is not None:
            return row.last_text
        return None
