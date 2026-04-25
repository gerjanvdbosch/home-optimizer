from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select

from home_optimizer.domain.clock import utc_now
from home_optimizer.domain.sensors import SensorSpec
from home_optimizer.domain.time import normalize_utc_timestamp
from home_optimizer.features.history_import.models import MinuteSample
from home_optimizer.infrastructure.database.orm_models import ImportChunk, Sample1m
from home_optimizer.infrastructure.database.session import Database


class TimeSeriesRepository:
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
        normalized_start = normalize_utc_timestamp(start_time)
        normalized_end = normalize_utc_timestamp(end_time)

        with self.database.session() as session:
            existing = session.get(
                ImportChunk,
                {
                    "source": self.source,
                    "name": spec.name,
                    "start_time_utc": normalized_start,
                },
            )

        return existing is not None and existing.end_time_utc == normalized_end

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
            start_time_utc=normalize_utc_timestamp(start_time),
            end_time_utc=normalize_utc_timestamp(end_time),
            row_count=row_count,
            imported_at_utc=normalize_utc_timestamp(utc_now()),
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

    def write_samples(self, samples: list[MinuteSample]) -> None:
        self.write_rows([self._to_orm_sample(sample) for sample in samples])

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
                    Sample1m.timestamp_minute_utc < normalize_utc_timestamp(before_time),
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

    @staticmethod
    def _to_orm_sample(sample: MinuteSample) -> Sample1m:
        return Sample1m(
            timestamp_minute_utc=sample.timestamp_minute.isoformat(),
            name=sample.name,
            source=sample.source,
            entity_id=sample.entity_id,
            category=sample.category,
            unit=sample.unit,
            mean_real=sample.mean_real,
            min_real=sample.min_real,
            max_real=sample.max_real,
            last_real=sample.last_real,
            last_text=sample.last_text,
            last_bool=sample.last_bool,
            sample_count=sample.sample_count,
        )
