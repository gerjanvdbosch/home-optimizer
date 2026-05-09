from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import Select, select
from sqlalchemy.orm import InstrumentedAttribute

from home_optimizer.domain.time import normalize_utc_timestamp
from home_optimizer.infrastructure.database.orm_models import Sample1m, Sample15m
from home_optimizer.infrastructure.database.session import Database


class DatasetRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def read_samples(
        self,
        *,
        interval_minutes: int,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        names: list[str] | None = None,
        sources: list[str] | None = None,
        categories: list[str] | None = None,
        entity_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        if interval_minutes == 1:
            frame_1m = self.read_samples_1m(
                start_time=start_time,
                end_time=end_time,
                names=names,
                sources=sources,
                categories=categories,
                entity_ids=entity_ids,
            )
            if not frame_1m.empty:
                return frame_1m
            return self.read_samples_15m(
                start_time=start_time,
                end_time=end_time,
                names=names,
                sources=sources,
                categories=categories,
                entity_ids=entity_ids,
            )
        if interval_minutes == 15:
            return self.read_samples_15m(
                start_time=start_time,
                end_time=end_time,
                names=names,
                sources=sources,
                categories=categories,
                entity_ids=entity_ids,
            )
        raise ValueError("interval_minutes must be 1 or 15")

    def read_samples_1m(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        names: list[str] | None = None,
        sources: list[str] | None = None,
        categories: list[str] | None = None,
        entity_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        return self._read_table(
            model=Sample1m,
            timestamp_column=Sample1m.timestamp_minute_utc,
            start_time=start_time,
            end_time=end_time,
            names=names,
            sources=sources,
            categories=categories,
            entity_ids=entity_ids,
        )

    def read_samples_15m(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        names: list[str] | None = None,
        sources: list[str] | None = None,
        categories: list[str] | None = None,
        entity_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        return self._read_table(
            model=Sample15m,
            timestamp_column=Sample15m.timestamp_15m_utc,
            start_time=start_time,
            end_time=end_time,
            names=names,
            sources=sources,
            categories=categories,
            entity_ids=entity_ids,
        )

    def _read_table(
        self,
        *,
        model: type[Sample1m] | type[Sample15m],
        timestamp_column: InstrumentedAttribute[str],
        start_time: datetime | None,
        end_time: datetime | None,
        names: list[str] | None,
        sources: list[str] | None,
        categories: list[str] | None,
        entity_ids: list[str] | None,
    ) -> pd.DataFrame:
        statement: Select[tuple[object]] = select(model)

        if start_time is not None:
            statement = statement.where(timestamp_column >= normalize_utc_timestamp(start_time))
        if end_time is not None:
            statement = statement.where(timestamp_column < normalize_utc_timestamp(end_time))
        if names:
            statement = statement.where(model.name.in_(names))
        if sources:
            statement = statement.where(model.source.in_(sources))
        if categories:
            statement = statement.where(model.category.in_(categories))
        if entity_ids:
            statement = statement.where(model.entity_id.in_(entity_ids))

        statement = statement.order_by(timestamp_column, model.name, model.source)

        with self.database.session() as session:
            return pd.read_sql(statement, session.connection())
