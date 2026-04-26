from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select

from home_optimizer.domain.charts import ChartPoint, ChartSeries, ChartTextPoint, ChartTextSeries
from home_optimizer.domain.time import normalize_utc_timestamp
from home_optimizer.infrastructure.database.orm_models import Sample1m
from home_optimizer.infrastructure.database.session import Database


class DashboardRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def read_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[ChartSeries]:
        if not names:
            return []

        value_expr = func.coalesce(
            Sample1m.mean_real,
            Sample1m.last_real,
            Sample1m.max_real,
            Sample1m.min_real,
        )
        with self.database.session() as session:
            rows = session.execute(
                select(
                    Sample1m.name,
                    Sample1m.timestamp_minute_utc,
                    Sample1m.unit,
                    value_expr.label("value"),
                )
                .where(
                    Sample1m.name.in_(names),
                    Sample1m.timestamp_minute_utc >= normalize_utc_timestamp(start_time),
                    Sample1m.timestamp_minute_utc < normalize_utc_timestamp(end_time),
                    value_expr.is_not(None),
                )
                .order_by(Sample1m.name, Sample1m.timestamp_minute_utc, Sample1m.source)
            ).all()

        points_by_name = {name: [] for name in names}
        units_by_name: dict[str, str | None] = {name: None for name in names}
        for name, timestamp, unit, value in rows:
            points_by_name[name].append(ChartPoint(timestamp=timestamp, value=float(value)))
            units_by_name[name] = units_by_name[name] or unit

        return [
            ChartSeries(name=name, unit=units_by_name[name], points=points_by_name[name])
            for name in names
        ]

    def read_text_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[ChartTextSeries]:
        if not names:
            return []

        with self.database.session() as session:
            rows = session.execute(
                select(
                    Sample1m.name,
                    Sample1m.timestamp_minute_utc,
                    Sample1m.last_text,
                )
                .where(
                    Sample1m.name.in_(names),
                    Sample1m.timestamp_minute_utc >= normalize_utc_timestamp(start_time),
                    Sample1m.timestamp_minute_utc < normalize_utc_timestamp(end_time),
                    Sample1m.last_text.is_not(None),
                )
                .order_by(Sample1m.name, Sample1m.timestamp_minute_utc, Sample1m.source)
            ).all()

        points_by_name = {name: [] for name in names}
        for name, timestamp, value in rows:
            points_by_name[name].append(ChartTextPoint(timestamp=timestamp, value=str(value)))

        return [ChartTextSeries(name=name, points=points_by_name[name]) for name in names]
