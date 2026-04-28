from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select

from home_optimizer.domain import NumericPoint, NumericSeries, TextPoint, TextSeries
from home_optimizer.domain.time import normalize_utc_timestamp
from home_optimizer.infrastructure.database.orm_models import ForecastValue, Sample1m
from home_optimizer.infrastructure.database.session import Database


class DashboardRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def read_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[NumericSeries]:
        if not names:
            return []

        value_expr = func.coalesce(
            Sample1m.mean_real,
            Sample1m.last_real,
            Sample1m.max_real,
            Sample1m.min_real,
            Sample1m.last_bool,
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
            points_by_name[name].append(NumericPoint(timestamp=timestamp, value=float(value)))
            units_by_name[name] = units_by_name[name] or unit

        return [
            NumericSeries(name=name, unit=units_by_name[name], points=points_by_name[name])
            for name in names
        ]

    def read_text_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[TextSeries]:
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
            points_by_name[name].append(TextPoint(timestamp=timestamp, value=str(value)))

        return [TextSeries(name=name, points=points_by_name[name]) for name in names]

    def read_forecast_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[NumericSeries]:
        if not names:
            return []

        with self.database.session() as session:
            latest_subquery = (
                select(
                    ForecastValue.name.label("name"),
                    ForecastValue.forecast_time_utc.label("forecast_time_utc"),
                    func.max(ForecastValue.created_at_utc).label("latest_created_at"),
                )
                .where(
                    ForecastValue.name.in_(names),
                    ForecastValue.forecast_time_utc >= normalize_utc_timestamp(start_time),
                    ForecastValue.forecast_time_utc < normalize_utc_timestamp(end_time),
                    ForecastValue.created_at_utc <= ForecastValue.forecast_time_utc,
                )
                .group_by(
                    ForecastValue.name,
                    ForecastValue.forecast_time_utc,
                )
                .subquery()
            )

            rows = session.execute(
                select(
                    ForecastValue.name,
                    ForecastValue.forecast_time_utc,
                    ForecastValue.unit,
                    ForecastValue.value,
                )
                .join(
                    latest_subquery,
                    (
                        (ForecastValue.name == latest_subquery.c.name)
                        & (
                            ForecastValue.forecast_time_utc
                            == latest_subquery.c.forecast_time_utc
                        )
                        & (
                            ForecastValue.created_at_utc
                            == latest_subquery.c.latest_created_at
                        )
                    ),
                )
                .order_by(
                    ForecastValue.name,
                    ForecastValue.forecast_time_utc,
                )
            ).all()

        points_by_name = {name: [] for name in names}
        units_by_name: dict[str, str | None] = {
            name: None for name in names
        }

        for name, timestamp, unit, value in rows:
            points_by_name[name].append(
                NumericPoint(
                    timestamp=timestamp,
                    value=float(value),
                )
            )
            units_by_name[name] = units_by_name[name] or unit

        return [
            NumericSeries(
                name=name,
                unit=units_by_name[name],
                points=points_by_name[name],
            )
            for name in names
        ]
