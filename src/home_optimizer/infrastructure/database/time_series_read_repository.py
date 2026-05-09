from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select

from home_optimizer.domain import (
    ELECTRICITY_PRICE,
    NumericPoint,
    NumericSeries,
    TextPoint,
    TextSeries,
    merge_numeric_with_fallback,
    merge_text_with_fallback,
)
from home_optimizer.domain.pricing import (
    PriceInterval,
    empty_electricity_price_series,
    price_series_from_intervals,
)
from home_optimizer.domain.time import normalize_utc_timestamp, parse_datetime
from home_optimizer.infrastructure.database.orm_models import (
    ElectricityPriceIntervalValue,
    ForecastValue,
    Sample1m,
    Sample15m,
)
from home_optimizer.infrastructure.database.session import Database


class TimeSeriesReadRepository:
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

        start_ts = normalize_utc_timestamp(start_time)
        end_ts = normalize_utc_timestamp(end_time)

        value_expr_1m = func.coalesce(
            Sample1m.mean_real,
            Sample1m.last_real,
            Sample1m.max_real,
            Sample1m.min_real,
            Sample1m.last_bool,
        )
        value_expr_15m = func.coalesce(
            Sample15m.mean_real,
            Sample15m.last_real,
            Sample15m.max_real,
            Sample15m.min_real,
            Sample15m.last_bool,
        )

        with self.database.session() as session:
            rows_1m = session.execute(
                select(
                    Sample1m.name,
                    Sample1m.timestamp_minute_utc,
                    Sample1m.unit,
                    value_expr_1m.label("value"),
                )
                .where(
                    Sample1m.name.in_(names),
                    Sample1m.timestamp_minute_utc >= start_ts,
                    Sample1m.timestamp_minute_utc < end_ts,
                    value_expr_1m.is_not(None),
                )
                .order_by(Sample1m.name, Sample1m.timestamp_minute_utc, Sample1m.source)
            ).all()

            rows_15m = session.execute(
                select(
                    Sample15m.name,
                    Sample15m.timestamp_15m_utc,
                    Sample15m.unit,
                    value_expr_15m.label("value"),
                )
                .where(
                    Sample15m.name.in_(names),
                    Sample15m.timestamp_15m_utc >= start_ts,
                    Sample15m.timestamp_15m_utc < end_ts,
                    value_expr_15m.is_not(None),
                )
                .order_by(Sample15m.name, Sample15m.timestamp_15m_utc, Sample15m.source)
            ).all()

        points_1m: dict[str, list[NumericPoint]] = {name: [] for name in names}
        units_by_name: dict[str, str | None] = {name: None for name in names}
        for name, timestamp, unit, value in rows_1m:
            points_1m[name].append(NumericPoint(timestamp=timestamp, value=float(value)))
            units_by_name[name] = units_by_name[name] or unit

        points_15m: dict[str, list[NumericPoint]] = {name: [] for name in names}
        for name, timestamp, unit, value in rows_15m:
            points_15m[name].append(NumericPoint(timestamp=timestamp, value=float(value)))
            units_by_name[name] = units_by_name[name] or unit

        return [
            NumericSeries(
                name=name,
                unit=units_by_name[name],
                points=merge_numeric_with_fallback(points_1m[name], points_15m[name]),
            )
            for name in names
        ]

    def sample_time_range(self) -> tuple[datetime | None, datetime | None]:
        with self.database.session() as session:
            earliest_1m, latest_1m = session.execute(
                select(
                    func.min(Sample1m.timestamp_minute_utc),
                    func.max(Sample1m.timestamp_minute_utc),
                )
            ).one()

            earliest_15m, latest_15m = session.execute(
                select(
                    func.min(Sample15m.timestamp_15m_utc),
                    func.max(Sample15m.timestamp_15m_utc),
                )
            ).one()

        candidates_earliest = [t for t in (earliest_1m, earliest_15m) if t is not None]
        candidates_latest = [t for t in (latest_1m, latest_15m) if t is not None]

        if not candidates_earliest or not candidates_latest:
            return None, None

        return parse_datetime(min(candidates_earliest)), parse_datetime(max(candidates_latest))

    def read_text_series(
        self,
        names: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> list[TextSeries]:
        if not names:
            return []

        start_ts = normalize_utc_timestamp(start_time)
        end_ts = normalize_utc_timestamp(end_time)

        with self.database.session() as session:
            rows_1m = session.execute(
                select(
                    Sample1m.name,
                    Sample1m.timestamp_minute_utc,
                    Sample1m.last_text,
                )
                .where(
                    Sample1m.name.in_(names),
                    Sample1m.timestamp_minute_utc >= start_ts,
                    Sample1m.timestamp_minute_utc < end_ts,
                    Sample1m.last_text.is_not(None),
                )
                .order_by(Sample1m.name, Sample1m.timestamp_minute_utc, Sample1m.source)
            ).all()

            rows_15m = session.execute(
                select(
                    Sample15m.name,
                    Sample15m.timestamp_15m_utc,
                    Sample15m.last_text,
                )
                .where(
                    Sample15m.name.in_(names),
                    Sample15m.timestamp_15m_utc >= start_ts,
                    Sample15m.timestamp_15m_utc < end_ts,
                    Sample15m.last_text.is_not(None),
                )
                .order_by(Sample15m.name, Sample15m.timestamp_15m_utc, Sample15m.source)
            ).all()

        points_1m: dict[str, list[TextPoint]] = {name: [] for name in names}
        for name, timestamp, value in rows_1m:
            points_1m[name].append(TextPoint(timestamp=timestamp, value=str(value)))

        points_15m: dict[str, list[TextPoint]] = {name: [] for name in names}
        for name, timestamp, value in rows_15m:
            points_15m[name].append(TextPoint(timestamp=timestamp, value=str(value)))

        return [
            TextSeries(
                name=name,
                points=merge_text_with_fallback(points_1m[name], points_15m[name]),
            )
            for name in names
        ]

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

    def read_electricity_price_series(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        source: str,
        interval_minutes: int = 15,
    ) -> NumericSeries:
        with self.database.session() as session:
            rows = session.execute(
                select(
                    ElectricityPriceIntervalValue.name,
                    ElectricityPriceIntervalValue.start_time_utc,
                    ElectricityPriceIntervalValue.end_time_utc,
                    ElectricityPriceIntervalValue.source,
                    ElectricityPriceIntervalValue.unit,
                    ElectricityPriceIntervalValue.value,
                )
                .where(
                    ElectricityPriceIntervalValue.name == ELECTRICITY_PRICE,
                    ElectricityPriceIntervalValue.source == source,
                    ElectricityPriceIntervalValue.end_time_utc
                    > normalize_utc_timestamp(start_time),
                    ElectricityPriceIntervalValue.start_time_utc
                    < normalize_utc_timestamp(end_time),
                )
                .order_by(ElectricityPriceIntervalValue.start_time_utc)
            ).all()

        if not rows:
            return empty_electricity_price_series()

        intervals = [
            PriceInterval(
                name=name,
                start_time_utc=parse_datetime(start_timestamp),
                end_time_utc=parse_datetime(end_timestamp),
                source=interval_source,
                unit=unit,
                value=float(value),
            )
            for name, start_timestamp, end_timestamp, interval_source, unit, value in rows
        ]
        return price_series_from_intervals(
            intervals,
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )
