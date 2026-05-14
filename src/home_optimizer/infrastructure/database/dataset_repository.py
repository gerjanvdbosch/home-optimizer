from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import Select, select
from sqlalchemy.orm import InstrumentedAttribute

from home_optimizer.domain import ForecastEntry, latest_forecast_entries, resample_forecast_entries
from home_optimizer.domain.time import parse_datetime
from home_optimizer.domain.time import normalize_utc_timestamp
from home_optimizer.infrastructure.database.orm_models import (
    ElectricityPriceIntervalValue,
    ForecastValue,
    Sample1m,
    Sample15m,
)
from home_optimizer.infrastructure.database.session import Database

_COVERAGE_TOLERANCE_MINUTES = 2
_FORECAST_SOURCE_INTERVAL_MINUTES = 15


def _frame_covers_range(
    frame: pd.DataFrame,
    timestamp_column: str,
    start_time: datetime,
    end_time: datetime,
) -> bool:
    if frame.empty or timestamp_column not in frame.columns:
        return False
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True).dropna()
    if timestamps.empty:
        return False
    tolerance = pd.Timedelta(minutes=_COVERAGE_TOLERANCE_MINUTES)
    start_ts = pd.Timestamp(start_time).tz_convert("UTC") if pd.Timestamp(start_time).tzinfo else pd.Timestamp(start_time, tz="UTC")
    end_ts = pd.Timestamp(end_time).tz_convert("UTC") if pd.Timestamp(end_time).tzinfo else pd.Timestamp(end_time, tz="UTC")
    return (
        timestamps.min() <= start_ts + tolerance
        and timestamps.max() >= end_ts - tolerance
    )


class DatasetRepository:
    def __init__(self, database: Database, *, forecast_interval_minutes: int = 10) -> None:
        self.database = database
        self.forecast_interval_minutes = forecast_interval_minutes

    def read_samples(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        names: list[str] | None = None,
        sources: list[str] | None = None,
        categories: list[str] | None = None,
        entity_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read measurement samples, preferring 1m resolution with fallback to 15m.

        When 1m data covers the full requested range it is returned exclusively.
        Otherwise 15m data is used as the primary source, but any existing 1m
        data is concatenated alongside it so that brief window-flag events
        (defrost, booster heater) that only appear in 1m samples are still
        captured by the max() aggregation in the service.
        """
        frame_1m = self.read_samples_1m(
            start_time=start_time,
            end_time=end_time,
            names=names,
            sources=sources,
            categories=categories,
            entity_ids=entity_ids,
        )
        if (
            start_time is not None
            and end_time is not None
            and _frame_covers_range(frame_1m, "timestamp_minute_utc", start_time, end_time)
        ):
            frame_1m["timestamp_utc"] = frame_1m["timestamp_minute_utc"]
            return frame_1m

        frame_15m = self.read_samples_15m(
            start_time=start_time,
            end_time=end_time,
            names=names,
            sources=sources,
            categories=categories,
            entity_ids=entity_ids,
        )
        frame_15m["timestamp_utc"] = frame_15m["timestamp_15m_utc"]

        if frame_1m.empty:
            return frame_15m

        # Include 1m data alongside 15m so brief flag events are still captured.
        frame_1m["timestamp_utc"] = frame_1m["timestamp_minute_utc"]
        return pd.concat([frame_15m, frame_1m], ignore_index=True)

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

    def read_forecast_values(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        names: list[str] | None = None,
        sources: list[str] | None = None,
        created_at_start_time: datetime | None = None,
        created_at_end_time: datetime | None = None,
    ) -> pd.DataFrame:
        query_start_time = (
            start_time - timedelta(minutes=_FORECAST_SOURCE_INTERVAL_MINUTES)
            if start_time is not None
            else None
        )
        query_end_time = (
            end_time + timedelta(minutes=_FORECAST_SOURCE_INTERVAL_MINUTES)
            if end_time is not None
            else None
        )
        statement: Select[tuple[object]] = select(ForecastValue)

        if query_start_time is not None:
            statement = statement.where(
                ForecastValue.forecast_time_utc >= normalize_utc_timestamp(query_start_time)
            )
        if query_end_time is not None:
            statement = statement.where(
                ForecastValue.forecast_time_utc < normalize_utc_timestamp(query_end_time)
            )
        if created_at_start_time is not None:
            statement = statement.where(
                ForecastValue.created_at_utc >= normalize_utc_timestamp(created_at_start_time)
            )
        if created_at_end_time is not None:
            statement = statement.where(
                ForecastValue.created_at_utc < normalize_utc_timestamp(created_at_end_time)
            )
        if names:
            statement = statement.where(ForecastValue.name.in_(names))
        if sources:
            statement = statement.where(ForecastValue.source.in_(sources))

        statement = statement.order_by(
            ForecastValue.forecast_time_utc,
            ForecastValue.name,
            ForecastValue.created_at_utc,
            ForecastValue.source,
        )

        with self.database.session() as session:
            frame = pd.read_sql(statement, session.connection())

        if frame.empty:
            return frame

        latest_entries = latest_forecast_entries(
            [
                ForecastEntry(
                    created_at_utc=parse_datetime(str(row["created_at_utc"])),
                    forecast_time_utc=parse_datetime(str(row["forecast_time_utc"])),
                    name=str(row["name"]),
                    value=float(row["value"]),
                    unit=str(row["unit"]) if row["unit"] is not None else None,
                    source=str(row["source"]),
                )
                for row in frame.to_dict(orient="records")
            ]
        )
        if start_time is not None and end_time is not None:
            latest_entries = resample_forecast_entries(
                latest_entries,
                start_time=start_time,
                end_time=end_time,
                interval_minutes=self.forecast_interval_minutes,
            )

        return pd.DataFrame(
            [
                {
                    "created_at_utc": normalize_utc_timestamp(entry.created_at_utc),
                    "forecast_time_utc": normalize_utc_timestamp(entry.forecast_time_utc),
                    "name": entry.name,
                    "source": entry.source,
                    "unit": entry.unit,
                    "value": entry.value,
                }
                for entry in latest_entries
            ],
            columns=[
                "created_at_utc",
                "forecast_time_utc",
                "name",
                "source",
                "unit",
                "value",
            ],
        )

    def read_electricity_price_intervals(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        names: list[str] | None = None,
        sources: list[str] | None = None,
    ) -> pd.DataFrame:
        statement: Select[tuple[object]] = select(ElectricityPriceIntervalValue)

        if start_time is not None:
            statement = statement.where(
                ElectricityPriceIntervalValue.end_time_utc > normalize_utc_timestamp(start_time)
            )
        if end_time is not None:
            statement = statement.where(
                ElectricityPriceIntervalValue.start_time_utc < normalize_utc_timestamp(end_time)
            )
        if names:
            statement = statement.where(ElectricityPriceIntervalValue.name.in_(names))
        if sources:
            statement = statement.where(ElectricityPriceIntervalValue.source.in_(sources))

        statement = statement.order_by(
            ElectricityPriceIntervalValue.start_time_utc,
            ElectricityPriceIntervalValue.name,
            ElectricityPriceIntervalValue.source,
        )

        with self.database.session() as session:
            return pd.read_sql(statement, session.connection())

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
