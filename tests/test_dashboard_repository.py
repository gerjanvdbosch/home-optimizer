from __future__ import annotations

from datetime import datetime, timezone

from home_optimizer.domain import NumericPoint
from home_optimizer.infrastructure.database.time_series_read_repository import (
    TimeSeriesReadRepository,
)
from home_optimizer.infrastructure.database.orm_models import ForecastValue, Sample1m
from home_optimizer.infrastructure.database.session import Database


def test_time_series_read_repository_reads_latest_forecast_batch(tmp_path) -> None:
    database = Database(str(tmp_path / "dashboard.sqlite"))
    database.init_schema()
    repository = TimeSeriesReadRepository(database)

    with database.session() as session:
        session.add_all(
            [
                ForecastValue(
                    created_at_utc="2026-04-25T10:00:00+00:00",
                    forecast_time_utc="2026-04-26T12:00:00+00:00",
                    name="temperature",
                    source="openmeteo",
                    unit="°C",
                    value=9.0,
                ),
                ForecastValue(
                    created_at_utc="2026-04-25T11:00:00+00:00",
                    forecast_time_utc="2026-04-26T12:00:00+00:00",
                    name="temperature",
                    source="openmeteo",
                    unit="°C",
                    value=12.5,
                ),
                ForecastValue(
                    created_at_utc="2026-04-25T11:00:00+00:00",
                    forecast_time_utc="2026-04-26T12:00:00+00:00",
                    name="gti_pv",
                    source="openmeteo",
                    unit="W/m2",
                    value=500.0,
                ),
                ForecastValue(
                    created_at_utc="2026-04-25T11:00:00+00:00",
                    forecast_time_utc="2026-04-26T12:00:00+00:00",
                    name="gti_living_room_windows",
                    source="openmeteo",
                    unit="W/m2",
                    value=220.0,
                ),
                ForecastValue(
                    created_at_utc="2026-04-25T11:00:00+00:00",
                    forecast_time_utc="2026-04-27T00:00:00+00:00",
                    name="temperature",
                    source="openmeteo",
                    unit="°C",
                    value=8.0,
                ),
            ]
        )
        session.commit()

    series = repository.read_forecast_series(
        names=["temperature", "gti_pv", "gti_living_room_windows"],
        start_time=datetime(2026, 4, 26, tzinfo=timezone.utc),
        end_time=datetime(2026, 4, 27, tzinfo=timezone.utc),
    )

    assert [item.name for item in series] == [
        "temperature",
        "gti_pv",
        "gti_living_room_windows",
    ]
    assert series[0].unit == "°C"
    assert series[0].points == [NumericPoint(timestamp="2026-04-26T12:00:00+00:00", value=12.5)]
    assert series[1].unit == "W/m2"
    assert series[1].points == [NumericPoint(timestamp="2026-04-26T12:00:00+00:00", value=500.0)]
    assert series[2].unit == "W/m2"
    assert series[2].points == [NumericPoint(timestamp="2026-04-26T12:00:00+00:00", value=220.0)]


def test_time_series_read_repository_returns_sample_time_range(tmp_path) -> None:
    database = Database(str(tmp_path / "dashboard.sqlite"))
    database.init_schema()
    repository = TimeSeriesReadRepository(database)

    with database.session() as session:
        session.add_all(
            [
                Sample1m(
                    timestamp_minute_utc="2026-04-25T00:00:00+00:00",
                    name="room_temperature",
                    source="home_assistant",
                    entity_id="sensor.room_temperature",
                    category="building",
                    unit="°C",
                    mean_real=20.0,
                    min_real=20.0,
                    max_real=20.0,
                    last_real=20.0,
                    last_text=None,
                    last_bool=None,
                    sample_count=1,
                ),
                Sample1m(
                    timestamp_minute_utc="2026-04-28T23:59:00+00:00",
                    name="room_temperature",
                    source="home_assistant",
                    entity_id="sensor.room_temperature",
                    category="building",
                    unit="°C",
                    mean_real=21.0,
                    min_real=21.0,
                    max_real=21.0,
                    last_real=21.0,
                    last_text=None,
                    last_bool=None,
                    sample_count=1,
                ),
            ]
        )
        session.commit()

    assert repository.sample_time_range() == (
        datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 4, 28, 23, 59, tzinfo=timezone.utc),
    )
