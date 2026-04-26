from __future__ import annotations

from datetime import datetime, timezone

from home_optimizer.domain.charts import ChartPoint
from home_optimizer.infrastructure.database.dashboard_repository import DashboardRepository
from home_optimizer.infrastructure.database.orm_models import ForecastValue
from home_optimizer.infrastructure.database.session import Database


def test_dashboard_repository_reads_latest_forecast_batch(tmp_path) -> None:
    database = Database(str(tmp_path / "dashboard.sqlite"))
    database.init_schema()
    repository = DashboardRepository(database)

    with database.session() as session:
        session.add_all(
            [
                ForecastValue(
                    created_at_utc="2026-04-25T10:00:00+00:00",
                    forecast_time_utc="2026-04-26T12:00:00+00:00",
                    name="temperature",
                    source="openmeteo",
                    unit="degC",
                    value=9.0,
                ),
                ForecastValue(
                    created_at_utc="2026-04-25T11:00:00+00:00",
                    forecast_time_utc="2026-04-26T12:00:00+00:00",
                    name="temperature",
                    source="openmeteo",
                    unit="degC",
                    value=12.5,
                ),
                ForecastValue(
                    created_at_utc="2026-04-25T11:00:00+00:00",
                    forecast_time_utc="2026-04-26T12:00:00+00:00",
                    name="gti_pv",
                    source="openmeteo",
                    unit="Wm2",
                    value=500.0,
                ),
                ForecastValue(
                    created_at_utc="2026-04-25T11:00:00+00:00",
                    forecast_time_utc="2026-04-26T12:00:00+00:00",
                    name="gti_living_room_windows",
                    source="openmeteo",
                    unit="Wm2",
                    value=220.0,
                ),
                ForecastValue(
                    created_at_utc="2026-04-25T11:00:00+00:00",
                    forecast_time_utc="2026-04-27T00:00:00+00:00",
                    name="temperature",
                    source="openmeteo",
                    unit="degC",
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
    assert series[0].unit == "degC"
    assert series[0].points == [ChartPoint(timestamp="2026-04-26T12:00:00+00:00", value=12.5)]
    assert series[1].unit == "Wm2"
    assert series[1].points == [ChartPoint(timestamp="2026-04-26T12:00:00+00:00", value=500.0)]
    assert series[2].unit == "Wm2"
    assert series[2].points == [ChartPoint(timestamp="2026-04-26T12:00:00+00:00", value=220.0)]

