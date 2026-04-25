from __future__ import annotations

from datetime import datetime, timezone

import httpx
from sqlalchemy import select

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.location import Location
from home_optimizer.features.forecast.service import OpenMeteoForecastService
from home_optimizer.infrastructure.database.forecast_repository import ForecastRepository
from home_optimizer.infrastructure.database.orm_models import ForecastValue
from home_optimizer.infrastructure.database.session import Database
from home_optimizer.infrastructure.weather.openmeteo import OpenMeteoGateway


def test_openmeteo_forecast_service_stores_requested_series(tmp_path) -> None:
    responses = [
        {
            "minutely_15": {
                "time": ["2026-04-25T12:00", "2026-04-25T12:15"],
                "temperature_2m": [12.5, 12.0],
                "relative_humidity_2m": [65, 66],
                "wind_speed_10m": [4.1, 4.0],
                "dew_point_2m": [6.0, 5.8],
                "direct_radiation": [300, 280],
                "diffuse_radiation": [120, 110],
            }
        },
        {
            "minutely_15": {
                "time": ["2026-04-25T12:00", "2026-04-25T12:15"],
                "global_tilted_irradiance": [500, 470],
            }
        },
        {
            "minutely_15": {
                "time": ["2026-04-25T12:00", "2026-04-25T12:15"],
                "global_tilted_irradiance": [220, 210],
            }
        },
    ]
    seen_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_queries.append(str(request.url))
        return httpx.Response(200, json=responses.pop(0))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    settings = AppSettings.from_options(
        {
            "database_path": str(tmp_path / "forecast.sqlite"),
            "pv_tilt": 50.0,
            "pv_azimuth": 148.0,
            "living_room_window_azimuth": 225.0,
        }
    )
    database = Database(settings.database_path)
    database.init_schema()
    gateway = OpenMeteoGateway(client=client)
    repository = ForecastRepository(database)
    service = OpenMeteoForecastService(
        gateway,
        Location(latitude=52.09, longitude=5.12),
        repository,
        enabled=settings.open_meteo_enabled,
        pv_tilt=settings.pv_tilt,
        pv_azimuth=settings.pv_azimuth,
        living_room_window_azimuth=settings.living_room_window_azimuth,
        forecast_steps=2,
    )

    written = service.refresh_forecast(datetime(2026, 4, 25, 11, 45, tzinfo=timezone.utc))

    assert written == 16
    assert len(seen_queries) == 3
    assert "latitude=52.09" in seen_queries[0]
    assert "longitude=5.12" in seen_queries[0]
    assert (
        "minutely_15=temperature_2m%2Crelative_humidity_2m%2Cwind_speed_10m"
        "%2Cdew_point_2m%2Cdirect_radiation%2Cdiffuse_radiation"
        in seen_queries[0]
    )
    assert "azimuth=-32.0" in seen_queries[1]
    assert "tilt=90.0" in seen_queries[2]
    assert "azimuth=45.0" in seen_queries[2]

    with database.session() as session:
        rows = session.execute(
            select(ForecastValue).order_by(ForecastValue.forecast_time_utc, ForecastValue.name)
        ).scalars().all()

    assert [row.name for row in rows] == [
        "dew_point",
        "diffuse_radiation",
        "direct_radiation",
        "gti_living_room_windows",
        "gti_pv",
        "humidity",
        "temperature",
        "wind",
        "dew_point",
        "diffuse_radiation",
        "direct_radiation",
        "gti_living_room_windows",
        "gti_pv",
        "humidity",
        "temperature",
        "wind",
    ]
    assert rows[0].created_at_utc == "2026-04-25T11:45:00+00:00"
    assert rows[0].forecast_time_utc == "2026-04-25T12:00:00+00:00"
    assert rows[0].unit == "degC"
    assert rows[0].source == "openmeteo"


def test_openmeteo_forecast_service_skips_without_home_coordinates(tmp_path) -> None:
    settings = AppSettings.from_options({"database_path": str(tmp_path / "forecast.sqlite")})
    database = Database(settings.database_path)
    database.init_schema()
    repository = ForecastRepository(database)
    gateway = OpenMeteoGateway(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )
    )
    service = OpenMeteoForecastService(
        gateway,
        None,
        repository,
        enabled=settings.open_meteo_enabled,
        pv_tilt=settings.pv_tilt,
        pv_azimuth=settings.pv_azimuth,
        living_room_window_azimuth=settings.living_room_window_azimuth,
    )

    assert service.enabled is True
    assert service.refresh_forecast() == 0
