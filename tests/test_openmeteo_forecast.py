from __future__ import annotations

from datetime import datetime, timezone

import httpx
from sqlalchemy import select

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.names import GTI_LIVING_ROOM_WINDOWS, GTI_PV
from home_optimizer.domain.location import Location
from home_optimizer.features.forecast.service import OpenMeteoForecastService
from home_optimizer.features.history_import.historical_weather_import_service import (
    HistoricalWeatherImportService,
)
from home_optimizer.features.history_import.weather_import_service import WeatherImportService
from home_optimizer.infrastructure.database.forecast_repository import ForecastRepository
from home_optimizer.infrastructure.database.historical_weather_repository import (
    HistoricalWeatherRepository,
)
from home_optimizer.infrastructure.database.orm_models import ForecastValue, HistoricalWeatherValue
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
        pv_tilt=settings.pv_tilt,
        pv_azimuth=settings.pv_azimuth,
        living_room_window_azimuth=settings.living_room_window_azimuth,
        poll_interval_seconds=settings.forecast_poll_interval_seconds,
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


def test_openmeteo_forecast_service_skips_when_database_forecast_is_fresh(tmp_path) -> None:
    responses = [
        {
            "minutely_15": {
                "time": ["2026-04-25T12:00"],
                "temperature_2m": [12.5],
                "relative_humidity_2m": [65],
                "wind_speed_10m": [4.1],
                "dew_point_2m": [6.0],
                "direct_radiation": [300],
                "diffuse_radiation": [120],
            }
        },
    ]
    seen_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_queries.append(str(request.url))
        return httpx.Response(200, json=responses.pop(0))

    settings = AppSettings.from_options(
        {
            "database_path": str(tmp_path / "forecast.sqlite"),
            "forecast_poll_interval_seconds": 3600,
        }
    )
    database = Database(settings.database_path)
    database.init_schema()
    gateway = OpenMeteoGateway(client=httpx.Client(transport=httpx.MockTransport(handler)))
    repository = ForecastRepository(database)
    service = OpenMeteoForecastService(
        gateway,
        Location(latitude=52.09, longitude=5.12),
        repository,
        pv_tilt=None,
        pv_azimuth=None,
        living_room_window_azimuth=None,
        poll_interval_seconds=settings.forecast_poll_interval_seconds,
        forecast_steps=1,
    )

    first_written = service.refresh_forecast(
        datetime(2026, 4, 25, 11, 45, tzinfo=timezone.utc)
    )
    second_written = service.refresh_forecast(
        datetime(2026, 4, 25, 12, 44, 59, tzinfo=timezone.utc)
    )

    with database.session() as session:
        rows = session.execute(select(ForecastValue)).scalars().all()

    assert first_written == 6
    assert second_written == 0
    assert len(seen_queries) == 1
    assert len(rows) == 6


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
        pv_tilt=settings.pv_tilt,
        pv_azimuth=settings.pv_azimuth,
        living_room_window_azimuth=settings.living_room_window_azimuth,
        poll_interval_seconds=settings.forecast_poll_interval_seconds,
    )

    assert service.enabled is True
    assert service.refresh_forecast() == 0


def test_weather_import_service_imports_only_missing_historical_rows(tmp_path) -> None:
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
        }
    ]
    seen_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_queries.append(str(request.url))
        return httpx.Response(200, json=responses.pop(0))

    settings = AppSettings.from_options(
        {
            "database_path": str(tmp_path / "forecast.sqlite"),
        }
    )
    database = Database(settings.database_path)
    database.init_schema()
    repository = ForecastRepository(database)
    with database.session() as session:
        session.add(
            ForecastValue(
                created_at_utc="2026-04-25T12:00:00+00:00",
                forecast_time_utc="2026-04-25T12:00:00+00:00",
                name="temperature",
                value=12.5,
                unit="degC",
                source="openmeteo",
            )
        )
        session.commit()

    service = WeatherImportService(
        OpenMeteoGateway(client=httpx.Client(transport=httpx.MockTransport(handler))),
        Location(latitude=52.09, longitude=5.12),
        repository,
        pv_tilt=None,
        pv_azimuth=None,
        living_room_window_azimuth=None,
        history_days_back=settings.history_import_max_days_back,
    )

    written = service.import_weather_data(
        datetime(2026, 4, 30, 10, 0, tzinfo=timezone.utc)
    )

    assert written == 11
    assert len(seen_queries) == 1
    assert f"past_days={settings.history_import_max_days_back}" in seen_queries[0]
    assert "forecast_minutely_15=192" not in seen_queries[0]

    with database.session() as session:
        rows = session.execute(
            select(ForecastValue).order_by(ForecastValue.forecast_time_utc, ForecastValue.name)
        ).scalars().all()

    assert rows[0].created_at_utc == "2026-04-25T12:00:00+00:00"
    assert all(row.created_at_utc == row.forecast_time_utc for row in rows[1:])


def test_forecast_repository_write_new_entries_chunks_large_insert(tmp_path) -> None:
    database = Database(str(tmp_path / "forecast.sqlite"))
    database.init_schema()
    repository = ForecastRepository(database)
    entries = [
        ForecastEntry(
            created_at_utc=f"2026-04-25T12:{(index % 60):02d}:00+00:00",
            forecast_time_utc=f"2026-04-26T{(index // 4) % 24:02d}:{((index % 4) * 15):02d}:00+00:00",
            name=f"temperature_{index}",
            source="openmeteo",
            unit="degC",
            value=float(index),
        )
        for index in range(220)
    ]

    inserted = repository.write_new_entries(entries)

    with database.session() as session:
        row_count = session.execute(select(ForecastValue)).scalars().all()

    assert inserted == 220
    assert len(row_count) == 220


def test_historical_weather_import_service_stores_archive_rows(tmp_path) -> None:
    responses = [
        {
            "hourly": {
                "time": ["2026-04-29T00:00", "2026-04-29T01:00"],
                "temperature_2m": [10.0, 10.5],
                "relative_humidity_2m": [60.0, 61.0],
                "wind_speed_10m": [3.0, 3.5],
                "dew_point_2m": [4.0, 4.5],
                "direct_radiation": [0.0, 10.0],
                "diffuse_radiation": [0.0, 8.0],
            }
        },
        {
            "hourly": {
                "time": ["2026-04-29T00:00", "2026-04-29T01:00"],
                "global_tilted_irradiance": [0.0, 12.0],
            }
        },
        {
            "hourly": {
                "time": ["2026-04-29T00:00", "2026-04-29T01:00"],
                "global_tilted_irradiance": [0.0, 5.0],
            }
        },
    ]
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json=responses.pop(0))

    database = Database(str(tmp_path / "historical-weather.sqlite"))
    database.init_schema()
    repository = HistoricalWeatherRepository(database)
    service = HistoricalWeatherImportService(
        gateway=OpenMeteoGateway(client=httpx.Client(transport=httpx.MockTransport(handler))),
        location=Location(latitude=52.09, longitude=5.12),
        repository=repository,
        pv_tilt=50.0,
        pv_azimuth=148.0,
        living_room_window_azimuth=225.0,
        history_days_back=1,
    )

    written = service.import_historical_weather(
        datetime(2026, 4, 29, 1, 30, tzinfo=timezone.utc)
    )

    assert written == 16
    assert len(seen_urls) == 3
    assert (
        "hourly=temperature_2m%2Crelative_humidity_2m%2Cwind_speed_10m%2Cdew_point_2m"
        "%2Cdirect_radiation%2Cdiffuse_radiation"
        in seen_urls[0]
    )
    assert "hourly=global_tilted_irradiance" in seen_urls[1]
    assert "hourly=global_tilted_irradiance" in seen_urls[2]
    assert "start_date=2026-04-28" in seen_urls[0]
    assert "end_date=2026-04-29" in seen_urls[0]

    with database.session() as session:
        rows = session.execute(
            select(HistoricalWeatherValue).order_by(
                HistoricalWeatherValue.timestamp_utc,
                HistoricalWeatherValue.name,
            )
        ).scalars().all()

    assert ("temperature", 10.0) in [(row.name, row.value) for row in rows]
    assert ("humidity", 60.0) in [(row.name, row.value) for row in rows]
    assert ("wind", 3.0) in [(row.name, row.value) for row in rows]
    assert ("dew_point", 4.0) in [(row.name, row.value) for row in rows]
    assert ("direct_radiation", 10.0) in [(row.name, row.value) for row in rows]
    assert ("diffuse_radiation", 8.0) in [(row.name, row.value) for row in rows]
    assert (GTI_PV, 12.0) in [(row.name, row.value) for row in rows]
    assert (GTI_LIVING_ROOM_WINDOWS, 5.0) in [(row.name, row.value) for row in rows]


def test_weather_import_service_filters_on_quarter_hour_window(tmp_path) -> None:
    responses = [
        {
            "minutely_15": {
                "time": [
                    "2026-04-20T10:00",
                    "2026-04-20T10:15",
                    "2026-04-30T10:00",
                    "2026-04-30T10:15",
                ],
                "temperature_2m": [10.0, 10.1, 15.0, 15.1],
                "relative_humidity_2m": [60, 60, 70, 70],
                "wind_speed_10m": [3.0, 3.0, 4.0, 4.0],
                "dew_point_2m": [4.0, 4.0, 7.0, 7.0],
                "direct_radiation": [0, 0, 300, 320],
                "diffuse_radiation": [0, 0, 100, 110],
            }
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses.pop(0))

    settings = AppSettings.from_options(
        {
            "database_path": str(tmp_path / "forecast.sqlite"),
        }
    )
    database = Database(settings.database_path)
    database.init_schema()
    repository = ForecastRepository(database)
    service = WeatherImportService(
        OpenMeteoGateway(client=httpx.Client(transport=httpx.MockTransport(handler))),
        Location(latitude=52.09, longitude=5.12),
        repository,
        pv_tilt=None,
        pv_azimuth=None,
        living_room_window_azimuth=None,
        history_days_back=settings.history_import_max_days_back,
    )

    written = service.import_weather_data(
        datetime(2026, 4, 30, 10, 7, 23, tzinfo=timezone.utc)
    )

    assert written == 18

    with database.session() as session:
        rows = session.execute(
            select(ForecastValue).where(ForecastValue.name == "temperature").order_by(
                ForecastValue.forecast_time_utc
            )
        ).scalars().all()

    assert [row.forecast_time_utc for row in rows] == [
        "2026-04-20T10:00:00+00:00",
        "2026-04-20T10:15:00+00:00",
        "2026-04-30T10:00:00+00:00",
    ]
