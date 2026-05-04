from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from home_optimizer.app import AppSettings
from home_optimizer.domain import (
    NumericPoint,
    NumericSeries,
    TextSeries,
    normalize_utc_timestamp,
    upsample_series_forward_fill,
)
from home_optimizer.web.services.dashboard_charts import (
    DashboardChartsService,
    adjusted_gti_with_shutter,
    current_timezone,
)


class FakeDashboardDataReader:
    def __init__(self, *, series: list[NumericSeries] | None = None) -> None:
        self._series = series or [
            NumericSeries(
                name="hp_supply_temperature",
                unit="°C",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=30.0)],
            ),
            NumericSeries(
                name="hp_return_temperature",
                unit="°C",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=32.5)],
            ),
        ]

    def read_series(self, names, start_time, end_time) -> list[NumericSeries]:
        if names == ["shutter_living_room"]:
            return []

        return self._series

    def read_text_series(self, names, start_time, end_time) -> list[TextSeries]:
        return []

    def read_forecast_series(self, names, start_time, end_time) -> list[NumericSeries]:
        return []

    def read_historical_weather_series(self, names, start_time, end_time) -> list[NumericSeries]:
        return []


def build_settings() -> AppSettings:
    return AppSettings.from_options({
        "database_path": "/tmp/home-optimizer-test.db",
        "room_target": [
            {"time": "00:00", "target": 19.0, "low_margin": 0.5, "high_margin": 1.5},
            {"time": "08:00", "target": 19.0, "low_margin": 0.5, "high_margin": 1.5},
            {"time": "14:00", "target": 19.0, "low_margin": 0.5, "high_margin": 1.5},
            {"time": "18:00", "target": 20.0, "low_margin": 0.5, "high_margin": 1.5},
            {"time": "22:00", "target": 19.0, "low_margin": 0.5, "high_margin": 1.5},
        ],
        "dhw_target": [
            {"time": "00:00", "target": 20.0, "low_margin": 5.0, "high_margin": 30.0},
            {"time": "10:00", "target": 20.0, "low_margin": 5.0, "high_margin": 35.0},
            {"time": "19:59", "target": 20.0, "low_margin": 5.0, "high_margin": 35.0},
            {"time": "20:00", "target": 50.0, "low_margin": 2.0, "high_margin": 5.0},
            {"time": "21:00", "target": 50.0, "low_margin": 2.0, "high_margin": 5.0},
            {"time": "21:01", "target": 20.0, "low_margin": 5.0, "high_margin": 30.0},
        ],
    })


def test_adjusted_gti_with_shutter_uses_latest_known_open_percentage() -> None:
    window_gti = NumericSeries(
        name="gti_living_room_windows",
        unit="W/m2",
        points=[
            NumericPoint(timestamp="2026-04-25T09:00:00+00:00", value=100.0),
            NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=200.0),
            NumericPoint(timestamp="2026-04-25T15:00:00+00:00", value=300.0),
        ],
    )
    shutter_position = NumericSeries(
        name="shutter_living_room",
        unit="percent",
        points=[
            NumericPoint(timestamp="2026-04-25T08:30:00+00:00", value=25.0),
            NumericPoint(timestamp="2026-04-25T13:30:00+00:00", value=80.0),
        ],
    )

    adjusted = adjusted_gti_with_shutter(window_gti, shutter_position)

    assert adjusted.name == "gti_living_room_windows_adjusted"
    assert adjusted.unit == "W/m2"
    assert adjusted.points == [
        NumericPoint(timestamp="2026-04-25T09:00:00+00:00", value=25.0),
        NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=50.0),
        NumericPoint(timestamp="2026-04-25T15:00:00+00:00", value=240.0),
    ]


def test_adjusted_gti_with_shutter_defaults_to_fully_open_when_no_position_is_known() -> None:
    window_gti = NumericSeries(
        name="gti_living_room_windows",
        unit="W/m2",
        points=[NumericPoint(timestamp="2026-04-25T09:00:00+00:00", value=150.0)],
    )
    shutter_position = NumericSeries(
        name="shutter_living_room",
        unit="percent",
        points=[],
    )

    adjusted = adjusted_gti_with_shutter(window_gti, shutter_position)

    assert adjusted.points == [NumericPoint(timestamp="2026-04-25T09:00:00+00:00", value=150.0)]


def test_upsample_series_forward_fill_expands_hourly_points_to_quarters() -> None:
    hourly_series = NumericSeries(
        name="gti_living_room_windows",
        unit="W/m2",
        points=[
            NumericPoint(timestamp="2026-04-25T10:00:00+00:00", value=100.0),
            NumericPoint(timestamp="2026-04-25T11:00:00+00:00", value=200.0),
        ],
    )

    upsampled = upsample_series_forward_fill(
        hourly_series,
        start_time=datetime(2026, 4, 25, 9, 45, tzinfo=timezone.utc),
        end_time=datetime(2026, 4, 25, 11, 30, tzinfo=timezone.utc),
        interval_minutes=15,
    )

    assert upsampled.name == "gti_living_room_windows"
    assert upsampled.unit == "W/m2"
    assert upsampled.points == [
        NumericPoint(timestamp="2026-04-25T10:00:00+00:00", value=100.0),
        NumericPoint(timestamp="2026-04-25T10:15:00+00:00", value=100.0),
        NumericPoint(timestamp="2026-04-25T10:30:00+00:00", value=100.0),
        NumericPoint(timestamp="2026-04-25T10:45:00+00:00", value=100.0),
        NumericPoint(timestamp="2026-04-25T11:00:00+00:00", value=200.0),
        NumericPoint(timestamp="2026-04-25T11:15:00+00:00", value=200.0),
    ]


def test_dashboard_charts_service_clamps_negative_delta_t_to_zero() -> None:
    service = DashboardChartsService(FakeDashboardDataReader(), build_settings())

    response = service.get_day_charts(date(2026, 4, 25))

    assert len(response.hp_delta_t.points) == 1
    assert response.hp_delta_t.points[0].timestamp == "2026-04-25T12:00:00+00:00"
    assert response.hp_delta_t.points[0].value == 0.0


def test_dashboard_charts_service_clamps_cop_to_ten() -> None:
    service = DashboardChartsService(
        FakeDashboardDataReader(
            series=[
                NumericSeries(
                    name="hp_supply_temperature",
                    unit="°C",
                    points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=40.0)],
                ),
                NumericSeries(
                    name="hp_return_temperature",
                    unit="°C",
                    points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=35.0)],
                ),
                NumericSeries(
                    name="hp_flow",
                    unit="Lmin",
                    points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=10.0)],
                ),
                NumericSeries(
                    name="hp_electric_power",
                    unit="kW",
                    points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=0.1)],
                ),
            ]
        ),
        build_settings(),
    )

    response = service.get_day_charts(date(2026, 4, 25))

    assert len(response.cop.points) == 1
    assert response.cop.points[0].timestamp == "2026-04-25T12:00:00+00:00"
    assert response.cop.points[0].value == 10.0


def test_dashboard_charts_service_includes_configured_room_and_dhw_targets() -> None:
    service = DashboardChartsService(FakeDashboardDataReader(series=[]), build_settings())

    response = service.get_day_charts(date(2026, 4, 25))
    local_timezone = current_timezone()
    start_time = datetime.combine(date(2026, 4, 25), time.min, tzinfo=local_timezone)
    end_time = start_time + timedelta(days=1)

    assert response.room_target_temperature.name == "room_target_temperature"
    assert response.room_target_min_temperature.points[0].timestamp == normalize_utc_timestamp(start_time)
    assert response.room_target_min_temperature.points[0].value == 18.5
    assert response.room_target_max_temperature.points[-1].timestamp == normalize_utc_timestamp(
        end_time - timedelta(minutes=15)
    )
    assert response.room_target_max_temperature.points[-1].value == 20.5
    assert response.dhw_target_temperature.points[80].timestamp == normalize_utc_timestamp(
        datetime.combine(date(2026, 4, 25), time(20, 0), tzinfo=local_timezone)
    )
    assert response.dhw_target_temperature.points[80].value == 50.0


