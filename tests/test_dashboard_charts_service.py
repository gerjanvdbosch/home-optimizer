from __future__ import annotations

from datetime import date, datetime, timezone

from home_optimizer.domain import (
    NumericPoint,
    NumericSeries,
    TextSeries,
    upsample_series_forward_fill,
)
from home_optimizer.web.services.dashboard_charts import (
    DashboardChartsService,
    adjusted_gti_with_shutter,
)


class FakeDashboardDataReader:
    def __init__(self, *, series: list[NumericSeries] | None = None) -> None:
        self._series = series or [
            NumericSeries(
                name="hp_supply_temperature",
                unit="degC",
                points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=30.0)],
            ),
            NumericSeries(
                name="hp_return_temperature",
                unit="degC",
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


def test_adjusted_gti_with_shutter_uses_latest_known_open_percentage() -> None:
    window_gti = NumericSeries(
        name="gti_living_room_windows",
        unit="Wm2",
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
    assert adjusted.unit == "Wm2"
    assert adjusted.points == [
        NumericPoint(timestamp="2026-04-25T09:00:00+00:00", value=25.0),
        NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=50.0),
        NumericPoint(timestamp="2026-04-25T15:00:00+00:00", value=240.0),
    ]


def test_adjusted_gti_with_shutter_defaults_to_fully_open_when_no_position_is_known() -> None:
    window_gti = NumericSeries(
        name="gti_living_room_windows",
        unit="Wm2",
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
        unit="Wm2",
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
    assert upsampled.unit == "Wm2"
    assert upsampled.points == [
        NumericPoint(timestamp="2026-04-25T10:00:00+00:00", value=100.0),
        NumericPoint(timestamp="2026-04-25T10:15:00+00:00", value=100.0),
        NumericPoint(timestamp="2026-04-25T10:30:00+00:00", value=100.0),
        NumericPoint(timestamp="2026-04-25T10:45:00+00:00", value=100.0),
        NumericPoint(timestamp="2026-04-25T11:00:00+00:00", value=200.0),
        NumericPoint(timestamp="2026-04-25T11:15:00+00:00", value=200.0),
    ]


def test_dashboard_charts_service_clamps_negative_delta_t_to_zero() -> None:
    service = DashboardChartsService(FakeDashboardDataReader())

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
                    unit="degC",
                    points=[NumericPoint(timestamp="2026-04-25T12:00:00+00:00", value=40.0)],
                ),
                NumericSeries(
                    name="hp_return_temperature",
                    unit="degC",
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
        )
    )

    response = service.get_day_charts(date(2026, 4, 25))

    assert len(response.cop.points) == 1
    assert response.cop.points[0].timestamp == "2026-04-25T12:00:00+00:00"
    assert response.cop.points[0].value == 10.0


