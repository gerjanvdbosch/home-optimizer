from __future__ import annotations

from datetime import datetime, time, timezone

from home_optimizer.domain import (
    BOOSTER_HEATER_ACTIVE,
    DHW_TARGET_MAX_TEMPERATURE,
    DHW_TARGET_MIN_TEMPERATURE,
    DHW_TARGET_TEMPERATURE,
    DEFROST_ACTIVE,
    ELECTRICITY_PRICE,
    HP_FLOW,
    HP_MODE,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    NumericPoint,
    NumericSeries,
    ROOM_TARGET_MAX_TEMPERATURE,
    ROOM_TARGET_MIN_TEMPERATURE,
    ROOM_TARGET_TEMPERATURE,
    TemperatureTargetWindow,
    TextPoint,
    TextSeries,
    build_daily_target_band_series,
    build_space_heating_thermal_output_series,
)
from home_optimizer.domain.pricing import DynamicPricing, FixedPricing
from home_optimizer.domain.series_transforms import build_daily_price_series, resolve_daily_price_series


def test_build_space_heating_thermal_output_series_filters_dhw_defrost_and_booster() -> None:
    thermal_output = build_space_heating_thermal_output_series(
        NumericSeries(
            name=HP_FLOW,
            unit="Lmin",
            points=[
                NumericPoint(timestamp="2026-04-25T00:00:00+00:00", value=10.0),
                NumericPoint(timestamp="2026-04-25T00:15:00+00:00", value=10.0),
                NumericPoint(timestamp="2026-04-25T00:30:00+00:00", value=10.0),
                NumericPoint(timestamp="2026-04-25T00:45:00+00:00", value=10.0),
            ],
        ),
        NumericSeries(
            name=HP_SUPPLY_TEMPERATURE,
            unit="degC",
            points=[
                NumericPoint(timestamp="2026-04-25T00:00:00+00:00", value=35.0),
                NumericPoint(timestamp="2026-04-25T00:15:00+00:00", value=35.0),
                NumericPoint(timestamp="2026-04-25T00:30:00+00:00", value=35.0),
                NumericPoint(timestamp="2026-04-25T00:45:00+00:00", value=35.0),
            ],
        ),
        NumericSeries(
            name=HP_RETURN_TEMPERATURE,
            unit="degC",
            points=[
                NumericPoint(timestamp="2026-04-25T00:00:00+00:00", value=30.0),
                NumericPoint(timestamp="2026-04-25T00:15:00+00:00", value=30.0),
                NumericPoint(timestamp="2026-04-25T00:30:00+00:00", value=30.0),
                NumericPoint(timestamp="2026-04-25T00:45:00+00:00", value=30.0),
            ],
        ),
        defrost_active=NumericSeries(
            name=DEFROST_ACTIVE,
            unit="bool",
            points=[NumericPoint(timestamp="2026-04-25T00:15:00+00:00", value=1.0)],
        ),
        booster_heater_active=NumericSeries(
            name=BOOSTER_HEATER_ACTIVE,
            unit="bool",
            points=[NumericPoint(timestamp="2026-04-25T00:45:00+00:00", value=1.0)],
        ),
        hp_mode=TextSeries(
            name=HP_MODE,
            points=[
                TextPoint(timestamp="2026-04-25T00:00:00+00:00", value="heat"),
                TextPoint(timestamp="2026-04-25T00:30:00+00:00", value="dhw"),
            ],
        ),
    )

    assert [point.timestamp for point in thermal_output.points] == [
        "2026-04-25T00:00:00+00:00",
    ]


def test_build_daily_target_band_series_creates_target_min_max_points() -> None:
    target, minimum, maximum = build_daily_target_band_series(
        [
            TemperatureTargetWindow(time=time(0, 0), target=19.0, low_margin=0.5, high_margin=1.5),
            TemperatureTargetWindow(time=time(18, 0), target=20.0, low_margin=0.5, high_margin=1.5),
        ],
        start_time=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 4, 26, 0, 0, tzinfo=timezone.utc),
        target_name=ROOM_TARGET_TEMPERATURE,
        minimum_name=ROOM_TARGET_MIN_TEMPERATURE,
        maximum_name=ROOM_TARGET_MAX_TEMPERATURE,
    )

    assert target.points == [
        NumericPoint(timestamp="2026-04-25T00:00:00+00:00", value=19.0),
        NumericPoint(timestamp="2026-04-25T18:00:00+00:00", value=20.0),
        NumericPoint(timestamp="2026-04-25T23:59:59+00:00", value=20.0),
    ]
    assert minimum.points == [
        NumericPoint(timestamp="2026-04-25T00:00:00+00:00", value=18.5),
        NumericPoint(timestamp="2026-04-25T18:00:00+00:00", value=19.5),
        NumericPoint(timestamp="2026-04-25T23:59:59+00:00", value=19.5),
    ]
    assert maximum.points == [
        NumericPoint(timestamp="2026-04-25T00:00:00+00:00", value=20.5),
        NumericPoint(timestamp="2026-04-25T18:00:00+00:00", value=21.5),
        NumericPoint(timestamp="2026-04-25T23:59:59+00:00", value=21.5),
    ]


def test_build_daily_target_band_series_uses_previous_day_window_before_first_change() -> None:
    target, minimum, maximum = build_daily_target_band_series(
        [
            TemperatureTargetWindow(time=time(8, 0), target=20.0, low_margin=2.0, high_margin=5.0),
            TemperatureTargetWindow(time=time(21, 0), target=50.0, low_margin=2.0, high_margin=5.0),
        ],
        start_time=datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc),
        target_name=DHW_TARGET_TEMPERATURE,
        minimum_name=DHW_TARGET_MIN_TEMPERATURE,
        maximum_name=DHW_TARGET_MAX_TEMPERATURE,
    )

    assert target.points[:2] == [
        NumericPoint(timestamp="2026-04-25T00:00:00+00:00", value=50.0),
        NumericPoint(timestamp="2026-04-25T08:00:00+00:00", value=20.0),
    ]
    assert minimum.points[0].value == 48.0
    assert maximum.points[0].value == 55.0


def test_build_daily_price_series_uses_peak_price_during_peak_hours() -> None:
    # 2026-05-04 is a Monday — peak day by default
    pricing = FixedPricing(peak_price=0.32, off_peak_price=0.21, feed_in_tariff=0.09)
    start = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 5, 0, 0, tzinfo=timezone.utc)

    series = build_daily_price_series(pricing, start_time=start, end_time=end)

    assert series.name == ELECTRICITY_PRICE
    assert series.unit == "EUR/kWh"
    assert len(series.points) == 96  # 24h × 4 quarters
    # 00:00 — off-peak
    assert series.points[0].value == 0.21
    # 07:00 = index 28 — peak
    assert series.points[28].value == 0.32
    # 23:00 = index 92 — off-peak (peak_end is 23:00, exclusive)
    assert series.points[92].value == 0.21


def test_build_daily_price_series_uses_off_peak_on_weekend() -> None:
    # 2026-05-03 is a Sunday — not a peak day by default
    pricing = FixedPricing(peak_price=0.32, off_peak_price=0.21, feed_in_tariff=0.09)
    start = datetime(2026, 5, 3, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)

    series = build_daily_price_series(pricing, start_time=start, end_time=end)

    assert all(p.value == 0.21 for p in series.points)


def test_build_daily_price_series_returns_empty_when_range_invalid() -> None:
    pricing = FixedPricing(peak_price=0.32, off_peak_price=0.21, feed_in_tariff=0.09)
    start = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)

    series = build_daily_price_series(pricing, start_time=start, end_time=start)

    assert series.points == []


def test_resolve_daily_price_series_returns_fetched_series_for_dynamic_pricing() -> None:
    pricing = DynamicPricing()
    fetched = NumericSeries(
        name=ELECTRICITY_PRICE,
        unit="EUR/kWh",
        points=[NumericPoint(timestamp="2026-05-04T00:00:00+00:00", value=0.18)],
    )
    start = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 5, 0, 0, tzinfo=timezone.utc)

    series = resolve_daily_price_series(pricing, start_time=start, end_time=end, fetched_series=fetched)

    assert series is fetched


def test_resolve_daily_price_series_returns_empty_for_dynamic_pricing_without_fetched_series() -> None:
    pricing = DynamicPricing()
    start = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 5, 0, 0, tzinfo=timezone.utc)

    series = resolve_daily_price_series(pricing, start_time=start, end_time=end)

    assert series.name == ELECTRICITY_PRICE
    assert series.points == []


def test_resolve_daily_price_series_generates_fixed_series_for_fixed_pricing() -> None:
    pricing = FixedPricing(peak_price=0.32, off_peak_price=0.21, feed_in_tariff=0.09)
    start = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 5, 0, 0, tzinfo=timezone.utc)

    series = resolve_daily_price_series(pricing, start_time=start, end_time=end)

    assert len(series.points) == 96
    assert series.points[28].value == 0.32  # 07:00 — peak


