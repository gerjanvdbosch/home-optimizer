from __future__ import annotations

from datetime import datetime, time, timezone

import pytest
from pydantic import ValidationError

from home_optimizer.domain import ELECTRICITY_PRICE, NumericPoint, NumericSeries
from home_optimizer.domain.pricing import (
    DynamicPricing,
    FixedPricing,
    build_fixed_price_intervals,
    build_daily_price_series,
    electricity_price_series,
    price_intervals_from_series,
    resolve_daily_price_series,
)


def test_build_daily_price_series_uses_peak_price_during_peak_hours() -> None:
    pricing = FixedPricing(peak_price=0.32, off_peak_price=0.21, feed_in_tariff=0.09)
    start = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 5, 0, 0, tzinfo=timezone.utc)

    series = build_daily_price_series(pricing, start_time=start, end_time=end)

    assert series.name == ELECTRICITY_PRICE
    assert series.unit == "EUR/kWh"
    assert len(series.points) == 96
    assert series.points[0].value == 0.21
    assert series.points[28].value == 0.32
    assert series.points[92].value == 0.21


def test_build_daily_price_series_uses_off_peak_on_weekend() -> None:
    pricing = FixedPricing(peak_price=0.32, off_peak_price=0.21, feed_in_tariff=0.09)
    start = datetime(2026, 5, 3, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)

    series = build_daily_price_series(pricing, start_time=start, end_time=end)

    assert all(point.value == 0.21 for point in series.points)


def test_build_daily_price_series_supports_overnight_peak_window() -> None:
    pricing = FixedPricing(
        peak_price=0.32,
        off_peak_price=0.21,
        feed_in_tariff=0.09,
        peak_start=time(23, 0),
        peak_end=time(7, 0),
        peak_tuesday=False,
    )
    start = datetime(2026, 5, 5, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 5, 8, 0, tzinfo=timezone.utc)

    series = build_daily_price_series(pricing, start_time=start, end_time=end)

    assert all(point.value == 0.32 for point in series.points[:28])
    assert all(point.value == 0.21 for point in series.points[28:])


def test_build_daily_price_series_returns_empty_when_range_invalid() -> None:
    pricing = FixedPricing(peak_price=0.32, off_peak_price=0.21, feed_in_tariff=0.09)
    start = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)

    series = build_daily_price_series(pricing, start_time=start, end_time=start)

    assert series == electricity_price_series(currency="EUR")


def test_price_intervals_from_series_merges_adjacent_equal_values() -> None:
    series = NumericSeries(
        name=ELECTRICITY_PRICE,
        unit="EUR/kWh",
        points=[
            NumericPoint(timestamp="2026-05-04T00:00:00+00:00", value=0.21),
            NumericPoint(timestamp="2026-05-04T00:15:00+00:00", value=0.21),
            NumericPoint(timestamp="2026-05-04T00:30:00+00:00", value=0.32),
        ],
    )

    intervals = price_intervals_from_series(series, source="nordpool")

    assert len(intervals) == 2
    assert intervals[0].start_time_utc == datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    assert intervals[0].end_time_utc == datetime(2026, 5, 4, 0, 30, tzinfo=timezone.utc)
    assert intervals[0].value == 0.21
    assert intervals[1].start_time_utc == datetime(2026, 5, 4, 0, 30, tzinfo=timezone.utc)
    assert intervals[1].end_time_utc == datetime(2026, 5, 4, 0, 45, tzinfo=timezone.utc)
    assert intervals[1].value == 0.32


def test_build_fixed_price_intervals_compresses_repeating_quarters() -> None:
    pricing = FixedPricing(peak_price=0.32, off_peak_price=0.21, feed_in_tariff=0.09)

    intervals = build_fixed_price_intervals(
        pricing,
        start_time=datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 5, 5, 0, 0, tzinfo=timezone.utc),
        source="fixed_pricing",
    )

    assert [interval.value for interval in intervals] == [0.21, 0.32, 0.21]
    assert intervals[0].start_time_utc == datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    assert intervals[0].end_time_utc == datetime(2026, 5, 4, 7, 0, tzinfo=timezone.utc)
    assert intervals[1].start_time_utc == datetime(2026, 5, 4, 7, 0, tzinfo=timezone.utc)
    assert intervals[1].end_time_utc == datetime(2026, 5, 4, 23, 0, tzinfo=timezone.utc)
    assert intervals[2].start_time_utc == datetime(2026, 5, 4, 23, 0, tzinfo=timezone.utc)
    assert intervals[2].end_time_utc == datetime(2026, 5, 5, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("interval_minutes", [0, -15])
def test_build_daily_price_series_rejects_invalid_interval(interval_minutes: int) -> None:
    pricing = FixedPricing(peak_price=0.32, off_peak_price=0.21, feed_in_tariff=0.09)
    start = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 5, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="interval_minutes"):
        build_daily_price_series(
            pricing,
            start_time=start,
            end_time=end,
            interval_minutes=interval_minutes,
        )


def test_fixed_pricing_rejects_empty_peak_window() -> None:
    with pytest.raises(ValidationError, match="non-empty tariff window"):
        FixedPricing(
            peak_price=0.32,
            off_peak_price=0.21,
            feed_in_tariff=0.09,
            peak_start=time(7, 0),
            peak_end=time(7, 0),
        )


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

    assert series == electricity_price_series(currency="EUR")


def test_resolve_daily_price_series_generates_fixed_series_for_fixed_pricing() -> None:
    pricing = FixedPricing(peak_price=0.32, off_peak_price=0.21, feed_in_tariff=0.09)
    start = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 5, 0, 0, tzinfo=timezone.utc)

    series = resolve_daily_price_series(pricing, start_time=start, end_time=end)

    assert len(series.points) == 96
    assert series.points[28].value == 0.32

