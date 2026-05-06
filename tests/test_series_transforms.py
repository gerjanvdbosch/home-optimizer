from __future__ import annotations

from datetime import datetime, time, timezone

from home_optimizer.domain import (
    BOOSTER_HEATER_ACTIVE,
    DHW_TARGET_MAX_TEMPERATURE,
    DHW_TARGET_MIN_TEMPERATURE,
    DHW_TARGET_TEMPERATURE,
    DEFROST_ACTIVE,
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
    build_daily_target_band_series,
)

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



