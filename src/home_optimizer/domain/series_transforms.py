from __future__ import annotations

from datetime import datetime, timedelta

from home_optimizer.domain import (
    FLOOR_HEAT_STATE,
    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
    THERMAL_OUTPUT,
    NumericPoint,
    NumericSeries,
)
from home_optimizer.domain.time import normalize_utc_timestamp, parse_datetime

DEFAULT_FLOOR_HEAT_STATE_ALPHA = 0.97


def latest_value_at(points: list[NumericPoint], timestamp: str) -> float | None:
    latest: float | None = None
    for point in points:
        if point.timestamp > timestamp:
            break
        latest = point.value
    return latest


def shutter_open_fraction_at(points: list[NumericPoint], timestamp: str) -> float:
    position = latest_value_at(points, timestamp)
    if position is None:
        return 1.0
    return max(0.0, min(position, 100.0)) / 100.0


def adjusted_gti_with_shutter(
    window_gti: NumericSeries,
    shutter_position: NumericSeries,
) -> NumericSeries:
    return NumericSeries(
        name=GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
        unit=window_gti.unit,
        points=[
            NumericPoint(
                timestamp=point.timestamp,
                value=point.value * shutter_open_fraction_at(shutter_position.points, point.timestamp),
            )
            for point in window_gti.points
        ],
    )


def upsample_series_forward_fill(
    series: NumericSeries,
    *,
    start_time: datetime,
    end_time: datetime,
    interval_minutes: int = 15,
) -> NumericSeries:
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than zero")
    if end_time <= start_time or not series.points:
        return NumericSeries(name=series.name, unit=series.unit, points=[])

    interval = timedelta(minutes=interval_minutes)
    source_points = [(parse_datetime(point.timestamp), point.value) for point in series.points]
    point_index = 0
    latest_value: float | None = None
    upsampled_points: list[NumericPoint] = []
    cursor = start_time

    while cursor < end_time:
        while point_index < len(source_points) and source_points[point_index][0] <= cursor:
            latest_value = source_points[point_index][1]
            point_index += 1

        if latest_value is not None:
            upsampled_points.append(
                NumericPoint(
                    timestamp=normalize_utc_timestamp(cursor),
                    value=latest_value,
                )
            )
        cursor += interval

    return NumericSeries(name=series.name, unit=series.unit, points=upsampled_points)


def build_thermal_output_series(
    flow: NumericSeries | None,
    supply: NumericSeries | None,
    return_s: NumericSeries | None,
    *,
    name: str = THERMAL_OUTPUT,
) -> NumericSeries:
    factor = 4186.0 / 60000.0
    if flow is None:
        return NumericSeries(name=name, unit="kW", points=[])

    thermal_points: list[NumericPoint] = []
    for flow_point in flow.points:
        supply_value = latest_value_at(supply.points, flow_point.timestamp) if supply else None
        return_value = latest_value_at(return_s.points, flow_point.timestamp) if return_s else None
        if supply_value is None or return_value is None:
            continue

        thermal_output = max(0.0, flow_point.value * (supply_value - return_value) * factor)
        thermal_points.append(NumericPoint(timestamp=flow_point.timestamp, value=thermal_output))

    return NumericSeries(name=name, unit="kW", points=thermal_points)


def build_floor_heat_state_series(
    thermal_output: NumericSeries,
    *,
    alpha: float = DEFAULT_FLOOR_HEAT_STATE_ALPHA,
    name: str = FLOOR_HEAT_STATE,
) -> NumericSeries:
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0.0, 1.0)")

    floor_points: list[NumericPoint] = []
    state = 0.0
    for point in thermal_output.points:
        state = alpha * state + (1.0 - alpha) * point.value
        floor_points.append(NumericPoint(timestamp=point.timestamp, value=state))

    return NumericSeries(name=name, unit=thermal_output.unit, points=floor_points)
