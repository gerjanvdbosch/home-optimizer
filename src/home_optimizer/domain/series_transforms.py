from __future__ import annotations

from datetime import datetime

from home_optimizer.domain import (
    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
    THERMAL_OUTPUT,
    NumericPoint,
    NumericSeries,
)
from home_optimizer.domain.time import ensure_utc


def latest_value_at(points: list[NumericPoint], timestamp: datetime | str) -> float | None:
    target_time = ensure_utc(timestamp)
    latest: float | None = None
    for point in points:
        if ensure_utc(point.timestamp) > target_time:
            break
        latest = point.value
    return latest


def shutter_open_fraction_at(points: list[NumericPoint], timestamp: datetime | str) -> float:
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
