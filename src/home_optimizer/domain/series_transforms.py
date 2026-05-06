from __future__ import annotations

from datetime import datetime, timedelta

from .names import GTI_LIVING_ROOM_WINDOWS_ADJUSTED, THERMAL_OUTPUT
from .series import NumericPoint, NumericSeries
from .target_schedule import TemperatureTargetWindow
from .time import normalize_utc_timestamp, parse_datetime


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
                value=point.value
                * shutter_open_fraction_at(
                    shutter_position.points,
                    point.timestamp,
                ),
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
        while point_index < len(source_points):
            point_time, point_value = source_points[point_index]
            if point_time > cursor:
                break
            latest_value = point_value
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


def build_daily_target_band_series(
    schedule: list[TemperatureTargetWindow],
    *,
    start_time: datetime,
    end_time: datetime,
    target_name: str,
    minimum_name: str,
    maximum_name: str,
    unit: str = "°C",
    interval_minutes: int | None = None,
) -> tuple[NumericSeries, NumericSeries, NumericSeries]:
    empty_target = NumericSeries(name=target_name, unit=unit, points=[])
    empty_minimum = NumericSeries(name=minimum_name, unit=unit, points=[])
    empty_maximum = NumericSeries(name=maximum_name, unit=unit, points=[])
    if end_time <= start_time or not schedule:
        return empty_target, empty_minimum, empty_maximum

    ordered_schedule = sorted(schedule, key=lambda window: window.time)
    active_window = ordered_schedule[-1]
    for window in ordered_schedule:
        change_time = datetime.combine(start_time.date(), window.time, tzinfo=start_time.tzinfo)
        if change_time <= start_time:
            active_window = window
            continue
        break

    target_points: list[NumericPoint] = []
    minimum_points: list[NumericPoint] = []
    maximum_points: list[NumericPoint] = []

    def append_points(timestamp: datetime, window: TemperatureTargetWindow) -> None:
        normalized_timestamp = normalize_utc_timestamp(timestamp)
        target_points.append(NumericPoint(timestamp=normalized_timestamp, value=window.target))
        minimum_points.append(NumericPoint(timestamp=normalized_timestamp, value=window.minimum))
        maximum_points.append(NumericPoint(timestamp=normalized_timestamp, value=window.maximum))

    append_points(start_time, active_window)

    for window in ordered_schedule:
        change_time = datetime.combine(start_time.date(), window.time, tzinfo=start_time.tzinfo)
        if not start_time < change_time < end_time:
            continue
        active_window = window
        append_points(change_time, active_window)

    final_timestamp = end_time - timedelta(seconds=1)
    if final_timestamp > start_time:
        append_points(final_timestamp, active_window)

    target = NumericSeries(name=target_name, unit=unit, points=target_points)
    minimum = NumericSeries(name=minimum_name, unit=unit, points=minimum_points)
    maximum = NumericSeries(name=maximum_name, unit=unit, points=maximum_points)

    if interval_minutes is not None:
        target = upsample_series_forward_fill(
            target,
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )
        minimum = upsample_series_forward_fill(
            minimum,
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )
        maximum = upsample_series_forward_fill(
            maximum,
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )

    return target, minimum, maximum


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
