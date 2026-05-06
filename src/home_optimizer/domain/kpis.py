from __future__ import annotations

from datetime import datetime

from .models import DomainModel
from .series import NumericPoint, NumericSeries
from .series_transforms import latest_value_at
from .time import parse_datetime


class DailyKpis(DomainModel):
    hp_electric_kwh: float | None = None
    total_import_kwh: float | None = None
    total_export_kwh: float | None = None
    pv_generation_kwh: float | None = None
    self_consumption_ratio: float | None = None
    electricity_cost_eur: float | None = None
    room_temperature_mae_c: float | None = None
    room_comfort_violation_degree_hours: float | None = None
    dhw_comfort_violation_minutes: float | None = None
    thermostat_setpoint_changes: int = 0
    compressor_starts: int = 0


def _sorted_points(series: NumericSeries | None) -> list[NumericPoint]:
    if series is None:
        return []
    return sorted(series.points, key=lambda point: point.timestamp)


def _duration_hours(start_time: datetime, end_time: datetime) -> float:
    return (end_time - start_time).total_seconds() / 3600.0


def integrate_power_series(
    series: NumericSeries | None,
    *,
    start_time: datetime,
    end_time: datetime,
    positive_only: bool = False,
    negative_only: bool = False,
) -> float | None:
    points = _sorted_points(series)
    if not points:
        return None

    total_kwh = 0.0
    for index, point in enumerate(points):
        point_time = parse_datetime(point.timestamp)
        if point_time >= end_time:
            break

        next_time = (
            parse_datetime(points[index + 1].timestamp)
            if index + 1 < len(points)
            else end_time
        )
        segment_start = max(point_time, start_time)
        segment_end = min(next_time, end_time)
        if segment_end <= segment_start:
            continue

        value = point.value
        if positive_only:
            value = max(value, 0.0)
        elif negative_only:
            value = max(-value, 0.0)

        total_kwh += value * _duration_hours(segment_start, segment_end)

    return total_kwh


def delta_kwh(series: NumericSeries | None) -> float | None:
    points = _sorted_points(series)
    if len(points) < 2:
        return None
    return max(points[-1].value - points[0].value, 0.0)


def weighted_absolute_error(
    measured: NumericSeries | None,
    target: NumericSeries,
    *,
    start_time: datetime,
    end_time: datetime,
) -> float | None:
    points = _sorted_points(measured)
    if not points:
        return None

    total_error = 0.0
    total_hours = 0.0
    for index, point in enumerate(points):
        point_time = parse_datetime(point.timestamp)
        if point_time >= end_time:
            break
        next_time = (
            parse_datetime(points[index + 1].timestamp)
            if index + 1 < len(points)
            else end_time
        )
        segment_start = max(point_time, start_time)
        segment_end = min(next_time, end_time)
        if segment_end <= segment_start:
            continue

        target_value = latest_value_at(target.points, point.timestamp)
        if target_value is None:
            continue

        duration_hours = _duration_hours(segment_start, segment_end)
        total_error += abs(point.value - target_value) * duration_hours
        total_hours += duration_hours

    if total_hours <= 0.0:
        return None
    return total_error / total_hours


def weighted_band_violation_degree_hours(
    measured: NumericSeries | None,
    minimum: NumericSeries,
    maximum: NumericSeries,
    *,
    start_time: datetime,
    end_time: datetime,
) -> float | None:
    points = _sorted_points(measured)
    if not points:
        return None

    violation_degree_hours = 0.0
    total_hours = 0.0
    for index, point in enumerate(points):
        point_time = parse_datetime(point.timestamp)
        if point_time >= end_time:
            break
        next_time = (
            parse_datetime(points[index + 1].timestamp)
            if index + 1 < len(points)
            else end_time
        )
        segment_start = max(point_time, start_time)
        segment_end = min(next_time, end_time)
        if segment_end <= segment_start:
            continue

        minimum_value = latest_value_at(minimum.points, point.timestamp)
        maximum_value = latest_value_at(maximum.points, point.timestamp)
        if minimum_value is None or maximum_value is None:
            continue

        violation = 0.0
        if point.value < minimum_value:
            violation = minimum_value - point.value
        elif point.value > maximum_value:
            violation = point.value - maximum_value

        duration_hours = _duration_hours(segment_start, segment_end)
        violation_degree_hours += violation * duration_hours
        total_hours += duration_hours

    if total_hours <= 0.0:
        return None
    return violation_degree_hours


def weighted_below_minimum_minutes(
    measured: NumericSeries | None,
    minimum: NumericSeries,
    *,
    start_time: datetime,
    end_time: datetime,
) -> float | None:
    points = _sorted_points(measured)
    if not points:
        return None

    violation_minutes = 0.0
    total_minutes = 0.0
    for index, point in enumerate(points):
        point_time = parse_datetime(point.timestamp)
        if point_time >= end_time:
            break
        next_time = (
            parse_datetime(points[index + 1].timestamp)
            if index + 1 < len(points)
            else end_time
        )
        segment_start = max(point_time, start_time)
        segment_end = min(next_time, end_time)
        if segment_end <= segment_start:
            continue

        minimum_value = latest_value_at(minimum.points, point.timestamp)
        if minimum_value is None:
            continue

        duration_minutes = (segment_end - segment_start).total_seconds() / 60.0
        if point.value < minimum_value:
            violation_minutes += duration_minutes
        total_minutes += duration_minutes

    if total_minutes <= 0.0:
        return None
    return violation_minutes


def count_setpoint_changes(series: NumericSeries | None) -> int:
    points = _sorted_points(series)
    changes = 0
    previous_value: float | None = None
    for point in points:
        if previous_value is not None and point.value != previous_value:
            changes += 1
        previous_value = point.value
    return changes


def count_compressor_starts(series: NumericSeries | None) -> int:
    points = _sorted_points(series)
    starts = 0
    previous_running = False
    for point in points:
        running = point.value > 0.0
        if running and not previous_running:
            starts += 1
        previous_running = running
    return starts


def clamp_ratio(value: float) -> float:
    return max(0.0, min(value, 1.0))


def cost_from_net_power(
    net_power: NumericSeries | None,
    price: NumericSeries,
    *,
    start_time: datetime,
    end_time: datetime,
    feed_in_tariff: float,
) -> float | None:
    points = _sorted_points(net_power)
    if not points:
        return None

    total_cost = 0.0
    for index, point in enumerate(points):
        point_time = parse_datetime(point.timestamp)
        if point_time >= end_time:
            break
        next_time = (
            parse_datetime(points[index + 1].timestamp)
            if index + 1 < len(points)
            else end_time
        )
        segment_start = max(point_time, start_time)
        segment_end = min(next_time, end_time)
        if segment_end <= segment_start:
            continue

        duration_hours = _duration_hours(segment_start, segment_end)
        interval_price = latest_value_at(price.points, point.timestamp)
        if interval_price is None:
            interval_price = 0.0

        if point.value >= 0.0:
            total_cost += point.value * duration_hours * interval_price
        else:
            total_cost -= abs(point.value) * duration_hours * feed_in_tariff

    return total_cost


def compute_daily_kpis(
    *,
    room_temperature: NumericSeries | None,
    room_target: NumericSeries,
    room_target_min: NumericSeries,
    room_target_max: NumericSeries,
    thermostat_setpoint: NumericSeries | None,
    compressor_frequency: NumericSeries | None,
    hp_electric_power: NumericSeries | None,
    hp_electric_total_kwh: NumericSeries | None,
    net_power: NumericSeries | None,
    import_total_kwh: NumericSeries | None,
    export_total_kwh: NumericSeries | None,
    pv_output_power: NumericSeries | None,
    pv_total_kwh: NumericSeries | None,
    dhw_top_temperature: NumericSeries | None,
    dhw_target_min: NumericSeries,
    electricity_price: NumericSeries,
    start_time: datetime,
    end_time: datetime,
    feed_in_tariff: float,
) -> DailyKpis:
    hp_electric_kwh = delta_kwh(hp_electric_total_kwh)
    if hp_electric_kwh is None:
        hp_electric_kwh = integrate_power_series(
            hp_electric_power,
            start_time=start_time,
            end_time=end_time,
            positive_only=True,
        )

    total_import_kwh = delta_kwh(import_total_kwh)
    total_export_kwh = delta_kwh(export_total_kwh)
    if total_import_kwh is None:
        total_import_kwh = integrate_power_series(
            net_power,
            start_time=start_time,
            end_time=end_time,
            positive_only=True,
        )
    if total_export_kwh is None:
        total_export_kwh = integrate_power_series(
            net_power,
            start_time=start_time,
            end_time=end_time,
            negative_only=True,
        )

    pv_generation_kwh = delta_kwh(pv_total_kwh)
    if pv_generation_kwh is None:
        pv_generation_kwh = integrate_power_series(
            pv_output_power,
            start_time=start_time,
            end_time=end_time,
            positive_only=True,
        )

    self_consumption_ratio: float | None = None
    if (
        pv_generation_kwh is not None
        and pv_generation_kwh > 0.0
        and total_export_kwh is not None
    ):
        self_consumption_ratio = clamp_ratio(
            (pv_generation_kwh - total_export_kwh) / pv_generation_kwh
        )

    return DailyKpis(
        hp_electric_kwh=hp_electric_kwh,
        total_import_kwh=total_import_kwh,
        total_export_kwh=total_export_kwh,
        pv_generation_kwh=pv_generation_kwh,
        self_consumption_ratio=self_consumption_ratio,
        electricity_cost_eur=cost_from_net_power(
            net_power,
            electricity_price,
            start_time=start_time,
            end_time=end_time,
            feed_in_tariff=feed_in_tariff,
        ),
        room_temperature_mae_c=weighted_absolute_error(
            room_temperature,
            room_target,
            start_time=start_time,
            end_time=end_time,
        ),
        room_comfort_violation_degree_hours=weighted_band_violation_degree_hours(
            room_temperature,
            room_target_min,
            room_target_max,
            start_time=start_time,
            end_time=end_time,
        ),
        dhw_comfort_violation_minutes=weighted_below_minimum_minutes(
            dhw_top_temperature,
            dhw_target_min,
            start_time=start_time,
            end_time=end_time,
        ),
        thermostat_setpoint_changes=count_setpoint_changes(thermostat_setpoint),
        compressor_starts=count_compressor_starts(compressor_frequency),
    )
