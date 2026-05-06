from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import Field

from .models import DomainModel
from .series import NumericPoint, NumericSeries
from .series_transforms import latest_value_at
from .time import parse_datetime


class DailyKpis(DomainModel):
    is_valid_for_control_evaluation: bool = True
    validity_reasons: list[str] = Field(default_factory=list)
    data_coverage_pct: float | None = None
    largest_data_gap_minutes: float | None = None
    hp_electric_kwh: float | None = None
    total_import_kwh: float | None = None
    total_export_kwh: float | None = None
    pv_generation_kwh: float | None = None
    outdoor_temperature_mean_c: float | None = None
    self_consumption_ratio: float | None = None
    electricity_cost_eur: float | None = None
    room_temperature_mae_c: float | None = None
    room_comfort_violation_degree_hours: float | None = None
    dhw_comfort_violation_minutes: float | None = None
    thermostat_setpoint_changes: int = 0
    compressor_starts: int = 0


class BaselineKpiSummary(DomainModel):
    number_of_days: int
    number_of_valid_days: int
    mean_hp_electric_kwh_per_day: float | None = None
    mean_electricity_cost_eur_per_day: float | None = None
    mean_room_temperature_mae_c: float | None = None
    total_comfort_violation_degree_hours: float = 0.0
    total_dhw_violation_minutes: float = 0.0
    mean_compressor_starts_per_day: float | None = None
    mean_self_consumption_ratio: float | None = None


def _sorted_points(series: NumericSeries | None) -> list[NumericPoint]:
    if series is None:
        return []
    return sorted(series.points, key=lambda point: point.timestamp)


def _duration_hours(start_time: datetime, end_time: datetime) -> float:
    return (end_time - start_time).total_seconds() / 3600.0


def _has_gap_longer_than(
    series: NumericSeries | None,
    *,
    start_time: datetime,
    end_time: datetime,
    max_gap: timedelta,
) -> bool:
    points = _sorted_points(series)
    if not points:
        return True

    previous_time = start_time
    for point in points:
        point_time = parse_datetime(point.timestamp)
        if point_time < start_time:
            continue
        if point_time > end_time:
            break
        if point_time - previous_time > max_gap:
            return True
        previous_time = point_time

    return end_time - previous_time > max_gap


def _largest_gap_duration(
    series: NumericSeries | None,
    *,
    start_time: datetime,
    end_time: datetime,
) -> timedelta | None:
    points = _sorted_points(series)
    if not points:
        return None

    largest_gap = timedelta(0)
    previous_time = start_time
    for point in points:
        point_time = parse_datetime(point.timestamp)
        if point_time < start_time:
            continue
        if point_time > end_time:
            break
        largest_gap = max(largest_gap, point_time - previous_time)
        previous_time = point_time

    largest_gap = max(largest_gap, end_time - previous_time)
    return largest_gap


def coverage_pct(
    series: NumericSeries | None,
    *,
    start_time: datetime,
    end_time: datetime,
    max_gap: timedelta,
) -> float | None:
    points = _sorted_points(series)
    total_seconds = (end_time - start_time).total_seconds()
    if not points or total_seconds <= 0.0:
        return None

    covered_seconds = 0.0
    previous_time = start_time
    for point in points:
        point_time = parse_datetime(point.timestamp)
        if point_time < start_time:
            continue
        if point_time > end_time:
            break
        covered_seconds += min(
            max((point_time - previous_time).total_seconds(), 0.0),
            max_gap.total_seconds(),
        )
        previous_time = point_time

    covered_seconds += min(
        max((end_time - previous_time).total_seconds(), 0.0),
        max_gap.total_seconds(),
    )
    return max(0.0, min((covered_seconds / total_seconds) * 100.0, 100.0))


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


def weighted_mean(
    series: NumericSeries | None,
    *,
    start_time: datetime,
    end_time: datetime,
) -> float | None:
    points = _sorted_points(series)
    if not points:
        return None

    weighted_total = 0.0
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
        duration_hours = _duration_hours(segment_start, segment_end)
        weighted_total += point.value * duration_hours
        total_hours += duration_hours

    if total_hours <= 0.0:
        return None
    return weighted_total / total_hours


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
    outdoor_temperature: NumericSeries | None,
    dhw_top_temperature: NumericSeries | None,
    dhw_target_min: NumericSeries,
    electricity_price: NumericSeries,
    start_time: datetime,
    end_time: datetime,
    feed_in_tariff: float,
) -> DailyKpis:
    validity_reasons: list[str] = []
    max_gap = timedelta(minutes=30)
    coverage_values: list[float] = []
    largest_gap_minutes = 0.0

    def record_series_quality(series: NumericSeries | None) -> None:
        nonlocal largest_gap_minutes
        coverage = coverage_pct(
            series,
            start_time=start_time,
            end_time=end_time,
            max_gap=max_gap,
        )
        if coverage is not None:
            coverage_values.append(coverage)
        largest_gap = _largest_gap_duration(
            series,
            start_time=start_time,
            end_time=end_time,
        )
        if largest_gap is not None:
            largest_gap_minutes = max(largest_gap_minutes, largest_gap.total_seconds() / 60.0)

    if not room_temperature or not room_temperature.points:
        validity_reasons.append("missing_room_temperature")
    elif _has_gap_longer_than(
        room_temperature,
        start_time=start_time,
        end_time=end_time,
        max_gap=max_gap,
    ):
        validity_reasons.append("room_temperature_gap_too_large")
    record_series_quality(room_temperature)

    if not thermostat_setpoint or not thermostat_setpoint.points:
        validity_reasons.append("missing_thermostat_setpoint")
    elif _has_gap_longer_than(
        thermostat_setpoint,
        start_time=start_time,
        end_time=end_time,
        max_gap=max_gap,
    ):
        validity_reasons.append("thermostat_setpoint_gap_too_large")
    record_series_quality(thermostat_setpoint)

    if not compressor_frequency or not compressor_frequency.points:
        validity_reasons.append("missing_compressor_frequency")
    elif _has_gap_longer_than(
        compressor_frequency,
        start_time=start_time,
        end_time=end_time,
        max_gap=max_gap,
    ):
        validity_reasons.append("compressor_frequency_gap_too_large")
    record_series_quality(compressor_frequency)

    if (
        (not hp_electric_total_kwh or not hp_electric_total_kwh.points)
        and (not hp_electric_power or not hp_electric_power.points)
    ):
        validity_reasons.append("missing_heatpump_electricity")
        record_series_quality(hp_electric_power)
        record_series_quality(hp_electric_total_kwh)
    elif (
        hp_electric_total_kwh
        and hp_electric_total_kwh.points
        and _has_gap_longer_than(
            hp_electric_total_kwh,
            start_time=start_time,
            end_time=end_time,
            max_gap=max_gap,
        )
    ):
        validity_reasons.append("hp_electric_total_kwh_gap_too_large")
        record_series_quality(hp_electric_total_kwh)
    elif (
        hp_electric_power
        and hp_electric_power.points
        and _has_gap_longer_than(
            hp_electric_power,
            start_time=start_time,
            end_time=end_time,
            max_gap=max_gap,
        )
    ):
        validity_reasons.append("hp_electric_power_gap_too_large")
        record_series_quality(hp_electric_power)
    else:
        record_series_quality(hp_electric_total_kwh or hp_electric_power)

    if (
        (not import_total_kwh or not import_total_kwh.points)
        and (not export_total_kwh or not export_total_kwh.points)
        and (not net_power or not net_power.points)
    ):
        validity_reasons.append("missing_grid_energy")
        record_series_quality(net_power)
        record_series_quality(import_total_kwh)
        record_series_quality(export_total_kwh)
    elif net_power and net_power.points and _has_gap_longer_than(
        net_power,
        start_time=start_time,
        end_time=end_time,
        max_gap=max_gap,
    ):
        validity_reasons.append("net_power_gap_too_large")
        record_series_quality(net_power)
    elif import_total_kwh and import_total_kwh.points and _has_gap_longer_than(
        import_total_kwh,
        start_time=start_time,
        end_time=end_time,
        max_gap=max_gap,
    ):
        validity_reasons.append("import_total_kwh_gap_too_large")
        record_series_quality(import_total_kwh)
    elif export_total_kwh and export_total_kwh.points and _has_gap_longer_than(
        export_total_kwh,
        start_time=start_time,
        end_time=end_time,
        max_gap=max_gap,
    ):
        validity_reasons.append("export_total_kwh_gap_too_large")
        record_series_quality(export_total_kwh)
    else:
        record_series_quality(net_power or import_total_kwh or export_total_kwh)

    if (
        (not pv_total_kwh or not pv_total_kwh.points)
        and (not pv_output_power or not pv_output_power.points)
    ):
        validity_reasons.append("missing_pv_generation")
        record_series_quality(pv_output_power)
        record_series_quality(pv_total_kwh)
    elif pv_total_kwh and pv_total_kwh.points and _has_gap_longer_than(
        pv_total_kwh,
        start_time=start_time,
        end_time=end_time,
        max_gap=max_gap,
    ):
        validity_reasons.append("pv_total_kwh_gap_too_large")
        record_series_quality(pv_total_kwh)
    elif pv_output_power and pv_output_power.points and _has_gap_longer_than(
        pv_output_power,
        start_time=start_time,
        end_time=end_time,
        max_gap=max_gap,
    ):
        validity_reasons.append("pv_output_power_gap_too_large")
        record_series_quality(pv_output_power)
    else:
        record_series_quality(pv_total_kwh or pv_output_power)

    if not outdoor_temperature or not outdoor_temperature.points:
        validity_reasons.append("missing_outdoor_temperature")
    elif _has_gap_longer_than(
        outdoor_temperature,
        start_time=start_time,
        end_time=end_time,
        max_gap=max_gap,
    ):
        validity_reasons.append("outdoor_temperature_gap_too_large")
    record_series_quality(outdoor_temperature)

    if not dhw_top_temperature or not dhw_top_temperature.points:
        validity_reasons.append("missing_dhw_temperature")
    elif _has_gap_longer_than(
        dhw_top_temperature,
        start_time=start_time,
        end_time=end_time,
        max_gap=max_gap,
    ):
        validity_reasons.append("dhw_top_temperature_gap_too_large")
    record_series_quality(dhw_top_temperature)

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
        is_valid_for_control_evaluation=not validity_reasons,
        validity_reasons=validity_reasons,
        data_coverage_pct=min(coverage_values) if coverage_values else None,
        largest_data_gap_minutes=largest_gap_minutes if coverage_values else None,
        hp_electric_kwh=hp_electric_kwh,
        total_import_kwh=total_import_kwh,
        total_export_kwh=total_export_kwh,
        pv_generation_kwh=pv_generation_kwh,
        outdoor_temperature_mean_c=weighted_mean(
            outdoor_temperature,
            start_time=start_time,
            end_time=end_time,
        ),
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


def compute_baseline_kpi_summary(daily_kpis: list[DailyKpis]) -> BaselineKpiSummary:
    valid_days = [
        day_kpis for day_kpis in daily_kpis if day_kpis.is_valid_for_control_evaluation
    ]

    def mean(values: list[float | None]) -> float | None:
        filtered = [value for value in values if value is not None]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    return BaselineKpiSummary(
        number_of_days=len(daily_kpis),
        number_of_valid_days=len(valid_days),
        mean_hp_electric_kwh_per_day=mean(
            [day_kpis.hp_electric_kwh for day_kpis in valid_days]
        ),
        mean_electricity_cost_eur_per_day=mean(
            [day_kpis.electricity_cost_eur for day_kpis in valid_days]
        ),
        mean_room_temperature_mae_c=mean(
            [day_kpis.room_temperature_mae_c for day_kpis in valid_days]
        ),
        total_comfort_violation_degree_hours=sum(
            day_kpis.room_comfort_violation_degree_hours or 0.0
            for day_kpis in valid_days
        ),
        total_dhw_violation_minutes=sum(
            day_kpis.dhw_comfort_violation_minutes or 0.0
            for day_kpis in valid_days
        ),
        mean_compressor_starts_per_day=mean(
            [float(day_kpis.compressor_starts) for day_kpis in valid_days]
        ),
        mean_self_consumption_ratio=mean(
            [day_kpis.self_consumption_ratio for day_kpis in valid_days]
        ),
    )
