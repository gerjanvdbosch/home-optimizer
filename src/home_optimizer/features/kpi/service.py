from __future__ import annotations

from datetime import date, datetime, time, timedelta

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain import (
    COMPRESSOR_FREQUENCY,
    DHW_TARGET_MAX_TEMPERATURE,
    DHW_TARGET_MIN_TEMPERATURE,
    DHW_TARGET_TEMPERATURE,
    DHW_TOP_TEMPERATURE,
    HP_ELECTRIC_POWER,
    HP_ELECTRIC_TOTAL_KWH,
    P1_EXPORT_TOTAL_KWH,
    P1_IMPORT_TOTAL_KWH,
    P1_NET_POWER,
    PV_OUTPUT_POWER,
    PV_TOTAL_KWH,
    ROOM_TARGET_MAX_TEMPERATURE,
    ROOM_TARGET_MIN_TEMPERATURE,
    ROOM_TARGET_TEMPERATURE,
    ROOM_TEMPERATURE,
    THERMOSTAT_SETPOINT,
    DailyKpis,
    FixedPricing,
    NumericPoint,
    NumericSeries,
    build_daily_price_series,
    build_daily_target_band_series,
    latest_value_at,
)
from home_optimizer.domain.pricing import (
    DEFAULT_DYNAMIC_PRICE_SOURCE,
    DEFAULT_FIXED_PRICE_SOURCE,
)
from home_optimizer.domain.time import current_local_timezone, parse_datetime

from .ports import KpiDataReader


def _series_by_name(series_list: list[NumericSeries]) -> dict[str, NumericSeries]:
    return {series.name: series for series in series_list}


def _sorted_points(series: NumericSeries | None) -> list[NumericPoint]:
    if series is None:
        return []
    return sorted(series.points, key=lambda point: point.timestamp)


def _duration_hours(start_time: datetime, end_time: datetime) -> float:
    return (end_time - start_time).total_seconds() / 3600.0


def _integrate_power_series(
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


def _delta_kwh(series: NumericSeries | None) -> float | None:
    points = _sorted_points(series)
    if len(points) < 2:
        return None
    return max(points[-1].value - points[0].value, 0.0)


def _latest_target_value(series: NumericSeries, timestamp: str) -> float | None:
    return latest_value_at(series.points, timestamp)


def _weighted_absolute_error(
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

        target_value = _latest_target_value(target, point.timestamp)
        if target_value is None:
            continue

        duration_hours = _duration_hours(segment_start, segment_end)
        total_error += abs(point.value - target_value) * duration_hours
        total_hours += duration_hours

    if total_hours <= 0.0:
        return None
    return total_error / total_hours


def _weighted_band_violation_degree_hours(
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

        minimum_value = _latest_target_value(minimum, point.timestamp)
        maximum_value = _latest_target_value(maximum, point.timestamp)
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


def _weighted_below_minimum_minutes(
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

        minimum_value = _latest_target_value(minimum, point.timestamp)
        if minimum_value is None:
            continue

        duration_minutes = (segment_end - segment_start).total_seconds() / 60.0
        if point.value < minimum_value:
            violation_minutes += duration_minutes
        total_minutes += duration_minutes

    if total_minutes <= 0.0:
        return None
    return violation_minutes


def _count_setpoint_changes(series: NumericSeries | None) -> int:
    points = _sorted_points(series)
    changes = 0
    previous_value: float | None = None
    for point in points:
        if previous_value is not None and point.value != previous_value:
            changes += 1
        previous_value = point.value
    return changes


def _count_compressor_starts(series: NumericSeries | None) -> int:
    points = _sorted_points(series)
    starts = 0
    previous_running = False
    for point in points:
        running = point.value > 0.0
        if running and not previous_running:
            starts += 1
        previous_running = running
    return starts


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _resolve_price_series(
    reader: KpiDataReader,
    settings: AppSettings,
    *,
    start_time: datetime,
    end_time: datetime,
) -> NumericSeries:
    fetched_series = reader.read_electricity_price_series(
        start_time=start_time,
        end_time=end_time,
        source=(
            DEFAULT_DYNAMIC_PRICE_SOURCE
            if settings.electricity_pricing.mode == "dynamic"
            else DEFAULT_FIXED_PRICE_SOURCE
        ),
    )
    if fetched_series.points:
        return fetched_series

    if isinstance(settings.electricity_pricing, FixedPricing):
        return build_daily_price_series(
            settings.electricity_pricing,
            start_time=start_time,
            end_time=end_time,
        )

    return NumericSeries(name="electricity_price", unit="EUR/kWh", points=[])


def _cost_from_net_power(
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
        interval_price = _latest_target_value(price, point.timestamp)
        if interval_price is None:
            interval_price = 0.0

        if point.value >= 0.0:
            total_cost += point.value * duration_hours * interval_price
        else:
            total_cost -= abs(point.value) * duration_hours * feed_in_tariff

    return total_cost


class DailyKpiService:
    def __init__(self, reader: KpiDataReader, settings: AppSettings) -> None:
        self.reader = reader
        self.settings = settings

    def get_day_kpis(self, day: date) -> DailyKpis:
        local_timezone = current_local_timezone()
        start_time = datetime.combine(day, time.min, tzinfo=local_timezone)
        end_time = start_time + timedelta(days=1)

        series = self.reader.read_series(
            names=[
                ROOM_TEMPERATURE,
                THERMOSTAT_SETPOINT,
                COMPRESSOR_FREQUENCY,
                HP_ELECTRIC_POWER,
                HP_ELECTRIC_TOTAL_KWH,
                P1_NET_POWER,
                P1_IMPORT_TOTAL_KWH,
                P1_EXPORT_TOTAL_KWH,
                PV_OUTPUT_POWER,
                PV_TOTAL_KWH,
                DHW_TOP_TEMPERATURE,
            ],
            start_time=start_time,
            end_time=end_time,
        )
        series_by_name = _series_by_name(series)

        room_target, room_target_min, room_target_max = build_daily_target_band_series(
            self.settings.room_target,
            start_time=start_time,
            end_time=end_time,
            target_name=ROOM_TARGET_TEMPERATURE,
            minimum_name=ROOM_TARGET_MIN_TEMPERATURE,
            maximum_name=ROOM_TARGET_MAX_TEMPERATURE,
            interval_minutes=15,
        )
        _, dhw_target_min, _ = build_daily_target_band_series(
            self.settings.dhw_target,
            start_time=start_time,
            end_time=end_time,
            target_name=DHW_TARGET_TEMPERATURE,
            minimum_name=DHW_TARGET_MIN_TEMPERATURE,
            maximum_name=DHW_TARGET_MAX_TEMPERATURE,
            interval_minutes=15,
        )

        hp_electric_kwh = _delta_kwh(series_by_name.get(HP_ELECTRIC_TOTAL_KWH))
        if hp_electric_kwh is None:
            hp_electric_kwh = _integrate_power_series(
                series_by_name.get(HP_ELECTRIC_POWER),
                start_time=start_time,
                end_time=end_time,
                positive_only=True,
            )

        total_import_kwh = _delta_kwh(series_by_name.get(P1_IMPORT_TOTAL_KWH))
        total_export_kwh = _delta_kwh(series_by_name.get(P1_EXPORT_TOTAL_KWH))
        if total_import_kwh is None:
            total_import_kwh = _integrate_power_series(
                series_by_name.get(P1_NET_POWER),
                start_time=start_time,
                end_time=end_time,
                positive_only=True,
            )
        if total_export_kwh is None:
            total_export_kwh = _integrate_power_series(
                series_by_name.get(P1_NET_POWER),
                start_time=start_time,
                end_time=end_time,
                negative_only=True,
            )

        pv_generation_kwh = _delta_kwh(series_by_name.get(PV_TOTAL_KWH))
        if pv_generation_kwh is None:
            pv_generation_kwh = _integrate_power_series(
                series_by_name.get(PV_OUTPUT_POWER),
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
            self_consumption_ratio = _clamp_ratio(
                (pv_generation_kwh - total_export_kwh) / pv_generation_kwh
            )

        feed_in_tariff = (
            self.settings.electricity_pricing.feed_in_tariff
            if isinstance(self.settings.electricity_pricing, FixedPricing)
            else 0.0
        )
        price_series = _resolve_price_series(
            self.reader,
            self.settings,
            start_time=start_time,
            end_time=end_time,
        )
        electricity_cost_eur = _cost_from_net_power(
            series_by_name.get(P1_NET_POWER),
            price_series,
            start_time=start_time,
            end_time=end_time,
            feed_in_tariff=feed_in_tariff,
        )

        return DailyKpis(
            hp_electric_kwh=hp_electric_kwh,
            total_import_kwh=total_import_kwh,
            total_export_kwh=total_export_kwh,
            pv_generation_kwh=pv_generation_kwh,
            self_consumption_ratio=self_consumption_ratio,
            electricity_cost_eur=electricity_cost_eur,
            room_temperature_mae_c=_weighted_absolute_error(
                series_by_name.get(ROOM_TEMPERATURE),
                room_target,
                start_time=start_time,
                end_time=end_time,
            ),
            room_comfort_violation_degree_hours=_weighted_band_violation_degree_hours(
                series_by_name.get(ROOM_TEMPERATURE),
                room_target_min,
                room_target_max,
                start_time=start_time,
                end_time=end_time,
            ),
            dhw_comfort_violation_minutes=_weighted_below_minimum_minutes(
                series_by_name.get(DHW_TOP_TEMPERATURE),
                dhw_target_min,
                start_time=start_time,
                end_time=end_time,
            ),
            thermostat_setpoint_changes=_count_setpoint_changes(
                series_by_name.get(THERMOSTAT_SETPOINT)
            ),
            compressor_starts=_count_compressor_starts(
                series_by_name.get(COMPRESSOR_FREQUENCY)
            ),
        )
