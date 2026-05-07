from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain import (
    BOOSTER_HEATER_ACTIVE,
    DEFROST_ACTIVE,
    DHW_BOTTOM_TEMPERATURE,
    DHW_TOP_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS,
    HP_ELECTRIC_POWER,
    HP_FLOW,
    HP_MODE,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    OUTDOOR_TEMPERATURE,
    P1_NET_POWER,
    PV_OUTPUT_POWER,
    ROOM_TARGET_MAX_TEMPERATURE,
    ROOM_TARGET_MIN_TEMPERATURE,
    ROOM_TARGET_TEMPERATURE,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
    FixedPricing,
    NumericPoint,
    NumericSeries,
    TemperatureTargetWindow,
    TextSeries,
    build_daily_price_series,
    build_thermal_output_series,
    ensure_utc,
    latest_value_at,
    normalize_utc_timestamp,
    shutter_open_fraction_at,
)
from home_optimizer.domain.pricing import (
    DEFAULT_DYNAMIC_PRICE_SOURCE,
    DEFAULT_FIXED_PRICE_SOURCE,
)
from home_optimizer.domain.time import parse_datetime
from home_optimizer.features.identification.models import (
    IdentificationDataset,
    IdentificationDatasetRow,
    IdentificationDatasetSummary,
)
from home_optimizer.features.identification.ports import IdentificationDataReader

_OCCUPIED_MARGIN_C = 0.25
_DHW_DRAW_DROP_THRESHOLD_C = 0.75
_MODE_SPACE_TOKENS = ("heat", "heating", "ufh")
_MODE_DHW_TOKENS = ("dhw", "sww")
_MODE_OFF_TOKENS = ("off", "idle", "standby", "none")
_MIN_PLAUSIBLE_COP = 1.0
_MAX_PLAUSIBLE_COP = 8.0


def _series_by_name(series_list: list[NumericSeries]) -> dict[str, NumericSeries]:
    return {series.name: series for series in series_list}


def _text_series_by_name(series_list: list[TextSeries]) -> dict[str, TextSeries]:
    return {series.name: series for series in series_list}


def _resolve_price_series(
    reader: IdentificationDataReader,
    settings: AppSettings,
    *,
    start_time: datetime,
    end_time: datetime,
    interval_minutes: int,
) -> NumericSeries:
    fetched_series = reader.read_electricity_price_series(
        start_time=start_time,
        end_time=end_time,
        source=(
            DEFAULT_DYNAMIC_PRICE_SOURCE
            if settings.electricity_pricing.mode == "dynamic"
            else DEFAULT_FIXED_PRICE_SOURCE
        ),
        interval_minutes=interval_minutes,
    )
    if fetched_series.points:
        return fetched_series

    if isinstance(settings.electricity_pricing, FixedPricing):
        return build_daily_price_series(
            settings.electricity_pricing,
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
        )

    return NumericSeries(name="electricity_price", unit="EUR/kWh", points=[])


def _active_target_window(
    schedule: list[TemperatureTargetWindow],
    timestamp: datetime,
) -> TemperatureTargetWindow | None:
    if not schedule:
        return None

    ordered_schedule = sorted(schedule, key=lambda window: window.time)
    local_timestamp = timestamp.astimezone(timestamp.tzinfo)
    active_window = ordered_schedule[-1]

    for window in ordered_schedule:
        change_time = datetime.combine(
            local_timestamp.date(),
            window.time,
            tzinfo=local_timestamp.tzinfo,
        )
        if change_time <= local_timestamp:
            active_window = window
            continue
        break

    return active_window


def _build_target_series(
    schedule: list[TemperatureTargetWindow],
    *,
    start_time: datetime,
    end_time: datetime,
    interval_minutes: int,
    series_name: str,
    extractor: Callable[[TemperatureTargetWindow], float],
) -> NumericSeries:
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be greater than zero")
    if end_time <= start_time or not schedule:
        return NumericSeries(name=series_name, unit="°C", points=[])

    interval = timedelta(minutes=interval_minutes)
    points: list[NumericPoint] = []
    cursor = start_time

    while cursor < end_time:
        active_window = _active_target_window(schedule, cursor)
        if active_window is not None:
            points.append(
                NumericPoint(
                    timestamp=normalize_utc_timestamp(cursor),
                    value=extractor(active_window),
                )
            )
        cursor += interval

    return NumericSeries(name=series_name, unit="°C", points=points)


def _classify_hp_mode(mode: str | None) -> tuple[int, int, int]:
    if mode is None:
        return 0, 0, 1

    normalized = mode.strip().lower()
    if any(token in normalized for token in _MODE_DHW_TOKENS):
        return 0, 1, 0
    if any(token in normalized for token in _MODE_SPACE_TOKENS):
        return 1, 0, 0
    if any(token in normalized for token in _MODE_OFF_TOKENS):
        return 0, 0, 1
    return 0, 0, 1


def _price_export_value(settings: AppSettings) -> float:
    if isinstance(settings.electricity_pricing, FixedPricing):
        return settings.electricity_pricing.feed_in_tariff
    return 0.0


def _occupied_flag(
    target_temperature: float | None,
    schedule: list[TemperatureTargetWindow],
) -> int:
    if target_temperature is None or not schedule:
        return 0

    minimum_target = min(window.target for window in schedule)
    return int(target_temperature > minimum_target + _OCCUPIED_MARGIN_C)


def _detect_dhw_draw(
    current_value: float | None,
    previous_value: float | None,
    *,
    mode_dhw: int,
) -> int:
    if mode_dhw:
        return 0
    if current_value is None or previous_value is None:
        return 0
    return int((previous_value - current_value) >= _DHW_DRAW_DROP_THRESHOLD_C)


def _empty_numeric_series() -> NumericSeries:
    return NumericSeries(name="", unit=None, points=[])


def _latest_numeric_value(
    series_by_name: dict[str, NumericSeries],
    name: str,
    timestamp: str,
) -> float | None:
    return latest_value_at(series_by_name.get(name, _empty_numeric_series()).points, timestamp)


def _window_has_positive_value(
    series_by_name: dict[str, NumericSeries],
    name: str,
    *,
    window_start: datetime,
    window_end: datetime,
) -> int:
    series = series_by_name.get(name)
    if series is None:
        return 0

    for point in series.points:
        point_time = parse_datetime(point.timestamp)
        if point_time < window_start:
            continue
        if point_time >= window_end:
            break
        if point.value > 0:
            return 1

    return 0


def _validate_row(
    *,
    mode_space: int,
    mode_dhw: int,
    mode_off: int,
    defrost_active: int,
    booster_heater_active: int,
    flow_l_min: float | None,
    thermal_output_estimate: float | None,
    cop_estimate: float | None,
) -> tuple[bool, bool, bool, list[str]]:
    reasons: list[str] = []
    active_mode_count = mode_space + mode_dhw + mode_off

    if active_mode_count != 1:
        reasons.append("invalid_mode_combination")
    if defrost_active:
        reasons.append("defrost_active")
    if booster_heater_active:
        reasons.append("booster_heater_active")
    if flow_l_min is not None and flow_l_min < 0:
        reasons.append("negative_flow")
    if thermal_output_estimate is not None and thermal_output_estimate < 0:
        reasons.append("negative_thermal_output")

    room_valid = active_mode_count == 1 and not defrost_active and not booster_heater_active
    dhw_valid = active_mode_count == 1 and not defrost_active and not booster_heater_active

    cop_valid = (
        active_mode_count == 1
        and not defrost_active
        and not booster_heater_active
        and thermal_output_estimate is not None
        and thermal_output_estimate > 0
        and cop_estimate is not None
        and _MIN_PLAUSIBLE_COP <= cop_estimate <= _MAX_PLAUSIBLE_COP
    )
    if cop_estimate is not None and not (_MIN_PLAUSIBLE_COP <= cop_estimate <= _MAX_PLAUSIBLE_COP):
        reasons.append("cop_out_of_range")
    if thermal_output_estimate is None or thermal_output_estimate <= 0:
        reasons.append("missing_or_nonpositive_thermal_output")

    return room_valid, dhw_valid, cop_valid, reasons


class IdentificationDatasetService:
    def __init__(self, reader: IdentificationDataReader, settings: AppSettings) -> None:
        self.reader = reader
        self.settings = settings

    def build_dataset(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 15,
    ) -> IdentificationDataset:
        start_time_utc = ensure_utc(start_time)
        end_time_utc = ensure_utc(end_time)
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")
        if end_time_utc <= start_time_utc:
            raise ValueError("end_time must be after start_time")

        numeric_series = _series_by_name(
            self.reader.read_series(
                names=[
                    ROOM_TEMPERATURE,
                    OUTDOOR_TEMPERATURE,
                    DHW_TOP_TEMPERATURE,
                    DHW_BOTTOM_TEMPERATURE,
                    HP_ELECTRIC_POWER,
                    PV_OUTPUT_POWER,
                    P1_NET_POWER,
                    DEFROST_ACTIVE,
                    BOOSTER_HEATER_ACTIVE,
                    SHUTTER_LIVING_ROOM,
                    THERMOSTAT_SETPOINT,
                    HP_SUPPLY_TEMPERATURE,
                    HP_RETURN_TEMPERATURE,
                    HP_FLOW,
                ],
                start_time=start_time_utc,
                end_time=end_time_utc,
            )
        )
        text_series = _text_series_by_name(
            self.reader.read_text_series(
                names=[HP_MODE],
                start_time=start_time_utc,
                end_time=end_time_utc,
            )
        )
        forecast_series = _series_by_name(
            self.reader.read_forecast_series(
                names=[GTI_LIVING_ROOM_WINDOWS],
                start_time=start_time_utc,
                end_time=end_time_utc,
            )
        )

        price_series = _resolve_price_series(
            self.reader,
            self.settings,
            start_time=start_time_utc,
            end_time=end_time_utc,
            interval_minutes=interval_minutes,
        )
        room_target = _build_target_series(
            self.settings.room_target,
            start_time=start_time_utc,
            end_time=end_time_utc,
            interval_minutes=interval_minutes,
            series_name=ROOM_TARGET_TEMPERATURE,
            extractor=lambda window: window.target,
        )
        room_target_min = _build_target_series(
            self.settings.room_target,
            start_time=start_time_utc,
            end_time=end_time_utc,
            interval_minutes=interval_minutes,
            series_name=ROOM_TARGET_MIN_TEMPERATURE,
            extractor=lambda window: window.minimum,
        )
        room_target_max = _build_target_series(
            self.settings.room_target,
            start_time=start_time_utc,
            end_time=end_time_utc,
            interval_minutes=interval_minutes,
            series_name=ROOM_TARGET_MAX_TEMPERATURE,
            extractor=lambda window: window.maximum,
        )
        thermal_output_series = build_thermal_output_series(
            numeric_series.get(HP_FLOW),
            numeric_series.get(HP_SUPPLY_TEMPERATURE),
            numeric_series.get(HP_RETURN_TEMPERATURE),
        )

        interval = timedelta(minutes=interval_minutes)
        rows: list[IdentificationDatasetRow] = []
        previous_dhw_top: float | None = None
        cursor = start_time_utc

        while cursor < end_time_utc:
            timestamp = normalize_utc_timestamp(cursor)
            next_cursor = min(cursor + interval, end_time_utc)
            room_temperature = _latest_numeric_value(
                numeric_series, ROOM_TEMPERATURE, timestamp
            )
            outdoor_temperature = _latest_numeric_value(
                numeric_series, OUTDOOR_TEMPERATURE, timestamp
            )
            dhw_top_temperature = _latest_numeric_value(
                numeric_series, DHW_TOP_TEMPERATURE, timestamp
            )
            dhw_bottom_temperature = _latest_numeric_value(
                numeric_series, DHW_BOTTOM_TEMPERATURE, timestamp
            )
            hp_electric_power = _latest_numeric_value(numeric_series, HP_ELECTRIC_POWER, timestamp)
            pv_output_power = _latest_numeric_value(numeric_series, PV_OUTPUT_POWER, timestamp)
            net_power = _latest_numeric_value(numeric_series, P1_NET_POWER, timestamp)
            defrost_active = _window_has_positive_value(
                numeric_series,
                DEFROST_ACTIVE,
                window_start=cursor,
                window_end=next_cursor,
            )
            booster_heater_active = _window_has_positive_value(
                numeric_series,
                BOOSTER_HEATER_ACTIVE,
                window_start=cursor,
                window_end=next_cursor,
            )
            shutter_position = _latest_numeric_value(numeric_series, SHUTTER_LIVING_ROOM, timestamp)
            thermostat_setpoint = _latest_numeric_value(
                numeric_series, THERMOSTAT_SETPOINT, timestamp
            )
            supply_temperature = _latest_numeric_value(
                numeric_series, HP_SUPPLY_TEMPERATURE, timestamp
            )
            return_temperature = _latest_numeric_value(
                numeric_series, HP_RETURN_TEMPERATURE, timestamp
            )
            flow_l_min = _latest_numeric_value(numeric_series, HP_FLOW, timestamp)
            thermal_output_estimate = latest_value_at(thermal_output_series.points, timestamp)
            solar_irradiance = _latest_numeric_value(
                forecast_series, GTI_LIVING_ROOM_WINDOWS, timestamp
            )
            price_import = latest_value_at(price_series.points, timestamp)
            room_target_temperature = latest_value_at(room_target.points, timestamp)
            room_target_min_temperature = latest_value_at(room_target_min.points, timestamp)
            room_target_max_temperature = latest_value_at(room_target_max.points, timestamp)

            hp_mode_raw = None
            mode_series = text_series.get(HP_MODE)
            if mode_series is not None:
                for point in mode_series.points:
                    if point.timestamp > timestamp:
                        break
                    hp_mode_raw = point.value

            mode_space, mode_dhw, mode_off = _classify_hp_mode(hp_mode_raw)
            hp_delta_t = None
            if supply_temperature is not None and return_temperature is not None:
                hp_delta_t = supply_temperature - return_temperature

            cop_estimate = None
            if (
                thermal_output_estimate is not None
                and hp_electric_power is not None
                and hp_electric_power > 0
            ):
                cop_estimate = thermal_output_estimate / hp_electric_power

            solar_gain_proxy = None
            shutter_fraction = shutter_open_fraction_at(
                numeric_series.get(SHUTTER_LIVING_ROOM, _empty_numeric_series()).points,
                timestamp,
            )
            if solar_irradiance is not None:
                solar_gain_proxy = solar_irradiance * shutter_fraction

            room_valid, dhw_valid, cop_valid, exclusion_reasons = _validate_row(
                mode_space=mode_space,
                mode_dhw=mode_dhw,
                mode_off=mode_off,
                defrost_active=defrost_active,
                booster_heater_active=booster_heater_active,
                flow_l_min=flow_l_min,
                thermal_output_estimate=thermal_output_estimate,
                cop_estimate=cop_estimate,
            )

            rows.append(
                IdentificationDatasetRow(
                    timestamp_utc=cursor,
                    room_temperature_c=room_temperature,
                    outdoor_temperature_c=outdoor_temperature,
                    dhw_top_temperature_c=dhw_top_temperature,
                    dhw_bottom_temperature_c=dhw_bottom_temperature,
                    hp_electric_power_kw=hp_electric_power,
                    hp_mode_raw=hp_mode_raw,
                    mode_space=mode_space,
                    mode_dhw=mode_dhw,
                    mode_off=mode_off,
                    defrost_active=defrost_active,
                    booster_heater_active=booster_heater_active,
                    pv_output_power_kw=pv_output_power,
                    net_power_kw=net_power,
                    shutter_position_pct=shutter_position,
                    thermostat_setpoint_c=thermostat_setpoint,
                    room_target_temperature_c=room_target_temperature,
                    room_target_min_temperature_c=room_target_min_temperature,
                    room_target_max_temperature_c=room_target_max_temperature,
                    supply_temperature_c=supply_temperature,
                    return_temperature_c=return_temperature,
                    flow_l_min=flow_l_min,
                    hp_delta_t_c=hp_delta_t,
                    thermal_output_estimate_kw=thermal_output_estimate,
                    cop_estimate=cop_estimate,
                    solar_irradiance_w_m2=solar_irradiance,
                    solar_gain_proxy_w_m2=solar_gain_proxy,
                    price_import_eur_kwh=price_import,
                    price_export_eur_kwh=_price_export_value(self.settings),
                    occupied_flag=_occupied_flag(
                        room_target_temperature,
                        self.settings.room_target,
                    ),
                    dhw_draw_detected=_detect_dhw_draw(
                        dhw_top_temperature,
                        previous_dhw_top,
                        mode_dhw=mode_dhw,
                    ),
                    is_valid_for_room_identification=room_valid,
                    is_valid_for_dhw_identification=dhw_valid,
                    is_valid_for_cop_identification=cop_valid,
                    exclusion_reasons=exclusion_reasons,
                )
            )
            previous_dhw_top = dhw_top_temperature
            cursor += interval

        return IdentificationDataset(
            interval_minutes=interval_minutes,
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
            rows=rows,
        )

    def summarize_dataset(
        self,
        dataset: IdentificationDataset,
    ) -> IdentificationDatasetSummary:
        exclusion_reason_counts: dict[str, int] = {}
        for row in dataset.rows:
            for reason in row.exclusion_reasons:
                exclusion_reason_counts[reason] = exclusion_reason_counts.get(reason, 0) + 1

        return IdentificationDatasetSummary(
            total_rows=len(dataset.rows),
            mode_space_rows=sum(row.mode_space for row in dataset.rows),
            mode_dhw_rows=sum(row.mode_dhw for row in dataset.rows),
            mode_off_rows=sum(row.mode_off for row in dataset.rows),
            defrost_rows=sum(row.defrost_active for row in dataset.rows),
            booster_rows=sum(row.booster_heater_active for row in dataset.rows),
            valid_room_rows=sum(int(row.is_valid_for_room_identification) for row in dataset.rows),
            valid_dhw_rows=sum(int(row.is_valid_for_dhw_identification) for row in dataset.rows),
            valid_cop_rows=sum(int(row.is_valid_for_cop_identification) for row in dataset.rows),
            exclusion_reason_counts=exclusion_reason_counts,
        )
