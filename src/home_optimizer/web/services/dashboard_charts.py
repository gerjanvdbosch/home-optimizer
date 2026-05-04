from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone, tzinfo

from home_optimizer.app import AppSettings
from home_optimizer.domain import (
    BASELOAD,
    BOOSTER_HEATER_ACTIVE,
    COMPRESSOR_FREQUENCY,
    COP,
    DEFROST_ACTIVE,
    DHW_BOTTOM_TEMPERATURE,
    DHW_TARGET_MAX_TEMPERATURE,
    DHW_TARGET_MIN_TEMPERATURE,
    DHW_TARGET_TEMPERATURE,
    DHW_TOP_TEMPERATURE,
    FORECAST_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS,
    GTI_PV,
    HP_DELTA_T,
    HP_ELECTRIC_POWER,
    HP_FLOW,
    HP_MODE,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TARGET_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    P1_NET_POWER,
    PV_OUTPUT_POWER,
    ROOM_TEMPERATURE,
    ROOM_TARGET_MAX_TEMPERATURE,
    ROOM_TARGET_MIN_TEMPERATURE,
    ROOM_TARGET_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMAL_OUTPUT,
    THERMOSTAT_SETPOINT,
    NumericPoint,
    NumericSeries,
    TextSeries,
    adjusted_gti_with_shutter,
    build_daily_target_band_series,
    latest_value_at,
    upsample_series_forward_fill,
)
from home_optimizer.web.mappers import series_response, text_series_response
from home_optimizer.web.ports import DashboardDataReader
from home_optimizer.web.schemas import DashboardChartsResponse


def current_timezone() -> tzinfo:
    return datetime.now().astimezone().tzinfo or timezone.utc


def empty_series(name: str, unit: str | None = None) -> NumericSeries:
    return NumericSeries(name=name, unit=unit, points=[])


def empty_text_series(name: str) -> TextSeries:
    return TextSeries(name=name, points=[])


def build_delta_series(
    supply: NumericSeries | None,
    return_s: NumericSeries | None,
    name: str,
) -> NumericSeries:
    unit = supply.unit if supply else "degC"
    delta = NumericSeries(name=name, unit=unit, points=[])
    if not supply or not return_s:
        return delta

    for sp in supply.points:
        rt = latest_value_at(return_s.points, sp.timestamp)
        if rt is None:
            continue
        delta.points.append(NumericPoint(timestamp=sp.timestamp, value=sp.value - rt))

    return delta


def clamp_series(
    series: NumericSeries,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> NumericSeries:
    return NumericSeries(
        name=series.name,
        unit=series.unit,
        points=[
            NumericPoint(
                timestamp=point.timestamp,
                value=min(
                    max(point.value, minimum) if minimum is not None else point.value,
                    maximum,
                )
                if maximum is not None
                else max(point.value, minimum)
                if minimum is not None
                else point.value,
            )
            for point in series.points
        ],
    )


def build_baseload_series(
    p1: NumericSeries | None,
    pv: NumericSeries | None,
    hp: NumericSeries | None,
    name: str,
) -> NumericSeries:
    unit = p1.unit if p1 else "kW"
    baseload = NumericSeries(name=name, unit=unit, points=[])
    if not p1 or not hp:
        return baseload

    for p in p1.points:
        hp_val = latest_value_at(hp.points, p.timestamp)
        if hp_val is None:
            continue
        pv_val = latest_value_at(pv.points, p.timestamp) if pv else 0.0
        baseload.points.append(NumericPoint(timestamp=p.timestamp, value=p.value + pv_val - hp_val))

    return baseload


def build_thermal_and_cop_series(
    flow: NumericSeries | None,
    supply: NumericSeries | None,
    return_s: NumericSeries | None,
    hp_power: NumericSeries | None,
    thermal_name: str,
    cop_name: str,
) -> tuple[NumericSeries, NumericSeries]:
    thermal_points: list[NumericPoint] = []
    cop_points: list[NumericPoint] = []
    factor = 4186.0 / 60000.0

    if not flow:
        return (
            NumericSeries(name=thermal_name, unit="kW", points=[]),
            NumericSeries(name=cop_name, unit=None, points=[]),
        )

    for fp in flow.points:
        supply_val = latest_value_at(supply.points, fp.timestamp) if supply else None
        return_val = latest_value_at(return_s.points, fp.timestamp) if return_s else None
        if supply_val is None or return_val is None:
            continue
        delta_t = supply_val - return_val
        q_kw = fp.value * delta_t * factor
        if q_kw < 0:
            q_kw = 0.0

        thermal_points.append(NumericPoint(timestamp=fp.timestamp, value=q_kw))
        elec = latest_value_at(hp_power.points, fp.timestamp) if hp_power else None
        if elec is None or elec <= 0:
            continue
        cop_points.append(NumericPoint(timestamp=fp.timestamp, value=q_kw / elec))

    thermal_series = NumericSeries(name=thermal_name, unit="kW", points=thermal_points)
    cop_series = NumericSeries(name=cop_name, unit=None, points=cop_points)
    return thermal_series, cop_series


class DashboardChartsService:
    def __init__(self, reader: DashboardDataReader, settings: AppSettings) -> None:
        self.reader = reader
        self.settings = settings

    def get_day_charts(
        self,
        chart_date: date,
    ) -> DashboardChartsResponse:
        local_timezone = current_timezone()
        start_time = datetime.combine(chart_date, time.min, tzinfo=local_timezone)
        end_time = start_time + timedelta(days=1)
        shutter_series = self.reader.read_series(
            names=[SHUTTER_LIVING_ROOM],
            start_time=start_time,
            end_time=end_time,
        )
        series = self.reader.read_series(
            names=[
                ROOM_TEMPERATURE,
                THERMOSTAT_SETPOINT,
                HP_FLOW,
                P1_NET_POWER,
                PV_OUTPUT_POWER,
                HP_SUPPLY_TEMPERATURE,
                HP_SUPPLY_TARGET_TEMPERATURE,
                HP_RETURN_TEMPERATURE,
                DHW_TOP_TEMPERATURE,
                DHW_BOTTOM_TEMPERATURE,
                HP_ELECTRIC_POWER,
                DEFROST_ACTIVE,
                BOOSTER_HEATER_ACTIVE,
                COMPRESSOR_FREQUENCY,
            ],
            start_time=start_time,
            end_time=end_time,
        )
        text_series = self.reader.read_text_series(
            names=[HP_MODE],
            start_time=start_time,
            end_time=end_time,
        )
        forecast_series = self.reader.read_forecast_series(
            names=[FORECAST_TEMPERATURE, GTI_PV, GTI_LIVING_ROOM_WINDOWS],
            start_time=start_time,
            end_time=end_time + timedelta(minutes=15),
        )
        historical_weather_series = self.reader.read_historical_weather_series(
            names=[FORECAST_TEMPERATURE, GTI_PV, GTI_LIVING_ROOM_WINDOWS],
            start_time=start_time,
            end_time=end_time,
        )
        series_by_name = {item.name: item for item in series}
        shutter_by_name = {item.name: item for item in shutter_series}
        text_series_by_name = {item.name: item for item in text_series}
        forecast_series_by_name = {item.name: item for item in forecast_series}
        historical_weather_series_by_name = {
            item.name: upsample_series_forward_fill(
                item,
                start_time=start_time,
                end_time=end_time,
                interval_minutes=15,
            )
            for item in historical_weather_series
        }
        adjusted_living_room_gti = adjusted_gti_with_shutter(
            forecast_series_by_name.get(
                GTI_LIVING_ROOM_WINDOWS,
                empty_series(GTI_LIVING_ROOM_WINDOWS, unit="Wm2"),
            ),
            shutter_by_name.get(
                SHUTTER_LIVING_ROOM,
                empty_series(SHUTTER_LIVING_ROOM, unit="percent"),
            ),
        )
        adjusted_historical_living_room_gti = adjusted_gti_with_shutter(
            historical_weather_series_by_name.get(
                GTI_LIVING_ROOM_WINDOWS,
                empty_series(GTI_LIVING_ROOM_WINDOWS, unit="Wm2"),
            ),
            shutter_by_name.get(
                SHUTTER_LIVING_ROOM,
                empty_series(SHUTTER_LIVING_ROOM, unit="percent"),
            ),
        )

        supply_series = series_by_name.get(HP_SUPPLY_TEMPERATURE)
        return_series = series_by_name.get(HP_RETURN_TEMPERATURE)
        delta_series = build_delta_series(supply_series, return_series, name=HP_DELTA_T)
        clamped_delta_series = clamp_series(delta_series, minimum=0.0)

        p1_series = series_by_name.get(P1_NET_POWER)
        pv_series = series_by_name.get(PV_OUTPUT_POWER)
        hp_power_series = series_by_name.get(HP_ELECTRIC_POWER)
        baseload_series = build_baseload_series(
            p1_series,
            pv_series,
            hp_power_series,
            name=BASELOAD,
        )
        room_target_series, room_target_min_series, room_target_max_series = (
            build_daily_target_band_series(
                self.settings.room_target,
                start_time=start_time,
                end_time=end_time,
                target_name=ROOM_TARGET_TEMPERATURE,
                minimum_name=ROOM_TARGET_MIN_TEMPERATURE,
                maximum_name=ROOM_TARGET_MAX_TEMPERATURE,
                interval_minutes=15,
            )
        )
        dhw_target_series, dhw_target_min_series, dhw_target_max_series = (
            build_daily_target_band_series(
                self.settings.dhw_target,
                start_time=start_time,
                end_time=end_time,
                target_name=DHW_TARGET_TEMPERATURE,
                minimum_name=DHW_TARGET_MIN_TEMPERATURE,
                maximum_name=DHW_TARGET_MAX_TEMPERATURE,
                interval_minutes=15,
            )
        )

        flow_series = series_by_name.get(HP_FLOW, empty_series(HP_FLOW, unit="Lmin"))
        thermal_series, cop_series = build_thermal_and_cop_series(
            flow_series,
            series_by_name.get(HP_SUPPLY_TEMPERATURE),
            series_by_name.get(HP_RETURN_TEMPERATURE),
            hp_power_series,
            thermal_name=THERMAL_OUTPUT,
            cop_name=COP,
        )
        clamped_cop_series = clamp_series(cop_series, maximum=10.0)

        return DashboardChartsResponse(
            date=chart_date.isoformat(),
            room_temperature=series_response(
                series_by_name.get(ROOM_TEMPERATURE, empty_series(ROOM_TEMPERATURE))
            ),
            thermostat_setpoint=series_response(
                series_by_name.get(THERMOSTAT_SETPOINT, empty_series(THERMOSTAT_SETPOINT))
            ),
            room_target_temperature=series_response(room_target_series),
            room_target_min_temperature=series_response(room_target_min_series),
            room_target_max_temperature=series_response(room_target_max_series),
            shutter_position=series_response(
                shutter_by_name.get(
                    SHUTTER_LIVING_ROOM,
                    empty_series(SHUTTER_LIVING_ROOM, unit="percent"),
                )
            ),
            dhw_temperatures=[
                series_response(
                    series_by_name.get(DHW_TOP_TEMPERATURE, empty_series(DHW_TOP_TEMPERATURE))
                ),
                series_response(
                    series_by_name.get(DHW_BOTTOM_TEMPERATURE, empty_series(DHW_BOTTOM_TEMPERATURE))
                ),
            ],
            dhw_target_temperature=series_response(dhw_target_series),
            dhw_target_min_temperature=series_response(dhw_target_min_series),
            dhw_target_max_temperature=series_response(dhw_target_max_series),
            heatpump_power=series_response(
                series_by_name.get(HP_ELECTRIC_POWER, empty_series(HP_ELECTRIC_POWER))
            ),
            heatpump_mode=text_series_response(
                text_series_by_name.get(HP_MODE, empty_text_series(HP_MODE))
            ),
            heatpump_statuses=[
                series_response(series_by_name.get(DEFROST_ACTIVE, empty_series(DEFROST_ACTIVE))),
                series_response(
                    series_by_name.get(BOOSTER_HEATER_ACTIVE, empty_series(BOOSTER_HEATER_ACTIVE))
                ),
            ],
            forecast_temperature=series_response(
                forecast_series_by_name.get(
                    FORECAST_TEMPERATURE,
                    empty_series(FORECAST_TEMPERATURE),
                )
            ),
            forecast_gti=[
                series_response(forecast_series_by_name.get(GTI_PV, empty_series(GTI_PV))),
                series_response(
                    forecast_series_by_name.get(
                        GTI_LIVING_ROOM_WINDOWS,
                        empty_series(GTI_LIVING_ROOM_WINDOWS),
                    )
                ),
                series_response(adjusted_living_room_gti),
            ],
            historical_weather_temperature=series_response(
                historical_weather_series_by_name.get(
                    FORECAST_TEMPERATURE,
                    empty_series(FORECAST_TEMPERATURE),
                )
            ),
            historical_weather_gti=[
                series_response(
                    historical_weather_series_by_name.get(GTI_PV, empty_series(GTI_PV))
                ),
                series_response(
                    historical_weather_series_by_name.get(
                        GTI_LIVING_ROOM_WINDOWS,
                        empty_series(GTI_LIVING_ROOM_WINDOWS),
                    )
                ),
                series_response(adjusted_historical_living_room_gti),
            ],
            hp_supply_temperature=series_response(
                series_by_name.get(HP_SUPPLY_TEMPERATURE, empty_series(HP_SUPPLY_TEMPERATURE))
            ),
            hp_supply_target_temperature=series_response(
                series_by_name.get(
                    HP_SUPPLY_TARGET_TEMPERATURE,
                    empty_series(HP_SUPPLY_TARGET_TEMPERATURE),
                )
            ),
            hp_return_temperature=series_response(
                series_by_name.get(HP_RETURN_TEMPERATURE, empty_series(HP_RETURN_TEMPERATURE))
            ),
            pv_output_power=series_response(
                series_by_name.get(PV_OUTPUT_POWER, empty_series(PV_OUTPUT_POWER))
            ),
            baseload=series_response(baseload_series),
            hp_delta_t=series_response(clamped_delta_series),
            thermal_output=series_response(thermal_series),
            cop=series_response(clamped_cop_series),
            hp_flow=series_response(flow_series),
            compressor_frequency=series_response(
                series_by_name.get(COMPRESSOR_FREQUENCY, empty_series(COMPRESSOR_FREQUENCY))
            ),
        )
