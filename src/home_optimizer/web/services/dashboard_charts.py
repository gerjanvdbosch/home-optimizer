from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone, tzinfo

from home_optimizer.domain import ChartPoint, ChartSeries, ChartTextSeries
from home_optimizer.web.mappers import series_response, text_series_response
from home_optimizer.web.ports import DashboardDataReader
from home_optimizer.web.schemas import DashboardChartsResponse


def current_timezone() -> tzinfo:
    return datetime.now().astimezone().tzinfo or timezone.utc


def adjusted_gti_with_shutter(
    window_gti: ChartSeries,
    shutter_position: ChartSeries,
) -> ChartSeries:
    return ChartSeries(
        name="gti_living_room_windows_adjusted",
        unit=window_gti.unit,
        points=[
            ChartPoint(
                timestamp=point.timestamp,
                value=point.value
                * shutter_open_fraction_at(shutter_position.points, point.timestamp),
            )
            for point in window_gti.points
        ],
    )


def shutter_open_fraction_at(points: list[ChartPoint], timestamp: str) -> float:
    position = latest_value_at(points, timestamp)
    if position is None:
        return 1.0
    return max(0.0, min(position, 100.0)) / 100.0


def latest_value_at(points: list[ChartPoint], timestamp: str) -> float | None:
    latest: float | None = None
    for point in points:
        if point.timestamp > timestamp:
            break
        latest = point.value
    return latest


def empty_series(name: str, unit: str | None = None) -> ChartSeries:
    return ChartSeries(name=name, unit=unit, points=[])


def empty_text_series(name: str) -> ChartTextSeries:
    return ChartTextSeries(name=name, points=[])


def build_delta_series(
    supply: ChartSeries | None,
    return_s: ChartSeries | None,
    name: str,
) -> ChartSeries:
    unit = supply.unit if supply else "degC"
    delta = ChartSeries(name=name, unit=unit, points=[])
    if not supply or not return_s:
        return delta

    for sp in supply.points:
        rt = latest_value_at(return_s.points, sp.timestamp)
        if rt is None:
            continue
        delta.points.append(ChartPoint(timestamp=sp.timestamp, value=sp.value - rt))

    return delta


def build_baseload_series(
    p1: ChartSeries | None,
    pv: ChartSeries | None,
    hp: ChartSeries | None,
    name: str,
) -> ChartSeries:
    unit = p1.unit if p1 else "kW"
    baseload = ChartSeries(name=name, unit=unit, points=[])
    if not p1 or not hp:
        return baseload

    for p in p1.points:
        hp_val = latest_value_at(hp.points, p.timestamp)
        if hp_val is None:
            continue
        pv_val = latest_value_at(pv.points, p.timestamp) if pv else 0.0
        baseload.points.append(ChartPoint(timestamp=p.timestamp, value=p.value + pv_val - hp_val))

    return baseload


def build_thermal_and_cop_series(
    flow: ChartSeries | None,
    supply: ChartSeries | None,
    return_s: ChartSeries | None,
    hp_power: ChartSeries | None,
    thermal_name: str,
    cop_name: str,
) -> tuple[ChartSeries, ChartSeries]:
    """Compute thermal output (kW) and COP series aligned to flow timestamps.

    Formula: Q_kW = flow_Lmin * (supply - return) * 4186 / 60000
    COP = Q_kW / electrical_input_kW when electrical_input_kW is available and non-zero.
    """
    thermal_points: list[ChartPoint] = []
    cop_points: list[ChartPoint] = []
    factor = 4186.0 / 60000.0

    if not flow:
        return (
            ChartSeries(name=thermal_name, unit="kW", points=[]),
            ChartSeries(name=cop_name, unit=None, points=[]),
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

        thermal_points.append(ChartPoint(timestamp=fp.timestamp, value=q_kw))
        elec = latest_value_at(hp_power.points, fp.timestamp) if hp_power else None
        if elec is None or elec <= 0:
            continue
        cop_points.append(ChartPoint(timestamp=fp.timestamp, value=q_kw / elec))

    thermal_series = ChartSeries(name=thermal_name, unit="kW", points=thermal_points)
    cop_series = ChartSeries(name=cop_name, unit=None, points=cop_points)
    return thermal_series, cop_series


class DashboardChartsService:
    def __init__(self, reader: DashboardDataReader) -> None:
        self.reader = reader

    def get_day_charts(
        self,
        chart_date: date,
    ) -> DashboardChartsResponse:
        local_timezone = current_timezone()
        start_time = datetime.combine(chart_date, time.min, tzinfo=local_timezone)
        end_time = start_time + timedelta(days=1)
        shutter_series = self.reader.read_series(
            names=["shutter_living_room"],
            start_time=start_time,
            end_time=end_time,
        )
        series = self.reader.read_series(
            names=[
                "room_temperature",
                "thermostat_setpoint",
                "hp_flow",
                "p1_net_power",
                "pv_output_power",
                "hp_supply_temperature",
                "hp_supply_target_temperature",
                "hp_return_temperature",
                "dhw_top_temperature",
                "dhw_bottom_temperature",
                "hp_electric_power",
                "defrost_active",
                "booster_heater_active",
                "compressor_frequency",
            ],
            start_time=start_time,
            end_time=end_time,
        )
        text_series = self.reader.read_text_series(
            names=["hp_mode"],
            start_time=start_time,
            end_time=end_time,
        )
        forecast_series = self.reader.read_forecast_series(
            names=["temperature", "gti_pv", "gti_living_room_windows"],
            start_time=start_time,
            end_time=end_time + timedelta(minutes=15),
        )
        series_by_name = {item.name: item for item in series}
        shutter_by_name = {item.name: item for item in shutter_series}
        text_series_by_name = {item.name: item for item in text_series}
        forecast_series_by_name = {item.name: item for item in forecast_series}
        adjusted_living_room_gti = adjusted_gti_with_shutter(
            forecast_series_by_name.get(
                "gti_living_room_windows",
                empty_series("gti_living_room_windows", unit="Wm2"),
            ),
            shutter_by_name.get(
                "shutter_living_room",
                empty_series("shutter_living_room", unit="percent"),
            ),
        )

        supply_series = series_by_name.get("hp_supply_temperature")
        return_series = series_by_name.get("hp_return_temperature")
        delta_series = build_delta_series(supply_series, return_series, name="hp_delta_t")

        p1_series = series_by_name.get("p1_net_power")
        pv_series = series_by_name.get("pv_output_power")
        hp_power_series = series_by_name.get("hp_electric_power")
        baseload_series = build_baseload_series(
            p1_series,
            pv_series,
            hp_power_series,
            name="baseload",
        )

        flow_series = series_by_name.get("hp_flow", empty_series("hp_flow", unit="Lmin"))
        thermal_series, cop_series = build_thermal_and_cop_series(
            flow_series,
            series_by_name.get("hp_supply_temperature"),
            series_by_name.get("hp_return_temperature"),
            hp_power_series,
            thermal_name="thermal_output",
            cop_name="cop",
        )

        return DashboardChartsResponse(
            date=chart_date.isoformat(),
            room_temperature=series_response(
                series_by_name.get("room_temperature", empty_series("room_temperature"))
            ),
            thermostat_setpoint=series_response(
                series_by_name.get("thermostat_setpoint", empty_series("thermostat_setpoint"))
            ),
            shutter_position=series_response(
                shutter_by_name.get(
                    "shutter_living_room",
                    empty_series("shutter_living_room", unit="percent"),
                )
            ),
            dhw_temperatures=[
                series_response(
                    series_by_name.get("dhw_top_temperature", empty_series("dhw_top_temperature"))
                ),
                series_response(
                    series_by_name.get(
                        "dhw_bottom_temperature",
                        empty_series("dhw_bottom_temperature"),
                    )
                ),
            ],
            heatpump_power=series_response(
                series_by_name.get("hp_electric_power", empty_series("hp_electric_power"))
            ),
            heatpump_mode=text_series_response(
                text_series_by_name.get("hp_mode", empty_text_series("hp_mode"))
            ),
            heatpump_statuses=[
                series_response(
                    series_by_name.get("defrost_active", empty_series("defrost_active"))
                ),
                series_response(
                    series_by_name.get(
                        "booster_heater_active",
                        empty_series("booster_heater_active"),
                    )
                ),
            ],
            forecast_temperature=series_response(
                forecast_series_by_name.get("temperature", empty_series("temperature"))
            ),
            forecast_gti=[
                series_response(
                    forecast_series_by_name.get("gti_pv", empty_series("gti_pv"))
                ),
                series_response(
                    forecast_series_by_name.get(
                        "gti_living_room_windows",
                        empty_series("gti_living_room_windows"),
                    )
                ),
                series_response(adjusted_living_room_gti),
            ],
            hp_supply_temperature=series_response(
                series_by_name.get("hp_supply_temperature", empty_series("hp_supply_temperature"))
            ),
            hp_supply_target_temperature=series_response(
                series_by_name.get(
                    "hp_supply_target_temperature",
                    empty_series("hp_supply_target_temperature"),
                )
            ),
            hp_return_temperature=series_response(
                series_by_name.get("hp_return_temperature", empty_series("hp_return_temperature"))
            ),
            pv_output_power=series_response(
                series_by_name.get("pv_output_power", empty_series("pv_output_power"))
            ),
            baseload=series_response(baseload_series),
            hp_delta_t=series_response(delta_series),
            thermal_output=series_response(thermal_series),
            cop=series_response(cop_series),
            hp_flow=series_response(flow_series),
            compressor_frequency=series_response(
                series_by_name.get(
                    "compressor_frequency",
                    empty_series("compressor_frequency"),
                )
            ),
        )
