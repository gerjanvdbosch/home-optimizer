from __future__ import annotations

from datetime import datetime

import pandas as pd

from home_optimizer.domain import (
    BOOSTER_HEATER_ACTIVE,
    DEFROST_ACTIVE,
    FLOOR_HEAT_STATE,
    GTI_LIVING_ROOM_WINDOWS,
    HP_FLOW,
    HP_MODE,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TARGET_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    NumericSeries,
    OUTDOOR_TEMPERATURE,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMAL_OUTPUT,
    THERMOSTAT_SETPOINT,
    adjusted_gti_with_shutter,
    build_floor_heat_state_series,
    build_space_heating_thermal_output_series,
    latest_value_at,
    upsample_series_forward_fill,
)
from home_optimizer.domain.time import parse_datetime

from ..ports import IdentificationDataReader
from ..schemas import IdentificationDataset
from ..room_temperature.state_filter import RoomTemperatureStateFilter
from .model import THERMAL_OUTPUT_FEATURE_NAMES


class ThermalOutputDatasetBuilder:
    def __init__(self, reader: IdentificationDataReader) -> None:
        self.reader = reader

    def build(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        interval_minutes: int = 15,
    ) -> IdentificationDataset:
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")

        series = self.reader.read_series(
            names=[
                ROOM_TEMPERATURE,
                OUTDOOR_TEMPERATURE,
                THERMOSTAT_SETPOINT,
                SHUTTER_LIVING_ROOM,
                HP_FLOW,
                HP_SUPPLY_TEMPERATURE,
                HP_SUPPLY_TARGET_TEMPERATURE,
                HP_RETURN_TEMPERATURE,
                DEFROST_ACTIVE,
                BOOSTER_HEATER_ACTIVE,
            ],
            start_time=start_time,
            end_time=end_time,
        )
        text_series = self.reader.read_text_series(
            names=[HP_MODE],
            start_time=start_time,
            end_time=end_time,
        )
        historical_weather_series = self.reader.read_historical_weather_series(
            names=[GTI_LIVING_ROOM_WINDOWS],
            start_time=start_time,
            end_time=end_time,
        )
        forecast_series = self.reader.read_forecast_series(
            names=[GTI_LIVING_ROOM_WINDOWS],
            start_time=start_time,
            end_time=end_time,
        )

        series_by_name = {item.name: item for item in series}
        text_by_name = {item.name: item for item in text_series}
        historical_weather_by_name = {item.name: item for item in historical_weather_series}
        forecast_by_name = {item.name: item for item in forecast_series}

        room_temperature = series_by_name.get(
            ROOM_TEMPERATURE,
            NumericSeries(name=ROOM_TEMPERATURE, unit="degC", points=[]),
        )
        if not room_temperature.points:
            raise ValueError("room_temperature series is empty")

        thermal_output = build_space_heating_thermal_output_series(
            series_by_name.get(HP_FLOW),
            series_by_name.get(HP_SUPPLY_TEMPERATURE),
            series_by_name.get(HP_RETURN_TEMPERATURE),
            defrost_active=series_by_name.get(DEFROST_ACTIVE),
            booster_heater_active=series_by_name.get(BOOSTER_HEATER_ACTIVE),
            hp_mode=text_by_name.get(HP_MODE),
        )
        if not thermal_output.points:
            raise ValueError("thermal_output series is empty")
        floor_heat_state = build_floor_heat_state_series(thermal_output)

        # Keep the same state filtering window as the room model so both layers align.
        _ = adjusted_gti_with_shutter(
            upsample_series_forward_fill(
                historical_weather_by_name.get(
                    GTI_LIVING_ROOM_WINDOWS,
                    forecast_by_name.get(
                        GTI_LIVING_ROOM_WINDOWS,
                        NumericSeries(name=GTI_LIVING_ROOM_WINDOWS, unit="Wm2", points=[]),
                    ),
                ),
                start_time=start_time,
                end_time=end_time,
                interval_minutes=interval_minutes,
            ),
            series_by_name.get(
                SHUTTER_LIVING_ROOM,
                NumericSeries(name=SHUTTER_LIVING_ROOM, unit="percent", points=[]),
            ),
        )
        state_filter = RoomTemperatureStateFilter(
            defrost_active=series_by_name.get(DEFROST_ACTIVE),
            booster_heater_active=series_by_name.get(BOOSTER_HEATER_ACTIVE),
            hp_mode=text_by_name.get(HP_MODE),
        )

        raw_rows = self._build_raw_rows(
            room_temperature=room_temperature,
            outdoor_temperature=series_by_name.get(
                OUTDOOR_TEMPERATURE,
                NumericSeries(name=OUTDOOR_TEMPERATURE, unit="degC", points=[]),
            ),
            thermostat_setpoint=series_by_name.get(
                THERMOSTAT_SETPOINT,
                NumericSeries(name=THERMOSTAT_SETPOINT, unit="degC", points=[]),
            ),
            supply_target_temperature=series_by_name.get(
                HP_SUPPLY_TARGET_TEMPERATURE,
                NumericSeries(name=HP_SUPPLY_TARGET_TEMPERATURE, unit="degC", points=[]),
            ),
            thermal_output=thermal_output,
            floor_heat_state=floor_heat_state,
            state_filter=state_filter,
        )
        frame = pd.DataFrame(raw_rows)
        if frame.empty:
            raise ValueError("identification dataset is empty after alignment")

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp").sort_index()

        resampled = frame.resample(f"{interval_minutes}min").agg(
            {
                OUTDOOR_TEMPERATURE: "mean",
                "heating_demand": "mean",
                THERMAL_OUTPUT: "mean",
                FLOOR_HEAT_STATE: "last",
                HP_SUPPLY_TARGET_TEMPERATURE: "mean",
            }
        )
        resampled["previous_thermal_output"] = resampled[THERMAL_OUTPUT].shift(1)
        resampled["previous_heating_demand"] = resampled["heating_demand"].shift(1)
        resampled[f"previous_{FLOOR_HEAT_STATE}"] = resampled[FLOOR_HEAT_STATE].shift(1)
        resampled = resampled.dropna()

        if len(resampled) < 3:
            raise ValueError("not enough aligned samples to identify a model")

        return IdentificationDataset(
            timestamps=[timestamp.isoformat() for timestamp in resampled.index.to_pydatetime()],
            feature_names=THERMAL_OUTPUT_FEATURE_NAMES,
            target_name=THERMAL_OUTPUT,
            features=resampled[THERMAL_OUTPUT_FEATURE_NAMES].to_numpy(dtype=float).tolist(),
            targets=resampled[THERMAL_OUTPUT].to_numpy(dtype=float).tolist(),
        )

    def _build_raw_rows(
        self,
        *,
        room_temperature: NumericSeries,
        outdoor_temperature: NumericSeries,
        thermostat_setpoint: NumericSeries,
        supply_target_temperature: NumericSeries,
        thermal_output: NumericSeries,
        floor_heat_state: NumericSeries,
        state_filter: RoomTemperatureStateFilter,
    ) -> list[dict[str, float | str]]:
        raw_rows: list[dict[str, float | str]] = []
        for thermal_point in thermal_output.points:
            timestamp = parse_datetime(thermal_point.timestamp)
            if not state_filter.is_valid(timestamp):
                continue

            room_temperature_value = latest_value_at(room_temperature.points, thermal_point.timestamp)
            outdoor_value = latest_value_at(outdoor_temperature.points, thermal_point.timestamp)
            setpoint_value = latest_value_at(thermostat_setpoint.points, thermal_point.timestamp)
            supply_target_value = latest_value_at(
                supply_target_temperature.points,
                thermal_point.timestamp,
            )
            floor_heat_state_value = latest_value_at(floor_heat_state.points, thermal_point.timestamp)
            if None in (
                room_temperature_value,
                outdoor_value,
                setpoint_value,
                supply_target_value,
                floor_heat_state_value,
            ):
                continue

            heating_demand = max(float(setpoint_value) - float(room_temperature_value), 0.0)
            raw_rows.append(
                {
                    "timestamp": thermal_point.timestamp,
                    OUTDOOR_TEMPERATURE: float(outdoor_value),
                    "heating_demand": heating_demand,
                    FLOOR_HEAT_STATE: float(floor_heat_state_value),
                    HP_SUPPLY_TARGET_TEMPERATURE: float(supply_target_value),
                    THERMAL_OUTPUT: thermal_point.value,
                }
            )
        return raw_rows
