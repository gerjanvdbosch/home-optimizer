from __future__ import annotations

from datetime import datetime

import pandas as pd

from home_optimizer.domain import (
    BOOSTER_HEATER_ACTIVE,
    DEFROST_ACTIVE,
    GTI_LIVING_ROOM_WINDOWS,
    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
    HP_MODE,
    NumericSeries,
    OUTDOOR_TEMPERATURE,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
    adjusted_gti_with_shutter,
    latest_value_at,
    upsample_series_forward_fill,
)
from home_optimizer.domain.time import parse_datetime

from ..ports import IdentificationDataReader
from ..schemas import IdentificationDataset
from .state_filter import RoomTemperatureStateFilter

ROOM_TEMPERATURE_FEATURE_NAMES = [
    "previous_room_temperature",
    OUTDOOR_TEMPERATURE,
    "previous_thermostat_setpoint",
    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
]


class RoomTemperatureDatasetBuilder:
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

        room_temperature = series_by_name[ROOM_TEMPERATURE]
        if not room_temperature.points:
            raise ValueError("room_temperature series is empty")

        adjusted_gti = adjusted_gti_with_shutter(
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
            adjusted_gti=adjusted_gti,
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
                THERMOSTAT_SETPOINT: "last",
                GTI_LIVING_ROOM_WINDOWS_ADJUSTED: "mean",
                ROOM_TEMPERATURE: "last",
            }
        )
        resampled["previous_room_temperature"] = resampled[ROOM_TEMPERATURE].shift(1)
        resampled["previous_thermostat_setpoint"] = resampled[THERMOSTAT_SETPOINT].shift(1)
        resampled = resampled.dropna()

        if len(resampled) < 3:
            raise ValueError("not enough aligned samples to identify a model")

        return IdentificationDataset(
            timestamps=[timestamp.isoformat() for timestamp in resampled.index.to_pydatetime()],
            feature_names=ROOM_TEMPERATURE_FEATURE_NAMES,
            target_name=ROOM_TEMPERATURE,
            features=resampled[ROOM_TEMPERATURE_FEATURE_NAMES].to_numpy(dtype=float).tolist(),
            targets=resampled[ROOM_TEMPERATURE].to_numpy(dtype=float).tolist(),
        )

    def _build_raw_rows(
        self,
        *,
        room_temperature: NumericSeries,
        outdoor_temperature: NumericSeries,
        thermostat_setpoint: NumericSeries,
        adjusted_gti: NumericSeries,
        state_filter: RoomTemperatureStateFilter,
    ) -> list[dict[str, float | str]]:
        raw_rows: list[dict[str, float | str]] = []
        for room_point in room_temperature.points:
            if not state_filter.is_valid(parse_datetime(room_point.timestamp)):
                continue

            outdoor_value = latest_value_at(outdoor_temperature.points, room_point.timestamp)
            setpoint_value = latest_value_at(thermostat_setpoint.points, room_point.timestamp)
            solar_gain = latest_value_at(adjusted_gti.points, room_point.timestamp)

            if None in (outdoor_value, setpoint_value, solar_gain):
                continue

            raw_rows.append(
                {
                    "timestamp": room_point.timestamp,
                    OUTDOOR_TEMPERATURE: float(outdoor_value),
                    THERMOSTAT_SETPOINT: float(setpoint_value),
                    GTI_LIVING_ROOM_WINDOWS_ADJUSTED: float(solar_gain),
                    ROOM_TEMPERATURE: room_point.value,
                }
            )
        return raw_rows
