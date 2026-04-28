from __future__ import annotations

from datetime import datetime, timedelta

from home_optimizer.domain import (
    FORECAST_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS,
    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
    NumericPoint,
    NumericSeries,
    OUTDOOR_TEMPERATURE,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
    adjusted_gti_with_shutter,
    latest_value_at,
)

from .ports import BuildingTemperatureModelReader, PredictionDataReader
from .schemas import BuildingTemperaturePrediction


class BuildingTemperaturePredictionService:
    def __init__(
        self,
        reader: PredictionDataReader,
        model_repository: BuildingTemperatureModelReader,
    ) -> None:
        self.reader = reader
        self.model_repository = model_repository

    def predict(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        thermostat_schedule: NumericSeries,
        shutter_schedule: NumericSeries | None = None,
    ) -> BuildingTemperaturePrediction:
        if end_time <= start_time:
            raise ValueError("end_time must be later than start_time")

        model = self.model_repository.latest()
        if model is None:
            raise ValueError("no stored building temperature model available")

        required_coefficients = {
            "previous_room_temperature",
            OUTDOOR_TEMPERATURE,
            THERMOSTAT_SETPOINT,
            GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
        }
        if not required_coefficients.issubset(model.coefficients):
            raise ValueError("stored building temperature model is missing prediction coefficients")

        interval = timedelta(minutes=model.interval_minutes)
        current_room_temperature = self._read_initial_room_temperature(
            start_time=start_time,
            interval=interval,
        )

        forecast_series = self.reader.read_forecast_series(
            names=[FORECAST_TEMPERATURE, GTI_LIVING_ROOM_WINDOWS],
            start_time=start_time,
            end_time=end_time + interval,
        )
        forecast_by_name = {item.name: item for item in forecast_series}
        outdoor_forecast = forecast_by_name.get(
            FORECAST_TEMPERATURE,
            NumericSeries(name=FORECAST_TEMPERATURE, unit="degC", points=[]),
        )
        adjusted_gti = adjusted_gti_with_shutter(
            forecast_by_name.get(
                GTI_LIVING_ROOM_WINDOWS,
                NumericSeries(name=GTI_LIVING_ROOM_WINDOWS, unit="Wm2", points=[]),
            ),
            shutter_schedule
            or NumericSeries(name=SHUTTER_LIVING_ROOM, unit="percent", points=[]),
        )

        prediction_points: list[NumericPoint] = []
        timestamp = start_time + interval
        while timestamp <= end_time:
            timestamp_iso = timestamp.isoformat()
            outdoor_temperature = latest_value_at(outdoor_forecast.points, timestamp_iso)
            thermostat_setpoint = latest_value_at(thermostat_schedule.points, timestamp_iso)
            solar_gain = latest_value_at(adjusted_gti.points, timestamp_iso)

            if None in (
                outdoor_temperature,
                thermostat_setpoint,
                solar_gain,
            ):
                raise ValueError(
                    f"missing prediction inputs at {timestamp_iso}; provide schedules and forecast coverage"
                )

            current_room_temperature = (
                model.intercept
                + model.coefficients["previous_room_temperature"] * current_room_temperature
                + model.coefficients[OUTDOOR_TEMPERATURE] * float(outdoor_temperature)
                + model.coefficients[THERMOSTAT_SETPOINT] * float(thermostat_setpoint)
                + model.coefficients[GTI_LIVING_ROOM_WINDOWS_ADJUSTED] * float(solar_gain)
            )
            prediction_points.append(
                NumericPoint(timestamp=timestamp_iso, value=current_room_temperature)
            )
            timestamp += interval

        return BuildingTemperaturePrediction(
            model_name=model.model_name,
            interval_minutes=model.interval_minutes,
            target_name=model.target_name,
            room_temperature=NumericSeries(
                name=ROOM_TEMPERATURE,
                unit="degC",
                points=prediction_points,
            ),
        )

    def _read_initial_room_temperature(
        self,
        *,
        start_time: datetime,
        interval: timedelta,
    ) -> float:
        room_series = self.reader.read_series(
            names=[ROOM_TEMPERATURE],
            start_time=start_time - (interval * 2),
            end_time=start_time + interval,
        )
        room_temperature = next(
            iter(room_series),
            NumericSeries(name=ROOM_TEMPERATURE, unit="degC", points=[]),
        )
        current_value = latest_value_at(room_temperature.points, start_time.isoformat())
        current_value = 20
        if current_value is None:
            raise ValueError("no room temperature available near prediction start_time")
        return float(current_value)
