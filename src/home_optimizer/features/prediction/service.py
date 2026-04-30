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
    normalize_utc_timestamp,
)
from home_optimizer.features.identification.room_temperature.model import MODEL_KIND

from .ports import IdentifiedModelReader, PredictionDataReader
from .schemas import RoomTemperaturePrediction, RoomTemperaturePredictionComparison


class RoomTemperaturePredictionService:
    def __init__(
        self,
        reader: PredictionDataReader,
        model_repository: IdentifiedModelReader,
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
    ) -> RoomTemperaturePrediction:
        if end_time <= start_time:
            raise ValueError("end_time must be later than start_time")

        model = self.model_repository.latest(model_kind=MODEL_KIND)
        if model is None:
            raise ValueError("no stored room temperature model available")

        required_coefficients = {
            "previous_room_temperature",
            "previous_thermostat_setpoint",
            OUTDOOR_TEMPERATURE,
            GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
        }
        if not required_coefficients.issubset(model.coefficients):
            raise ValueError("stored room temperature model is missing prediction coefficients")

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
            timestamp_iso = normalize_utc_timestamp(timestamp)
            previous_timestamp_iso = normalize_utc_timestamp(timestamp - interval)
            outdoor_temperature = latest_value_at(outdoor_forecast.points, timestamp_iso)
            thermostat_setpoint = latest_value_at(
                thermostat_schedule.points,
                previous_timestamp_iso,
            )
            solar_gain = latest_value_at(adjusted_gti.points, timestamp_iso)

            if None in (
                outdoor_temperature,
                thermostat_setpoint,
                solar_gain,
            ):
                raise ValueError(
                    "missing prediction inputs at "
                    f"{normalize_utc_timestamp(timestamp)}; provide schedules and forecast coverage"
                )

            current_room_temperature = (
                model.intercept
                + model.coefficients["previous_room_temperature"] * current_room_temperature
                + model.coefficients[OUTDOOR_TEMPERATURE] * float(outdoor_temperature)
                + model.coefficients["previous_thermostat_setpoint"] * float(thermostat_setpoint)
                + model.coefficients[GTI_LIVING_ROOM_WINDOWS_ADJUSTED] * float(solar_gain)
            )
            prediction_points.append(
                NumericPoint(timestamp=timestamp_iso, value=current_room_temperature)
            )
            timestamp += interval

        return RoomTemperaturePrediction(
            model_name=model.model_name,
            interval_minutes=model.interval_minutes,
            target_name=model.target_name,
            room_temperature=NumericSeries(
                name=ROOM_TEMPERATURE,
                unit="degC",
                points=prediction_points,
            ),
        )

    def predict_vs_actual(
        self,
        start_time: datetime,
        end_time: datetime,
        *,
        thermostat_schedule: NumericSeries,
        shutter_schedule: NumericSeries | None = None,
    ) -> RoomTemperaturePredictionComparison:
        prediction = self.predict(
            start_time=start_time,
            end_time=end_time,
            thermostat_schedule=thermostat_schedule,
            shutter_schedule=shutter_schedule,
        )
        actual_series = self.reader.read_series(
            names=[ROOM_TEMPERATURE],
            start_time=start_time,
            end_time=end_time,
        )
        actual_room_temperature_raw = next(
            iter(actual_series),
            NumericSeries(name=ROOM_TEMPERATURE, unit="degC", points=[]),
        )
        actual_room_temperature = NumericSeries(
            name=actual_room_temperature_raw.name,
            unit=actual_room_temperature_raw.unit,
            points=[
                NumericPoint(
                    timestamp=point.timestamp,
                    value=float(
                        latest_value_at(actual_room_temperature_raw.points, point.timestamp)
                    ),
                )
                for point in prediction.room_temperature.points
                if latest_value_at(actual_room_temperature_raw.points, point.timestamp) is not None
            ],
        )
        return RoomTemperaturePredictionComparison(
            model_name=prediction.model_name,
            interval_minutes=prediction.interval_minutes,
            target_name=prediction.target_name,
            predicted_room_temperature=prediction.room_temperature,
            actual_room_temperature=actual_room_temperature,
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
        current_value = latest_value_at(
            room_temperature.points,
            normalize_utc_timestamp(start_time),
        )
        if current_value is None:
            raise ValueError("no room temperature available near prediction start_time")
        return float(current_value)
