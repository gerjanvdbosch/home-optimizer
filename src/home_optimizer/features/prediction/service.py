from __future__ import annotations

from datetime import datetime, timedelta

from home_optimizer.domain import (
    DEFAULT_FLOOR_HEAT_STATE_ALPHA,
    FLOOR_HEAT_STATE,
    FORECAST_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS,
    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
    HP_FLOW,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    NumericPoint,
    NumericSeries,
    OUTDOOR_TEMPERATURE,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    adjusted_gti_with_shutter,
    build_floor_heat_state_series,
    build_thermal_output_series,
    latest_value_at,
    normalize_utc_timestamp,
)
from home_optimizer.features.identification.room_temperature.model import (
    FLOOR_HEAT_STATE_FEATURE_NAME,
    MODEL_KIND,
    MODEL_NAME,
)
from home_optimizer.features.identification.thermal_output.model import (
    MODEL_KIND as THERMAL_OUTPUT_MODEL_KIND,
)

from .ports import IdentifiedModelReader, PredictionDataReader
from .schemas import (
    RoomTemperatureControlInputs,
    RoomTemperaturePrediction,
    RoomTemperaturePredictionComparison,
)


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
        control_inputs: RoomTemperatureControlInputs,
        model_name: str = MODEL_NAME,
    ) -> RoomTemperaturePrediction:
        if end_time <= start_time:
            raise ValueError("end_time must be later than start_time")

        model = self.model_repository.latest(model_kind=MODEL_KIND, model_name=model_name)
        if model is None:
            raise ValueError("no stored room temperature model available")

        required_coefficients = {
            "previous_room_temperature",
            OUTDOOR_TEMPERATURE,
            GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
            FLOOR_HEAT_STATE_FEATURE_NAME,
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
            (
                control_inputs.shutter_position.schedule
                if control_inputs.shutter_position is not None
                else NumericSeries(name=SHUTTER_LIVING_ROOM, unit="percent", points=[])
            ),
        )
        historical_floor_heat_state = self._read_historical_floor_heat_state_series(
            start_time=start_time - timedelta(days=1),
            end_time=start_time,
        )
        floor_heat_state_value = latest_value_at(
            historical_floor_heat_state.points,
            normalize_utc_timestamp(start_time),
        )
        if floor_heat_state_value is None:
            floor_heat_state_value = 0.0
        thermal_output_model = self.model_repository.latest(model_kind=THERMAL_OUTPUT_MODEL_KIND)
        previous_thermal_output = 0.0
        if thermal_output_model is not None:
            previous_thermal_output = self._read_initial_thermal_output(start_time=start_time)

        prediction_points: list[NumericPoint] = []
        timestamp = start_time + interval
        while timestamp <= end_time:
            timestamp_iso = normalize_utc_timestamp(timestamp)
            previous_timestamp_iso = normalize_utc_timestamp(timestamp - interval)
            outdoor_temperature = latest_value_at(outdoor_forecast.points, timestamp_iso)
            solar_gain = latest_value_at(adjusted_gti.points, timestamp_iso)
            thermostat_setpoint = latest_value_at(
                control_inputs.thermostat_setpoint.schedule.points,
                previous_timestamp_iso,
            )

            if None in (
                outdoor_temperature,
                solar_gain,
                thermostat_setpoint,
            ):
                raise ValueError(
                    "missing prediction inputs at "
                    f"{normalize_utc_timestamp(timestamp)}; provide schedules and forecast coverage"
                )

            if thermal_output_model is not None:
                predicted_thermal_output = self._predict_thermal_output_step(
                    model=thermal_output_model,
                    previous_thermal_output=previous_thermal_output,
                    previous_room_temperature=current_room_temperature,
                    thermostat_setpoint=float(thermostat_setpoint),
                    outdoor_temperature=float(outdoor_temperature),
                )
                previous_thermal_output = predicted_thermal_output
                floor_heat_state_value = (
                    DEFAULT_FLOOR_HEAT_STATE_ALPHA * float(floor_heat_state_value)
                    + (1.0 - DEFAULT_FLOOR_HEAT_STATE_ALPHA) * predicted_thermal_output
                )

            current_room_temperature = (
                model.intercept
                + model.coefficients["previous_room_temperature"] * current_room_temperature
                + model.coefficients[OUTDOOR_TEMPERATURE] * float(outdoor_temperature)
                + model.coefficients[GTI_LIVING_ROOM_WINDOWS_ADJUSTED] * float(solar_gain)
                + model.coefficients[FLOOR_HEAT_STATE_FEATURE_NAME] * float(floor_heat_state_value)
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
        control_inputs: RoomTemperatureControlInputs,
        model_name: str = MODEL_NAME,
    ) -> RoomTemperaturePredictionComparison:
        prediction = self.predict(
            start_time=start_time,
            end_time=end_time,
            control_inputs=control_inputs,
            model_name=model_name,
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

    def _predict_thermal_output_step(
        self,
        *,
        model,
        previous_thermal_output: float,
        previous_room_temperature: float,
        thermostat_setpoint: float,
        outdoor_temperature: float,
    ) -> float:
        required_coefficients = {
            "previous_thermal_output",
            "previous_heating_demand",
            OUTDOOR_TEMPERATURE,
        }
        if not required_coefficients.issubset(model.coefficients):
            raise ValueError("stored thermal output model is missing prediction coefficients")
        previous_heating_demand = max(thermostat_setpoint - previous_room_temperature, 0.0)
        return max(
            0.0,
            model.intercept
            + model.coefficients["previous_thermal_output"] * previous_thermal_output
            + model.coefficients["previous_heating_demand"] * previous_heating_demand
            + model.coefficients[OUTDOOR_TEMPERATURE] * outdoor_temperature,
        )

    def _read_initial_thermal_output(
        self,
        *,
        start_time: datetime,
    ) -> float:
        historical_thermal_output = self._read_historical_thermal_output_series(
            start_time=start_time - timedelta(days=1),
            end_time=start_time,
        )
        value = latest_value_at(
            historical_thermal_output.points,
            normalize_utc_timestamp(start_time),
        )
        return float(value) if value is not None else 0.0

    def _read_historical_floor_heat_state_series(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> NumericSeries:
        return build_floor_heat_state_series(
            self._read_historical_thermal_output_series(start_time=start_time, end_time=end_time),
            name=FLOOR_HEAT_STATE,
        )

    def _read_historical_thermal_output_series(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
    ) -> NumericSeries:
        source_series = self.reader.read_series(
            names=[HP_FLOW, HP_SUPPLY_TEMPERATURE, HP_RETURN_TEMPERATURE],
            start_time=start_time,
            end_time=end_time,
        )
        by_name = {series.name: series for series in source_series}
        return build_thermal_output_series(
            by_name.get(HP_FLOW),
            by_name.get(HP_SUPPLY_TEMPERATURE),
            by_name.get(HP_RETURN_TEMPERATURE),
        )
