from __future__ import annotations

from datetime import datetime, timedelta

from home_optimizer.domain import (
    NumericPoint,
    NumericSeries,
    THERMOSTAT_SETPOINT,
    latest_value_at,
    normalize_utc_timestamp,
)
from home_optimizer.features.prediction.service import RoomTemperaturePredictionService

from .model import StateSpaceThermalModel, StateSpaceThermalState
from .schemas import (
    StateSpaceActuatorSensitivityResult,
    StateSpaceActuatorSensitivityRow,
    StateSpaceSetpointPredictionRequest,
)
from .service import StateSpaceSetpointPredictionService


class StateSpaceActuatorSensitivityService:
    def __init__(
        self,
        *,
        prediction_service: RoomTemperaturePredictionService,
        setpoint_prediction_service: StateSpaceSetpointPredictionService | None = None,
    ) -> None:
        self.prediction_service = prediction_service
        self.setpoint_prediction_service = (
            setpoint_prediction_service or StateSpaceSetpointPredictionService()
        )

    def inspect(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        setpoints: list[float],
        shutter_position=None,
        model_name: str = "linear_2state_room_temperature",
    ) -> StateSpaceActuatorSensitivityResult:
        if end_time <= start_time:
            raise ValueError("end_time must be later than start_time")
        if not setpoints:
            raise ValueError("setpoints must not be empty")

        context = self.prediction_service.prepare_prediction_context(
            start_time=start_time,
            end_time=end_time,
            shutter_position=shutter_position,
            model_name=model_name,
        )
        if context.thermal_output_model is None:
            raise ValueError("no stored thermal output model available")

        thermal_model = StateSpaceThermalModel.from_identified_model(context.model)
        interval = timedelta(minutes=thermal_model.interval_minutes)
        first_prediction_timestamp = normalize_utc_timestamp(start_time + interval)
        first_supply_target_temperature = latest_value_at(
            context.supply_target_temperature_series.points,
            first_prediction_timestamp,
        )

        rows: list[StateSpaceActuatorSensitivityRow] = []
        for setpoint in sorted(set(float(value) for value in setpoints)):
            schedule = self._constant_setpoint_schedule(
                start_time=start_time,
                end_time=end_time,
                interval_minutes=thermal_model.interval_minutes,
                value=setpoint,
            )
            prediction = self.setpoint_prediction_service.predict(
                thermal_model=thermal_model,
                thermal_output_model=context.thermal_output_model,
                request=StateSpaceSetpointPredictionRequest(
                    start_time=start_time,
                    end_time=end_time,
                    initial_state=StateSpaceThermalState(
                        room_temperature=context.initial_room_temperature,
                        floor_heat_state=context.initial_floor_heat_state,
                    ),
                    initial_thermal_output=context.initial_thermal_output,
                    thermostat_setpoint_schedule=schedule,
                    outdoor_temperature_series=context.outdoor_forecast,
                    solar_gain_series=context.adjusted_gti,
                    supply_target_temperature_series=context.supply_target_temperature_series,
                ),
            )
            thermal_outputs = [point.value for point in prediction.thermal_output.points]
            room_temperatures = [point.value for point in prediction.room_temperature.points]
            rows.append(
                StateSpaceActuatorSensitivityRow(
                    thermostat_setpoint=setpoint,
                    initial_heating_demand=max(
                        setpoint - context.initial_room_temperature,
                        0.0,
                    ),
                    first_predicted_thermal_output=(
                        thermal_outputs[0] if thermal_outputs else None
                    ),
                    peak_predicted_thermal_output=max(thermal_outputs) if thermal_outputs else None,
                    average_predicted_thermal_output=(
                        sum(thermal_outputs) / len(thermal_outputs) if thermal_outputs else None
                    ),
                    final_room_temperature=room_temperatures[-1] if room_temperatures else None,
                    maximum_room_temperature=max(room_temperatures) if room_temperatures else None,
                )
            )

        return StateSpaceActuatorSensitivityResult(
            model_name=context.model.model_name,
            thermal_output_model_name=context.thermal_output_model.model_name,
            interval_minutes=thermal_model.interval_minutes,
            start_time=start_time,
            end_time=end_time,
            initial_room_temperature=context.initial_room_temperature,
            initial_floor_heat_state=context.initial_floor_heat_state,
            initial_thermal_output=context.initial_thermal_output,
            first_supply_target_temperature=(
                float(first_supply_target_temperature)
                if first_supply_target_temperature is not None
                else None
            ),
            rows=rows,
        )

    @staticmethod
    def _constant_setpoint_schedule(
        *,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
        value: float,
    ) -> NumericSeries:
        interval = timedelta(minutes=interval_minutes)
        points: list[NumericPoint] = []
        cursor = start_time
        while cursor <= end_time:
            points.append(
                NumericPoint(
                    timestamp=normalize_utc_timestamp(cursor),
                    value=float(value),
                )
            )
            cursor += interval
        return NumericSeries(name=THERMOSTAT_SETPOINT, unit="degC", points=points)
