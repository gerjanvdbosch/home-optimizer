from __future__ import annotations

from datetime import timedelta

from home_optimizer.domain import (
    NumericPoint,
    NumericSeries,
    THERMOSTAT_SETPOINT,
    normalize_utc_timestamp,
)
from home_optimizer.domain.control import ShutterPositionControl
from home_optimizer.features.mpc.control_oriented import (
    StateSpaceSetpointMpcPlanRequest,
    StateSpaceSetpointMpcService,
    StateSpaceSetpointPredictionRequest,
    StateSpaceSetpointPredictionService,
    StateSpaceThermalModel,
    StateSpaceThermalState,
)

from .schemas import (
    ThermostatSetpointMpcEvaluationResult,
    ThermostatSetpointMpcPlanRequest,
    ThermostatSetpointPlanEvaluation,
)


class ThermostatSetpointMpcPlanner:
    COMFORT_UNDERSHOOT_PENALTY = 1.0
    COMFORT_OVERSHOOT_PENALTY = 2.0
    THERMAL_OUTPUT_USAGE_PENALTY = 0.01
    THERMAL_OUTPUT_CHANGE_PENALTY = 0.01
    THERMAL_OUTPUT_TRACKING_PENALTY = 0.05

    def __init__(
        self,
        *,
        prediction_service,
        setpoint_mpc_service: StateSpaceSetpointMpcService | None = None,
        setpoint_prediction_service: StateSpaceSetpointPredictionService | None = None,
    ) -> None:
        self.prediction_service = prediction_service
        self.setpoint_mpc_service = setpoint_mpc_service or StateSpaceSetpointMpcService()
        self.setpoint_prediction_service = (
            setpoint_prediction_service or StateSpaceSetpointPredictionService()
        )

    def propose_plan(
        self,
        request: ThermostatSetpointMpcPlanRequest,
        *,
        shutter_position: ShutterPositionControl | None = None,
    ) -> ThermostatSetpointMpcEvaluationResult:
        context = self.prediction_service.prepare_prediction_context(
            start_time=request.start_time,
            end_time=request.end_time,
            shutter_position=shutter_position,
        )
        if context.thermal_output_model is None:
            raise ValueError("no stored thermal output model available")

        thermal_model = StateSpaceThermalModel.from_identified_model(context.model)
        optimization_result = self.setpoint_mpc_service.optimize(
            thermal_model=thermal_model,
            thermal_output_model=context.thermal_output_model,
            request=StateSpaceSetpointMpcPlanRequest(
                start_time=request.start_time,
                end_time=request.end_time,
                initial_state=StateSpaceThermalState(
                    room_temperature=context.initial_room_temperature,
                    floor_heat_state=context.initial_floor_heat_state,
                ),
                initial_thermal_output=context.initial_thermal_output,
                allowed_setpoints=request.allowed_setpoints,
                allowed_thermal_outputs=self._estimate_allowed_thermal_outputs(
                    thermal_model=thermal_model,
                    thermal_output_model=context.thermal_output_model,
                    request=request,
                    context=context,
                ),
                move_block_times=request.switch_times,
                outdoor_temperature_series=context.outdoor_forecast,
                solar_gain_series=context.adjusted_gti,
                supply_target_temperature_series=context.supply_target_temperature_series,
                comfort_min_temperature=request.comfort_min_temperature,
                comfort_max_temperature=request.comfort_max_temperature,
                comfort_undershoot_penalty=self.COMFORT_UNDERSHOOT_PENALTY,
                comfort_overshoot_penalty=self.COMFORT_OVERSHOOT_PENALTY,
                thermal_output_usage_penalty=self.THERMAL_OUTPUT_USAGE_PENALTY,
                thermal_output_change_penalty=self.THERMAL_OUTPUT_CHANGE_PENALTY,
                thermal_output_tracking_penalty=self.THERMAL_OUTPUT_TRACKING_PENALTY,
                setpoint_change_penalty=request.setpoint_change_penalty,
                previous_applied_thermal_output=context.initial_thermal_output,
                previous_applied_setpoint=request.previous_applied_setpoint,
            ),
        )

        recommended = optimization_result.recommended_plan
        mapped_plan = ThermostatSetpointPlanEvaluation(
            plan_name=recommended.plan_name,
            thermostat_setpoint_schedule=recommended.thermostat_setpoint_schedule,
            predicted_room_temperature=recommended.predicted_room_temperature,
            total_cost=recommended.total_cost,
            comfort_violation_cost=recommended.comfort_violation_cost,
            setpoint_change_cost=recommended.setpoint_change_cost,
            minimum_predicted_temperature=recommended.minimum_predicted_temperature,
            maximum_predicted_temperature=recommended.maximum_predicted_temperature,
        )
        return ThermostatSetpointMpcEvaluationResult(
            model_name=optimization_result.model_name,
            interval_minutes=optimization_result.interval_minutes,
            plan_results=[mapped_plan],
            recommended_plan=mapped_plan,
        )

    def _estimate_allowed_thermal_outputs(
        self,
        *,
        thermal_model: StateSpaceThermalModel,
        thermal_output_model,
        request: ThermostatSetpointMpcPlanRequest,
        context,
    ) -> list[float]:
        if not request.allowed_setpoints:
            raise ValueError("allowed_setpoints must not be empty")
        max_setpoint = max(request.allowed_setpoints)
        schedule = self._constant_setpoint_schedule(
            start_time=request.start_time,
            end_time=request.end_time,
            interval_minutes=thermal_model.interval_minutes,
            value=max_setpoint,
        )
        prediction = self.setpoint_prediction_service.predict(
            thermal_model=thermal_model,
            thermal_output_model=thermal_output_model,
            request=StateSpaceSetpointPredictionRequest(
                start_time=request.start_time,
                end_time=request.end_time,
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
        max_output = max(
            [point.value for point in prediction.thermal_output.points] + [context.initial_thermal_output, 0.0]
        )
        return [0.0, max_output]

    @staticmethod
    def _constant_setpoint_schedule(
        *,
        start_time,
        end_time,
        interval_minutes: int,
        value: float,
    ) -> NumericSeries:
        interval = timedelta(minutes=interval_minutes)
        points: list[NumericPoint] = []
        cursor = start_time + interval
        while cursor <= end_time:
            points.append(
                NumericPoint(
                    timestamp=normalize_utc_timestamp(cursor),
                    value=float(value),
                )
            )
            cursor += interval
        return NumericSeries(name=THERMOSTAT_SETPOINT, unit="degC", points=points)
