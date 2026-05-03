from __future__ import annotations

from datetime import timedelta
from dataclasses import dataclass

from home_optimizer.domain import (
    HP_SUPPLY_TARGET_TEMPERATURE,
    FLOOR_HEAT_STATE,
    ROOM_TEMPERATURE,
    THERMAL_OUTPUT,
    THERMOSTAT_SETPOINT,
    NumericPoint,
    NumericSeries,
    latest_value_at,
    normalize_utc_timestamp,
)
from home_optimizer.features.identification.thermal_output.model import predict_thermal_output

from .model import (
    StateSpaceThermalControlInput,
    StateSpaceThermalDisturbance,
    StateSpaceThermalModel,
)
from .schemas import (
    OPTIMIZED_CONTROL_PLAN_NAME,
    StateSpaceThermalMpcPlanRequest,
    StateSpaceThermalMpcPlanResult,
    StateSpaceThermalPlanEvaluation,
    StateSpaceThermalPredictionRequest,
    StateSpaceThermalPredictionResult,
    StateSpaceSetpointPredictionRequest,
    StateSpaceSetpointPredictionResult,
    StateSpaceSetpointMpcPlanRequest,
    StateSpaceSetpointMpcPlanResult,
    StateSpaceSetpointPlanEvaluation,
)


class StateSpaceThermalPredictionService:
    def predict(
        self,
        *,
        model: StateSpaceThermalModel,
        request: StateSpaceThermalPredictionRequest,
    ) -> StateSpaceThermalPredictionResult:
        if request.end_time <= request.start_time:
            raise ValueError("end_time must be later than start_time")

        interval = timedelta(minutes=model.interval_minutes)
        timestamps: list[str] = []
        control_inputs: list[StateSpaceThermalControlInput] = []
        disturbances: list[StateSpaceThermalDisturbance] = []

        cursor = request.start_time + interval
        while cursor <= request.end_time:
            timestamp = normalize_utc_timestamp(cursor)
            thermal_output = latest_value_at(
                request.thermal_output_schedule.points,
                timestamp,
            )
            outdoor_temperature = latest_value_at(
                request.outdoor_temperature_series.points,
                timestamp,
            )
            solar_gain = latest_value_at(
                request.solar_gain_series.points,
                timestamp,
            )
            if None in (thermal_output, outdoor_temperature, solar_gain):
                raise ValueError(
                    "missing state-space prediction input at "
                    f"{timestamp}; provide thermal_output, outdoor_temperature and solar_gain coverage"
                )

            timestamps.append(timestamp)
            control_inputs.append(
                StateSpaceThermalControlInput(thermal_output=float(thermal_output))
            )
            disturbances.append(
                StateSpaceThermalDisturbance(
                    outdoor_temperature=float(outdoor_temperature),
                    solar_gain=float(solar_gain),
                )
            )
            cursor += interval

        states = model.simulate(
            initial_state=request.initial_state,
            control_inputs=control_inputs,
            disturbances=disturbances,
        )

        return StateSpaceThermalPredictionResult(
            model_name=model.model_name,
            interval_minutes=model.interval_minutes,
            room_temperature=NumericSeries(
                name=ROOM_TEMPERATURE,
                unit="degC",
                points=[
                    NumericPoint(timestamp=timestamp, value=state.room_temperature)
                    for timestamp, state in zip(timestamps, states, strict=True)
                ],
            ),
            floor_heat_state=NumericSeries(
                name=FLOOR_HEAT_STATE,
                unit=request.thermal_output_schedule.unit,
                points=[
                    NumericPoint(timestamp=timestamp, value=state.floor_heat_state)
                    for timestamp, state in zip(timestamps, states, strict=True)
                ],
            ),
        )


class StateSpaceThermalMpcService:
    def __init__(self, prediction_service: StateSpaceThermalPredictionService | None = None) -> None:
        self.prediction_service = prediction_service or StateSpaceThermalPredictionService()

    def optimize(
        self,
        *,
        model: StateSpaceThermalModel,
        request: StateSpaceThermalMpcPlanRequest,
    ) -> StateSpaceThermalMpcPlanResult:
        if request.end_time <= request.start_time:
            raise ValueError("end_time must be later than start_time")
        if not request.allowed_thermal_outputs:
            raise ValueError("allowed_thermal_outputs must not be empty")
        if request.comfort_min_temperature > request.comfort_max_temperature:
            raise ValueError("comfort_min_temperature must be <= comfort_max_temperature")
        if request.comfort_undershoot_penalty < 0:
            raise ValueError("comfort_undershoot_penalty must be >= 0")
        if request.comfort_overshoot_penalty < 0:
            raise ValueError("comfort_overshoot_penalty must be >= 0")
        if request.thermal_output_usage_penalty < 0:
            raise ValueError("thermal_output_usage_penalty must be >= 0")
        if request.thermal_output_change_penalty < 0:
            raise ValueError("thermal_output_change_penalty must be >= 0")

        lower_bound = min(request.allowed_thermal_outputs)
        upper_bound = max(request.allowed_thermal_outputs)
        block_starts = self._control_block_starts(
            start_time=request.start_time,
            end_time=request.end_time,
            move_block_times=request.move_block_times,
        )
        initial_value = request.previous_applied_thermal_output
        if initial_value is None:
            initial_value = lower_bound
        initial_value = float(min(upper_bound, max(lower_bound, initial_value)))
        block_values = [initial_value for _ in block_starts]

        for _ in range(3):
            changed = False
            for index in range(len(block_values)):
                optimized_value = self._optimize_block_value(
                    model=model,
                    request=request,
                    block_starts=block_starts,
                    block_values=block_values,
                    block_index=index,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
                if abs(optimized_value - block_values[index]) > 1e-3:
                    changed = True
                block_values[index] = optimized_value
            if not changed:
                break

        optimized_plan = self._evaluate_block_values(
            model=model,
            request=request,
            block_starts=block_starts,
            block_values=block_values,
        )
        return StateSpaceThermalMpcPlanResult(
            model_name=model.model_name,
            interval_minutes=model.interval_minutes,
            plan_results=[optimized_plan.evaluation],
            recommended_plan=optimized_plan.evaluation,
        )

    def _optimize_block_value(
        self,
        *,
        model: StateSpaceThermalModel,
        request: StateSpaceThermalMpcPlanRequest,
        block_starts: list,
        block_values: list[float],
        block_index: int,
        lower_bound: float,
        upper_bound: float,
    ) -> float:
        current = block_values[block_index]
        candidates = [lower_bound, current, upper_bound]
        best_value = min(
            candidates,
            key=lambda candidate: self._evaluate_cost_for_value(
                model=model,
                request=request,
                block_starts=block_starts,
                block_values=block_values,
                block_index=block_index,
                candidate_value=candidate,
            ),
        )
        left = lower_bound
        right = upper_bound
        for _ in range(8):
            left_probe = right - (right - left) / 1.61803398875
            right_probe = left + (right - left) / 1.61803398875
            left_cost = self._evaluate_cost_for_value(
                model=model,
                request=request,
                block_starts=block_starts,
                block_values=block_values,
                block_index=block_index,
                candidate_value=left_probe,
            )
            right_cost = self._evaluate_cost_for_value(
                model=model,
                request=request,
                block_starts=block_starts,
                block_values=block_values,
                block_index=block_index,
                candidate_value=right_probe,
            )
            if left_cost <= right_cost:
                right = right_probe
                if left_cost < self._evaluate_cost_for_value(
                    model=model,
                    request=request,
                    block_starts=block_starts,
                    block_values=block_values,
                    block_index=block_index,
                    candidate_value=best_value,
                ):
                    best_value = left_probe
            else:
                left = left_probe
                if right_cost < self._evaluate_cost_for_value(
                    model=model,
                    request=request,
                    block_starts=block_starts,
                    block_values=block_values,
                    block_index=block_index,
                    candidate_value=best_value,
                ):
                    best_value = right_probe
        return float(min(upper_bound, max(lower_bound, best_value)))

    def _evaluate_cost_for_value(
        self,
        *,
        model: StateSpaceThermalModel,
        request: StateSpaceThermalMpcPlanRequest,
        block_starts: list,
        block_values: list[float],
        block_index: int,
        candidate_value: float,
    ) -> float:
        return self._evaluate_block_values(
            model=model,
            request=request,
            block_starts=block_starts,
            block_values=self._updated_block_values(
                block_values,
                block_index,
                candidate_value,
            ),
        ).evaluation.total_cost

    def _evaluate_block_values(
        self,
        *,
        model: StateSpaceThermalModel,
        request: StateSpaceThermalMpcPlanRequest,
        block_starts: list,
        block_values: list[float],
    ) -> "_OptimizedThermalPlan":
        schedule = self._build_schedule(
            start_time=request.start_time,
            end_time=request.end_time,
            interval_minutes=model.interval_minutes,
            block_starts=block_starts,
            block_values=block_values,
        )
        prediction = self.prediction_service.predict(
            model=model,
            request=StateSpaceThermalPredictionRequest(
                start_time=request.start_time,
                end_time=request.end_time,
                initial_state=request.initial_state,
                thermal_output_schedule=schedule,
                outdoor_temperature_series=request.outdoor_temperature_series,
                solar_gain_series=request.solar_gain_series,
            ),
        )
        predicted_values = [point.value for point in prediction.room_temperature.points]
        comfort_cost = self._mean_squared_comfort_violation(
            predicted_values,
            comfort_min_temperature=request.comfort_min_temperature,
            comfort_max_temperature=request.comfort_max_temperature,
            undershoot_penalty=request.comfort_undershoot_penalty,
            overshoot_penalty=request.comfort_overshoot_penalty,
        )
        usage_cost = request.thermal_output_usage_penalty * self._mean_squared_values(
            point.value for point in schedule.points
        )
        change_cost = self._thermal_output_change_cost(
            schedule,
            penalty=request.thermal_output_change_penalty,
            previous_applied_thermal_output=request.previous_applied_thermal_output,
        )
        evaluation = StateSpaceThermalPlanEvaluation(
            plan_name=OPTIMIZED_CONTROL_PLAN_NAME,
            thermal_output_schedule=schedule,
            predicted_room_temperature=prediction.room_temperature,
            predicted_floor_heat_state=prediction.floor_heat_state,
            total_cost=comfort_cost + usage_cost + change_cost,
            comfort_violation_cost=comfort_cost,
            thermal_output_usage_cost=usage_cost,
            thermal_output_change_cost=change_cost,
            minimum_predicted_temperature=min(predicted_values) if predicted_values else None,
            maximum_predicted_temperature=max(predicted_values) if predicted_values else None,
        )
        return _OptimizedThermalPlan(
            model_name=prediction.model_name,
            interval_minutes=prediction.interval_minutes,
            evaluation=evaluation,
        )

    @staticmethod
    def _control_block_starts(*, start_time, end_time, move_block_times) -> list:
        starts = [start_time]
        for timestamp in sorted(move_block_times):
            if start_time < timestamp <= end_time:
                starts.append(timestamp)
        return starts

    @staticmethod
    def _build_schedule(
        *,
        start_time,
        end_time,
        interval_minutes: int,
        block_starts,
        block_values: list[float],
    ) -> NumericSeries:
        interval = timedelta(minutes=interval_minutes)
        points: list[NumericPoint] = []
        cursor = start_time + interval
        while cursor <= end_time:
            active_block_index = 0
            for index, block_start in enumerate(block_starts):
                if block_start <= cursor:
                    active_block_index = index
                else:
                    break
            points.append(
                NumericPoint(
                    timestamp=normalize_utc_timestamp(cursor),
                    value=float(block_values[active_block_index]),
                )
            )
            cursor += interval
        return NumericSeries(name="thermal_output", unit="kW", points=points)

    @staticmethod
    def _updated_block_values(block_values: list[float], block_index: int, candidate_value: float) -> list[float]:
        updated = block_values.copy()
        updated[block_index] = float(candidate_value)
        return updated

    @staticmethod
    def _thermal_output_change_cost(
        schedule: NumericSeries,
        *,
        penalty: float,
        previous_applied_thermal_output: float | None,
    ) -> float:
        if penalty <= 0 or not schedule.points:
            return 0.0
        previous_value = (
            float(previous_applied_thermal_output)
            if previous_applied_thermal_output is not None
            else schedule.points[0].value
        )
        deltas: list[float] = []
        for point in schedule.points:
            deltas.append(point.value - previous_value)
            previous_value = point.value
        return penalty * StateSpaceThermalMpcService._mean_squared_values(deltas)

    @staticmethod
    def _mean_squared_values(values) -> float:
        values_list = [float(value) for value in values]
        if not values_list:
            return 0.0
        return sum(value**2 for value in values_list) / len(values_list)

    @staticmethod
    def _mean_squared_comfort_violation(
        predicted_values: list[float],
        *,
        comfort_min_temperature: float,
        comfort_max_temperature: float,
        undershoot_penalty: float,
        overshoot_penalty: float,
    ) -> float:
        if not predicted_values:
            return 0.0
        violations: list[float] = []
        for value in predicted_values:
            if value < comfort_min_temperature:
                violations.append(
                    (comfort_min_temperature - value) * undershoot_penalty
                )
            elif value > comfort_max_temperature:
                violations.append(
                    (value - comfort_max_temperature) * overshoot_penalty
                )
            else:
                violations.append(0.0)
        return StateSpaceThermalMpcService._mean_squared_values(violations)


@dataclass(frozen=True)
class _OptimizedThermalPlan:
    model_name: str
    interval_minutes: int
    evaluation: StateSpaceThermalPlanEvaluation


class StateSpaceSetpointPredictionService:
    def predict(
        self,
        *,
        thermal_model: StateSpaceThermalModel,
        thermal_output_model,
        request: StateSpaceSetpointPredictionRequest,
    ) -> StateSpaceSetpointPredictionResult:
        required_coefficients = {
            "previous_thermal_output",
            "previous_heating_demand",
            f"previous_{FLOOR_HEAT_STATE}",
            "outdoor_temperature",
            HP_SUPPLY_TARGET_TEMPERATURE,
        }
        if not required_coefficients.issubset(thermal_output_model.coefficients):
            raise ValueError("stored thermal output model is missing prediction coefficients")
        if request.end_time <= request.start_time:
            raise ValueError("end_time must be later than start_time")

        interval = timedelta(minutes=thermal_model.interval_minutes)
        current_state = request.initial_state
        previous_thermal_output = request.initial_thermal_output
        room_points: list[NumericPoint] = []
        floor_points: list[NumericPoint] = []
        thermal_output_points: list[NumericPoint] = []

        cursor = request.start_time + interval
        while cursor <= request.end_time:
            timestamp = normalize_utc_timestamp(cursor)
            thermostat_setpoint = latest_value_at(
                request.thermostat_setpoint_schedule.points,
                timestamp,
            )
            outdoor_temperature = latest_value_at(
                request.outdoor_temperature_series.points,
                timestamp,
            )
            solar_gain = latest_value_at(
                request.solar_gain_series.points,
                timestamp,
            )
            supply_target_temperature = latest_value_at(
                request.supply_target_temperature_series.points,
                timestamp,
            )
            if None in (
                thermostat_setpoint,
                outdoor_temperature,
                solar_gain,
                supply_target_temperature,
            ):
                raise ValueError(
                    "missing setpoint prediction input at "
                    f"{timestamp}; provide setpoint, outdoor_temperature, solar_gain and supply_target coverage"
                )

            previous_heating_demand = max(
                float(thermostat_setpoint) - current_state.room_temperature,
                0.0,
            )
            predicted_thermal_output = predict_thermal_output(
                coefficients=thermal_output_model.coefficients,
                intercept=thermal_output_model.intercept,
                previous_thermal_output=previous_thermal_output,
                previous_heating_demand=previous_heating_demand,
                previous_floor_heat_state=current_state.floor_heat_state,
                outdoor_temperature=float(outdoor_temperature),
                supply_target_temperature=float(supply_target_temperature),
            )
            current_state = thermal_model.step(
                current_state,
                control_input=StateSpaceThermalControlInput(
                    thermal_output=predicted_thermal_output
                ),
                disturbance=StateSpaceThermalDisturbance(
                    outdoor_temperature=float(outdoor_temperature),
                    solar_gain=float(solar_gain),
                ),
            )
            previous_thermal_output = predicted_thermal_output
            room_points.append(
                NumericPoint(timestamp=timestamp, value=current_state.room_temperature)
            )
            floor_points.append(
                NumericPoint(timestamp=timestamp, value=current_state.floor_heat_state)
            )
            thermal_output_points.append(
                NumericPoint(timestamp=timestamp, value=predicted_thermal_output)
            )
            cursor += interval

        return StateSpaceSetpointPredictionResult(
            model_name=thermal_model.model_name,
            interval_minutes=thermal_model.interval_minutes,
            thermal_output_model_name=thermal_output_model.model_name,
            room_temperature=NumericSeries(
                name=ROOM_TEMPERATURE,
                unit="degC",
                points=room_points,
            ),
            floor_heat_state=NumericSeries(
                name=FLOOR_HEAT_STATE,
                unit="kW",
                points=floor_points,
            ),
            thermal_output=NumericSeries(
                name=THERMAL_OUTPUT,
                unit="kW",
                points=thermal_output_points,
            ),
        )


class StateSpaceSetpointMpcService:
    def __init__(
        self,
        *,
        thermal_mpc_service: StateSpaceThermalMpcService | None = None,
        setpoint_prediction_service: StateSpaceSetpointPredictionService | None = None,
    ) -> None:
        self.thermal_mpc_service = thermal_mpc_service or StateSpaceThermalMpcService()
        self.setpoint_prediction_service = (
            setpoint_prediction_service or StateSpaceSetpointPredictionService()
        )

    def optimize(
        self,
        *,
        thermal_model: StateSpaceThermalModel,
        thermal_output_model,
        request: StateSpaceSetpointMpcPlanRequest,
    ) -> StateSpaceSetpointMpcPlanResult:
        if not request.allowed_setpoints:
            raise ValueError("allowed_setpoints must not be empty")
        if request.comfort_undershoot_penalty < 0:
            raise ValueError("comfort_undershoot_penalty must be >= 0")
        if request.comfort_overshoot_penalty < 0:
            raise ValueError("comfort_overshoot_penalty must be >= 0")
        if request.thermal_output_tracking_penalty < 0:
            raise ValueError("thermal_output_tracking_penalty must be >= 0")
        if request.setpoint_change_penalty < 0:
            raise ValueError("setpoint_change_penalty must be >= 0")

        thermal_plan = self.thermal_mpc_service.optimize(
            model=thermal_model,
            request=StateSpaceThermalMpcPlanRequest(
                start_time=request.start_time,
                end_time=request.end_time,
                initial_state=request.initial_state,
                allowed_thermal_outputs=request.allowed_thermal_outputs,
                move_block_times=request.move_block_times,
                outdoor_temperature_series=request.outdoor_temperature_series,
                solar_gain_series=request.solar_gain_series,
                comfort_min_temperature=request.comfort_min_temperature,
                comfort_max_temperature=request.comfort_max_temperature,
                comfort_undershoot_penalty=request.comfort_undershoot_penalty,
                comfort_overshoot_penalty=request.comfort_overshoot_penalty,
                thermal_output_usage_penalty=request.thermal_output_usage_penalty,
                thermal_output_change_penalty=request.thermal_output_change_penalty,
                previous_applied_thermal_output=request.previous_applied_thermal_output,
            ),
        )
        target_thermal_output_schedule = thermal_plan.recommended_plan.thermal_output_schedule

        block_starts = self.thermal_mpc_service._control_block_starts(
            start_time=request.start_time,
            end_time=request.end_time,
            move_block_times=request.move_block_times,
        )
        initial_value = (
            request.previous_applied_setpoint
            if request.previous_applied_setpoint is not None
            else min(request.allowed_setpoints)
        )
        block_values = [float(initial_value) for _ in block_starts]

        for _ in range(3):
            changed = False
            for index in range(len(block_values)):
                optimized_value = self._optimize_block_value(
                    thermal_model=thermal_model,
                    thermal_output_model=thermal_output_model,
                    request=request,
                    target_thermal_output_schedule=target_thermal_output_schedule,
                    block_starts=block_starts,
                    block_values=block_values,
                    block_index=index,
                )
                if abs(optimized_value - block_values[index]) > 1e-3:
                    changed = True
                block_values[index] = optimized_value
            if not changed:
                break

        recommended_plan = self._evaluate_setpoint_block_values(
            thermal_model=thermal_model,
            thermal_output_model=thermal_output_model,
            request=request,
            target_thermal_output_schedule=target_thermal_output_schedule,
            block_starts=block_starts,
            block_values=block_values,
        )

        return StateSpaceSetpointMpcPlanResult(
            model_name=thermal_model.model_name,
            interval_minutes=thermal_model.interval_minutes,
            thermal_output_model_name=thermal_output_model.model_name,
            thermal_plan=thermal_plan.recommended_plan,
            plan_results=[recommended_plan],
            recommended_plan=recommended_plan,
        )

    def _optimize_block_value(
        self,
        *,
        thermal_model: StateSpaceThermalModel,
        thermal_output_model,
        request: StateSpaceSetpointMpcPlanRequest,
        target_thermal_output_schedule: NumericSeries,
        block_starts: list,
        block_values: list[float],
        block_index: int,
    ) -> float:
        candidates = sorted(set(float(value) for value in request.allowed_setpoints))
        return min(
            candidates,
            key=lambda candidate: self._evaluate_setpoint_block_values(
                thermal_model=thermal_model,
                thermal_output_model=thermal_output_model,
                request=request,
                target_thermal_output_schedule=target_thermal_output_schedule,
                block_starts=block_starts,
                block_values=self.thermal_mpc_service._updated_block_values(
                    block_values,
                    block_index,
                    candidate,
                ),
            ).total_cost,
        )

    def _evaluate_setpoint_block_values(
        self,
        *,
        thermal_model: StateSpaceThermalModel,
        thermal_output_model,
        request: StateSpaceSetpointMpcPlanRequest,
        target_thermal_output_schedule: NumericSeries,
        block_starts: list,
        block_values: list[float],
    ) -> StateSpaceSetpointPlanEvaluation:
        thermostat_setpoint_schedule = self._build_setpoint_schedule(
            start_time=request.start_time,
            end_time=request.end_time,
            interval_minutes=thermal_model.interval_minutes,
            block_starts=block_starts,
            block_values=block_values,
        )
        prediction = self.setpoint_prediction_service.predict(
            thermal_model=thermal_model,
            thermal_output_model=thermal_output_model,
            request=StateSpaceSetpointPredictionRequest(
                start_time=request.start_time,
                end_time=request.end_time,
                initial_state=request.initial_state,
                initial_thermal_output=request.initial_thermal_output,
                thermostat_setpoint_schedule=thermostat_setpoint_schedule,
                outdoor_temperature_series=request.outdoor_temperature_series,
                solar_gain_series=request.solar_gain_series,
                supply_target_temperature_series=request.supply_target_temperature_series,
            ),
        )
        predicted_temperatures = [point.value for point in prediction.room_temperature.points]
        comfort_cost = self.thermal_mpc_service._mean_squared_comfort_violation(
            predicted_temperatures,
            comfort_min_temperature=request.comfort_min_temperature,
            comfort_max_temperature=request.comfort_max_temperature,
            undershoot_penalty=request.comfort_undershoot_penalty,
            overshoot_penalty=request.comfort_overshoot_penalty,
        )
        tracking_errors: list[float] = []
        for target_point, predicted_point in zip(
            target_thermal_output_schedule.points,
            prediction.thermal_output.points,
            strict=True,
        ):
            tracking_errors.append(predicted_point.value - target_point.value)
        tracking_cost = (
            request.thermal_output_tracking_penalty
            * self.thermal_mpc_service._mean_squared_values(tracking_errors)
        )
        setpoint_change_cost = self._setpoint_change_cost(
            thermostat_setpoint_schedule,
            penalty=request.setpoint_change_penalty,
            previous_applied_setpoint=request.previous_applied_setpoint,
        )
        return StateSpaceSetpointPlanEvaluation(
            plan_name=OPTIMIZED_CONTROL_PLAN_NAME,
            thermostat_setpoint_schedule=thermostat_setpoint_schedule,
            target_thermal_output_schedule=target_thermal_output_schedule,
            predicted_thermal_output=prediction.thermal_output,
            predicted_room_temperature=prediction.room_temperature,
            predicted_floor_heat_state=prediction.floor_heat_state,
            total_cost=comfort_cost + tracking_cost + setpoint_change_cost,
            comfort_violation_cost=comfort_cost,
            thermal_output_tracking_cost=tracking_cost,
            setpoint_change_cost=setpoint_change_cost,
            minimum_predicted_temperature=min(predicted_temperatures) if predicted_temperatures else None,
            maximum_predicted_temperature=max(predicted_temperatures) if predicted_temperatures else None,
        )

    @staticmethod
    def _build_setpoint_schedule(
        *,
        start_time,
        end_time,
        interval_minutes: int,
        block_starts,
        block_values: list[float],
    ) -> NumericSeries:
        interval = timedelta(minutes=interval_minutes)
        points: list[NumericPoint] = []
        cursor = start_time
        while cursor <= end_time:
            active_block_index = 0
            for index, block_start in enumerate(block_starts):
                if block_start <= cursor:
                    active_block_index = index
                else:
                    break
            points.append(
                NumericPoint(
                    timestamp=normalize_utc_timestamp(cursor),
                    value=float(block_values[active_block_index]),
                )
            )
            cursor += interval
        return NumericSeries(name=THERMOSTAT_SETPOINT, unit="degC", points=points)

    @staticmethod
    def _setpoint_change_cost(
        schedule: NumericSeries,
        *,
        penalty: float,
        previous_applied_setpoint: float | None,
    ) -> float:
        if penalty <= 0 or not schedule.points:
            return 0.0
        previous_value = (
            float(previous_applied_setpoint)
            if previous_applied_setpoint is not None
            else schedule.points[0].value
        )
        deltas: list[float] = []
        for point in schedule.points:
            deltas.append(point.value - previous_value)
            previous_value = point.value
        return penalty * StateSpaceThermalMpcService._mean_squared_values(deltas)
