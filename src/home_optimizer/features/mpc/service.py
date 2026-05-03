from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from home_optimizer.domain import NumericPoint, NumericSeries, normalize_utc_timestamp
from home_optimizer.domain.control import ShutterPositionControl, ThermostatSetpointControl
from home_optimizer.domain.names import THERMOSTAT_SETPOINT
from home_optimizer.features.prediction.schemas import (
    RoomTemperatureControlInputs,
    RoomTemperaturePrediction,
)

from .schemas import (
    ThermostatSetpointMpcEvaluationResult,
    ThermostatSetpointPlanEvaluation,
)

OPTIMIZED_PLAN_NAME = "optimized_plan"


class ThermostatSetpointMpcOptimizer:
    def __init__(self, prediction_service) -> None:
        self.prediction_service = prediction_service

    def optimize(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
        allowed_setpoints: list[float],
        move_block_times: list[datetime],
        shutter_position: ShutterPositionControl | None = None,
        comfort_min_temperature: float = 19.0,
        comfort_max_temperature: float = 21.0,
        setpoint_change_penalty: float = 0.1,
        previous_applied_setpoint: float | None = None,
    ) -> ThermostatSetpointMpcEvaluationResult:
        if not allowed_setpoints:
            raise ValueError("allowed_setpoints must not be empty")
        if comfort_min_temperature > comfort_max_temperature:
            raise ValueError("comfort_min_temperature must be <= comfort_max_temperature")
        if setpoint_change_penalty < 0:
            raise ValueError("setpoint_change_penalty must be >= 0")
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be greater than zero")

        lower_bound = min(allowed_setpoints)
        upper_bound = max(allowed_setpoints)
        control_block_starts = self._control_block_starts(
            start_time=start_time,
            end_time=end_time,
            move_block_times=move_block_times,
        )
        current_value = previous_applied_setpoint if previous_applied_setpoint is not None else upper_bound
        initial_value = min(upper_bound, max(lower_bound, float(current_value)))
        block_values = [initial_value for _ in control_block_starts]
        context = self._prepare_context(
            start_time=start_time,
            end_time=end_time,
            shutter_position=shutter_position,
        )

        for _ in range(3):
            changed = False
            for index in range(len(block_values)):
                optimized_value = self._optimize_block_value(
                    block_index=index,
                    block_values=block_values,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    start_time=start_time,
                    end_time=end_time,
                    interval_minutes=interval_minutes,
                    control_block_starts=control_block_starts,
                    shutter_position=shutter_position,
                    comfort_min_temperature=comfort_min_temperature,
                    comfort_max_temperature=comfort_max_temperature,
                    setpoint_change_penalty=setpoint_change_penalty,
                    previous_applied_setpoint=previous_applied_setpoint,
                    context=context,
                )
                if abs(optimized_value - block_values[index]) > 1e-3:
                    changed = True
                block_values[index] = optimized_value
            if not changed:
                break

        optimized_plan = self._evaluate_block_values(
            block_values=block_values,
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
            control_block_starts=control_block_starts,
            shutter_position=shutter_position,
            comfort_min_temperature=comfort_min_temperature,
            comfort_max_temperature=comfort_max_temperature,
            setpoint_change_penalty=setpoint_change_penalty,
            previous_applied_setpoint=previous_applied_setpoint,
            context=context,
        )
        return ThermostatSetpointMpcEvaluationResult(
            model_name=optimized_plan.model_name,
            interval_minutes=optimized_plan.interval_minutes,
            plan_results=[optimized_plan.evaluation],
            recommended_plan=optimized_plan.evaluation,
        )

    def _prepare_context(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        shutter_position: ShutterPositionControl | None,
    ):
        if hasattr(self.prediction_service, "prepare_prediction_context"):
            return self.prediction_service.prepare_prediction_context(
                start_time=start_time,
                end_time=end_time,
                shutter_position=shutter_position,
            )
        return None

    def _optimize_block_value(
        self,
        *,
        block_index: int,
        block_values: list[float],
        lower_bound: float,
        upper_bound: float,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
        control_block_starts: list[datetime],
        shutter_position: ShutterPositionControl | None,
        comfort_min_temperature: float,
        comfort_max_temperature: float,
        setpoint_change_penalty: float,
        previous_applied_setpoint: float | None,
        context,
    ) -> float:
        current = block_values[block_index]
        candidates = [lower_bound, current, upper_bound]
        best_value = min(
            candidates,
            key=lambda candidate: self._evaluate_block_values(
                block_values=self._updated_block_values(block_values, block_index, candidate),
                start_time=start_time,
                end_time=end_time,
                interval_minutes=interval_minutes,
                control_block_starts=control_block_starts,
                shutter_position=shutter_position,
                comfort_min_temperature=comfort_min_temperature,
                comfort_max_temperature=comfort_max_temperature,
                setpoint_change_penalty=setpoint_change_penalty,
                previous_applied_setpoint=previous_applied_setpoint,
                context=context,
            ).evaluation.total_cost,
        )
        left = lower_bound
        right = upper_bound
        for _ in range(8):
            left_probe = right - (right - left) / 1.61803398875
            right_probe = left + (right - left) / 1.61803398875
            left_cost = self._evaluate_block_values(
                block_values=self._updated_block_values(block_values, block_index, left_probe),
                start_time=start_time,
                end_time=end_time,
                interval_minutes=interval_minutes,
                control_block_starts=control_block_starts,
                shutter_position=shutter_position,
                comfort_min_temperature=comfort_min_temperature,
                comfort_max_temperature=comfort_max_temperature,
                setpoint_change_penalty=setpoint_change_penalty,
                previous_applied_setpoint=previous_applied_setpoint,
                context=context,
            ).evaluation.total_cost
            right_cost = self._evaluate_block_values(
                block_values=self._updated_block_values(block_values, block_index, right_probe),
                start_time=start_time,
                end_time=end_time,
                interval_minutes=interval_minutes,
                control_block_starts=control_block_starts,
                shutter_position=shutter_position,
                comfort_min_temperature=comfort_min_temperature,
                comfort_max_temperature=comfort_max_temperature,
                setpoint_change_penalty=setpoint_change_penalty,
                previous_applied_setpoint=previous_applied_setpoint,
                context=context,
            ).evaluation.total_cost
            if left_cost <= right_cost:
                right = right_probe
                if left_cost < self._evaluate_cost_for_value(
                    block_values=block_values,
                    block_index=block_index,
                    candidate_value=best_value,
                    start_time=start_time,
                    end_time=end_time,
                    interval_minutes=interval_minutes,
                    control_block_starts=control_block_starts,
                    shutter_position=shutter_position,
                    comfort_min_temperature=comfort_min_temperature,
                    comfort_max_temperature=comfort_max_temperature,
                    setpoint_change_penalty=setpoint_change_penalty,
                    previous_applied_setpoint=previous_applied_setpoint,
                    context=context,
                ):
                    best_value = left_probe
            else:
                left = left_probe
                if right_cost < self._evaluate_cost_for_value(
                    block_values=block_values,
                    block_index=block_index,
                    candidate_value=best_value,
                    start_time=start_time,
                    end_time=end_time,
                    interval_minutes=interval_minutes,
                    control_block_starts=control_block_starts,
                    shutter_position=shutter_position,
                    comfort_min_temperature=comfort_min_temperature,
                    comfort_max_temperature=comfort_max_temperature,
                    setpoint_change_penalty=setpoint_change_penalty,
                    previous_applied_setpoint=previous_applied_setpoint,
                    context=context,
                ):
                    best_value = right_probe
        return float(min(upper_bound, max(lower_bound, best_value)))

    def _evaluate_cost_for_value(
        self,
        *,
        block_values: list[float],
        block_index: int,
        candidate_value: float,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
        control_block_starts: list[datetime],
        shutter_position: ShutterPositionControl | None,
        comfort_min_temperature: float,
        comfort_max_temperature: float,
        setpoint_change_penalty: float,
        previous_applied_setpoint: float | None,
        context,
    ) -> float:
        return self._evaluate_block_values(
            block_values=self._updated_block_values(block_values, block_index, candidate_value),
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
            control_block_starts=control_block_starts,
            shutter_position=shutter_position,
            comfort_min_temperature=comfort_min_temperature,
            comfort_max_temperature=comfort_max_temperature,
            setpoint_change_penalty=setpoint_change_penalty,
            previous_applied_setpoint=previous_applied_setpoint,
            context=context,
        ).evaluation.total_cost

    def _evaluate_block_values(
        self,
        *,
        block_values: list[float],
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
        control_block_starts: list[datetime],
        shutter_position: ShutterPositionControl | None,
        comfort_min_temperature: float,
        comfort_max_temperature: float,
        setpoint_change_penalty: float,
        previous_applied_setpoint: float | None,
        context,
    ) -> "_OptimizedPlanResult":
        schedule = self._build_schedule(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=interval_minutes,
            control_block_starts=control_block_starts,
            block_values=block_values,
        )
        thermostat_setpoint = ThermostatSetpointControl.from_schedule(schedule)
        prediction = self._predict(
            start_time=start_time,
            end_time=end_time,
            thermostat_setpoint=thermostat_setpoint,
            shutter_position=shutter_position,
            context=context,
        )
        comfort_violation_cost = 0.0
        predicted_values: list[float] = []
        for point in prediction.room_temperature.points:
            predicted_values.append(point.value)
            if point.value < comfort_min_temperature:
                comfort_violation_cost += (comfort_min_temperature - point.value) ** 2
            elif point.value > comfort_max_temperature:
                comfort_violation_cost += (point.value - comfort_max_temperature) ** 2
        setpoint_change_cost = self._setpoint_change_cost(
            schedule,
            penalty=setpoint_change_penalty,
            previous_applied_setpoint=previous_applied_setpoint,
        )
        evaluation = ThermostatSetpointPlanEvaluation(
            plan_name=OPTIMIZED_PLAN_NAME,
            thermostat_setpoint_schedule=schedule,
            predicted_room_temperature=prediction.room_temperature,
            total_cost=comfort_violation_cost + setpoint_change_cost,
            comfort_violation_cost=comfort_violation_cost,
            setpoint_change_cost=setpoint_change_cost,
            minimum_predicted_temperature=min(predicted_values) if predicted_values else None,
            maximum_predicted_temperature=max(predicted_values) if predicted_values else None,
        )
        return _OptimizedPlanResult(
            model_name=prediction.model_name,
            interval_minutes=prediction.interval_minutes,
            evaluation=evaluation,
        )

    def _predict(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        thermostat_setpoint: ThermostatSetpointControl,
        shutter_position: ShutterPositionControl | None,
        context,
    ) -> RoomTemperaturePrediction:
        if context is not None and hasattr(self.prediction_service, "predict_with_context"):
            return self.prediction_service.predict_with_context(
                context=context,
                thermostat_setpoint=thermostat_setpoint,
            )
        return self.prediction_service.predict(
            start_time=start_time,
            end_time=end_time,
            control_inputs=RoomTemperatureControlInputs(
                thermostat_setpoint=thermostat_setpoint,
                shutter_position=shutter_position,
            ),
        )

    @staticmethod
    def _control_block_starts(
        *,
        start_time: datetime,
        end_time: datetime,
        move_block_times: list[datetime],
    ) -> list[datetime]:
        block_starts = [start_time]
        block_starts.extend(
            sorted(
                {
                    item
                    for item in move_block_times
                    if start_time < item <= end_time
                }
            )
        )
        return block_starts

    @staticmethod
    def _build_schedule(
        *,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
        control_block_starts: list[datetime],
        block_values: list[float],
    ) -> NumericSeries:
        interval = timedelta(minutes=interval_minutes)
        points: list[NumericPoint] = []
        timestamp = start_time
        while timestamp <= end_time:
            block_index = max(
                index
                for index, block_start in enumerate(control_block_starts)
                if block_start <= timestamp
            )
            points.append(
                NumericPoint(
                    timestamp=normalize_utc_timestamp(timestamp),
                    value=float(block_values[block_index]),
                )
            )
            timestamp += interval
        return NumericSeries(name=THERMOSTAT_SETPOINT, unit="degC", points=points)

    @staticmethod
    def _updated_block_values(
        block_values: list[float],
        block_index: int,
        candidate_value: float,
    ) -> list[float]:
        updated = list(block_values)
        updated[block_index] = float(candidate_value)
        return updated

    @staticmethod
    def _setpoint_change_cost(
        schedule: NumericSeries,
        *,
        penalty: float,
        previous_applied_setpoint: float | None = None,
    ) -> float:
        if not schedule.points:
            return 0.0

        previous_value = previous_applied_setpoint
        change_cost = 0.0
        for point in schedule.points:
            if previous_value is None:
                previous_value = point.value
                continue
            change_cost += abs(point.value - previous_value) * penalty
            previous_value = point.value
        return change_cost


ThermostatSetpointMpcEvaluator = ThermostatSetpointMpcOptimizer


@dataclass(frozen=True)
class _OptimizedPlanResult:
    model_name: str
    interval_minutes: int
    evaluation: ThermostatSetpointPlanEvaluation
