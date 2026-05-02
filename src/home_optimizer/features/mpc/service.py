from __future__ import annotations

from datetime import datetime

from home_optimizer.domain import NumericSeries, latest_value_at, normalize_utc_timestamp
from home_optimizer.domain.control import ShutterPositionControl, ThermostatSetpointControl
from home_optimizer.features.prediction.schemas import (
    RoomTemperatureControlInputs,
    RoomTemperaturePrediction,
)

from .schemas import (
    ThermostatSetpointCandidateEvaluation,
    ThermostatSetpointMpcEvaluationResult,
)


class ThermostatSetpointMpcEvaluator:
    def __init__(self, prediction_service) -> None:
        self.prediction_service = prediction_service

    def evaluate_candidates(
        self,
        *,
        start_time: datetime,
        end_time: datetime,
        thermostat_setpoint_candidates: list[ThermostatSetpointControl],
        shutter_position: ShutterPositionControl | None = None,
        comfort_min_temperature: float = 19.0,
        comfort_max_temperature: float = 21.0,
        setpoint_change_penalty: float = 0.1,
    ) -> ThermostatSetpointMpcEvaluationResult:
        if not thermostat_setpoint_candidates:
            raise ValueError("thermostat_setpoint_candidates must not be empty")
        if comfort_min_temperature > comfort_max_temperature:
            raise ValueError("comfort_min_temperature must be <= comfort_max_temperature")
        if setpoint_change_penalty < 0:
            raise ValueError("setpoint_change_penalty must be >= 0")

        candidate_results: list[ThermostatSetpointCandidateEvaluation] = []
        model_name: str | None = None
        interval_minutes: int | None = None

        for index, thermostat_setpoint in enumerate(thermostat_setpoint_candidates, start=1):
            prediction = self.prediction_service.predict(
                start_time=start_time,
                end_time=end_time,
                control_inputs=RoomTemperatureControlInputs(
                    thermostat_setpoint=thermostat_setpoint,
                    shutter_position=shutter_position,
                ),
            )
            model_name = model_name or prediction.model_name
            interval_minutes = interval_minutes or prediction.interval_minutes
            candidate_results.append(
                self._candidate_result(
                    candidate_name=f"candidate_{index}",
                    thermostat_setpoint=thermostat_setpoint,
                    prediction=prediction,
                    comfort_min_temperature=comfort_min_temperature,
                    comfort_max_temperature=comfort_max_temperature,
                    setpoint_change_penalty=setpoint_change_penalty,
                    start_time=start_time,
                )
            )

        best_candidate = min(candidate_results, key=lambda item: item.total_cost)
        return ThermostatSetpointMpcEvaluationResult(
            model_name=model_name or "",
            interval_minutes=interval_minutes or 0,
            candidate_results=candidate_results,
            best_candidate=best_candidate,
        )

    def _candidate_result(
        self,
        *,
        candidate_name: str,
        thermostat_setpoint: ThermostatSetpointControl,
        prediction: RoomTemperaturePrediction,
        comfort_min_temperature: float,
        comfort_max_temperature: float,
        setpoint_change_penalty: float,
        start_time: datetime,
    ) -> ThermostatSetpointCandidateEvaluation:
        comfort_violation_cost = 0.0
        predicted_values: list[float] = []
        for point in prediction.room_temperature.points:
            predicted_values.append(point.value)
            if point.value < comfort_min_temperature:
                comfort_violation_cost += (comfort_min_temperature - point.value) ** 2
            elif point.value > comfort_max_temperature:
                comfort_violation_cost += (point.value - comfort_max_temperature) ** 2

        setpoint_change_cost = self._setpoint_change_cost(
            thermostat_setpoint.schedule,
            start_time=start_time,
            penalty=setpoint_change_penalty,
        )
        return ThermostatSetpointCandidateEvaluation(
            candidate_name=candidate_name,
            thermostat_setpoint_schedule=thermostat_setpoint.schedule,
            predicted_room_temperature=prediction.room_temperature,
            total_cost=comfort_violation_cost + setpoint_change_cost,
            comfort_violation_cost=comfort_violation_cost,
            setpoint_change_cost=setpoint_change_cost,
            minimum_predicted_temperature=min(predicted_values) if predicted_values else None,
            maximum_predicted_temperature=max(predicted_values) if predicted_values else None,
        )

    @staticmethod
    def _setpoint_change_cost(
        schedule: NumericSeries,
        *,
        start_time: datetime,
        penalty: float,
    ) -> float:
        if not schedule.points:
            return 0.0

        previous_value = latest_value_at(schedule.points, normalize_utc_timestamp(start_time))
        change_cost = 0.0
        for point in schedule.points:
            if previous_value is None:
                previous_value = point.value
                continue
            change_cost += abs(point.value - previous_value) * penalty
            previous_value = point.value
        return change_cost
