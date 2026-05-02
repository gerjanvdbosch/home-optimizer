from __future__ import annotations

from datetime import datetime, timezone

from home_optimizer.features.mpc import (
    ThermostatSetpointCandidateGenerator,
    ThermostatSetpointMpcEvaluator,
    ThermostatSetpointMpcPlanRequest,
    ThermostatSetpointMpcPlanner,
)
from home_optimizer.features.prediction.schemas import RoomTemperaturePrediction
from home_optimizer.domain import NumericPoint, NumericSeries, ROOM_TEMPERATURE


class FakePredictionService:
    def predict(self, start_time, end_time, *, control_inputs):
        schedule = control_inputs.thermostat_setpoint.schedule
        avg_setpoint = sum(point.value for point in schedule.points) / len(schedule.points)
        predicted_value = 18.7 + ((avg_setpoint - 19.0) * 0.9)
        points = [
            NumericPoint(timestamp="2026-05-02T10:15:00+00:00", value=predicted_value),
            NumericPoint(timestamp="2026-05-02T10:30:00+00:00", value=predicted_value),
        ]
        return RoomTemperaturePrediction(
            model_name="linear_2state_room_temperature",
            interval_minutes=15,
            target_name=ROOM_TEMPERATURE,
            room_temperature=NumericSeries(name=ROOM_TEMPERATURE, unit="degC", points=points),
        )


def test_mpc_planner_proposes_best_schedule_from_generated_candidates() -> None:
    planner = ThermostatSetpointMpcPlanner(
        ThermostatSetpointCandidateGenerator(),
        ThermostatSetpointMpcEvaluator(FakePredictionService()),
    )

    result = planner.propose_plan(
        ThermostatSetpointMpcPlanRequest(
            start_time=datetime(2026, 5, 2, 10, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 5, 2, 10, 30, tzinfo=timezone.utc),
            interval_minutes=15,
            allowed_setpoints=[19.0, 20.0, 21.0],
            switch_times=[datetime(2026, 5, 2, 10, 15, tzinfo=timezone.utc)],
            comfort_min_temperature=19.4,
            comfort_max_temperature=20.2,
            setpoint_change_penalty=0.05,
        )
    )

    assert result.model_name == "linear_2state_room_temperature"
    assert result.best_candidate.total_cost == min(
        candidate.total_cost for candidate in result.candidate_results
    )
    assert result.best_candidate.thermostat_setpoint_schedule.name == "thermostat_setpoint"
    assert len(result.candidate_results) == 9
