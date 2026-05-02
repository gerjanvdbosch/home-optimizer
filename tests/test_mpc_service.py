from __future__ import annotations

from datetime import datetime, timedelta, timezone

from home_optimizer.domain import (
    NumericPoint,
    NumericSeries,
    ROOM_TEMPERATURE,
    ShutterPositionControl,
    ThermostatSetpointControl,
)
from home_optimizer.features.mpc import ThermostatSetpointMpcEvaluator
from home_optimizer.features.prediction.schemas import RoomTemperaturePrediction


class FakePredictionService:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def predict(self, start_time, end_time, *, control_inputs):
        self.calls.append(control_inputs)
        schedule = control_inputs.thermostat_setpoint.schedule
        avg_setpoint = sum(point.value for point in schedule.points) / len(schedule.points)
        base_time = start_time
        points = [
            NumericPoint(
                timestamp=(base_time + timedelta(minutes=15 * step)).isoformat(),
                value=18.8 + ((avg_setpoint - 19.0) * 0.8),
            )
            for step in range(1, 5)
        ]
        return RoomTemperaturePrediction(
            model_name="linear_2state_room_temperature",
            interval_minutes=15,
            target_name=ROOM_TEMPERATURE,
            room_temperature=NumericSeries(name=ROOM_TEMPERATURE, unit="degC", points=points),
        )


def test_mpc_evaluator_ranks_thermostat_candidates_by_cost() -> None:
    evaluator = ThermostatSetpointMpcEvaluator(FakePredictionService())
    low_schedule = ThermostatSetpointControl.from_schedule(
        NumericSeries(
            name="thermostat_setpoint",
            unit="degC",
            points=[
                NumericPoint(timestamp="2026-05-02T10:00:00+00:00", value=19.0),
                NumericPoint(timestamp="2026-05-02T10:15:00+00:00", value=19.0),
                NumericPoint(timestamp="2026-05-02T10:30:00+00:00", value=19.0),
            ],
        )
    )
    high_schedule = ThermostatSetpointControl.from_schedule(
        NumericSeries(
            name="thermostat_setpoint",
            unit="degC",
            points=[
                NumericPoint(timestamp="2026-05-02T10:00:00+00:00", value=20.5),
                NumericPoint(timestamp="2026-05-02T10:15:00+00:00", value=20.5),
                NumericPoint(timestamp="2026-05-02T10:30:00+00:00", value=20.5),
            ],
        )
    )
    shutter_control = ShutterPositionControl.from_schedule(
        NumericSeries(
            name="shutter_living_room",
            unit="percent",
            points=[NumericPoint(timestamp="2026-05-02T10:00:00+00:00", value=100.0)],
        )
    )

    result = evaluator.evaluate_candidates(
        start_time=datetime(2026, 5, 2, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 5, 2, 11, 0, tzinfo=timezone.utc),
        thermostat_setpoint_candidates=[low_schedule, high_schedule],
        shutter_position=shutter_control,
        comfort_min_temperature=19.0,
        comfort_max_temperature=21.0,
    )

    assert result.model_name == "linear_2state_room_temperature"
    assert result.interval_minutes == 15
    assert len(result.candidate_results) == 2
    assert result.best_candidate.candidate_name == "candidate_2"
    assert result.best_candidate.total_cost < result.candidate_results[0].total_cost


def test_mpc_evaluator_penalizes_setpoint_changes() -> None:
    evaluator = ThermostatSetpointMpcEvaluator(FakePredictionService())
    flat_schedule = ThermostatSetpointControl.from_schedule(
        NumericSeries(
            name="thermostat_setpoint",
            unit="degC",
            points=[
                NumericPoint(timestamp="2026-05-02T10:00:00+00:00", value=20.0),
                NumericPoint(timestamp="2026-05-02T10:15:00+00:00", value=20.0),
                NumericPoint(timestamp="2026-05-02T10:30:00+00:00", value=20.0),
            ],
        )
    )
    stepped_schedule = ThermostatSetpointControl.from_schedule(
        NumericSeries(
            name="thermostat_setpoint",
            unit="degC",
            points=[
                NumericPoint(timestamp="2026-05-02T10:00:00+00:00", value=20.0),
                NumericPoint(timestamp="2026-05-02T10:15:00+00:00", value=21.0),
                NumericPoint(timestamp="2026-05-02T10:30:00+00:00", value=20.0),
            ],
        )
    )

    result = evaluator.evaluate_candidates(
        start_time=datetime(2026, 5, 2, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 5, 2, 11, 0, tzinfo=timezone.utc),
        thermostat_setpoint_candidates=[flat_schedule, stepped_schedule],
        setpoint_change_penalty=0.5,
    )

    assert result.candidate_results[0].setpoint_change_cost == 0.0
    assert result.candidate_results[1].setpoint_change_cost > 0.0
