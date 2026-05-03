from __future__ import annotations

from datetime import datetime, timedelta, timezone

from home_optimizer.domain import NumericPoint, NumericSeries, ROOM_TEMPERATURE
from home_optimizer.features.mpc import ThermostatSetpointMpcOptimizer
from home_optimizer.features.prediction.schemas import RoomTemperaturePrediction


class FakePredictionService:
    def predict(self, start_time, end_time, *, control_inputs):
        schedule = control_inputs.thermostat_setpoint.schedule
        avg_setpoint = sum(point.value for point in schedule.points) / len(schedule.points)
        points = [
            NumericPoint(
                timestamp=(start_time + timedelta(minutes=15 * step)).isoformat(),
                value=avg_setpoint,
            )
            for step in range(1, 5)
        ]
        return RoomTemperaturePrediction(
            model_name="linear_2state_room_temperature",
            interval_minutes=15,
            target_name=ROOM_TEMPERATURE,
            room_temperature=NumericSeries(name=ROOM_TEMPERATURE, unit="degC", points=points),
        )


def test_mpc_optimizer_returns_single_optimized_plan() -> None:
    optimizer = ThermostatSetpointMpcOptimizer(FakePredictionService())

    result = optimizer.optimize(
        start_time=datetime(2026, 5, 2, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 5, 2, 11, 0, tzinfo=timezone.utc),
        interval_minutes=15,
        allowed_setpoints=[19.0, 20.0, 21.0],
        move_block_times=[datetime(2026, 5, 2, 10, 30, tzinfo=timezone.utc)],
        comfort_min_temperature=21.0,
        comfort_max_temperature=21.0,
        setpoint_change_penalty=0.05,
    )

    assert result.model_name == "linear_2state_room_temperature"
    assert result.interval_minutes == 15
    assert len(result.plan_results) == 1
    assert result.recommended_plan.plan_name == "optimized_plan"
    assert result.recommended_plan.total_cost == 0.0
    assert all(
        point.value == 21.0
        for point in result.recommended_plan.thermostat_setpoint_schedule.points
    )


def test_mpc_optimizer_respects_previous_applied_setpoint_when_costs_are_equal() -> None:
    optimizer = ThermostatSetpointMpcOptimizer(FakePredictionService())

    result = optimizer.optimize(
        start_time=datetime(2026, 5, 2, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 5, 2, 10, 30, tzinfo=timezone.utc),
        interval_minutes=15,
        allowed_setpoints=[19.0, 20.0, 21.0],
        move_block_times=[],
        comfort_min_temperature=19.0,
        comfort_max_temperature=21.0,
        setpoint_change_penalty=0.5,
        previous_applied_setpoint=19.0,
    )

    assert result.recommended_plan.setpoint_change_cost == 0.0
    assert all(
        point.value == 19.0
        for point in result.recommended_plan.thermostat_setpoint_schedule.points
    )
