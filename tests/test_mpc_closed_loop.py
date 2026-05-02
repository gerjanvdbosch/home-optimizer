from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from home_optimizer.domain import (
    NumericPoint,
    NumericSeries,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
)
from home_optimizer.features.mpc import (
    ThermostatSetpointCandidateEvaluation,
    ThermostatSetpointMpcClosedLoopService,
    ThermostatSetpointMpcEvaluationResult,
)


class FakeClosedLoopReader:
    def read_series(self, names, start_time, end_time) -> list[NumericSeries]:
        base_time = datetime(2026, 5, 2, 0, 0, tzinfo=timezone.utc)
        series_by_name = {
            THERMOSTAT_SETPOINT: NumericSeries(
                name=THERMOSTAT_SETPOINT,
                unit="degC",
                points=[
                    NumericPoint(
                        timestamp=(base_time + timedelta(minutes=15 * step)).isoformat(),
                        value=20.0,
                    )
                    for step in range(-1, 96)
                ],
            ),
            SHUTTER_LIVING_ROOM: NumericSeries(
                name=SHUTTER_LIVING_ROOM,
                unit="percent",
                points=[
                    NumericPoint(
                        timestamp=(base_time + timedelta(minutes=15 * step)).isoformat(),
                        value=100.0,
                    )
                    for step in range(-1, 96)
                ],
            ),
        }
        return [series_by_name[name] for name in names]


class FakeClosedLoopPlanner:
    def __init__(self) -> None:
        self.calls: list[tuple[datetime, datetime]] = []

    def propose_plan(self, request, *, shutter_position=None) -> ThermostatSetpointMpcEvaluationResult:
        self.calls.append((request.start_time, request.end_time))
        schedule = NumericSeries(
            name=THERMOSTAT_SETPOINT,
            unit="degC",
            points=[
                NumericPoint(timestamp=request.start_time.isoformat(), value=19.5),
                NumericPoint(
                    timestamp=(request.start_time + timedelta(minutes=request.interval_minutes)).isoformat(),
                    value=19.5,
                ),
            ],
        )
        predicted_room_temperature = NumericSeries(
            name=ROOM_TEMPERATURE,
            unit="degC",
            points=[
                NumericPoint(
                    timestamp=(request.start_time + timedelta(minutes=request.interval_minutes)).isoformat(),
                    value=19.6,
                )
            ],
        )
        best_candidate = ThermostatSetpointCandidateEvaluation(
            candidate_name="candidate_1",
            thermostat_setpoint_schedule=schedule,
            predicted_room_temperature=predicted_room_temperature,
            total_cost=0.2,
            comfort_violation_cost=0.1,
            setpoint_change_cost=0.1,
            minimum_predicted_temperature=19.6,
            maximum_predicted_temperature=19.6,
        )
        return ThermostatSetpointMpcEvaluationResult(
            model_name="linear_2state_room_temperature",
            interval_minutes=request.interval_minutes,
            candidate_results=[best_candidate],
            best_candidate=best_candidate,
        )


def test_closed_loop_service_replans_each_interval_and_applies_first_step() -> None:
    service = ThermostatSetpointMpcClosedLoopService(
        FakeClosedLoopReader(),
        FakeClosedLoopPlanner(),
    )

    result = service.evaluate_by_day(
        start_date=date(2026, 5, 2),
        end_date=date(2026, 5, 2),
        allowed_setpoints=[19.0, 19.5, 20.0],
        horizon_hours=6,
        interval_minutes=15,
        switch_interval_hours=2,
        comfort_min_temperature=19.0,
        comfort_max_temperature=21.0,
        setpoint_change_penalty=0.1,
        timezone_info=timezone.utc,
    )

    assert result.model_name == "linear_2state_room_temperature"
    assert result.successful_days == 1
    assert result.failed_days == 0
    day_result = result.day_results[0]
    assert len(day_result.step_results) == 95
    assert len(day_result.applied_thermostat_setpoint_schedule.points) == 95
    assert len(day_result.predicted_room_temperature.points) == 95
    assert day_result.average_total_cost == 0.2
    assert day_result.average_comfort_violation_cost == 0.1
    assert day_result.average_setpoint_change_cost == 0.1
    assert day_result.under_comfort_count == 0
    assert day_result.over_comfort_count == 0
