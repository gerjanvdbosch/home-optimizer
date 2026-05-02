from __future__ import annotations

from datetime import date, datetime

from home_optimizer.domain import NumericSeries
from home_optimizer.domain.models import DomainModel

DEFAULT_MPC_HORIZON_HOURS = 6


class ThermostatSetpointCandidateEvaluation(DomainModel):
    candidate_name: str
    thermostat_setpoint_schedule: NumericSeries
    predicted_room_temperature: NumericSeries
    total_cost: float
    comfort_violation_cost: float
    setpoint_change_cost: float
    minimum_predicted_temperature: float | None
    maximum_predicted_temperature: float | None


class ThermostatSetpointMpcEvaluationResult(DomainModel):
    model_name: str
    interval_minutes: int
    candidate_results: list[ThermostatSetpointCandidateEvaluation]
    best_candidate: ThermostatSetpointCandidateEvaluation


class ThermostatSetpointMpcPlanRequest(DomainModel):
    start_time: datetime
    end_time: datetime
    interval_minutes: int
    allowed_setpoints: list[float]
    switch_times: list[datetime]
    comfort_min_temperature: float = 19.0
    comfort_max_temperature: float = 21.0
    setpoint_change_penalty: float = 0.1


class ThermostatSetpointMpcClosedLoopStepResult(DomainModel):
    step_start_time: datetime
    applied_setpoint: float
    best_candidate_name: str
    best_candidate_total_cost: float
    predicted_next_room_temperature: float


class ThermostatSetpointMpcClosedLoopDayResult(DomainModel):
    day: date
    horizon_hours: int
    interval_minutes: int
    applied_thermostat_setpoint_schedule: NumericSeries
    measured_thermostat_setpoint_schedule: NumericSeries
    predicted_room_temperature: NumericSeries
    average_total_cost: float
    average_comfort_violation_cost: float
    average_setpoint_change_cost: float
    minimum_predicted_temperature: float | None
    maximum_predicted_temperature: float | None
    under_comfort_count: int
    over_comfort_count: int
    step_results: list[ThermostatSetpointMpcClosedLoopStepResult]
    error: str | None = None


class ThermostatSetpointMpcClosedLoopResult(DomainModel):
    model_name: str
    interval_minutes: int
    horizon_hours: int
    start_date: date
    end_date: date
    total_days: int
    successful_days: int
    failed_days: int
    average_total_cost: float | None
    average_comfort_violation_cost: float | None
    average_setpoint_change_cost: float | None
    day_results: list[ThermostatSetpointMpcClosedLoopDayResult]
