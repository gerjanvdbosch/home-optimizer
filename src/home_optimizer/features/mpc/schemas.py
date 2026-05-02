from __future__ import annotations

from home_optimizer.domain import NumericSeries
from home_optimizer.domain.models import DomainModel


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
