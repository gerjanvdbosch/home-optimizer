from __future__ import annotations

from home_optimizer.domain.control import ShutterPositionControl

from .candidates import ThermostatSetpointCandidateGenerator
from .schemas import (
    ThermostatSetpointMpcEvaluationResult,
    ThermostatSetpointMpcPlanRequest,
)
from .service import ThermostatSetpointMpcEvaluator


class ThermostatSetpointMpcPlanner:
    def __init__(
        self,
        candidate_generator: ThermostatSetpointCandidateGenerator,
        evaluator: ThermostatSetpointMpcEvaluator,
    ) -> None:
        self.candidate_generator = candidate_generator
        self.evaluator = evaluator

    def propose_plan(
        self,
        request: ThermostatSetpointMpcPlanRequest,
        *,
        shutter_position: ShutterPositionControl | None = None,
    ) -> ThermostatSetpointMpcEvaluationResult:
        constant_candidates = self.candidate_generator.generate_constant_candidates(
            start_time=request.start_time,
            end_time=request.end_time,
            interval_minutes=request.interval_minutes,
            allowed_setpoints=request.allowed_setpoints,
        )
        switch_candidates = self.candidate_generator.generate_single_switch_candidates(
            start_time=request.start_time,
            end_time=request.end_time,
            interval_minutes=request.interval_minutes,
            allowed_setpoints=request.allowed_setpoints,
            switch_times=request.switch_times,
        )
        return self.evaluator.evaluate_candidates(
            start_time=request.start_time,
            end_time=request.end_time,
            thermostat_setpoint_candidates=constant_candidates + switch_candidates,
            shutter_position=shutter_position,
            comfort_min_temperature=request.comfort_min_temperature,
            comfort_max_temperature=request.comfort_max_temperature,
            setpoint_change_penalty=request.setpoint_change_penalty,
        )
