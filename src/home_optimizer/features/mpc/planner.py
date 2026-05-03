from __future__ import annotations

from home_optimizer.domain.control import ShutterPositionControl

from .schemas import (
    ThermostatSetpointMpcEvaluationResult,
    ThermostatSetpointMpcPlanRequest,
)
from .service import ThermostatSetpointMpcOptimizer


class ThermostatSetpointMpcPlanner:
    def __init__(
        self,
        optimizer: ThermostatSetpointMpcOptimizer,
    ) -> None:
        self.optimizer = optimizer

    def propose_plan(
        self,
        request: ThermostatSetpointMpcPlanRequest,
        *,
        shutter_position: ShutterPositionControl | None = None,
    ) -> ThermostatSetpointMpcEvaluationResult:
        return self.optimizer.optimize(
            start_time=request.start_time,
            end_time=request.end_time,
            interval_minutes=request.interval_minutes,
            allowed_setpoints=request.allowed_setpoints,
            move_block_times=request.switch_times,
            shutter_position=shutter_position,
            comfort_min_temperature=request.comfort_min_temperature,
            comfort_max_temperature=request.comfort_max_temperature,
            setpoint_change_penalty=request.setpoint_change_penalty,
            previous_applied_setpoint=request.previous_applied_setpoint,
        )
