from __future__ import annotations

from typing import Callable

from home_optimizer.features.mpc.models import (
    LinearThermalControlModel,
    MpcControllerRequest,
    MpcHorizonStep,
    MpcInitialState,
    MpcPlan,
    MpcProblem,
)
from home_optimizer.features.mpc.space_heating_mpc import SpaceHeatingMpcSolver


class SpaceHeatingMpcControllerService:
    def __init__(
        self,
        *,
        solver: SpaceHeatingMpcSolver | None = None,
        control_model_provider: Callable[[], LinearThermalControlModel] | None = None,
        horizon_provider: Callable[[], list[MpcHorizonStep]] | None = None,
        initial_state_provider: Callable[[], MpcInitialState] | None = None,
    ) -> None:
        self.solver = solver or SpaceHeatingMpcSolver()
        self.control_model_provider = control_model_provider
        self.horizon_provider = horizon_provider
        self.initial_state_provider = initial_state_provider

    def plan(
        self,
        request: MpcControllerRequest,
        *,
        control_model: LinearThermalControlModel | None = None,
        initial_state: MpcInitialState | None = None,
        horizon: list[MpcHorizonStep] | None = None,
    ) -> MpcPlan:
        resolved_control_model = self._resolve_control_model(control_model)
        resolved_initial_state = self._resolve_initial_state(initial_state)
        resolved_horizon = self._resolve_horizon(horizon or request.horizon)

        problem = MpcProblem(
            interval_minutes=request.interval_minutes,
            control_model=resolved_control_model,
            initial_state=resolved_initial_state,
            horizon=resolved_horizon,
            constraints=request.constraints,
            objective_weights=request.objective_weights,
            max_solver_seconds=request.max_solver_seconds,
        )
        return self.solver.solve(problem)

    def _resolve_control_model(
        self,
        control_model: LinearThermalControlModel | None,
    ) -> LinearThermalControlModel:
        if control_model is not None:
            return control_model
        if self.control_model_provider is None:
            raise ValueError(
                "control_model is required when no control_model_provider is configured"
            )
        return self.control_model_provider()

    def _resolve_initial_state(
        self,
        initial_state: MpcInitialState | None,
    ) -> MpcInitialState:
        if initial_state is not None:
            return initial_state
        if self.initial_state_provider is None:
            raise ValueError(
                "initial_state is required when no initial_state_provider is configured"
            )
        return self.initial_state_provider()

    def _resolve_horizon(self, horizon: list[MpcHorizonStep] | None) -> list[MpcHorizonStep]:
        if horizon is not None:
            return horizon
        if self.horizon_provider is None:
            raise ValueError("horizon is required when no horizon_provider is configured")
        return self.horizon_provider()
