from __future__ import annotations

from typing import Callable

from home_optimizer.features.modeling import RoomRcModel, TrainedLinearRoomModel
from home_optimizer.features.mpc.control_model import to_control_model
from home_optimizer.features.mpc.explain import explain_heating_plan
from home_optimizer.features.mpc.flexibility_assessor import SpaceHeatingFlexibilityAssessor
from home_optimizer.features.mpc.horizon_builder import MpcHorizonBuilder
from home_optimizer.features.mpc.models import (
    ControlModelConversionOptions,
    ExecutionTargetStep,
    LinearThermalControlModel,
    MpcControllerRequest,
    MpcHorizonBuildRequest,
    MpcHorizonStep,
    MpcInitialState,
    MpcPlan,
    MpcProblem,
    PreheatSchedule,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
    ThermalFlexibilityState,
)
from home_optimizer.features.mpc.preheat_scheduler import SpaceHeatingPreheatScheduler
from home_optimizer.features.mpc.space_heating_mpc import SpaceHeatingMpcSolver


class SpaceHeatingMpcControllerService:
    def __init__(
        self,
        *,
        solver: SpaceHeatingMpcSolver | None = None,
        control_model_provider: Callable[
            [],
            LinearThermalControlModel | Rc2StateThermalControlModel,
        ] | None = None,
        horizon_provider: Callable[[], list[MpcHorizonStep]] | None = None,
        initial_state_provider: Callable[
            [],
            MpcInitialState | Rc2StateMpcInitialState,
        ] | None = None,
        horizon_builder: MpcHorizonBuilder | None = None,
        preheat_scheduler: SpaceHeatingPreheatScheduler | None = None,
        flexibility_assessor: SpaceHeatingFlexibilityAssessor | None = None,
    ) -> None:
        self.solver = solver or SpaceHeatingMpcSolver()
        self.control_model_provider = control_model_provider
        self.horizon_provider = horizon_provider
        self.initial_state_provider = initial_state_provider
        self.horizon_builder = horizon_builder or MpcHorizonBuilder()
        self.preheat_scheduler = preheat_scheduler or SpaceHeatingPreheatScheduler()
        self.flexibility_assessor = flexibility_assessor or SpaceHeatingFlexibilityAssessor()

    def plan(
        self,
        request: MpcControllerRequest,
        *,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel | None = None,
        initial_state: MpcInitialState | Rc2StateMpcInitialState | None = None,
        horizon: list[MpcHorizonStep] | None = None,
    ) -> MpcPlan:
        resolved_control_model = self._resolve_control_model(control_model)
        resolved_initial_state = self._resolve_initial_state(initial_state)
        resolved_horizon = self._resolve_horizon(horizon or request.horizon)
        resolved_preheat_plan = None
        resolved_flexibility: ThermalFlexibilityState | None = None
        resolved_preheat_schedule: PreheatSchedule | None = None
        execution_targets: list[ExecutionTargetStep] | None = None
        resolved_flexibility = self.flexibility_assessor.assess(
            interval_minutes=request.interval_minutes,
            control_model=resolved_control_model,
            initial_state=resolved_initial_state,
            horizon=resolved_horizon,
            constraints=request.constraints,
        )
        resolved_preheat_schedule = self.preheat_scheduler.build_schedule(
            flexibility_state=resolved_flexibility,
            constraints=request.constraints,
            interval_minutes=request.interval_minutes,
            control_model=resolved_control_model,
            initial_state=resolved_initial_state,
            horizon=resolved_horizon,
        )
        execution_targets = self.flexibility_assessor.build_execution_targets(
            flexibility_state=resolved_flexibility,
            schedule=resolved_preheat_schedule,
        )
        resolved_horizon = self._apply_execution_targets(
            horizon=resolved_horizon,
            execution_targets=execution_targets,
            preheat_schedule=resolved_preheat_schedule,
        )

        problem = MpcProblem(
            interval_minutes=request.interval_minutes,
            control_mode=request.control_mode,
            control_model=resolved_control_model,
            initial_state=resolved_initial_state,
            horizon=resolved_horizon,
            preheat_plan=resolved_preheat_plan,
            thermal_flexibility=resolved_flexibility,
            preheat_schedule=resolved_preheat_schedule,
            execution_targets=execution_targets,
            constraints=request.constraints,
            objective_weights=request.objective_weights,
            max_solver_seconds=request.max_solver_seconds,
        )
        plan = self.solver.solve(problem)
        return plan.model_copy(
            update={
                "control_mode": request.control_mode,
                "preheat_plan": resolved_preheat_plan,
                "thermal_flexibility": resolved_flexibility,
                "preheat_schedule": resolved_preheat_schedule,
                "heating_explanation": explain_heating_plan(
                    plan=plan,
                    control_model=resolved_control_model,
                    initial_state=resolved_initial_state,
                    horizon=resolved_horizon,
                )
            }
        )

    def build_horizon(self, request: MpcHorizonBuildRequest) -> list[MpcHorizonStep]:
        return self.horizon_builder.build(request)

    def plan_from_source_model(
        self,
        request: MpcControllerRequest,
        *,
        source_model: TrainedLinearRoomModel | RoomRcModel,
        initial_state: MpcInitialState | Rc2StateMpcInitialState,
        horizon: list[MpcHorizonStep] | None = None,
        conversion_options: ControlModelConversionOptions | None = None,
    ) -> MpcPlan:
        return self.plan(
            request,
            control_model=to_control_model(
                source_model,
                options=conversion_options,
            ),
            initial_state=initial_state,
            horizon=horizon,
        )

    def _resolve_control_model(
        self,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel | None,
    ) -> LinearThermalControlModel | Rc2StateThermalControlModel:
        if control_model is not None:
            return control_model
        if self.control_model_provider is None:
            raise ValueError(
                "control_model is required when no control_model_provider is configured"
            )
        return self.control_model_provider()

    def _resolve_initial_state(
        self,
        initial_state: MpcInitialState | Rc2StateMpcInitialState | None,
    ) -> MpcInitialState | Rc2StateMpcInitialState:
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

    @staticmethod
    def _apply_preheat_plan(
        *,
        horizon: list[MpcHorizonStep],
        preheat_plan,
    ) -> list[MpcHorizonStep]:
        if not preheat_plan or not preheat_plan.steps:
            return horizon
        updated_horizon: list[MpcHorizonStep] = []
        for step, preheat_step in zip(horizon, preheat_plan.steps, strict=False):
            updated_horizon.append(
                step.model_copy(
                    update={
                        "economic_target_c": preheat_step.economic_target_c,
                        "preheat_active": preheat_step.preheat_active,
                        "preheat_opportunity_score": preheat_step.preheat_opportunity_score,
                        "max_preheat_target_c": preheat_step.max_preheat_target_c,
                        "preheat_budget_share_kwh": preheat_step.preheat_budget_share_kwh,
                        "preheat_block_id": preheat_step.preheat_block_id,
                        "preheat_block_budget_kwh": preheat_step.preheat_block_budget_kwh,
                        "preheat_block_max_starts": preheat_step.preheat_block_max_starts,
                    }
                )
            )
        return updated_horizon

    @staticmethod
    def _apply_execution_targets(
        *,
        horizon: list[MpcHorizonStep],
        execution_targets: list[ExecutionTargetStep] | None,
        preheat_schedule: PreheatSchedule | None,
    ) -> list[MpcHorizonStep]:
        if not execution_targets:
            return horizon
        blocks_by_id = {
            block.block_id: block for block in (preheat_schedule.blocks if preheat_schedule else [])
        }
        updated_horizon: list[MpcHorizonStep] = []
        for step, target_step in zip(horizon, execution_targets, strict=False):
            block = (
                blocks_by_id.get(target_step.active_preheat_block_id)
                if target_step.active_preheat_block_id is not None
                else None
            )
            updated_horizon.append(
                step.model_copy(
                    update={
                        "economic_target_c": target_step.economic_target_c,
                        "preheat_active": target_step.active_preheat_block_id is not None,
                        "preheat_block_id": target_step.active_preheat_block_id,
                        "preheat_opportunity_score": 1.0 if target_step.active_preheat_block_id is not None else 0.0,
                        "max_preheat_target_c": target_step.max_preheat_target_c,
                        "preheat_budget_share_kwh": target_step.block_budget_share_kwh,
                        "preheat_block_budget_kwh": block.planned_charge_kwh if block is not None else 0.0,
                        "preheat_block_cumulative_target_kwh": target_step.block_cumulative_budget_target_kwh,
                        "preheat_block_max_starts": block.max_starts if block is not None else 0,
                    }
                )
            )
        return updated_horizon
