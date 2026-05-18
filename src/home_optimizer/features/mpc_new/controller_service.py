from __future__ import annotations

from typing import Callable

from home_optimizer.features.mpc.explain import explain_heating_plan
from home_optimizer.features.mpc.flexibility_assessor import SpaceHeatingFlexibilityAssessor
from home_optimizer.features.mpc.models import (
    LinearThermalControlModel,
    MpcHorizonStep,
    MpcInitialState,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
)
from home_optimizer.features.mpc_new.models import (
    IntentAwareMpcControllerRequest,
    IntentAwareMpcPlan,
    IntentAwareMpcProblem,
    RunExecutionState,
    RunIntentExecutionTargetStep,
    RunIntentPlan,
)
from home_optimizer.features.mpc_new.planner import RunSelectionPlanner
from home_optimizer.features.mpc_new.sequencer import IntentDrivenSequencer
from home_optimizer.features.mpc_new.solver import IntentAwareMpcSolver


class IntentAwareMpcControllerService:
    def __init__(
        self,
        *,
        solver: IntentAwareMpcSolver | None = None,
        control_model_provider: Callable[
            [],
            LinearThermalControlModel | Rc2StateThermalControlModel,
        ] | None = None,
        horizon_provider: Callable[[], list[MpcHorizonStep]] | None = None,
        initial_state_provider: Callable[
            [],
            MpcInitialState | Rc2StateMpcInitialState,
        ] | None = None,
        flexibility_assessor: SpaceHeatingFlexibilityAssessor | None = None,
        planner: RunSelectionPlanner | None = None,
        sequencer: IntentDrivenSequencer | None = None,
    ) -> None:
        self.solver = solver or IntentAwareMpcSolver()
        self.control_model_provider = control_model_provider
        self.horizon_provider = horizon_provider
        self.initial_state_provider = initial_state_provider
        self.flexibility_assessor = flexibility_assessor or SpaceHeatingFlexibilityAssessor()
        self.planner = planner or RunSelectionPlanner()
        self.sequencer = sequencer or IntentDrivenSequencer()

    def plan(
        self,
        request: IntentAwareMpcControllerRequest,
        *,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel | None = None,
        initial_state: MpcInitialState | Rc2StateMpcInitialState | None = None,
        horizon: list[MpcHorizonStep] | None = None,
    ) -> IntentAwareMpcPlan:
        resolved_control_model = self._resolve_control_model(control_model)
        resolved_initial_state = self._resolve_initial_state(initial_state)
        resolved_horizon = self._resolve_horizon(horizon or request.horizon)
        flexibility = self.flexibility_assessor.assess(
            interval_minutes=request.interval_minutes,
            control_model=resolved_control_model,
            initial_state=resolved_initial_state,
            horizon=resolved_horizon,
            constraints=request.constraints,
        )
        intent_plan = self.planner.build_plan(
            flexibility_state=flexibility,
            constraints=request.constraints,
            interval_minutes=request.interval_minutes,
            control_model=resolved_control_model,
            initial_state=resolved_initial_state,
            horizon=resolved_horizon,
            planning_policy=request.planning_policy,
            previous_plan=request.previous_intent_plan,
            execution_state=request.run_execution_state,
        )
        execution_targets, projected_state = self.sequencer.build_execution_targets(
            horizon=resolved_horizon,
            flexibility_state=flexibility,
            intent_plan=intent_plan,
            constraints=request.constraints,
            execution_state=request.run_execution_state,
            interval_minutes=request.interval_minutes,
        )
        problem = IntentAwareMpcProblem(
            interval_minutes=request.interval_minutes,
            control_mode=request.control_mode,
            control_model=resolved_control_model,
            initial_state=resolved_initial_state,
            horizon=resolved_horizon,
            thermal_flexibility=flexibility,
            run_intent_plan=intent_plan,
            execution_targets=execution_targets,
            constraints=request.constraints,
            objective_weights=request.objective_weights,
            max_solver_seconds=request.max_solver_seconds,
        )
        annotated_horizon = self.solver.annotate_horizon(problem)
        base_plan = self.solver.solve(problem)
        diagnostics = self._summarize_plan(
            plan_intents=intent_plan,
            execution_targets=execution_targets,
            plan_steps=base_plan.steps,
        )
        return IntentAwareMpcPlan(
            control_mode=request.control_mode,
            status=base_plan.status,
            termination_condition=base_plan.termination_condition,
            feasible=base_plan.feasible,
            objective_value=base_plan.objective_value,
            solve_time_seconds=base_plan.solve_time_seconds,
            objective_breakdown=base_plan.objective_breakdown,
            steps=base_plan.steps,
            thermal_flexibility=flexibility,
            run_intent_plan=intent_plan,
            run_execution_state=projected_state,
            execution_targets=execution_targets,
            diagnostics=diagnostics | {
                "heating_explanation": explain_heating_plan(
                    plan=base_plan,
                    control_model=resolved_control_model,
                    initial_state=resolved_initial_state,
                    horizon=annotated_horizon,
                )
            },
        )

    def advance_execution_state(
        self,
        *,
        state: RunExecutionState,
        executed_step: MpcHorizonStep,
        executed_target: RunIntentExecutionTargetStep,
        executed_hp_on: bool,
        interval_minutes: int,
        preheat_charge_kwh: float,
    ) -> RunExecutionState:
        return self.sequencer.advance_state(
            state=state,
            executed_step=executed_step,
            executed_target=executed_target,
            executed_hp_on=executed_hp_on,
            interval_minutes=interval_minutes,
            preheat_charge_kwh=preheat_charge_kwh,
        )

    def _resolve_control_model(
        self,
        control_model: LinearThermalControlModel | Rc2StateThermalControlModel | None,
    ) -> LinearThermalControlModel | Rc2StateThermalControlModel:
        if control_model is not None:
            return control_model
        if self.control_model_provider is None:
            raise ValueError("control_model is required when no provider is configured")
        return self.control_model_provider()

    def _resolve_initial_state(
        self,
        initial_state: MpcInitialState | Rc2StateMpcInitialState | None,
    ) -> MpcInitialState | Rc2StateMpcInitialState:
        if initial_state is not None:
            return initial_state
        if self.initial_state_provider is None:
            raise ValueError("initial_state is required when no provider is configured")
        return self.initial_state_provider()

    def _resolve_horizon(self, horizon: list[MpcHorizonStep] | None) -> list[MpcHorizonStep]:
        if horizon is not None:
            return horizon
        if self.horizon_provider is None:
            raise ValueError("horizon is required when no provider is configured")
        return self.horizon_provider()

    @staticmethod
    def _summarize_plan(
        *,
        plan_intents: RunIntentPlan,
        execution_targets: list[RunIntentExecutionTargetStep],
        plan_steps,
    ) -> dict[str, float | int]:
        starts_outside_intents = 0
        comfort_fallback_run_count = 0
        short_run_count = 0
        run_durations: list[int] = []
        current_run_length = 0
        for target, step in zip(execution_targets, plan_steps, strict=False):
            if step.start and target.active_intent_id is None:
                starts_outside_intents += 1
            if step.start and target.start_reason_hint == "comfort_low_risk":
                comfort_fallback_run_count += 1
            if step.hp_on:
                current_run_length += 1
            elif current_run_length > 0:
                run_durations.append(current_run_length)
                if current_run_length <= 1:
                    short_run_count += 1
                current_run_length = 0
        if current_run_length > 0:
            run_durations.append(current_run_length)
            if current_run_length <= 1:
                short_run_count += 1
        starts_blocked_no_intent = sum(
            int(target.starts_blocked_no_intent) for target in execution_targets
        )
        return {
            "selected_intent_count": plan_intents.selected_intent_count,
            "active_intent_count": sum(
                int(target.active_intent_id is not None) for target in execution_targets
            ),
            "replaced_intent_count": sum(
                int(intent.replacement_reason is not None) for intent in plan_intents.intents
            ),
            "comfort_fallback_run_count": comfort_fallback_run_count,
            "starts_outside_intents": starts_outside_intents,
            "starts_blocked_no_intent": starts_blocked_no_intent,
            "short_run_count": short_run_count,
            "average_run_duration": (
                float(sum(run_durations) / len(run_durations)) if run_durations else 0.0
            ),
            "post_solar_hold_prediction_error": 0.0,
        }
