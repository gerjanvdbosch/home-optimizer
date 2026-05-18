from __future__ import annotations

from typing import Callable

from home_optimizer.features.mpc.explain import explain_heating_plan
from home_optimizer.features.mpc.flexibility_assessor import SpaceHeatingFlexibilityAssessor
from home_optimizer.features.mpc.models import (
    ExecutionTargetStep,
    LinearThermalControlModel,
    MpcHorizonStep,
    MpcInitialState,
    MpcProblem,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
)
from home_optimizer.features.mpc.space_heating_mpc import SpaceHeatingMpcSolver
from home_optimizer.features.mpc_new.models import (
    IntentAwareMpcControllerRequest,
    IntentAwareMpcPlan,
    RunExecutionState,
    RunIntentExecutionTargetStep,
    RunIntentPlan,
)
from home_optimizer.features.mpc_new.planner import RunSelectionPlanner
from home_optimizer.features.mpc_new.sequencer import IntentDrivenSequencer


class IntentAwareMpcControllerService:
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
        flexibility_assessor: SpaceHeatingFlexibilityAssessor | None = None,
        planner: RunSelectionPlanner | None = None,
        sequencer: IntentDrivenSequencer | None = None,
    ) -> None:
        self.solver = solver or SpaceHeatingMpcSolver()
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
        updated_horizon, legacy_targets = self._apply_intents_and_targets(
            horizon=resolved_horizon,
            execution_targets=execution_targets,
            intent_plan=intent_plan,
            flexibility=flexibility,
        )
        problem = MpcProblem(
            interval_minutes=request.interval_minutes,
            control_mode=request.control_mode,
            control_model=resolved_control_model,
            initial_state=resolved_initial_state,
            horizon=updated_horizon,
            thermal_flexibility=flexibility,
            execution_targets=legacy_targets,
            sequencer_state=None,
            constraints=request.constraints,
            objective_weights=request.objective_weights,
            max_solver_seconds=request.max_solver_seconds,
        )
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
            legacy_execution_targets=legacy_targets,
            diagnostics=diagnostics | {
                "heating_explanation": explain_heating_plan(
                    plan=base_plan,
                    control_model=resolved_control_model,
                    initial_state=resolved_initial_state,
                    horizon=updated_horizon,
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
    def _apply_intents_and_targets(
        *,
        horizon: list[MpcHorizonStep],
        execution_targets: list[RunIntentExecutionTargetStep],
        intent_plan: RunIntentPlan,
        flexibility,
    ) -> tuple[list[MpcHorizonStep], list[ExecutionTargetStep]]:
        intents_by_id = {intent.intent_id: intent for intent in intent_plan.intents}
        updated_horizon: list[MpcHorizonStep] = []
        legacy_targets: list[ExecutionTargetStep] = []
        for index, (step, target) in enumerate(zip(horizon, execution_targets, strict=False)):
            intent = intents_by_id.get(target.active_intent_id or target.eligible_intent_id)
            economic_target_c = (
                flexibility.steps[index].economic_target_c
                if index < len(flexibility.steps)
                else float(step.economic_target_c or step.temp_min_c)
            )
            preheat_active = intent is not None
            preheat_budget_share_kwh = 0.0
            preheat_block_budget_kwh = 0.0
            if intent is not None:
                window_steps = max(
                    1,
                    sum(
                        1
                        for horizon_step in horizon
                        if intent.start_window_start_utc
                        <= horizon_step.timestamp_utc
                        <= intent.start_window_end_utc
                    ),
                )
                preheat_budget_share_kwh = intent.target_charge_kwh / window_steps
                preheat_block_budget_kwh = intent.target_charge_kwh
            updated_horizon.append(
                step.model_copy(
                    update={
                        "economic_target_c": economic_target_c,
                        "preheat_active": preheat_active,
                        "preheat_opportunity_score": float(
                            intent.score if intent is not None else 0.0
                        ),
                        "max_preheat_target_c": (
                            intent.max_preheat_target_c
                            if intent is not None
                            else economic_target_c
                        ),
                        "preheat_budget_share_kwh": preheat_budget_share_kwh,
                        "preheat_block_id": intent.source_block_id if intent is not None else None,
                        "preheat_block_budget_kwh": preheat_block_budget_kwh,
                        "preheat_block_cumulative_target_kwh": (
                            preheat_budget_share_kwh
                            * (index + 1 if preheat_active else 0)
                        ),
                        "preheat_block_max_starts": intent.max_starts if intent is not None else 0,
                        "active_run_id": target.active_run_id,
                        "hp_must_be_on": target.hp_must_be_on,
                        "hp_must_be_off": target.hp_must_be_off,
                        "hp_start_allowed": target.hp_start_allowed,
                        "start_reason_hint": target.start_reason_hint,
                        "stop_reason_hint": target.stop_reason_hint,
                        "committed_on_until_utc": target.committed_on_until_utc,
                        "locked_off_until_utc": target.locked_off_until_utc,
                        "sequencer_mode": target.mode,
                    }
                )
            )
            legacy_targets.append(
                ExecutionTargetStep(
                    timestamp_utc=target.timestamp_utc,
                    economic_target_c=economic_target_c,
                    preheat_target_c=(
                        intent.max_preheat_target_c
                        if intent is not None
                        else economic_target_c
                    ),
                    active_preheat_block_id=intent.source_block_id if intent is not None else None,
                    remaining_block_budget_kwh=target.target_charge_remaining_kwh,
                    block_budget_share_kwh=preheat_budget_share_kwh,
                    block_cumulative_budget_target_kwh=(
                        preheat_budget_share_kwh
                        * (index + 1 if preheat_active else 0)
                    ),
                    storage_target_kwh=preheat_block_budget_kwh,
                    max_preheat_target_c=(
                        intent.max_preheat_target_c
                        if intent is not None
                        else economic_target_c
                    ),
                    start_allowed_for_preheat=target.hp_start_allowed and intent is not None,
                    start_reason_hint=target.start_reason_hint,
                    sequencer_mode=target.mode,
                    active_run_id=target.active_run_id,
                    hp_must_be_on=target.hp_must_be_on,
                    hp_must_be_off=target.hp_must_be_off,
                    hp_start_allowed=target.hp_start_allowed,
                    stop_reason_hint=target.stop_reason_hint,
                    committed_on_until_utc=target.committed_on_until_utc,
                    locked_off_until_utc=target.locked_off_until_utc,
                    starts_blocked_by_lockout=bool(
                        target.locked_off_until_utc is not None
                        and target.timestamp_utc < target.locked_off_until_utc
                    ),
                    starts_blocked_by_max_starts=False,
                    starts_blocked_by_existing_commitment=target.active_intent_id is not None,
                )
            )
        return updated_horizon, legacy_targets

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
