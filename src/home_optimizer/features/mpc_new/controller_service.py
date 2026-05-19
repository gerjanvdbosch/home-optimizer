from __future__ import annotations

from typing import Callable

from home_optimizer.features.modeling import RoomRcModel, TrainedLinearRoomModel
from home_optimizer.features.mpc.control_model import to_control_model
from home_optimizer.features.mpc.explain import explain_heating_plan
from home_optimizer.features.mpc.models import (
    ControlModelConversionOptions,
    MpcControllerRequest,
    MpcHorizonStep,
    MpcInitialState,
    Rc2StateMpcInitialState,
    Rc2StateThermalControlModel,
)
from home_optimizer.features.mpc_new.assessor import IntentPlanningAssessor
from home_optimizer.features.mpc_new.models import (
    AuthorityInvariantReport,
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
            Rc2StateThermalControlModel,
        ]
        | None = None,
        horizon_provider: Callable[[], list[MpcHorizonStep]] | None = None,
        initial_state_provider: Callable[
            [],
            MpcInitialState | Rc2StateMpcInitialState,
        ]
        | None = None,
        planning_assessor: IntentPlanningAssessor | None = None,
        planner: RunSelectionPlanner | None = None,
        sequencer: IntentDrivenSequencer | None = None,
    ) -> None:
        self.solver = solver or IntentAwareMpcSolver()
        self.control_model_provider = control_model_provider
        self.horizon_provider = horizon_provider
        self.initial_state_provider = initial_state_provider
        self.planning_assessor = planning_assessor or IntentPlanningAssessor()
        self.planner = planner or RunSelectionPlanner()
        self.sequencer = sequencer or IntentDrivenSequencer()

    def plan(
        self,
        request: IntentAwareMpcControllerRequest,
        *,
        control_model: Rc2StateThermalControlModel | None = None,
        initial_state: MpcInitialState | Rc2StateMpcInitialState | None = None,
        horizon: list[MpcHorizonStep] | None = None,
    ) -> IntentAwareMpcPlan:
        resolved_control_model = self._resolve_control_model(control_model)
        resolved_initial_state = self._resolve_initial_state(initial_state)
        resolved_horizon = self._resolve_horizon(horizon or request.horizon)
        planning_state = self.planning_assessor.assess(
            interval_minutes=request.interval_minutes,
            control_model=resolved_control_model,
            initial_state=resolved_initial_state,
            horizon=resolved_horizon,
            constraints=request.constraints,
        )
        intent_plan = self.planner.build_plan(
            planning_state=planning_state,
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
            planning_state=planning_state,
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
            intent_planning_state=planning_state,
            run_intent_plan=intent_plan,
            execution_targets=execution_targets,
            constraints=request.constraints,
            objective_weights=request.objective_weights,
            max_solver_seconds=request.max_solver_seconds,
        )
        annotated_horizon = self.solver.annotate_horizon(problem)
        try:
            base_plan = self.solver.solve(problem)
        except ValueError as exc:
            base_plan = self._solver_error_plan(str(exc))
        invariant_report = self._build_invariant_report(
            plan_steps=base_plan.steps,
            execution_targets=execution_targets,
            execution_state=request.run_execution_state,
        )
        diagnostics = self._summarize_plan(
            plan_intents=intent_plan,
            execution_targets=execution_targets,
            plan_steps=base_plan.steps,
            invariant_report=invariant_report,
        )
        return IntentAwareMpcPlan(
            control_mode=request.control_mode,
            status=base_plan.status,
            termination_condition=base_plan.termination_condition,
            feasible=base_plan.feasible,
            objective_value=base_plan.objective_value,
            solve_time_seconds=base_plan.solve_time_seconds,
            heating_explanation=explain_heating_plan(
                plan=base_plan,
                control_model=resolved_control_model,
                initial_state=resolved_initial_state,
                horizon=annotated_horizon,
            ),
            objective_breakdown=base_plan.objective_breakdown,
            steps=base_plan.steps,
            intent_planning_state=planning_state,
            run_intent_plan=intent_plan,
            run_execution_state=projected_state,
            execution_targets=execution_targets,
            start_stop_ledger=(
                request.run_execution_state.start_stop_ledger
                if request.run_execution_state is not None
                else []
            ),
            invariant_report=invariant_report,
            diagnostics=diagnostics
            | {
                "heating_explanation": explain_heating_plan(
                    plan=base_plan,
                    control_model=resolved_control_model,
                    initial_state=resolved_initial_state,
                    horizon=annotated_horizon,
                )
            },
        )

    def plan_from_source_model(
        self,
        request: IntentAwareMpcControllerRequest | MpcControllerRequest,
        *,
        source_model: TrainedLinearRoomModel | RoomRcModel,
        initial_state: MpcInitialState | Rc2StateMpcInitialState,
        horizon: list[MpcHorizonStep] | None = None,
        conversion_options: ControlModelConversionOptions | None = None,
    ) -> IntentAwareMpcPlan:
        normalized_request = (
            request
            if isinstance(request, IntentAwareMpcControllerRequest)
            else IntentAwareMpcControllerRequest(
                interval_minutes=request.interval_minutes,
                horizon=request.horizon,
                control_mode=request.control_mode,
                constraints=request.constraints,
                objective_weights=request.objective_weights,
                max_solver_seconds=request.max_solver_seconds,
            )
        )
        return self.plan(
            normalized_request,
            control_model=to_control_model(
                source_model,
                options=conversion_options,
            ),
            initial_state=initial_state,
            horizon=horizon,
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
        control_model: Rc2StateThermalControlModel | None,
    ) -> Rc2StateThermalControlModel:
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
        invariant_report: AuthorityInvariantReport,
    ) -> dict[str, float | int]:
        comfort_fallback_run_count = 0
        short_run_count = 0
        run_durations: list[int] = []
        current_run_length = 0
        for target, step in zip(execution_targets, plan_steps, strict=False):
            if step.start and target.start_reason_hint == "emergency_comfort_low":
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
            "starts_outside_intents": invariant_report.starts_outside_intents,
            "starts_blocked_no_intent": starts_blocked_no_intent,
            "short_run_count": short_run_count,
            "average_run_duration": (
                float(sum(run_durations) / len(run_durations)) if run_durations else 0.0
            ),
            "post_solar_hold_prediction_error": 0.0,
            "total_starts": invariant_report.total_starts,
            "total_stops": invariant_report.total_stops,
            "emergency_starts": invariant_report.emergency_starts,
            "external_starts": invariant_report.external_starts,
            "start_stop_violation_count": invariant_report.start_stop_violation_count,
        }

    @staticmethod
    def _solver_error_plan(message: str):
        from home_optimizer.features.mpc.models import MpcObjectiveBreakdown, MpcPlan

        return MpcPlan(
            status="error",
            termination_condition="infeasible",
            feasible=False,
            objective_breakdown=MpcObjectiveBreakdown(),
            heating_explanation=message,
            steps=[],
        )

    @staticmethod
    def _build_invariant_report(
        *,
        plan_steps,
        execution_targets: list[RunIntentExecutionTargetStep],
        execution_state: RunExecutionState | None,
    ) -> AuthorityInvariantReport:
        total_starts = 0
        total_stops = 0
        starts_by_reason: dict[str, int] = {}
        stops_by_reason: dict[str, int] = {}
        starts_outside_intents = 0
        emergency_starts = 0
        external_starts = 0
        violation_breakdown: dict[str, int] = {}
        for step, target in zip(plan_steps, execution_targets, strict=False):
            if step.start:
                total_starts += 1
                if step.start_reason is not None:
                    starts_by_reason[step.start_reason] = starts_by_reason.get(
                        step.start_reason, 0
                    ) + 1
                if target.active_intent_id is None and step.start_reason != "emergency_comfort_low":
                    starts_outside_intents += 1
                if step.start_reason == "emergency_comfort_low":
                    emergency_starts += 1
                if step.start_reason == "external_plant":
                    external_starts += 1
                if not target.hp_start_allowed and not target.hp_must_be_on:
                    violation_breakdown["hp_start_allowed_false_but_start_true"] = (
                        violation_breakdown.get("hp_start_allowed_false_but_start_true", 0) + 1
                    )
            if step.stop:
                total_stops += 1
                if step.stop_reason is not None:
                    stops_by_reason[step.stop_reason] = stops_by_reason.get(
                        step.stop_reason, 0
                    ) + 1
            if target.hp_must_be_on and not step.hp_on:
                violation_breakdown["hp_must_be_on_true_but_hp_on_false"] = (
                    violation_breakdown.get("hp_must_be_on_true_but_hp_on_false", 0) + 1
                )
            if target.hp_must_be_off and step.hp_on:
                violation_breakdown["hp_must_be_off_true_but_hp_on_true"] = (
                    violation_breakdown.get("hp_must_be_off_true_but_hp_on_true", 0) + 1
                )
            if step.start and step.start_reason is None:
                violation_breakdown["start_without_valid_reason"] = (
                    violation_breakdown.get("start_without_valid_reason", 0) + 1
                )
            if step.stop and step.stop_reason is None:
                violation_breakdown["stop_without_valid_reason"] = (
                    violation_breakdown.get("stop_without_valid_reason", 0) + 1
                )
        if execution_state is not None:
            for violation in execution_state.authority_violations:
                violation_breakdown[violation.violation] = (
                    violation_breakdown.get(violation.violation, 0) + 1
                )
        return AuthorityInvariantReport(
            total_starts=total_starts,
            total_stops=total_stops,
            starts_by_reason=starts_by_reason,
            stops_by_reason=stops_by_reason,
            starts_outside_intents=starts_outside_intents,
            emergency_starts=emergency_starts,
            external_starts=external_starts,
            start_stop_violation_count=sum(violation_breakdown.values()),
            violation_breakdown=violation_breakdown,
        )
