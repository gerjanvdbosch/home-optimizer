from __future__ import annotations

from typing import Any

from home_optimizer.features.mpc.models import MpcProblem
from home_optimizer.features.mpc.space_heating_mpc import SpaceHeatingMpcSolver
from home_optimizer.features.mpc_new.models import IntentAwareMpcProblem


class IntentAwareMpcSolver(SpaceHeatingMpcSolver):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_intent_problem: IntentAwareMpcProblem | None = None

    def solve(self, problem: IntentAwareMpcProblem):
        annotated_horizon = self.annotate_horizon(problem)
        self._current_intent_problem = problem
        try:
            return super().solve(
                MpcProblem(
                    interval_minutes=problem.interval_minutes,
                    control_mode=problem.control_mode,
                    control_model=problem.control_model,
                    initial_state=problem.initial_state,
                    horizon=annotated_horizon,
                    thermal_flexibility=None,
                    constraints=problem.constraints,
                    objective_weights=problem.objective_weights,
                    max_solver_seconds=problem.max_solver_seconds,
                )
            )
        finally:
            self._current_intent_problem = None

    @staticmethod
    def annotate_horizon(problem: IntentAwareMpcProblem):
        intents_by_id = {
            intent.intent_id: intent
            for intent in (problem.run_intent_plan.intents if problem.run_intent_plan else [])
        }
        annotated_horizon = []
        for index, (step, target) in enumerate(
            zip(problem.horizon, problem.execution_targets, strict=False)
        ):
            intent = intents_by_id.get(target.active_intent_id or target.eligible_intent_id)
            economic_target_c = (
                problem.intent_planning_state.steps[index].economic_target_c
                if problem.intent_planning_state is not None
                and index < len(problem.intent_planning_state.steps)
                else float(step.economic_target_c or step.temp_min_c)
            )
            annotated_horizon.append(
                step.model_copy(
                    update={
                        "economic_target_c": economic_target_c,
                        "preheat_active": intent is not None,
                        "preheat_opportunity_score": float(
                            intent.score if intent is not None else 0.0
                        ),
                        "max_preheat_target_c": (
                            intent.max_preheat_target_c
                            if intent is not None
                            else economic_target_c
                        ),
                        "preheat_budget_share_kwh": 0.0,
                        "preheat_block_id": None,
                        "preheat_block_budget_kwh": 0.0,
                        "preheat_block_cumulative_target_kwh": 0.0,
                        "preheat_block_max_starts": 0,
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
        return annotated_horizon

    def _apply_preheat_block_start_limits(
        self,
        *,
        model: Any,
        problem: MpcProblem,
        pyo: Any,
    ) -> None:
        intent_problem = self._current_intent_problem
        if intent_problem is None or intent_problem.run_intent_plan is None:
            return
        intent_step_indices: dict[str, list[int]] = {}
        for index, target in enumerate(intent_problem.execution_targets):
            intent_id = target.eligible_intent_id or target.active_intent_id
            if intent_id is None:
                continue
            intent_step_indices.setdefault(intent_id, []).append(index)
        if not intent_step_indices:
            return
        model.INTENTS = pyo.Set(initialize=list(intent_step_indices))
        max_starts_by_intent = {
            intent.intent_id: intent.max_starts
            for intent in intent_problem.run_intent_plan.intents
            if intent.intent_id in intent_step_indices
        }
        model.intent_max_starts = pyo.Param(
            model.INTENTS,
            initialize=max_starts_by_intent,
        )
        model.intent_start_limit = pyo.Constraint(
            model.INTENTS,
            rule=lambda model_ref, intent_id: sum(
                model_ref.start[index] for index in intent_step_indices[intent_id]
            ) <= model_ref.intent_max_starts[intent_id],
        )

    def _apply_preheat_block_budget_constraints(
        self,
        *,
        model: Any,
        problem: MpcProblem,
        pyo: Any,
    ) -> None:
        intent_problem = self._current_intent_problem
        if intent_problem is None or intent_problem.run_intent_plan is None:
            model.preheat_budget_shortfall = pyo.Var(domain=pyo.NonNegativeReals)
            model.preheat_budget_shortfall.fix(0.0)
            return
        intent_step_indices: dict[str, list[int]] = {}
        for index, target in enumerate(intent_problem.execution_targets):
            intent_id = target.eligible_intent_id or target.active_intent_id
            if intent_id is None:
                continue
            intent_step_indices.setdefault(intent_id, []).append(index)
        if not intent_step_indices:
            model.preheat_budget_shortfall = pyo.Var(domain=pyo.NonNegativeReals)
            model.preheat_budget_shortfall.fix(0.0)
            return
        if not hasattr(model, "INTENTS"):
            model.INTENTS = pyo.Set(initialize=list(intent_step_indices))
        model.preheat_budget_shortfall = pyo.Var(
            model.INTENTS,
            domain=pyo.NonNegativeReals,
        )
        model.intent_charge_budget = pyo.Param(
            model.INTENTS,
            initialize={
                intent.intent_id: intent.target_charge_kwh
                for intent in intent_problem.run_intent_plan.intents
                if intent.intent_id in intent_step_indices
            },
        )
        cumulative_targets: dict[int, float] = {}
        for intent in intent_problem.run_intent_plan.intents:
            indices = intent_step_indices.get(intent.intent_id, [])
            if not indices:
                continue
            per_step_target = intent.target_charge_kwh / max(len(indices), 1)
            for position, index in enumerate(indices, start=1):
                cumulative_targets[index] = per_step_target * position
        model.intent_step_target = pyo.Param(
            model.T,
            initialize=cumulative_targets,
            default=0.0,
        )
        model.intent_charge_limit = pyo.Constraint(
            model.INTENTS,
            rule=lambda model_ref, intent_id: sum(
                model_ref.preheat_charge[index] for index in intent_step_indices[intent_id]
            ) <= model_ref.intent_charge_budget[intent_id],
        )
        model.intent_budget_shortfall = pyo.Constraint(
            model.INTENTS,
            rule=lambda model_ref, intent_id: model_ref.preheat_budget_shortfall[intent_id]
            >= (
                model_ref.intent_charge_budget[intent_id]
                - sum(model_ref.preheat_charge[index] for index in intent_step_indices[intent_id])
            ),
        )
        model.preheat_budget_cumulative_shortfall = pyo.Var(
            model.T,
            domain=pyo.NonNegativeReals,
        )
        model.preheat_budget_cumulative_shortfall_limit = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.preheat_budget_cumulative_shortfall[t]
            >= (
                model_ref.intent_step_target[t]
                - sum(model_ref.preheat_charge[index] for index in range(t + 1))
            ),
        )

    def _preheat_budget_shortfall_expression(
        self,
        *,
        model: Any,
        problem: MpcProblem,
        pyo: Any,
    ) -> Any:
        if not hasattr(model, "INTENTS"):
            return 0.0
        block_shortfall_term = sum(
            problem.objective_weights.preheat_budget_shortfall
            * model.preheat_budget_shortfall[intent_id]
            for intent_id in model.INTENTS
        )
        cumulative_shortfall_term = (
            0.5
            * problem.objective_weights.preheat_budget_shortfall
            * sum(
                model.preheat_budget_cumulative_shortfall[t]
                for t in model.T
                if float(model.intent_step_target[t]) > 0.0
            )
        )
        return block_shortfall_term + cumulative_shortfall_term
