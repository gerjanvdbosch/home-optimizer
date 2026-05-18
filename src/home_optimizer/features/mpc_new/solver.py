from __future__ import annotations

from home_optimizer.features.mpc.models import MpcPlan, MpcProblem
from home_optimizer.features.mpc.space_heating_mpc import SpaceHeatingMpcSolver
from home_optimizer.features.mpc_new.models import IntentAwareMpcProblem


class IntentAwareMpcSolver:
    def __init__(
        self,
        *,
        base_solver: SpaceHeatingMpcSolver | None = None,
    ) -> None:
        self.base_solver = base_solver or SpaceHeatingMpcSolver()

    def solve(self, problem: IntentAwareMpcProblem) -> MpcPlan:
        annotated_horizon = self.annotate_horizon(problem)
        base_problem = MpcProblem(
            interval_minutes=problem.interval_minutes,
            control_mode=problem.control_mode,
            control_model=problem.control_model,
            initial_state=problem.initial_state,
            horizon=annotated_horizon,
            thermal_flexibility=problem.thermal_flexibility,
            constraints=problem.constraints,
            objective_weights=problem.objective_weights,
            max_solver_seconds=problem.max_solver_seconds,
        )
        return self.base_solver.solve(base_problem)

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
                problem.thermal_flexibility.steps[index].economic_target_c
                if problem.thermal_flexibility is not None
                and index < len(problem.thermal_flexibility.steps)
                else float(step.economic_target_c or step.temp_min_c)
            )
            preheat_active = intent is not None
            preheat_budget_share_kwh = 0.0
            preheat_block_budget_kwh = 0.0
            preheat_block_cumulative_target_kwh = 0.0
            if intent is not None:
                window_indices = [
                    idx
                    for idx, horizon_step in enumerate(problem.horizon)
                    if (
                        intent.start_window_start_utc
                        <= horizon_step.timestamp_utc
                        <= intent.start_window_end_utc
                    )
                ]
                window_steps = max(len(window_indices), 1)
                preheat_budget_share_kwh = intent.target_charge_kwh / window_steps
                preheat_block_budget_kwh = intent.target_charge_kwh
                if index in window_indices:
                    relative_position = window_indices.index(index) + 1
                    preheat_block_cumulative_target_kwh = (
                        preheat_budget_share_kwh * relative_position
                    )
            annotated_horizon.append(
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
                            preheat_block_cumulative_target_kwh
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
        return annotated_horizon
