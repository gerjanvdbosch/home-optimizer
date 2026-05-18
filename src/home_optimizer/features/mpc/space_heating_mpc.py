from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from home_optimizer.features.mpc.explain import (
    rollout_with_full_heating,
    rollout_without_heating,
)
from home_optimizer.features.mpc.models import (
    MpcHorizonStep,
    MpcObjectiveBreakdown,
    MpcPlan,
    MpcPlanStep,
    MpcProblem,
    Rc2StateThermalControlModel,
)


@dataclass(slots=True)
class _ObjectiveContext:
    useful_preheat_targets_c: list[float]
    pv_self_consumable_kw: list[float]
    pv_opportunity_scores: list[float]
    comfort_high_big_m_c: list[float]
    unnecessary_heating_big_m_c: list[float]
    q_heat_eff_big_m_kw: float
    terminal_target_c: float


def _load_pyomo() -> Any:
    try:
        import pyomo.environ as pyo
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pyomo is required for space-heating MPC. Install project dependencies including "
            "'pyomo' and 'highspy'."
        ) from exc
    return pyo


class SpaceHeatingMpcSolver:
    def __init__(
        self,
        *,
        solver_name: str | None = None,
        solver_candidates: tuple[str, ...] = ("highs", "appsi_highs"),
    ) -> None:
        self.solver_name = solver_name
        self.solver_candidates = solver_candidates

    def solve(self, problem: MpcProblem) -> MpcPlan:
        pyo = _load_pyomo()
        model = self._build_model(problem, pyo)
        solver = self._build_solver(pyo, problem.max_solver_seconds)

        started_at = perf_counter()
        try:
            results = solver.solve(model)
        except Exception as exc:
            if exc.__class__.__name__ == "NoFeasibleSolutionError":
                return MpcPlan(
                    status="error",
                    termination_condition="infeasible",
                    feasible=False,
                    solve_time_seconds=perf_counter() - started_at,
                    objective_breakdown=MpcObjectiveBreakdown(),
                    steps=[],
                )
            raise
        solve_time_seconds = perf_counter() - started_at

        solver_status = str(results.solver.status)
        termination_condition = str(results.solver.termination_condition)
        feasible = termination_condition.lower() in {
            "optimal",
            "feasible",
            "locallyoptimal",
        }

        if not feasible:
            return MpcPlan(
                status=solver_status,
                termination_condition=termination_condition,
                feasible=False,
                solve_time_seconds=solve_time_seconds,
                objective_breakdown=MpcObjectiveBreakdown(),
                steps=[],
            )

        objective_breakdown = MpcObjectiveBreakdown(
            comfort_low=float(pyo.value(model.comfort_low_term)),
            active_comfort_high=float(pyo.value(model.active_comfort_high_term)),
            passive_comfort_high=float(pyo.value(model.passive_comfort_high_term)),
            tracking_under_target=float(pyo.value(model.tracking_under_target_term)),
            tracking_over_target=float(pyo.value(model.tracking_over_target_term)),
            unnecessary_heating=float(pyo.value(model.unnecessary_heating_term)),
            terminal=float(pyo.value(model.terminal_term)),
            start=float(pyo.value(model.start_term)),
            runtime=float(pyo.value(model.runtime_term)),
            energy_cost=float(pyo.value(model.energy_term - model.energy_baseline_term)),
            pv_self_consumption_reward=float(pyo.value(model.pv_self_consumption_reward_term)),
            captured_pv_kwh=float(pyo.value(model.captured_pv_kwh_term)),
            preheat_budget_shortfall=float(pyo.value(model.preheat_budget_shortfall_term)),
        )
        objective_value = float(pyo.value(model.objective))
        steps: list[MpcPlanStep] = []
        for index, horizon_step in enumerate(problem.horizon):
            hp_on = bool(round(pyo.value(model.hp_on[index])))
            q_heat_eff_kw = float(pyo.value(model.q_heat_eff[index]))
            effective_heating_kw = q_heat_eff_kw
            site_energy_cost = float(
                (
                    (horizon_step.import_price_eur_kwh * pyo.value(model.grid_import[index]))
                    - (horizon_step.export_price_eur_kwh * pyo.value(model.grid_export[index]))
                )
                * problem.dt_hours
            )
            baseline_energy_cost = float(
                _baseline_site_energy_cost(horizon_step) * problem.dt_hours
            )
            steps.append(
                MpcPlanStep(
                    timestamp_utc=horizon_step.timestamp_utc,
                    hp_on=hp_on,
                    start=bool(round(pyo.value(model.start[index]))),
                    stop=bool(round(pyo.value(model.stop[index]))),
                    predicted_room_temp_c=float(pyo.value(model.room_temp[index])),
                    economic_target_c=float(
                        horizon_step.economic_target_c or horizon_step.temp_min_c
                    ),
                    useful_preheat_target_c=float(pyo.value(model.useful_preheat_target[index])),
                    preheat_active=bool(horizon_step.preheat_active),
                    preheat_opportunity_score=float(horizon_step.preheat_opportunity_score),
                    preheat_budget_share_kwh=float(horizon_step.preheat_budget_share_kwh),
                    preheat_charge_kwh=float(pyo.value(model.preheat_charge[index])),
                    preheat_block_id=horizon_step.preheat_block_id,
                    preheat_block_budget_kwh=float(horizon_step.preheat_block_budget_kwh),
                    q_heat_eff_kw=q_heat_eff_kw,
                    sequencer_mode=horizon_step.sequencer_mode,
                    active_run_id=horizon_step.active_run_id,
                    hp_must_be_on=bool(horizon_step.hp_must_be_on),
                    hp_must_be_off=bool(horizon_step.hp_must_be_off),
                    hp_start_allowed=bool(horizon_step.hp_start_allowed),
                    start_reason=horizon_step.start_reason_hint,
                    stop_reason=horizon_step.stop_reason_hint,
                    committed_on_until_utc=horizon_step.committed_on_until_utc,
                    locked_off_until_utc=horizon_step.locked_off_until_utc,
                    starts_used_in_block=int(horizon_step.starts_used_in_block),
                    run_budget_used_kwh=float(horizon_step.run_budget_used_kwh),
                    starts_blocked_by_lockout=bool(horizon_step.starts_blocked_by_lockout),
                    starts_blocked_by_max_starts=bool(horizon_step.starts_blocked_by_max_starts),
                    starts_blocked_by_existing_commitment=bool(
                        horizon_step.starts_blocked_by_existing_commitment
                    ),
                    temp_min_c=horizon_step.temp_min_c,
                    temp_max_c=horizon_step.temp_max_c,
                    slack_low_c=float(pyo.value(model.slack_low[index])),
                    slack_high_c=float(pyo.value(model.slack_high[index])),
                    effective_heating_kw=effective_heating_kw,
                    price_eur_kwh=horizon_step.import_price_eur_kwh,
                    estimated_energy_cost_eur=site_energy_cost - baseline_energy_cost,
                )
            )

        return MpcPlan(
            status=solver_status,
            termination_condition=termination_condition,
            feasible=True,
            objective_value=objective_value,
            solve_time_seconds=solve_time_seconds,
            objective_breakdown=objective_breakdown,
            steps=steps,
        )

    def _build_solver(self, pyo: Any, max_solver_seconds: float | None) -> Any:
        solver_names = (self.solver_name,) if self.solver_name else self.solver_candidates
        last_error: Exception | None = None
        for solver_name in solver_names:
            if solver_name is None:
                continue
            solver = pyo.SolverFactory(solver_name)
            if solver is None:
                continue
            try:
                available = bool(solver.available(exception_flag=False))
            except Exception as exc:  # pragma: no cover - solver-specific behavior
                last_error = exc
                continue
            if not available:
                continue
            if max_solver_seconds is not None:
                self._apply_time_limit(solver, max_solver_seconds)
            return solver

        if last_error is not None:
            raise RuntimeError("Unable to initialize a HiGHS solver for MPC") from last_error
        raise RuntimeError(
            "No available HiGHS solver found for space-heating MPC. "
            "Install 'highspy' and ensure Pyomo can access 'appsi_highs' or 'highs'."
        )

    @staticmethod
    def _apply_time_limit(solver: Any, max_solver_seconds: float) -> None:
        if hasattr(solver, "config") and hasattr(solver.config, "time_limit"):
            solver.config.time_limit = max_solver_seconds
            return
        options = getattr(solver, "options", None)
        if options is not None:
            options["time_limit"] = max_solver_seconds

    def _build_model(self, problem: MpcProblem, pyo: Any) -> Any:
        if isinstance(problem.control_model, Rc2StateThermalControlModel):
            return self._build_2state_model(problem, pyo)
        return self._build_linear_model(problem, pyo)

    def _build_linear_model(self, problem: MpcProblem, pyo: Any) -> Any:
        model = pyo.ConcreteModel(name="space_heating_mpc")
        horizon_size = len(problem.horizon)
        objective_context = self._build_objective_context(problem)
        model.T = pyo.RangeSet(0, horizon_size - 1)
        model.T_transition = pyo.RangeSet(0, max(0, horizon_size - 2))

        model.hp_on = pyo.Var(model.T, domain=pyo.Binary)
        model.start = pyo.Var(model.T, domain=pyo.Binary)
        model.stop = pyo.Var(model.T, domain=pyo.Binary)
        model.grid_import = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.grid_export = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.room_temp = pyo.Var(model.T, domain=pyo.Reals)
        model.q_heat_eff = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.slack_low = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.slack_high = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.active_comfort_high = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.passive_comfort_high = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.active_heating_state = pyo.Var(model.T, domain=pyo.Binary)
        model.track_under = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.track_over = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.unnecessary_heat_excess = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.preheat_charge = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.terminal_under = pyo.Var(domain=pyo.NonNegativeReals)
        model.useful_preheat_target = pyo.Param(
            model.T,
            initialize={
                index: objective_context.useful_preheat_targets_c[index]
                for index in range(horizon_size)
            },
        )
        model.unnecessary_heating_big_m = pyo.Param(
            model.T,
            initialize={
                index: objective_context.unnecessary_heating_big_m_c[index]
                for index in range(horizon_size)
            },
        )
        model.comfort_high_big_m = pyo.Param(
            model.T,
            initialize={
                index: objective_context.comfort_high_big_m_c[index]
                for index in range(horizon_size)
            },
        )
        model.pv_self_consumable_kw = pyo.Param(
            model.T,
            initialize={
                index: objective_context.pv_self_consumable_kw[index]
                for index in range(horizon_size)
            },
        )
        model.pv_opportunity_score = pyo.Param(
            model.T,
            initialize={
                index: objective_context.pv_opportunity_scores[index]
                for index in range(horizon_size)
            },
        )
        model.q_heat_eff_active_big_m = pyo.Param(
            initialize=objective_context.q_heat_eff_big_m_kw
        )
        model.terminal_target = pyo.Param(initialize=objective_context.terminal_target_c)

        initial_hp_on = 1 if problem.initial_state.hp_on else 0
        model.initial_room_temp = pyo.Param(initialize=problem.initial_state.room_temp_c)
        model.initial_q_heat_eff = pyo.Param(initialize=problem.initial_state.q_heat_eff_kw)
        model.initial_hp_on = pyo.Param(initialize=initial_hp_on)
        model.room_temp[0].fix(problem.initial_state.room_temp_c)

        def dynamics_rule(model_ref: Any, t: int) -> Any:
            step = problem.horizon[t]
            return model_ref.room_temp[t + 1] == (
                (problem.control_model.a * model_ref.room_temp[t])
                + (problem.control_model.b_out * step.outdoor_temp_c)
                + (problem.control_model.b_solar * step.solar_gain_kw)
                + (problem.control_model.b_heat * model_ref.q_heat_eff[t])
                + (problem.control_model.b_occ * step.occupied)
                + problem.control_model.c
            )

        if horizon_size >= 2:
            model.dynamics = pyo.Constraint(model.T_transition, rule=dynamics_rule)

        model.heat_actuator = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.q_heat_eff[t]
            == (
                problem.control_model.actuator_alpha
                * (model_ref.initial_q_heat_eff if t == 0 else model_ref.q_heat_eff[t - 1])
            )
            + (
                (1.0 - problem.control_model.actuator_alpha)
                * problem.horizon[t].effective_heating_kw_forecast
                * model_ref.hp_on[t]
            ),
        )

        def comfort_low_rule(model_ref: Any, t: int) -> Any:
            return (
                problem.horizon[t].temp_min_c - model_ref.slack_low[t]
                <= model_ref.room_temp[t]
            )

        def comfort_high_rule(model_ref: Any, t: int) -> Any:
            return (
                model_ref.room_temp[t]
                <= problem.horizon[t].temp_max_c + model_ref.slack_high[t]
            )

        model.comfort_low = pyo.Constraint(model.T, rule=comfort_low_rule)
        model.comfort_high = pyo.Constraint(model.T, rule=comfort_high_rule)
        model.comfort_high_split = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.slack_high[t]
            == model_ref.active_comfort_high[t] + model_ref.passive_comfort_high[t],
        )
        model.active_heating_from_hp = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.active_heating_state[t] >= model_ref.hp_on[t],
        )
        model.active_heating_from_q = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.q_heat_eff[t]
            <= problem.objective_weights.q_heat_eff_active_threshold_kw
            + (model_ref.q_heat_eff_active_big_m * model_ref.active_heating_state[t]),
        )
        model.active_comfort_high_gate = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.active_comfort_high[t]
            <= (model_ref.comfort_high_big_m[t] * model_ref.active_heating_state[t]),
        )
        model.passive_comfort_high_gate = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.passive_comfort_high[t]
            <= (model_ref.comfort_high_big_m[t] * (1 - model_ref.active_heating_state[t])),
        )
        model.tracking_under = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.track_under[t]
            >= (model_ref.useful_preheat_target[t] - model_ref.room_temp[t]),
        )
        model.tracking_over = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.track_over[t]
            >= (model_ref.room_temp[t] - model_ref.useful_preheat_target[t]),
        )
        model.unnecessary_heating = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.unnecessary_heat_excess[t]
            >= (
                model_ref.room_temp[t]
                - model_ref.useful_preheat_target[t]
                - (
                    model_ref.unnecessary_heating_big_m[t]
                    * (1 - model_ref.active_heating_state[t])
                )
            ),
        )

        def transition_rule(model_ref: Any, t: int) -> Any:
            previous_hp_on = model_ref.initial_hp_on if t == 0 else model_ref.hp_on[t - 1]
            return model_ref.start[t] - model_ref.stop[t] == model_ref.hp_on[t] - previous_hp_on

        model.transition = pyo.Constraint(model.T, rule=transition_rule)
        model.start_stop_mutex = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.start[t] + model_ref.stop[t] <= 1,
        )
        self._apply_execution_constraints(model=model, problem=problem, pyo=pyo)
        model.grid_balance = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: (
                model_ref.grid_import[t] - model_ref.grid_export[t]
                == problem.horizon[t].base_load_power_forecast_kw
                + (problem.horizon[t].hp_electric_power_forecast_kw * model_ref.hp_on[t])
                - problem.horizon[t].pv_available_power_forecast_kw
            ),
        )
        model.preheat_charge_limit = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.preheat_charge[t]
            <= (problem.dt_hours * model_ref.pv_self_consumable_kw[t] * model_ref.hp_on[t]),
        )

        if problem.constraints.min_on_steps > 1:
            model.min_on = pyo.Constraint(
                model.T,
                rule=lambda model_ref, t: sum(
                    model_ref.start[k]
                    for k in range(max(0, t - problem.constraints.min_on_steps + 1), t + 1)
                )
                <= model_ref.hp_on[t],
            )

        if problem.constraints.min_off_steps > 1:
            model.min_off = pyo.Constraint(
                model.T,
                rule=lambda model_ref, t: sum(
                    model_ref.stop[k]
                    for k in range(max(0, t - problem.constraints.min_off_steps + 1), t + 1)
                )
                <= 1 - model_ref.hp_on[t],
            )

        self._apply_preheat_block_start_limits(
            model=model,
            problem=problem,
            pyo=pyo,
        )
        self._apply_preheat_block_budget_constraints(
            model=model,
            problem=problem,
            pyo=pyo,
        )

        remaining_on_steps = 0
        if problem.initial_state.hp_on:
            remaining_on_steps = max(
                0,
                problem.constraints.min_on_steps - problem.initial_state.on_steps,
            )
        for t in range(min(remaining_on_steps, horizon_size)):
            model.hp_on[t].fix(1)
            model.stop[t].fix(0)

        remaining_off_steps = 0
        if not problem.initial_state.hp_on:
            remaining_off_steps = max(
                0,
                problem.constraints.min_off_steps - problem.initial_state.off_steps,
            )
        for t in range(min(remaining_off_steps, horizon_size)):
            model.hp_on[t].fix(0)
            model.start[t].fix(0)

        model.terminal_tracking = pyo.Constraint(
            expr=model.terminal_under
            >= (model.terminal_target - model.room_temp[horizon_size - 1])
        )

        comfort_low_term = sum(
            problem.objective_weights.comfort_low
            * problem.dt_hours
            * model.slack_low[t]
            for t in range(horizon_size)
        )
        active_comfort_high_term = sum(
            float(problem.objective_weights.active_comfort_high or 0.0)
            * problem.dt_hours
            * model.active_comfort_high[t]
            for t in range(horizon_size)
        )
        passive_comfort_high_term = sum(
            problem.objective_weights.passive_comfort_high
            * problem.dt_hours
            * model.passive_comfort_high[t]
            for t in range(horizon_size)
        )
        start_term = sum(
            problem.objective_weights.start * model.start[t] for t in range(horizon_size)
        )
        tracking_under_target_term = sum(
            problem.objective_weights.tracking_under_target
            * model.track_under[t]
            for t in range(horizon_size)
        )
        tracking_over_target_term = sum(
            problem.objective_weights.tracking_over_target
            * model.track_over[t]
            for t in range(horizon_size)
        )
        unnecessary_heating_term = sum(
            problem.objective_weights.unnecessary_heating
            * model.unnecessary_heat_excess[t]
            for t in range(horizon_size)
        )
        energy_term = sum(
            problem.objective_weights.energy
            * problem.dt_hours
            * (
                (problem.horizon[t].import_price_eur_kwh * model.grid_import[t])
                - (problem.horizon[t].export_price_eur_kwh * model.grid_export[t])
            )
            for t in range(horizon_size)
        )
        energy_baseline_term = sum(
            problem.objective_weights.energy
            * problem.dt_hours
            * _baseline_site_energy_cost(problem.horizon[t])
            for t in range(horizon_size)
        )
        pv_self_consumption_reward_term = sum(
            problem.objective_weights.pv_self_consumption
            * model.preheat_charge[t]
            * model.pv_opportunity_score[t]
            for t in range(horizon_size)
        )
        captured_pv_kwh_term = sum(
            model.preheat_charge[t]
            for t in range(horizon_size)
        )
        runtime_term = sum(
            problem.objective_weights.runtime * model.hp_on[t] for t in range(horizon_size)
        )
        terminal_term = problem.objective_weights.terminal * model.terminal_under
        preheat_budget_shortfall_term = self._preheat_budget_shortfall_expression(
            model=model,
            problem=problem,
            pyo=pyo,
        )
        model.comfort_low_term = pyo.Expression(expr=comfort_low_term)
        model.active_comfort_high_term = pyo.Expression(expr=active_comfort_high_term)
        model.passive_comfort_high_term = pyo.Expression(expr=passive_comfort_high_term)
        model.comfort_high_term = pyo.Expression(
            expr=model.active_comfort_high_term + model.passive_comfort_high_term
        )
        model.tracking_under_target_term = pyo.Expression(expr=tracking_under_target_term)
        model.tracking_over_target_term = pyo.Expression(expr=tracking_over_target_term)
        model.unnecessary_heating_term = pyo.Expression(expr=unnecessary_heating_term)
        model.terminal_term = pyo.Expression(expr=terminal_term)
        model.start_term = pyo.Expression(expr=start_term)
        model.energy_term = pyo.Expression(expr=energy_term)
        model.energy_baseline_term = pyo.Expression(expr=energy_baseline_term)
        model.pv_self_consumption_reward_term = pyo.Expression(expr=pv_self_consumption_reward_term)
        model.captured_pv_kwh_term = pyo.Expression(expr=captured_pv_kwh_term)
        model.preheat_budget_shortfall_term = pyo.Expression(expr=preheat_budget_shortfall_term)
        model.runtime_term = pyo.Expression(expr=runtime_term)
        model.objective = pyo.Objective(
            expr=(
                model.comfort_low_term
                + model.comfort_high_term
                + model.tracking_under_target_term
                + model.tracking_over_target_term
                + model.unnecessary_heating_term
                + model.terminal_term
                + model.start_term
                + (model.energy_term - model.energy_baseline_term)
                + model.preheat_budget_shortfall_term
                - model.pv_self_consumption_reward_term
                + model.runtime_term
            ),
            sense=pyo.minimize,
        )
        return model

    def _build_2state_model(self, problem: MpcProblem, pyo: Any) -> Any:
        model = pyo.ConcreteModel(name="space_heating_mpc_2state")
        horizon_size = len(problem.horizon)
        objective_context = self._build_objective_context(problem)
        model.T = pyo.RangeSet(0, horizon_size - 1)
        model.T_transition = pyo.RangeSet(0, max(0, horizon_size - 2))

        model.hp_on = pyo.Var(model.T, domain=pyo.Binary)
        model.start = pyo.Var(model.T, domain=pyo.Binary)
        model.stop = pyo.Var(model.T, domain=pyo.Binary)
        model.grid_import = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.grid_export = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.room_temp = pyo.Var(model.T, domain=pyo.Reals)
        model.mass_temp = pyo.Var(model.T, domain=pyo.Reals)
        model.q_heat_eff = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.slack_low = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.slack_high = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.active_comfort_high = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.passive_comfort_high = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.active_heating_state = pyo.Var(model.T, domain=pyo.Binary)
        model.track_under = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.track_over = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.unnecessary_heat_excess = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.preheat_charge = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.terminal_under = pyo.Var(domain=pyo.NonNegativeReals)
        model.useful_preheat_target = pyo.Param(
            model.T,
            initialize={
                index: objective_context.useful_preheat_targets_c[index]
                for index in range(horizon_size)
            },
        )
        model.unnecessary_heating_big_m = pyo.Param(
            model.T,
            initialize={
                index: objective_context.unnecessary_heating_big_m_c[index]
                for index in range(horizon_size)
            },
        )
        model.comfort_high_big_m = pyo.Param(
            model.T,
            initialize={
                index: objective_context.comfort_high_big_m_c[index]
                for index in range(horizon_size)
            },
        )
        model.pv_self_consumable_kw = pyo.Param(
            model.T,
            initialize={
                index: objective_context.pv_self_consumable_kw[index]
                for index in range(horizon_size)
            },
        )
        model.pv_opportunity_score = pyo.Param(
            model.T,
            initialize={
                index: objective_context.pv_opportunity_scores[index]
                for index in range(horizon_size)
            },
        )
        model.q_heat_eff_active_big_m = pyo.Param(
            initialize=objective_context.q_heat_eff_big_m_kw
        )
        model.terminal_target = pyo.Param(initialize=objective_context.terminal_target_c)

        initial_hp_on = 1 if problem.initial_state.hp_on else 0
        model.initial_room_temp = pyo.Param(initialize=problem.initial_state.room_temp_c)
        model.initial_mass_temp = pyo.Param(initialize=problem.initial_state.mass_temp_c)
        model.initial_q_heat_eff = pyo.Param(initialize=problem.initial_state.q_heat_eff_kw)
        model.initial_hp_on = pyo.Param(initialize=initial_hp_on)
        model.room_temp[0].fix(problem.initial_state.room_temp_c)
        model.mass_temp[0].fix(problem.initial_state.mass_temp_c)

        def room_dynamics_rule(model_ref: Any, t: int) -> Any:
            step = problem.horizon[t]
            return model_ref.room_temp[t + 1] == (
                (problem.control_model.a11 * model_ref.room_temp[t])
                + (problem.control_model.a12 * model_ref.mass_temp[t])
                + (problem.control_model.b_out_room * step.outdoor_temp_c)
                + (problem.control_model.b_heat_room * model_ref.q_heat_eff[t])
                + (problem.control_model.b_solar_direct_room * step.solar_gain_kw)
                + (
                    problem.control_model.b_solar_filtered_room
                    * float(step.solar_gain_mass_kw)
                )
                + (problem.control_model.b_occ_room * step.occupied)
                + (problem.control_model.b_hour_sin_room * step.hour_sin)
                + (problem.control_model.b_hour_cos_room * step.hour_cos)
                + problem.control_model.c_room
            )

        def mass_dynamics_rule(model_ref: Any, t: int) -> Any:
            step = problem.horizon[t]
            return model_ref.mass_temp[t + 1] == (
                (problem.control_model.a21 * model_ref.room_temp[t])
                + (problem.control_model.a22 * model_ref.mass_temp[t])
                + (problem.control_model.b_out_mass * step.outdoor_temp_c)
                + (problem.control_model.b_heat_mass * model_ref.q_heat_eff[t])
                + (problem.control_model.b_solar_direct_mass * step.solar_gain_kw)
                + (
                    problem.control_model.b_solar_filtered_mass
                    * float(step.solar_gain_mass_kw)
                )
                + (problem.control_model.b_occ_mass * step.occupied)
                + (problem.control_model.b_hour_sin_mass * step.hour_sin)
                + (problem.control_model.b_hour_cos_mass * step.hour_cos)
                + problem.control_model.c_mass
            )

        if horizon_size >= 2:
            model.room_dynamics = pyo.Constraint(model.T_transition, rule=room_dynamics_rule)
            model.mass_dynamics = pyo.Constraint(model.T_transition, rule=mass_dynamics_rule)

        model.heat_actuator = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.q_heat_eff[t]
            == (
                problem.control_model.actuator_alpha
                * (model_ref.initial_q_heat_eff if t == 0 else model_ref.q_heat_eff[t - 1])
            )
            + (
                (1.0 - problem.control_model.actuator_alpha)
                * problem.horizon[t].effective_heating_kw_forecast
                * model_ref.hp_on[t]
            ),
        )

        def comfort_low_rule(model_ref: Any, t: int) -> Any:
            return (
                problem.horizon[t].temp_min_c - model_ref.slack_low[t]
                <= model_ref.room_temp[t]
            )

        def comfort_high_rule(model_ref: Any, t: int) -> Any:
            return (
                model_ref.room_temp[t]
                <= problem.horizon[t].temp_max_c + model_ref.slack_high[t]
            )

        model.comfort_low = pyo.Constraint(model.T, rule=comfort_low_rule)
        model.comfort_high = pyo.Constraint(model.T, rule=comfort_high_rule)
        model.comfort_high_split = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.slack_high[t]
            == model_ref.active_comfort_high[t] + model_ref.passive_comfort_high[t],
        )
        model.active_heating_from_hp = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.active_heating_state[t] >= model_ref.hp_on[t],
        )
        model.active_heating_from_q = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.q_heat_eff[t]
            <= problem.objective_weights.q_heat_eff_active_threshold_kw
            + (model_ref.q_heat_eff_active_big_m * model_ref.active_heating_state[t]),
        )
        model.active_comfort_high_gate = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.active_comfort_high[t]
            <= (model_ref.comfort_high_big_m[t] * model_ref.active_heating_state[t]),
        )
        model.passive_comfort_high_gate = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.passive_comfort_high[t]
            <= (model_ref.comfort_high_big_m[t] * (1 - model_ref.active_heating_state[t])),
        )
        model.tracking_under = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.track_under[t]
            >= (model_ref.useful_preheat_target[t] - model_ref.room_temp[t]),
        )
        model.tracking_over = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.track_over[t]
            >= (model_ref.room_temp[t] - model_ref.useful_preheat_target[t]),
        )
        model.unnecessary_heating = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.unnecessary_heat_excess[t]
            >= (
                model_ref.room_temp[t]
                - model_ref.useful_preheat_target[t]
                - (
                    model_ref.unnecessary_heating_big_m[t]
                    * (1 - model_ref.active_heating_state[t])
                )
            ),
        )

        def transition_rule(model_ref: Any, t: int) -> Any:
            previous_hp_on = model_ref.initial_hp_on if t == 0 else model_ref.hp_on[t - 1]
            return model_ref.start[t] - model_ref.stop[t] == model_ref.hp_on[t] - previous_hp_on

        model.transition = pyo.Constraint(model.T, rule=transition_rule)
        model.start_stop_mutex = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.start[t] + model_ref.stop[t] <= 1,
        )
        self._apply_execution_constraints(model=model, problem=problem, pyo=pyo)
        model.grid_balance = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: (
                model_ref.grid_import[t] - model_ref.grid_export[t]
                == problem.horizon[t].base_load_power_forecast_kw
                + (problem.horizon[t].hp_electric_power_forecast_kw * model_ref.hp_on[t])
                - problem.horizon[t].pv_available_power_forecast_kw
            ),
        )
        model.preheat_charge_limit = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.preheat_charge[t]
            <= (problem.dt_hours * model_ref.pv_self_consumable_kw[t] * model_ref.hp_on[t]),
        )

        if problem.constraints.min_on_steps > 1:
            model.min_on = pyo.Constraint(
                model.T,
                rule=lambda model_ref, t: sum(
                    model_ref.start[k]
                    for k in range(max(0, t - problem.constraints.min_on_steps + 1), t + 1)
                )
                <= model_ref.hp_on[t],
            )

        if problem.constraints.min_off_steps > 1:
            model.min_off = pyo.Constraint(
                model.T,
                rule=lambda model_ref, t: sum(
                    model_ref.stop[k]
                    for k in range(max(0, t - problem.constraints.min_off_steps + 1), t + 1)
                )
                <= 1 - model_ref.hp_on[t],
            )

        self._apply_preheat_block_start_limits(
            model=model,
            problem=problem,
            pyo=pyo,
        )
        self._apply_preheat_block_budget_constraints(
            model=model,
            problem=problem,
            pyo=pyo,
        )

        remaining_on_steps = 0
        if problem.initial_state.hp_on:
            remaining_on_steps = max(
                0,
                problem.constraints.min_on_steps - problem.initial_state.on_steps,
            )
        for t in range(min(remaining_on_steps, horizon_size)):
            model.hp_on[t].fix(1)
            model.stop[t].fix(0)

        remaining_off_steps = 0
        if not problem.initial_state.hp_on:
            remaining_off_steps = max(
                0,
                problem.constraints.min_off_steps - problem.initial_state.off_steps,
            )
        for t in range(min(remaining_off_steps, horizon_size)):
            model.hp_on[t].fix(0)
            model.start[t].fix(0)

        model.terminal_tracking = pyo.Constraint(
            expr=model.terminal_under
            >= (model.terminal_target - model.room_temp[horizon_size - 1])
        )

        comfort_low_term = sum(
            problem.objective_weights.comfort_low
            * problem.dt_hours
            * model.slack_low[t]
            for t in range(horizon_size)
        )
        active_comfort_high_term = sum(
            float(problem.objective_weights.active_comfort_high or 0.0)
            * problem.dt_hours
            * model.active_comfort_high[t]
            for t in range(horizon_size)
        )
        passive_comfort_high_term = sum(
            problem.objective_weights.passive_comfort_high
            * problem.dt_hours
            * model.passive_comfort_high[t]
            for t in range(horizon_size)
        )
        start_term = sum(
            problem.objective_weights.start * model.start[t] for t in range(horizon_size)
        )
        tracking_under_target_term = sum(
            problem.objective_weights.tracking_under_target
            * model.track_under[t]
            for t in range(horizon_size)
        )
        tracking_over_target_term = sum(
            problem.objective_weights.tracking_over_target
            * model.track_over[t]
            for t in range(horizon_size)
        )
        unnecessary_heating_term = sum(
            problem.objective_weights.unnecessary_heating
            * model.unnecessary_heat_excess[t]
            for t in range(horizon_size)
        )
        energy_term = sum(
            problem.objective_weights.energy
            * problem.dt_hours
            * (
                (problem.horizon[t].import_price_eur_kwh * model.grid_import[t])
                - (problem.horizon[t].export_price_eur_kwh * model.grid_export[t])
            )
            for t in range(horizon_size)
        )
        energy_baseline_term = sum(
            problem.objective_weights.energy
            * problem.dt_hours
            * _baseline_site_energy_cost(problem.horizon[t])
            for t in range(horizon_size)
        )
        pv_self_consumption_reward_term = sum(
            problem.objective_weights.pv_self_consumption
            * model.preheat_charge[t]
            * model.pv_opportunity_score[t]
            for t in range(horizon_size)
        )
        captured_pv_kwh_term = sum(
            model.preheat_charge[t]
            for t in range(horizon_size)
        )
        runtime_term = sum(
            problem.objective_weights.runtime * model.hp_on[t] for t in range(horizon_size)
        )
        terminal_term = problem.objective_weights.terminal * model.terminal_under
        preheat_budget_shortfall_term = self._preheat_budget_shortfall_expression(
            model=model,
            problem=problem,
            pyo=pyo,
        )
        model.comfort_low_term = pyo.Expression(expr=comfort_low_term)
        model.active_comfort_high_term = pyo.Expression(expr=active_comfort_high_term)
        model.passive_comfort_high_term = pyo.Expression(expr=passive_comfort_high_term)
        model.comfort_high_term = pyo.Expression(
            expr=model.active_comfort_high_term + model.passive_comfort_high_term
        )
        model.tracking_under_target_term = pyo.Expression(expr=tracking_under_target_term)
        model.tracking_over_target_term = pyo.Expression(expr=tracking_over_target_term)
        model.unnecessary_heating_term = pyo.Expression(expr=unnecessary_heating_term)
        model.terminal_term = pyo.Expression(expr=terminal_term)
        model.start_term = pyo.Expression(expr=start_term)
        model.energy_term = pyo.Expression(expr=energy_term)
        model.energy_baseline_term = pyo.Expression(expr=energy_baseline_term)
        model.pv_self_consumption_reward_term = pyo.Expression(expr=pv_self_consumption_reward_term)
        model.captured_pv_kwh_term = pyo.Expression(expr=captured_pv_kwh_term)
        model.preheat_budget_shortfall_term = pyo.Expression(expr=preheat_budget_shortfall_term)
        model.runtime_term = pyo.Expression(expr=runtime_term)
        model.objective = pyo.Objective(
            expr=(
                model.comfort_low_term
                + model.comfort_high_term
                + model.tracking_under_target_term
                + model.tracking_over_target_term
                + model.unnecessary_heating_term
                + model.terminal_term
                + model.start_term
                + (model.energy_term - model.energy_baseline_term)
                + model.preheat_budget_shortfall_term
                - model.pv_self_consumption_reward_term
                + model.runtime_term
            ),
            sense=pyo.minimize,
        )
        return model

    def _build_objective_context(self, problem: MpcProblem) -> _ObjectiveContext:
        pv_surplus_available_kw = [
            max(
                0.0,
                float(step.pv_available_power_forecast_kw)
                - float(step.base_load_power_forecast_kw),
            )
            for step in problem.horizon
        ]
        pv_opportunity_scores = [
            float(step.preheat_opportunity_score) if step.preheat_active else 0.0
            for step in problem.horizon
        ]
        no_heat_rollout = rollout_without_heating(
            control_model=problem.control_model,
            initial_state=problem.initial_state,
            horizon=problem.horizon,
        )
        full_heat_rollout = rollout_with_full_heating(
            control_model=problem.control_model,
            initial_state=problem.initial_state,
            horizon=problem.horizon,
        )
        useful_preheat_targets_c: list[float] = []
        for step in problem.horizon:
            comfort_floor_c = float(step.temp_min_c)
            economic_target_c = float(
                step.economic_target_c or step.target_temp_c or step.temp_min_c
            )
            if not step.preheat_active:
                useful_preheat_targets_c.append(max(comfort_floor_c, economic_target_c))
                continue
            useful_preheat_targets_c.append(
                min(
                    float(step.max_preheat_target_c or step.temp_max_c),
                    max(comfort_floor_c, economic_target_c),
                )
            )

        pv_self_consumable_kw = [
            min(
                float(step.hp_electric_power_forecast_kw),
                pv_surplus_available_kw[index],
            ) * (1.0 if step.preheat_active else 0.0)
            for index, step in enumerate(problem.horizon)
        ]
        unnecessary_heating_big_m_c = [
            max(
                float(problem.horizon[index].temp_max_c) - useful_preheat_targets_c[index],
                max(
                    (
                        no_heat_rollout[future_index] - useful_preheat_targets_c[index]
                        for future_index in range(index, len(problem.horizon))
                    ),
                    default=0.0,
                ),
                max(
                    (
                        full_heat_rollout[future_index] - useful_preheat_targets_c[index]
                        for future_index in range(index, len(problem.horizon))
                    ),
                    default=0.0,
                ),
                float(problem.horizon[index].temp_max_c - problem.horizon[index].temp_min_c),
                0.0,
            )
            for index in range(len(problem.horizon))
        ]
        comfort_high_big_m_c = [
            max(
                float(problem.horizon[index].temp_max_c - problem.horizon[index].temp_min_c),
                max(
                    (
                        no_heat_rollout[future_index] - float(problem.horizon[index].temp_max_c)
                        for future_index in range(index, len(problem.horizon))
                    ),
                    default=0.0,
                ),
                max(
                    (
                        full_heat_rollout[future_index] - float(problem.horizon[index].temp_max_c)
                        for future_index in range(index, len(problem.horizon))
                    ),
                    default=0.0,
                ),
                0.0,
            )
            for index in range(len(problem.horizon))
        ]
        q_heat_eff_big_m_kw = max(
            (
                float(step.effective_heating_kw_forecast)
                for step in problem.horizon
            ),
            default=0.0,
        )
        return _ObjectiveContext(
            useful_preheat_targets_c=useful_preheat_targets_c,
            pv_self_consumable_kw=pv_self_consumable_kw,
            pv_opportunity_scores=pv_opportunity_scores,
            comfort_high_big_m_c=comfort_high_big_m_c,
            unnecessary_heating_big_m_c=unnecessary_heating_big_m_c,
            q_heat_eff_big_m_kw=q_heat_eff_big_m_kw,
            terminal_target_c=useful_preheat_targets_c[-1],
        )

    @staticmethod
    def _preheat_block_step_indices(problem: MpcProblem) -> dict[int, list[int]]:
        block_indices: dict[int, list[int]] = {}
        for index, step in enumerate(problem.horizon):
            if step.preheat_block_id is None or step.preheat_block_max_starts <= 0:
                continue
            block_indices.setdefault(step.preheat_block_id, []).append(index)
        return block_indices

    def _apply_preheat_block_start_limits(
        self,
        *,
        model: Any,
        problem: MpcProblem,
        pyo: Any,
    ) -> None:
        block_indices = self._preheat_block_step_indices(problem)
        if not block_indices:
            return
        model.PREHEAT_BLOCKS = pyo.Set(initialize=list(block_indices))
        model.preheat_block_max_starts = pyo.Param(
            model.PREHEAT_BLOCKS,
            initialize={
                block_id: problem.horizon[indices[0]].preheat_block_max_starts
                for block_id, indices in block_indices.items()
            },
        )
        model.preheat_block_start_limit = pyo.Constraint(
            model.PREHEAT_BLOCKS,
            rule=lambda model_ref, block_id: sum(
                model_ref.start[index] for index in block_indices[block_id]
            ) <= model_ref.preheat_block_max_starts[block_id],
        )

    def _apply_preheat_block_budget_constraints(
        self,
        *,
        model: Any,
        problem: MpcProblem,
        pyo: Any,
    ) -> None:
        block_indices = self._preheat_block_step_indices(problem)
        if not block_indices:
            model.preheat_budget_shortfall = pyo.Var(domain=pyo.NonNegativeReals)
            model.preheat_budget_shortfall.fix(0.0)
            return
        model.preheat_budget_shortfall = pyo.Var(
            model.PREHEAT_BLOCKS,
            domain=pyo.NonNegativeReals,
        )
        model.preheat_block_budget = pyo.Param(
            model.PREHEAT_BLOCKS,
            initialize={
                block_id: problem.horizon[indices[0]].preheat_block_budget_kwh
                for block_id, indices in block_indices.items()
            },
        )
        model.preheat_block_step_target = pyo.Param(
            model.T,
            initialize={
                index: float(problem.horizon[index].preheat_block_cumulative_target_kwh)
                for index in range(len(problem.horizon))
            },
            default=0.0,
        )
        model.preheat_block_charge_limit = pyo.Constraint(
            model.PREHEAT_BLOCKS,
            rule=lambda model_ref, block_id: sum(
                model_ref.preheat_charge[index] for index in block_indices[block_id]
            ) <= model_ref.preheat_block_budget[block_id],
        )
        model.preheat_block_budget_shortfall = pyo.Constraint(
            model.PREHEAT_BLOCKS,
            rule=lambda model_ref, block_id: model_ref.preheat_budget_shortfall[block_id]
            >= (
                model_ref.preheat_block_budget[block_id]
                - sum(model_ref.preheat_charge[index] for index in block_indices[block_id])
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
                model_ref.preheat_block_step_target[t]
                - sum(model_ref.preheat_charge[index] for index in range(t + 1))
            ),
        )

    @staticmethod
    def _apply_execution_constraints(
        *,
        model: Any,
        problem: MpcProblem,
        pyo: Any,
    ) -> None:
        forced_on = [index for index, step in enumerate(problem.horizon) if step.hp_must_be_on]
        forced_off = [index for index, step in enumerate(problem.horizon) if step.hp_must_be_off]
        blocked_start = [
            index
            for index, step in enumerate(problem.horizon)
            if (not step.hp_start_allowed) and not step.hp_must_be_on
        ]
        overlap = set(forced_on).intersection(forced_off)
        if overlap:
            raise ValueError(f"conflicting sequencer constraints for steps: {sorted(overlap)}")
        if forced_on:
            model.sequencer_force_on = pyo.Constraint(
                forced_on,
                rule=lambda model_ref, t: model_ref.hp_on[t] == 1,
            )
        if forced_off:
            model.sequencer_force_off = pyo.Constraint(
                forced_off,
                rule=lambda model_ref, t: model_ref.hp_on[t] == 0,
            )
        if blocked_start:
            model.sequencer_block_start = pyo.Constraint(
                blocked_start,
                rule=lambda model_ref, t: model_ref.start[t] == 0,
            )

    @staticmethod
    def _preheat_budget_shortfall_expression(
        *,
        model: Any,
        problem: MpcProblem,
        pyo: Any,
    ) -> Any:
        if not hasattr(model, "PREHEAT_BLOCKS"):
            return 0.0
        block_shortfall_term = sum(
            problem.objective_weights.preheat_budget_shortfall
            * model.preheat_budget_shortfall[block_id]
            for block_id in model.PREHEAT_BLOCKS
        )
        cumulative_shortfall_term = 0.5 * problem.objective_weights.preheat_budget_shortfall * sum(
            model.preheat_budget_cumulative_shortfall[t]
            for t in model.T
            if float(model.preheat_block_step_target[t]) > 0.0
        )
        return block_shortfall_term + cumulative_shortfall_term


def _baseline_site_energy_cost(horizon_step: MpcHorizonStep) -> float:
    baseline_net_power_kw = (
        horizon_step.base_load_power_forecast_kw
        - horizon_step.pv_available_power_forecast_kw
    )
    baseline_import_kw = max(baseline_net_power_kw, 0.0)
    baseline_export_kw = max(-baseline_net_power_kw, 0.0)
    return float(
        (horizon_step.import_price_eur_kwh * baseline_import_kw)
        - (horizon_step.export_price_eur_kwh * baseline_export_kw)
    )
