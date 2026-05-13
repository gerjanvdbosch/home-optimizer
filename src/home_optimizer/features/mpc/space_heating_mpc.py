from __future__ import annotations

from time import perf_counter
from typing import Any

from home_optimizer.features.mpc.models import (
    MpcObjectiveBreakdown,
    MpcPlan,
    MpcPlanStep,
    MpcProblem,
)


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
        solver_candidates: tuple[str, ...] = ("appsi_highs", "highs"),
    ) -> None:
        self.solver_name = solver_name
        self.solver_candidates = solver_candidates

    def solve(self, problem: MpcProblem) -> MpcPlan:
        pyo = _load_pyomo()
        model = self._build_model(problem, pyo)
        solver = self._build_solver(pyo, problem.max_solver_seconds)

        started_at = perf_counter()
        results = solver.solve(model)
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
            comfort_high=float(pyo.value(model.comfort_high_term)),
            temperature_tracking=0.0,
            terminal=0.0,
            start=float(pyo.value(model.start_term)),
            runtime=float(pyo.value(model.runtime_term)),
            energy=float(pyo.value(model.energy_term)),
        )
        objective_value = float(pyo.value(model.objective))
        steps: list[MpcPlanStep] = []
        for index, horizon_step in enumerate(problem.horizon):
            hp_on = bool(round(pyo.value(model.hp_on[index])))
            effective_heating_kw = float(horizon_step.effective_heating_kw_forecast * hp_on)
            estimated_energy_cost = float(
                problem.objective_weights.energy
                * horizon_step.price_eur_kwh
                * effective_heating_kw
                * problem.dt_hours
            )
            steps.append(
                MpcPlanStep(
                    timestamp_utc=horizon_step.timestamp_utc,
                    hp_on=hp_on,
                    start=bool(round(pyo.value(model.start[index]))),
                    stop=bool(round(pyo.value(model.stop[index]))),
                    predicted_room_temp_c=float(pyo.value(model.room_temp[index])),
                    temp_min_c=horizon_step.temp_min_c,
                    temp_max_c=horizon_step.temp_max_c,
                    slack_low_c=float(pyo.value(model.slack_low[index])),
                    slack_high_c=float(pyo.value(model.slack_high[index])),
                    effective_heating_kw=effective_heating_kw,
                    price_eur_kwh=horizon_step.price_eur_kwh,
                    estimated_energy_cost_eur=estimated_energy_cost,
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
        model = pyo.ConcreteModel(name="space_heating_mpc")
        horizon_size = len(problem.horizon)
        model.T = pyo.RangeSet(0, horizon_size - 1)
        model.T_transition = pyo.RangeSet(0, max(0, horizon_size - 2))

        model.hp_on = pyo.Var(model.T, domain=pyo.Binary)
        model.start = pyo.Var(model.T, domain=pyo.Binary)
        model.stop = pyo.Var(model.T, domain=pyo.Binary)
        model.room_temp = pyo.Var(model.T, domain=pyo.Reals)
        model.slack_low = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.slack_high = pyo.Var(model.T, domain=pyo.NonNegativeReals)

        initial_hp_on = 1 if problem.initial_state.hp_on else 0
        model.initial_room_temp = pyo.Param(initialize=problem.initial_state.room_temp_c)
        model.initial_hp_on = pyo.Param(initialize=initial_hp_on)
        model.room_temp[0].fix(problem.initial_state.room_temp_c)

        def dynamics_rule(model_ref: Any, t: int) -> Any:
            step = problem.horizon[t]
            return model_ref.room_temp[t + 1] == (
                (problem.control_model.a * model_ref.room_temp[t])
                + (problem.control_model.b_out * step.outdoor_temp_c)
                + (problem.control_model.b_solar * step.solar_gain_kw)
                + (
                    problem.control_model.b_heat
                    * step.effective_heating_kw_forecast
                    * model_ref.hp_on[t]
                )
                + (problem.control_model.b_occ * step.occupied)
                + problem.control_model.c
            )

        if horizon_size >= 2:
            model.dynamics = pyo.Constraint(model.T_transition, rule=dynamics_rule)

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

        def transition_rule(model_ref: Any, t: int) -> Any:
            previous_hp_on = model_ref.initial_hp_on if t == 0 else model_ref.hp_on[t - 1]
            return model_ref.start[t] - model_ref.stop[t] == model_ref.hp_on[t] - previous_hp_on

        model.transition = pyo.Constraint(model.T, rule=transition_rule)
        model.start_stop_mutex = pyo.Constraint(
            model.T,
            rule=lambda model_ref, t: model_ref.start[t] + model_ref.stop[t] <= 1,
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

        comfort_low_term = sum(
            problem.objective_weights.comfort_low * model.slack_low[t] for t in range(horizon_size)
        )
        comfort_high_term = sum(
            problem.objective_weights.comfort_high * model.slack_high[t]
            for t in range(horizon_size)
        )
        start_term = sum(
            problem.objective_weights.start * model.start[t] for t in range(horizon_size)
        )
        energy_term = sum(
            problem.objective_weights.energy
            * problem.horizon[t].price_eur_kwh
            * problem.horizon[t].effective_heating_kw_forecast
            * problem.dt_hours
            * model.hp_on[t]
            for t in range(horizon_size)
        )
        runtime_term = sum(
            problem.objective_weights.runtime * model.hp_on[t] for t in range(horizon_size)
        )
        model.comfort_low_term = pyo.Expression(expr=comfort_low_term)
        model.comfort_high_term = pyo.Expression(expr=comfort_high_term)
        model.start_term = pyo.Expression(expr=start_term)
        model.energy_term = pyo.Expression(expr=energy_term)
        model.runtime_term = pyo.Expression(expr=runtime_term)
        model.objective = pyo.Objective(
            expr=(
                model.comfort_low_term
                + model.comfort_high_term
                + model.start_term
                + model.energy_term
                + model.runtime_term
            ),
            sense=pyo.minimize,
        )
        return model
