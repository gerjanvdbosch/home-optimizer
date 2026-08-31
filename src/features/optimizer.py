import logging
from typing import cast

import pyomo.environ as pyo

from domain.models import MPCConfig, MPCInput, MPCResult

logger = logging.getLogger(__name__)


class MPCOptimizer:
    def __init__(
        self,
        config: MPCConfig,
        solver_name: str = "appsi_highs",
    ) -> None:
        self.config = config
        self.solver_name = solver_name

    def solve(self, data: MPCInput) -> MPCResult:
        self._validate_input(data)

        model = self._build_model(data)
        solver = self._get_solver()

        try:
            results = solver.solve(model, tee=False)
        except Exception as exc:
            raise Exception(f"Optimization failed: {exc}") from exc

        self._check_result(results)

        schedule = tuple(int(round(pyo.value(model.boiler_on[t]))) for t in model.T)

        # 1. Lees BEIDE temperatuurlagen uit (T_top en T_bottom)
        temperatures_top = tuple(
            round(float(pyo.value(model.T_top[t])), 2) for t in model.T
        )
        temperatures_bottom = tuple(
            round(float(pyo.value(model.T_bottom[t])), 2) for t in model.T
        )
        objective_value = float(pyo.value(model.objective))

        logger.info("Gepland temperatuurverloop Top:    %s", temperatures_top)
        logger.info("Gepland temperatuurverloop Bottom: %s", temperatures_bottom)

        return MPCResult(
            schedule=schedule,
            temperatures_top=temperatures_top,
            temperatures_bottom=temperatures_bottom,
            objective_value=objective_value,
            solver_status=str(results.solver.status),
            termination_condition=str(results.solver.termination_condition),
        )

    def _build_model(self, data: MPCInput) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel(name="home_energy_mpc")

        # Time index
        horizon = len(data.solar_forecast_kw)
        model.T = pyo.RangeSet(0, horizon - 1)

        # Parameters
        solar_values = {t: max(0.0, float(data.solar_forecast_kw[t])) for t in model.T}
        model.solar_kw = pyo.Param(
            model.T, initialize=solar_values, within=pyo.NonNegativeReals
        )

        solar_used_kwh = {
            t: min(solar_values[t], self.config.boiler_power) * self.config.step_hours
            for t in model.T
        }
        model.solar_used_kwh = pyo.Param(
            model.T, initialize=solar_used_kwh, within=pyo.NonNegativeReals
        )

        # Variables
        model.boiler_on = pyo.Var(model.T, domain=pyo.Binary)

        # Objective
        model.objective = pyo.Objective(
            expr=sum(model.boiler_on[t] * model.solar_used_kwh[t] for t in model.T),
            sense=pyo.maximize,
        )

        # Constraints
        self._add_contiguous_run_constraint(model, data)
        self._add_temperature_constraints(model, data)

        return model

    def _add_contiguous_run_constraint(
        self, model: pyo.ConcreteModel, data: MPCInput
    ) -> None:
        required_steps = self.config.boiler_steps
        horizon = len(model.T)

        if data.boiler_on:
            valid_starts = [0]
        else:
            valid_starts = list(range(0, horizon - required_steps + 1))

        model.S = pyo.Set(initialize=valid_starts)
        model.start_choice = pyo.Var(model.S, domain=pyo.Binary)

        model.one_start = pyo.Constraint(
            expr=sum(model.start_choice[s] for s in model.S) <= 1
        )

        def on_link_rule(m, t):
            active_starts = [s for s in valid_starts if s <= t < s + required_steps]
            return m.boiler_on[t] == sum(m.start_choice[s] for s in active_starts)

        model.on_link = pyo.Constraint(model.T, rule=on_link_rule)

    def _add_temperature_constraints(
        self, model: pyo.ConcreteModel, data: MPCInput
    ) -> None:
        model.T_top = pyo.Var(model.T, bounds=(20.0, 75.0))
        model.T_bottom = pyo.Var(model.T, bounds=(10.0, 75.0))

        # Startcondities van de echte sensoren
        model.init_top = pyo.Constraint(expr=model.T_top[0] == data.current_temp_top)
        model.init_bottom = pyo.Constraint(
            expr=model.T_bottom[0] == data.current_temp_bottom
        )

        thm = data.thermal_model

        # 1. Dynamica voor Bovenkant (T_top)
        def top_dynamics_rule(m, t):
            if t == len(m.T) - 1:
                return pyo.Constraint.Skip
            heat_input = m.boiler_on[t] * thm.dhw_freq
            amb_loss = (1.0 - thm.a_top_top - thm.a_top_bottom) * thm.t_ambient

            return m.T_top[t + 1] == (
                thm.a_top_top * m.T_top[t]
                + thm.a_top_bottom * m.T_bottom[t]
                + amb_loss
                + (thm.c_top * heat_input)
            )

        model.top_dynamics = pyo.Constraint(model.T, rule=top_dynamics_rule)

        # 2. Dynamica voor Onderkant (T_bottom)
        def bottom_dynamics_rule(m, t):
            if t == len(m.T) - 1:
                return pyo.Constraint.Skip
            heat_input = m.boiler_on[t] * thm.dhw_freq
            amb_loss = (1.0 - thm.a_bottom_top - thm.a_bottom_bottom) * thm.t_ambient

            return m.T_bottom[t + 1] == (
                thm.a_bottom_top * m.T_top[t]
                + thm.a_bottom_bottom * m.T_bottom[t]
                + amb_loss
                + (thm.c_bottom * heat_input)
            )

        model.bottom_dynamics = pyo.Constraint(model.T, rule=bottom_dynamics_rule)

    def _validate_input(self, data: MPCInput) -> None:
        if not data.solar_forecast_kw:
            raise Exception("Solar forecast is empty.")

        if any(value is None for value in data.solar_forecast_kw):
            raise Exception("Solar forecast contains None values.")

        if data.current_temp_top is None:
            raise Exception("Current boiler temperature is missing.")

        required_steps = self.config.boiler_steps
        horizon = len(data.solar_forecast_kw)

        if required_steps > horizon:
            raise Exception(
                f"Boiler requires {required_steps} steps, but the horizon only has {horizon} steps."
            )

    def _get_solver(self):
        candidates = [self.solver_name, "appsi_highs", "highspy", "cbc", "glpk"]
        checked: set[str] = set()

        for name in candidates:
            if name in checked:
                continue
            checked.add(name)

            try:
                solver = pyo.SolverFactory(name)
                if solver is not None and solver.available():
                    return solver
            except Exception as exc:
                logger.debug("Solver '%s' unavailable: %s", name, exc)

        raise Exception(
            "No suitable MILP solver is available. Install HiGHS/highspy, CBC or GLPK."
        )

    @staticmethod
    def _check_result(results) -> None:
        termination = results.solver.termination_condition
        valid_termination = {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        }

        if termination not in valid_termination:
            raise Exception(
                "Optimization did not produce a usable solution. "
                f"Status={results.solver.status}, "
                f"Termination={termination}"
            )
