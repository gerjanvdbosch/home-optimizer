import logging

import pyomo.environ as pyo

from domain.boiler_model import MPCConfig, MPCInput, MPCResult

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
        solver = pyo.SolverFactory(self.solver_name)

        try:
            results = solver.solve(model, tee=False)
        except Exception as exc:
            raise RuntimeError(f"Optimization failed: {exc}") from exc

        self._check_result(results)

        schedule = tuple(int(round(pyo.value(model.boiler_on[t]))) for t in model.T)

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

        horizon = len(data.solar_forecast_kw)
        model.T = pyo.RangeSet(0, horizon - 1)

        # 1. Exogene Zonne-energie Parameters
        solar_values = {t: max(0.0, float(data.solar_forecast_kw[t])) for t in model.T}
        solar_used_kwh = {
            t: min(solar_values[t], self.config.boiler_power) * self.config.step_hours
            for t in model.T
        }
        model.solar_used_kwh = pyo.Param(
            model.T, initialize=solar_used_kwh, within=pyo.NonNegativeReals
        )

        # 2. Beslissingsvariabele
        model.boiler_on = pyo.Var(model.T, domain=pyo.Binary)

        # 3. Systeembeperkingen toevoegen
        self._add_contiguous_run_constraint(model, data)

        # DELEGEER DE THERMISCHE FYSIKA NAAR HET MODEL (DRY & MODULAIR)
        data.thermal_model.apply_pyomo_constraints(model, data)

        # 4. Doelfunctie: Maximaliseer zonne-energie MIN de inverter afknijp-boete
        model.objective = pyo.Objective(
            expr=(
                sum(model.boiler_on[t] * model.solar_used_kwh[t] for t in model.T)
                - sum(0.001 * model.q_curtail[t] for t in model.T)
            ),
            sense=pyo.maximize,
        )

        return model

    def _add_contiguous_run_constraint(
        self, model: pyo.ConcreteModel, data: MPCInput
    ) -> None:
        required_steps = int(round(self.config.boiler_steps))
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

    def _validate_input(self, data: MPCInput) -> None:
        if not data.solar_forecast_kw:
            raise ValueError("Solar forecast is empty.")
        if any(value is None for value in data.solar_forecast_kw):
            raise ValueError("Solar forecast contains None values.")
        if data.current_temp_top is None:
            raise ValueError("Current boiler temperature (top) is missing.")
        if data.current_temp_bottom is None:
            raise ValueError("Current boiler temperature (bottom) is missing.")
        if data.thermal_model is None:
            raise ValueError("Thermal model is missing.")

        required_steps = self.config.boiler_steps
        horizon = len(data.solar_forecast_kw)
        if required_steps > horizon:
            raise ValueError(
                f"Boiler requires {required_steps} steps, "
                f"but the horizon only has {horizon} steps."
            )

    @staticmethod
    def _check_result(results) -> None:
        termination = results.solver.termination_condition
        valid_termination = {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        }
        if termination not in valid_termination:
            raise RuntimeError(
                f"Optimization did not produce a usable solution. Status={results.solver.status}, "
                f"Termination={termination}"
            )
