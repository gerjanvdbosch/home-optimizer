from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import pyomo.environ as pyo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MPCConfig:
    step_hours: float = 0.25
    boiler_power_kw: float = 2.0
    boiler_energy_kwh: float = 2.0
    max_starts: int = 1

    @property
    def boiler_steps(self) -> int:
        if self.step_hours <= 0:
            raise ValueError("step_hours must be greater than zero.")
        if self.boiler_power_kw <= 0:
            raise ValueError("boiler_power_kw must be greater than zero.")
        if self.boiler_energy_kwh <= 0:
            raise ValueError("boiler_energy_kwh must be greater than zero.")

        energy_per_step = self.boiler_power_kw * self.step_hours
        steps = self.boiler_energy_kwh / energy_per_step

        if not steps.is_integer():
            raise ValueError(
                "boiler_energy_kwh must be an exact multiple of "
                "boiler_power_kw * step_hours."
            )
        return int(steps)


@dataclass(frozen=True)
class MPCInput:
    solar_forecast_kw: Sequence[float]
    boiler_on: bool = False


@dataclass(frozen=True)
class MPCResult:
    schedule: tuple[int, ...]
    objective_value: float
    solver_status: str
    termination_condition: str


class MPCOptimizer:
    """
    Bouwt en lost het Pyomo-model voor de energie-optimalisatie op.
    """

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
        objective_value = float(pyo.value(model.objective))

        return MPCResult(
            schedule=schedule,
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
            t: min(solar_values[t], self.config.boiler_power_kw)
            * self.config.step_hours
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
            expr=sum(model.start_choice[s] for s in model.S) == 1
        )

        def on_link_rule(m, t):
            active_starts = [s for s in valid_starts if s <= t < s + required_steps]
            return m.boiler_on[t] == sum(m.start_choice[s] for s in active_starts)

        model.on_link = pyo.Constraint(model.T, rule=on_link_rule)

    def _validate_input(self, data: MPCInput) -> None:
        if not data.solar_forecast_kw:
            raise Exception("Solar forecast is empty.")

        if any(value is None for value in data.solar_forecast_kw):
            raise Exception("Solar forecast contains None values.")

        required_steps = self.config.boiler_steps
        horizon = len(data.solar_forecast_kw)

        if required_steps > horizon:
            raise Exception(
                f"Boiler requires {required_steps} "
                f"steps, but the horizon only has "
                f"{horizon} steps."
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
