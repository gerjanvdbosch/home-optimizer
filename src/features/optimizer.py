import logging

import numpy as np
import pyomo.environ as pyo
from highspy import Highs

from domain.types import BoilerThermalModel, MPCConfig, MPCInput, MPCResult

logger = logging.getLogger(__name__)


class MPCOptimizer:
    """
    Mixed-integer MPC optimizer for the DHW tank.

    The thermal model itself is kept outside Pyomo. This class only:
    1. validates MPC input,
    2. builds the Pyomo optimization model,
    3. solves it with HiGHS,
    4. extracts the optimal control trajectory.

    Thermal dynamics are supplied by BoilerThermalModel.
    """

    def __init__(
        self,
        thermal_model: BoilerThermalModel,
        config: MPCConfig | None = None,
    ) -> None:
        self.thermal_model = thermal_model
        self.config = config or MPCConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, data: MPCInput) -> MPCResult:
        """
        Solve the MPC optimization problem.

        Parameters
        ----------
        data:
            Current measured state and future forecasts.

        Returns
        -------
        MPCResult
            Optimal temperature trajectory and boiler schedule.
        """

        self._validate_input(data)

        model = self._build_model(data)

        solver = Highs()

        result = solver.solve(model)

        termination = result.termination_condition

        if termination not in {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.feasible,
        }:
            raise RuntimeError(
                f"MPC optimization failed. Termination condition: {termination}"
            )

        return self._extract_result(model, data)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_input(self, data: MPCInput) -> None:
        """
        Validate MPC input data.

        All forecast arrays must have identical length.
        """

        fields = {
            "T_ambient_forecast": data.T_ambient_forecast,
            "T_cold_forecast": data.T_cold_forecast,
            "flow_forecast": data.flow_forecast,
            "T_setpoint_forecast": data.T_setpoint_forecast,
            "solar_available_w": data.solar_available_w,
        }

        lengths = {name: len(values) for name, values in fields.items()}

        if len(set(lengths.values())) != 1:
            raise ValueError(
                "All forecast arrays must have the same length. "
                f"Received lengths: {lengths}"
            )

        horizon = len(data.T_ambient_forecast)

        if horizon < 2:
            raise ValueError("MPC horizon must contain at least 2 steps.")

        if self.config.step_hours <= 0:
            raise ValueError("step_hours must be greater than zero.")

        if self.config.boiler_power_w < 0:
            raise ValueError("boiler_power_w cannot be negative.")

        if self.config.boiler_min_runtime_steps < 1:
            raise ValueError("boiler_min_runtime_steps must be at least 1.")

    # ------------------------------------------------------------------
    # Pyomo model
    # ------------------------------------------------------------------

    def _build_model(self, data: MPCInput) -> pyo.ConcreteModel:
        """
        Build the complete Pyomo MPC model.
        """

        horizon = len(data.T_ambient_forecast)

        model = pyo.ConcreteModel()

        # --------------------------------------------------------------
        # Time index
        # --------------------------------------------------------------

        model.K = pyo.RangeSet(0, horizon - 1)

        # --------------------------------------------------------------
        # State variables
        # --------------------------------------------------------------

        model.T_top = pyo.Var(
            model.K,
            bounds=(0.0, 100.0),
        )

        model.T_mid = pyo.Var(
            model.K,
            bounds=(0.0, 100.0),
        )

        model.T_bottom = pyo.Var(
            model.K,
            bounds=(0.0, 100.0),
        )

        # --------------------------------------------------------------
        # Binary boiler variables
        # --------------------------------------------------------------

        model.boiler_on = pyo.Var(
            model.K,
            domain=pyo.Binary,
        )

        model.boiler_start = pyo.Var(
            model.K,
            domain=pyo.Binary,
        )

        # --------------------------------------------------------------
        # Soft-constraint slack variables
        # --------------------------------------------------------------

        model.top_slack = pyo.Var(
            model.K,
            domain=pyo.NonNegativeReals,
        )

        model.bottom_slack = pyo.Var(
            model.K,
            domain=pyo.NonNegativeReals,
        )

        # --------------------------------------------------------------
        # Initial state
        # --------------------------------------------------------------

        initial_mid = self._get_initial_mid_temperature(data)

        model.initial_top = pyo.Constraint(
            expr=model.T_top[0] == float(data.T_top_current)
        )

        model.initial_mid = pyo.Constraint(expr=model.T_mid[0] == float(initial_mid))

        model.initial_bottom = pyo.Constraint(
            expr=model.T_bottom[0] == float(data.T_bottom_current)
        )

        # --------------------------------------------------------------
        # Temperature constraints
        # --------------------------------------------------------------

        def top_temperature_rule(
            m: pyo.ConcreteModel,
            k: int,
        ):
            """
            Keep the top of the tank above the requested temperature.

            The constraint is soft, so the optimizer can violate it if
            necessary, at a large penalty.
            """

            required_temperature = max(
                float(data.T_setpoint_forecast[k]),
                self.config.boiler_min_top_temperature,
            )

            return m.T_top[k] + m.top_slack[k] >= required_temperature

        model.top_temperature_constraint = pyo.Constraint(
            model.K,
            rule=top_temperature_rule,
        )

        def bottom_temperature_rule(
            m: pyo.ConcreteModel,
            k: int,
        ):
            """
            Keep the bottom of the tank above its minimum temperature.

            This is deliberately NOT tied to the top setpoint. The
            thermal stratification of the tank should be preserved.
            """

            return (
                m.T_bottom[k] + m.bottom_slack[k]
                >= self.config.boiler_min_bottom_temperature
            )

        model.bottom_temperature_constraint = pyo.Constraint(
            model.K,
            rule=bottom_temperature_rule,
        )

        # --------------------------------------------------------------
        # Boiler startup logic
        # --------------------------------------------------------------

        initial_boiler_on = int(
            bool(
                getattr(
                    data,
                    "boiler_on_current",
                    False,
                )
            )
        )

        def startup_rule(
            m: pyo.ConcreteModel,
            k: int,
        ):
            if k == 0:
                return m.boiler_start[k] == (m.boiler_on[k] - initial_boiler_on)

            return m.boiler_start[k] == (m.boiler_on[k] - m.boiler_on[k - 1])

        model.startup_constraint = pyo.Constraint(
            model.K,
            rule=startup_rule,
        )

        # --------------------------------------------------------------
        # Minimum runtime
        # --------------------------------------------------------------

        model.minimum_runtime = pyo.ConstraintList()

        min_runtime = self.config.boiler_min_runtime_steps

        for start in range(horizon):
            for offset in range(min_runtime):
                k = start + offset

                if k >= horizon:
                    continue

                model.minimum_runtime.add(
                    model.boiler_on[k] >= model.boiler_start[start]
                )

        # --------------------------------------------------------------
        # Thermal dynamics
        # --------------------------------------------------------------

        model.thermal_dynamics = pyo.ConstraintList()

        dt_seconds = self.config.step_hours * 3600.0

        for k in range(horizon - 1):
            ambient = float(data.T_ambient_forecast[k])

            cold = float(data.T_cold_forecast[k])

            flow_lpm = float(data.flow_forecast[k])

            # Exact zero-order-hold discretization.
            #
            # Inputs are:
            #   u[0] = heater power [W]
            #   u[1] = ambient temperature [°C]
            #   u[2] = cold-water temperature [°C]
            A_d, B_d, d_d = self.thermal_model.discrete_matrices(
                dt_seconds=dt_seconds,
                flow_lpm=flow_lpm,
            )

            x_current = [
                model.T_top[k],
                model.T_mid[k],
                model.T_bottom[k],
            ]

            x_next = [
                model.T_top[k + 1],
                model.T_mid[k + 1],
                model.T_bottom[k + 1],
            ]

            heater_power = self.config.boiler_power_w * model.boiler_on[k]

            u = [
                heater_power,
                ambient,
                cold,
            ]

            # Three thermal states:
            #   0 = top
            #   1 = middle
            #   2 = bottom
            for i in range(3):
                rhs = float(d_d[i])

                # A_d * x
                for j in range(3):
                    rhs += float(A_d[i, j]) * x_current[j]

                # B_d * u
                for j in range(3):
                    rhs += float(B_d[i, j]) * u[j]

                model.thermal_dynamics.add(x_next[i] == rhs)

        # --------------------------------------------------------------
        # Objective
        # --------------------------------------------------------------

        model.objective = pyo.Objective(
            expr=self._build_objective(model, data),
            sense=pyo.minimize,
        )

        return model

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _build_objective(
        self,
        model: pyo.ConcreteModel,
        data: MPCInput,
    ):
        """
        Build MPC objective.

        Objective:

            energy cost
          - solar priority
          + switching penalty
          + temperature slack penalty

        Energy is calculated in kWh.
        """

        objective = 0.0

        dt_hours = self.config.step_hours

        for k in model.K:
            boiler_on = model.boiler_on[k]

            # ----------------------------------------------------------
            # Boiler energy
            # ----------------------------------------------------------

            boiler_energy_kwh = (
                self.config.boiler_power_w * dt_hours * boiler_on / 1000.0
            )

            objective += self.config.weight_energy * boiler_energy_kwh

            # ----------------------------------------------------------
            # Solar priority
            # ----------------------------------------------------------

            solar_available_w = max(
                0.0,
                float(data.solar_available_w[k]),
            )

            # The boiler cannot consume more PV power than its own
            # maximum electrical power.
            solar_usable_w = min(
                solar_available_w,
                self.config.boiler_power_w,
            )

            solar_energy_kwh = solar_usable_w * dt_hours * boiler_on / 1000.0

            # More available solar makes boiler operation more
            # attractive.
            objective -= self.config.weight_solar_priority * solar_energy_kwh

            # ----------------------------------------------------------
            # Switching penalty
            # ----------------------------------------------------------

            objective += self.config.weight_switching * model.boiler_start[k]

            # ----------------------------------------------------------
            # Soft temperature constraints
            # ----------------------------------------------------------

            objective += self.config.weight_temperature_slack * model.top_slack[k]

            objective += self.config.weight_temperature_slack * model.bottom_slack[k]

        return objective

    # ------------------------------------------------------------------
    # Initial hidden state
    # ------------------------------------------------------------------

    def _get_initial_mid_temperature(
        self,
        data: MPCInput,
    ) -> float:
        """
        Determine the initial hidden middle-node temperature.

        Priority:
        1. measured/provided T_mid_current,
        2. identified model initial T_mid_0,
        3. average of top and bottom temperature.
        """

        current_mid = getattr(
            data,
            "T_mid_current",
            None,
        )

        if current_mid is not None:
            return float(current_mid)

        model_mid = getattr(
            self.thermal_model,
            "T_mid_0",
            None,
        )

        if model_mid is not None:
            return float(model_mid)

        return (float(data.T_top_current) + float(data.T_bottom_current)) / 2.0

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def _extract_result(
        self,
        model: pyo.ConcreteModel,
        data: MPCInput,
    ) -> MPCResult:
        """
        Extract the optimized MPC trajectory from Pyomo.
        """

        horizon = len(data.T_ambient_forecast)

        T_top = np.array(
            [pyo.value(model.T_top[k]) for k in range(horizon)],
            dtype=float,
        )

        T_mid = np.array(
            [pyo.value(model.T_mid[k]) for k in range(horizon)],
            dtype=float,
        )

        T_bottom = np.array(
            [pyo.value(model.T_bottom[k]) for k in range(horizon)],
            dtype=float,
        )

        boiler_on = np.array(
            [round(float(pyo.value(model.boiler_on[k]))) for k in range(horizon)],
            dtype=int,
        )

        boiler_start = np.array(
            [round(float(pyo.value(model.boiler_start[k]))) for k in range(horizon)],
            dtype=int,
        )

        top_slack = np.array(
            [float(pyo.value(model.top_slack[k])) for k in range(horizon)],
            dtype=float,
        )

        bottom_slack = np.array(
            [float(pyo.value(model.bottom_slack[k])) for k in range(horizon)],
            dtype=float,
        )

        # --------------------------------------------------------------
        # Boiler power and energy
        # --------------------------------------------------------------

        boiler_power_w = self.config.boiler_power_w * boiler_on

        boiler_energy_kwh = float(
            np.sum(boiler_power_w * self.config.step_hours) / 1000.0
        )

        # --------------------------------------------------------------
        # Objective value
        # --------------------------------------------------------------

        objective_value = float(pyo.value(model.objective))

        # --------------------------------------------------------------
        # Result object
        # --------------------------------------------------------------

        return MPCResult(
            T_top=T_top,
            T_mid=T_mid,
            T_bottom=T_bottom,
            boiler_on=boiler_on,
            boiler_start=boiler_start,
            boiler_power_w=boiler_power_w,
            boiler_energy_kwh=boiler_energy_kwh,
            top_slack=top_slack,
            bottom_slack=bottom_slack,
            objective_value=objective_value,
        )
