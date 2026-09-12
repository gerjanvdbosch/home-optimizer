import logging

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers.highs import Highs

from domain.types import (
    BoilerThermalModel,
    HeatPumpCOPModel,
    MPCConfig,
    MPCInput,
    MPCResult,
)
from features.boiler import CP_WATER_J_PER_KG_K, RHO_WATER_KG_PER_L, discretize_zoh
from features.cop import HeatPumpCOPIdentifier

logger = logging.getLogger(__name__)


def _lumped_state_space(
    volume_l: float,
    ua_total_w_per_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous state-space for a single lumped tank node: dT/dt = A T + B u,
    u = [T_ambient, Q_in_effective, Q_tap_forecast].

    Used only for MPC planning, not for the calibrated identification/validation
    model (see features/boiler.py, which keeps top and bottom separate). Mixing
    during active heating was found to saturate at the sampling-resolution
    ceiling there (UA_mix_active pinned at its bound), meaning the tank is
    practically fully mixed within one MPC step - so a single node using the
    well-identified UA_top+UA_bottom sum and q_in_nominal_w is a defensible
    simplification. It also keeps these dynamics linear in the binary boiler_on
    decision: the full two-node model would need a disjunctive/big-M
    reformulation to let UA_mix switch with boiler_on, for precision in the
    individual UA_top/UA_bottom split that isn't there anyway.

    Q_tap_forecast is an additional heat-sink term (cold mains water entering,
    warm water drawn out) - the third B column carries a negative coefficient
    since, unlike Q_in, it removes energy from the tank: C dT/dt = Q_in -
    UA*(T-T_ambient) - Q_tap.
    """

    c_total = RHO_WATER_KG_PER_L * volume_l * CP_WATER_J_PER_KG_K

    a = np.array([[-ua_total_w_per_k / c_total]])
    b = np.array(
        [[ua_total_w_per_k / c_total, 1.0 / c_total, -1.0 / c_total]]
    )

    return a, b


class MPCOptimizer:
    def __init__(
        self,
        thermal_model: BoilerThermalModel,
        config: MPCConfig,
        cop_model: HeatPumpCOPModel | None = None,
    ) -> None:
        self.thermal_model = thermal_model
        self.config = config
        # None until a cop_dhw model has actually been calibrated (see
        # HeatPumpCOPIdentifier) - _power_line_coefficients() falls back to
        # the flat boiler_electrical_power_w assumption until then.
        self.cop_model = cop_model

    def solve(self, data: MPCInput) -> MPCResult:
        self._validate_input(data)

        model = self._build_model(data)

        solver = Highs()
        results = solver.solve(model)

        if results.termination_condition != TerminationCondition.optimal:
            raise RuntimeError(
                "MPC optimization failed. Termination condition: "
                f"{results.termination_condition}"
            )

        return self._extract_result(model, data, results.termination_condition)

    def _validate_input(self, data: MPCInput) -> None:
        horizon = len(data.solar_forecast_w)

        if horizon < 2:
            raise ValueError("MPC horizon must contain at least 2 steps.")

        if len(data.target_temperature_top) != horizon:
            raise ValueError(
                "target_temperature_top must have the same length as "
                f"solar_forecast_w ({horizon}), got "
                f"{len(data.target_temperature_top)}."
            )

        # Empty means "no forecast available" (treated as no draws elsewhere) -
        # only a non-empty, mismatched length is an actual bug.
        if data.tap_forecast_w and len(data.tap_forecast_w) != horizon:
            raise ValueError(
                "tap_forecast_w must be empty or have the same length as "
                f"solar_forecast_w ({horizon}), got {len(data.tap_forecast_w)}."
            )

        # Same "empty means no forecast" convention as tap_forecast_w above.
        if (
            data.outdoor_temperature_forecast
            and len(data.outdoor_temperature_forecast) != horizon
        ):
            raise ValueError(
                "outdoor_temperature_forecast must be empty or have the same "
                f"length as solar_forecast_w ({horizon}), got "
                f"{len(data.outdoor_temperature_forecast)}."
            )

        if self.config.step_hours <= 0:
            raise ValueError("step_hours must be greater than zero.")

        if self.config.boiler_electrical_power_w < 0:
            raise ValueError("boiler_electrical_power_w cannot be negative.")

        if self.config.boiler_min_runtime_steps < 1:
            raise ValueError("boiler_min_runtime_steps must be at least 1.")

    def _build_model(self, data: MPCInput) -> pyo.ConcreteModel:
        horizon = len(data.solar_forecast_w)

        model = pyo.ConcreteModel()

        model.K = pyo.RangeSet(0, horizon - 1)

        model.T = pyo.Var(model.K, bounds=(0.0, 100.0))

        model.boiler_on = pyo.Var(model.K, domain=pyo.Binary)

        model.boiler_start = pyo.Var(model.K, domain=pyo.Binary)

        model.slack = pyo.Var(model.K, domain=pyo.NonNegativeReals)

        # Equal-volume-node assumption, same as the calibrated identification
        # model: the average of the two measured sensors approximates the tank's
        # current total stored thermal energy per unit mass.
        initial_temperature = (data.current_temp_top + data.current_temp_bottom) / 2.0

        model.initial_temperature = pyo.Constraint(
            expr=model.T[0] == float(initial_temperature)
        )

        def temperature_rule(m: pyo.ConcreteModel, k: int):
            return m.T[k] + m.slack[k] >= float(data.target_temperature_top[k])

        model.temperature_constraint = pyo.Constraint(
            model.K,
            rule=temperature_rule,
        )

        initial_boiler_on = int(data.boiler_on_current)

        # Inequality, not equality: boiler_start must be 1 on a real 0->1
        # transition (RHS=1, forcing boiler_start[k]>=1), but on a 1->0 stop the
        # RHS is -1 and boiler_start[k]=0 already satisfies ">=-1" trivially. An
        # equality here would force boiler_start=-1 on every stop, which is
        # infeasible against its own binary domain - making any schedule that
        # ever turns the boiler back off unsolvable, and forcing it to stay on
        # forever once started (confirmed: this was the actual cause of an
        # apparently-wasteful "never stops heating" result before this fix).
        # weight_switching in the objective still drives it to 0 except at real
        # starts, since setting it higher only adds cost.
        def startup_rule(m: pyo.ConcreteModel, k: int):
            if k == 0:
                return m.boiler_start[k] >= (m.boiler_on[k] - initial_boiler_on)

            return m.boiler_start[k] >= (m.boiler_on[k] - m.boiler_on[k - 1])

        model.startup_constraint = pyo.Constraint(
            model.K,
            rule=startup_rule,
        )

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

        # Exact zero-order-hold dynamics for the lumped tank node - one constant
        # (A_d, B_d) pair for the whole horizon, since (unlike the two-node
        # calibration model) this simplified model has no on/off mixing-regime
        # switch: only the heat input, not the loss coefficient, depends on
        # boiler_on, so it stays linear without per-step matrix recomputation.
        a, b = _lumped_state_space(
            self.thermal_model.volume_l,
            self.thermal_model.ua_top_w_per_k + self.thermal_model.ua_bottom_w_per_k,
        )
        a_d, b_d = discretize_zoh(a, b, self.config.step_hours * 3600.0)

        model.thermal_dynamics = pyo.ConstraintList()

        tap_forecast_w = data.tap_forecast_w or (0.0,) * horizon

        for k in range(horizon - 1):
            model.thermal_dynamics.add(
                model.T[k + 1]
                == a_d[0, 0] * model.T[k]
                + b_d[0, 0] * float(data.ambient_temperature)
                + b_d[0, 1] * self.thermal_model.q_in_nominal_w * model.boiler_on[k]
                + b_d[0, 2] * float(tap_forecast_w[k])
            )

        # active_power_w[k] represents boiler_on[k] * max(0, electrical_power_w[k]
        # - solar[k]) - the grid draw actually costed at step k. electrical
        # power is linear in T[k] (see _power_line_coefficients), so this
        # would ordinarily need a McCormick linearization to multiply by the
        # binary boiler_on[k]; folding the max(0, ...) and the on/off gating
        # into one big-M lower bound (below) avoids a second, separate
        # linearization for that product.
        model.active_power_w = pyo.Var(model.K, domain=pyo.NonNegativeReals)

        model.active_power_constraint = pyo.ConstraintList()

        for k in range(horizon):
            alpha, beta = self._power_line_coefficients(data, k)

            solar_available_w = max(0.0, float(data.solar_forecast_w[k]))

            # Safe upper bound on (alpha + beta*T - solar) for T within its
            # own declared bounds and solar >= 0 - large enough that the
            # constraint is always non-binding once relaxed by
            # (1 - boiler_on[k]), so active_power_w[k] is free to fall to 0
            # (via the objective's minimization) whenever boiler_on[k] = 0.
            t_lower, t_upper = model.T[k].bounds
            big_m = max(alpha + beta * t_lower, alpha + beta * t_upper) + 1.0

            model.active_power_constraint.add(
                model.active_power_w[k]
                >= (alpha + beta * model.T[k] - solar_available_w)
                - big_m * (1 - model.boiler_on[k])
            )

        model.objective = pyo.Objective(
            expr=self._build_objective(model, data),
            sense=pyo.minimize,
        )

        return model

    def _power_line_coefficients(self, data: MPCInput, k: int) -> tuple[float, float]:
        """Coefficients (alpha, beta) of a linear approximation
        electrical_power_w[k] = alpha + beta * T for step k, where T stands
        for the MPC's own tank temperature at that step (model.T[k]).

        The true relationship (via HeatPumpCOPModel.cop(), a ratio of
        temperatures) is concave in T, not linear - and a concave function
        cannot be represented by inequality "tangent cut" constraints under
        minimization the way a convex one can (tangent lines of a concave
        function are upper bounds; minimizing gives the solver no pressure
        to rise to meet them, so it would just drive the variable to 0
        instead of the true curve). Representing it exactly would need a
        real piecewise-linear formulation (SOS2 or binary-selected
        segments), adding real complexity. Real data on this installation
        confirmed electrical power is very close to linear in supply
        temperature over a DHW cycle's normal active-heating range
        (HeatPumpCOPIdentifier.POWER_FIT_T_LOW_C to ...HIGH_C) - so a single
        secant line through the two ends of that range is an adequate, much
        simpler stand-in, and (being a genuine straight line, not a bound)
        is usable directly inside the objective, exactly as used for
        reporting (see _extract_result) - the same numbers the optimizer
        actually costs with are the ones shown.

        Q_th at each reference point comes from the calibrated
        q_th_at_power_fit_low_w/high_w (real, calorimetric thermal output
        measured near that supply temperature - see
        HeatPumpCOPIdentifier.calibrate()), not BoilerThermalModel's fixed
        q_in_nominal_w: real data confirmed Q_th is not constant across a
        compressor run (it rises from a low start, peaks mid-cycle, then
        falls as the compressor modulates down approaching setpoint), and
        q_in_nominal_w - calibrated for the tank's temperature *trajectory*,
        a different purpose - understated real electrical draw through the
        middle of a cycle by using a constant well below the true mid-cycle
        Q_th.

        T[k] (rather than a single fixed reference temperature) is what
        this line is evaluated against: the heat pump's actual supply
        temperature physically tracks the tank it is currently charging (it
        must stay hotter to keep pushing heat in), which is exactly what
        T[k] represents, already decision-consistent with the rest of the
        model. The margin between T[k] and the real supply temperature is
        not directly measured (cop_dhw has no tank-temperature column to
        calibrate it against), so it is approximated as
        reference_supply_temperature_c's own margin above the highest
        configured target - the same "how much hotter does supply run than
        the target it's aiming for" gap already implied by that calibrated
        value, floored at 0 so supply is never modelled as colder than the
        tank it is heating.

        Falls back to (boiler_electrical_power_w, 0.0) - flat, independent
        of T - when no calibrated model or outdoor-temperature forecast
        exists for this step.
        """

        if self.cop_model is None or k >= len(data.outdoor_temperature_forecast):
            return self.config.boiler_electrical_power_w, 0.0

        margin = max(
            self.cop_model.reference_supply_temperature_c
            - max(data.target_temperature_top),
            0.0,
        )
        T_outdoor = data.outdoor_temperature_forecast[k]

        def power_at(T_tank_c: float, q_th_w: float) -> float:
            cop = self.cop_model.cop(T_outdoor, T_tank_c + margin)
            cop = min(
                max(cop, HeatPumpCOPIdentifier.MIN_COP), HeatPumpCOPIdentifier.MAX_COP
            )

            return q_th_w / cop

        power_low = power_at(
            HeatPumpCOPIdentifier.POWER_FIT_T_LOW_C,
            self.cop_model.q_th_at_power_fit_low_w,
        )
        power_high = power_at(
            HeatPumpCOPIdentifier.POWER_FIT_T_HIGH_C,
            self.cop_model.q_th_at_power_fit_high_w,
        )

        fit_range_c = (
            HeatPumpCOPIdentifier.POWER_FIT_T_HIGH_C
            - HeatPumpCOPIdentifier.POWER_FIT_T_LOW_C
        )
        beta = (power_high - power_low) / fit_range_c
        alpha = power_low - beta * HeatPumpCOPIdentifier.POWER_FIT_T_LOW_C

        return alpha, beta

    def _build_objective(
        self,
        model: pyo.ConcreteModel,
        data: MPCInput,
    ):
        objective = 0.0

        dt_hours = self.config.step_hours

        for k in model.K:
            grid_energy_kwh = model.active_power_w[k] * dt_hours / 1000.0

            objective += self.config.price_eur_per_kwh * grid_energy_kwh

            objective += self.config.weight_switching * model.boiler_start[k]

            objective += self.config.weight_temperature_slack * model.slack[k]

        return objective

    def _extract_result(
        self,
        model: pyo.ConcreteModel,
        data: MPCInput,
        termination_condition: TerminationCondition,
    ) -> MPCResult:
        horizon = len(data.solar_forecast_w)

        temperatures = tuple(float(pyo.value(model.T[k])) for k in range(horizon))

        schedule = tuple(
            round(float(pyo.value(model.boiler_on[k]))) for k in range(horizon)
        )

        objective_value = float(pyo.value(model.objective))

        # Same (alpha, beta) line the objective itself was built from (see
        # _power_line_coefficients) - the reported curve is exactly what the
        # optimizer costed with, not a separate display-only estimate.
        power_coefficients = [
            self._power_line_coefficients(data, k) for k in range(horizon)
        ]
        electrical_power_w = tuple(
            alpha + beta * temperatures[k]
            for k, (alpha, beta) in enumerate(power_coefficients)
        )

        return MPCResult(
            schedule=schedule,
            temperatures=temperatures,
            electrical_power_w=electrical_power_w,
            objective_value=objective_value,
            solver_status=str(termination_condition),
            termination_condition=str(termination_condition),
        )
