"""CVXPY MPC problem assembly separated from solve orchestration.

This builder supports both:

* a continuous convex MPC formulation for modulating thermal power, and
* a mixed-integer formulation with explicit on/off variables, minimum power,
  and switch-tracking auxiliaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .supervisors import HeatPumpTopologySupervisor, LegionellaSupervisor
from ..types.control import DHWMPCParameters, MPCParameters
from ..types.forecast import DHWForecastHorizon, ForecastHorizon

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class MpcDecisionVariables:
    """All CVXPY decision variables used by one MPC problem instance."""

    x: object
    u_ufh: object
    u_dhw: object | None
    p_import: object | None
    p_export: object | None
    z_ufh: object | None
    z_dhw: object | None
    sw_ufh: object | None
    sw_dhw: object | None
    s_lo_ufh: object
    s_hi_ufh: object
    s_dhw_lo: object | None
    s_dhw_hi: object | None
    s_dhw_target: object | None
    s_leg: object | None
    d_room: object | None
    d_room_terminal: object | None


@dataclass(frozen=True, slots=True)
class MpcBuildArtifacts:
    """Assembled MPC problem and references to its decision variables."""

    problem: object
    variables: MpcDecisionVariables


@dataclass(frozen=True, slots=True)
class MpcProblemBuilder:
    """Build one MPC problem from fixed forecasts, models, and supervisors."""

    ufh_parameters: MPCParameters
    dhw_parameters: DHWMPCParameters | None
    topology_supervisor: HeatPumpTopologySupervisor
    legionella_supervisor: LegionellaSupervisor | None
    dhw_enabled: bool

    def build(
        self,
        *,
        x0: np.ndarray,
        ufh_forecast: ForecastHorizon,
        dhw_forecast: DHWForecastHorizon | None,
        prev_u_ufh: float,
        prev_u_dhw: float,
        a_list: list[np.ndarray],
        b_matrix: np.ndarray,
        e_list: list[np.ndarray],
        disturbance_matrix: np.ndarray,
        cop_ufh: np.ndarray,
        cop_dhw: np.ndarray | None,
        dt_hours: float,
        mixed_integer_mode: bool,
    ) -> MpcBuildArtifacts:
        """Return the fully assembled MPC problem."""
        assert cp is not None

        n_horizon = self.ufh_parameters.horizon_steps
        n_states = a_list[0].shape[0]
        references_c = ufh_forecast.room_temperature_ref_c
        prices = ufh_forecast.price_eur_per_kwh
        feed_in_prices = ufh_forecast.feed_in_price_eur_per_kwh
        pv_kw = ufh_forecast.pv_kw
        has_pv = bool(np.any(pv_kw > 0.0))
        rho_ufh = self.ufh_parameters.rho_factor * max(self.ufh_parameters.Q_c, 1.0)
        legionella_required = (
            dhw_forecast.legionella_required if dhw_forecast is not None else np.zeros(n_horizon, dtype=bool)
        )

        variables = self._create_variables(
            n_horizon=n_horizon,
            n_states=n_states,
            has_pv=has_pv,
            mixed_integer_mode=mixed_integer_mode,
        )
        x = variables.x
        u_ufh = variables.u_ufh
        u_dhw = variables.u_dhw
        p_import = variables.p_import
        p_export = variables.p_export
        z_ufh = variables.z_ufh
        z_dhw = variables.z_dhw
        sw_ufh = variables.sw_ufh
        sw_dhw = variables.sw_dhw
        s_lo_ufh = variables.s_lo_ufh
        s_hi_ufh = variables.s_hi_ufh
        s_dhw_lo = variables.s_dhw_lo
        s_dhw_hi = variables.s_dhw_hi
        s_dhw_target = variables.s_dhw_target
        s_leg = variables.s_leg
        d_room = variables.d_room
        d_room_terminal = variables.d_room_terminal

        constraints: list = [x[:, 0] == x0]
        cost_terms = []

        for step_index in range(n_horizon):
            u_step = (
                cp.hstack([u_ufh[step_index : step_index + 1], u_dhw[step_index : step_index + 1]])
                if self.dhw_enabled and u_dhw is not None
                else u_ufh[step_index : step_index + 1]
            )
            constraints.append(
                x[:, step_index + 1]
                == a_list[step_index] @ x[:, step_index]
                + b_matrix @ u_step
                + e_list[step_index] @ disturbance_matrix[step_index]
            )

            p_elec_ufh = u_ufh[step_index] * (1.0 / float(cop_ufh[step_index]))
            if self.dhw_enabled:
                if u_dhw is None or cop_dhw is None:
                    raise ValueError("Combined MPC problem requires DHW variables and COP forecast.")
                p_elec_dhw = u_dhw[step_index] * (1.0 / float(cop_dhw[step_index]))
                p_elec_total = p_elec_ufh + p_elec_dhw
            else:
                p_elec_total = p_elec_ufh

            if has_pv:
                if p_import is None or p_export is None:
                    raise ValueError("PV-enabled MPC problem requires import/export variables.")
                constraints.extend(
                    [
                        p_import[step_index] >= p_elec_total - float(pv_kw[step_index]),
                        p_export[step_index] >= float(pv_kw[step_index]) - p_elec_total,
                    ]
                )

            previous_u_ufh = prev_u_ufh if step_index == 0 else u_ufh[step_index - 1]
            previous_u_dhw = None
            if self.dhw_enabled and u_dhw is not None:
                previous_u_dhw = prev_u_dhw if step_index == 0 else u_dhw[step_index - 1]
            self.topology_supervisor.apply_step_constraints(
                constraints=constraints,
                step_index=step_index,
                u_ufh=u_ufh,
                previous_u_ufh=previous_u_ufh,
                total_electrical_power_kw=p_elec_total,
                u_dhw=u_dhw,
                previous_u_dhw=previous_u_dhw,
                ufh_on=z_ufh,
                dhw_on=z_dhw,
                ufh_switch=sw_ufh,
                dhw_switch=sw_dhw,
            )

            constraints.extend(
                [
                    x[0, step_index + 1] >= self.ufh_parameters.T_min - s_lo_ufh[step_index],
                    x[0, step_index + 1] <= self.ufh_parameters.T_max + s_hi_ufh[step_index],
                ]
            )
            if mixed_integer_mode:
                if d_room is None:
                    raise ValueError("Mixed-integer MPC requires room deviation variables.")
                constraints.extend(
                    [
                        d_room[step_index] >= x[0, step_index] - references_c[step_index],
                        d_room[step_index] >= references_c[step_index] - x[0, step_index],
                    ]
                )

            if self.dhw_enabled:
                if (
                    self.dhw_parameters is None
                    or self.legionella_supervisor is None
                    or s_dhw_lo is None
                    or s_dhw_hi is None
                    or s_dhw_target is None
                    or s_leg is None
                ):
                    raise ValueError("Combined MPC problem requires DHW parameters and slack variables.")
                self.legionella_supervisor.apply_step_constraints(
                    constraints=constraints,
                    predicted_state_matrix=x,
                    step_index=step_index,
                    lower_slack=s_dhw_lo,
                    upper_slack=s_dhw_hi,
                    legionella_slack=s_leg,
                    legionella_required=bool(legionella_required[step_index]),
                )
                target_top_c = (
                    float(dhw_forecast.target_top_c[step_index])
                    if dhw_forecast is not None and dhw_forecast.target_top_c is not None
                    else float(self.dhw_parameters.T_dhw_target)
                )
                constraints.append(
                    s_dhw_target[step_index] >= target_top_c - x[2, step_index + 1]
                )

            energy_cost = (
                prices[step_index] * p_import[step_index] * dt_hours
                - feed_in_prices[step_index] * p_export[step_index] * dt_hours
                if has_pv and p_import is not None and p_export is not None
                else prices[step_index] * p_elec_total * dt_hours
            )
            if mixed_integer_mode:
                if d_room is None:
                    raise ValueError("Mixed-integer MPC requires room deviation variables.")
                cost_step = (
                    self.ufh_parameters.Q_c * d_room[step_index]
                    + energy_cost
                    + self.ufh_parameters.R_c * u_ufh[step_index]
                    + rho_ufh * (s_lo_ufh[step_index] + s_hi_ufh[step_index])
                )
                if sw_ufh is not None:
                    cost_step = cost_step + self.ufh_parameters.switch_penalty_eur * sw_ufh[step_index]
                if self.dhw_enabled:
                    assert self.dhw_parameters is not None
                    assert (
                        s_dhw_lo is not None
                        and s_dhw_hi is not None
                        and s_dhw_target is not None
                        and s_leg is not None
                    )
                    cost_step = cost_step + self.dhw_parameters.comfort_rho_factor * (
                        s_dhw_lo[step_index] + s_dhw_hi[step_index]
                    )
                    cost_step = cost_step + self.dhw_parameters.target_rho_factor * s_dhw_target[step_index]
                    cost_step = cost_step + self.dhw_parameters.legionella_rho_factor * s_leg[step_index]
                    if sw_dhw is not None and u_dhw is not None:
                        cost_step = (
                            cost_step
                            + self.dhw_parameters.switch_penalty_eur * sw_dhw[step_index]
                            + self.ufh_parameters.R_c * u_dhw[step_index]
                        )
            else:
                cost_step = (
                    self.ufh_parameters.Q_c * cp.square(x[0, step_index] - references_c[step_index])
                    + energy_cost
                    + self.ufh_parameters.R_c * cp.square(u_ufh[step_index])
                    + rho_ufh
                    * (cp.square(s_lo_ufh[step_index]) + cp.square(s_hi_ufh[step_index]))
                )
                if self.dhw_enabled:
                    assert self.dhw_parameters is not None
                    assert (
                        s_dhw_lo is not None
                        and s_dhw_hi is not None
                        and s_dhw_target is not None
                        and s_leg is not None
                    )
                    cost_step = cost_step + self.dhw_parameters.comfort_rho_factor * (
                        cp.square(s_dhw_lo[step_index]) + cp.square(s_dhw_hi[step_index])
                    )
                    cost_step = cost_step + self.dhw_parameters.target_rho_factor * cp.square(
                        s_dhw_target[step_index]
                    )
                    cost_step = cost_step + self.dhw_parameters.legionella_rho_factor * cp.square(
                        s_leg[step_index]
                    )
            cost_terms.append(cost_step)

        if mixed_integer_mode:
            if d_room_terminal is None:
                raise ValueError("Mixed-integer MPC requires a terminal room deviation variable.")
            constraints.extend(
                [
                    d_room_terminal >= x[0, n_horizon] - references_c[n_horizon],
                    d_room_terminal >= references_c[n_horizon] - x[0, n_horizon],
                ]
            )
            objective = cp.Minimize(cp.sum(cost_terms) + self.ufh_parameters.Q_N * d_room_terminal)
        else:
            objective = cp.Minimize(
                cp.sum(cost_terms)
                + self.ufh_parameters.Q_N * cp.square(x[0, n_horizon] - references_c[n_horizon])
            )
        if self.dhw_enabled:
            if self.dhw_parameters is None:
                raise ValueError("Combined MPC problem requires DHW parameters.")
            constraints.append(x[2, n_horizon] >= float(self.dhw_parameters.terminal_top_min))
        return MpcBuildArtifacts(problem=cp.Problem(objective, constraints), variables=variables)

    def _create_variables(
        self,
        *,
        n_horizon: int,
        n_states: int,
        has_pv: bool,
        mixed_integer_mode: bool = False,
    ) -> MpcDecisionVariables:
        """Allocate all CVXPY decision variables used by the MPC."""
        assert cp is not None
        return MpcDecisionVariables(
            x=cp.Variable((n_states, n_horizon + 1)),
            u_ufh=cp.Variable(n_horizon),
            u_dhw=cp.Variable(n_horizon) if self.dhw_enabled else None,
            p_import=cp.Variable(n_horizon, nonneg=True) if has_pv else None,
            p_export=cp.Variable(n_horizon, nonneg=True) if has_pv else None,
            z_ufh=cp.Variable(n_horizon, boolean=True) if mixed_integer_mode else None,
            z_dhw=cp.Variable(n_horizon, boolean=True) if mixed_integer_mode and self.dhw_enabled else None,
            sw_ufh=cp.Variable(n_horizon, nonneg=True) if mixed_integer_mode else None,
            sw_dhw=cp.Variable(n_horizon, nonneg=True) if mixed_integer_mode and self.dhw_enabled else None,
            s_lo_ufh=cp.Variable(n_horizon, nonneg=True),
            s_hi_ufh=cp.Variable(n_horizon, nonneg=True),
            s_dhw_lo=cp.Variable(n_horizon, nonneg=True) if self.dhw_enabled else None,
            s_dhw_hi=cp.Variable(n_horizon, nonneg=True) if self.dhw_enabled else None,
            s_dhw_target=cp.Variable(n_horizon, nonneg=True) if self.dhw_enabled else None,
            s_leg=cp.Variable(n_horizon, nonneg=True) if self.dhw_enabled else None,
            d_room=cp.Variable(n_horizon, nonneg=True) if mixed_integer_mode else None,
            d_room_terminal=cp.Variable(nonneg=True) if mixed_integer_mode else None,
        )
