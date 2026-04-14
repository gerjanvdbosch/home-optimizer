"""Unified MPC implementation for UFH-only and UFH + DHW operation.

This module is the single canonical MPC entry point for the project.

Supported modes
---------------
* UFH-only: ``MPCController(..., dhw_model=None, params=MPCParameters)``
* Combined: ``MPCController(..., dhw_model=DHWModel(...), params=CombinedMPCParameters)``

Backward-compatible adapter/alias names remain available:
* ``UFHMPCController`` → UFH-only convenience wrapper
* ``CombinedMPCController`` → alias of ``MPCController``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from .dhw_model import DHWModel
from .thermal_model import ThermalModel
from .types import (
    CombinedMPCParameters,
    DHWForecastHorizon,
    DHWMPCParameters,
    ForecastHorizon,
    MPCParameters,
)

try:
    import cvxpy as cp

    _CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    cp = None  # type: ignore[assignment]
    _CVXPY_AVAILABLE = False

_AnyMPCParams = Union[MPCParameters, CombinedMPCParameters]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CombinedMPCSolution:
    """Result of a unified MPC solve call.

    Attributes
    ----------
    ufh_control_sequence_kw:     Optimal P_UFH sequence [kW], length N.
    dhw_control_sequence_kw:     Optimal P_dhw sequence [kW], length N
                                 (all zeros in UFH-only mode).
    predicted_states_c:          Predicted state trajectory, shape (N+1, n_states).
                                 n_states = 2 (UFH-only) or 4 (combined).
    objective_value:             Value of the MPC cost function J.
    solver_status:               Status string from the optimisation backend.
    max_ufh_comfort_violation_c: Largest room-temperature soft-constraint violation [K].
    max_dhw_comfort_violation_c: Largest DHW top-layer soft-constraint violation [K].
    max_legionella_violation_c:  Largest legionella soft-constraint violation [K].
    used_fallback:               True when the greedy fallback solver was used.
    """

    ufh_control_sequence_kw: np.ndarray
    dhw_control_sequence_kw: np.ndarray
    predicted_states_c: np.ndarray
    objective_value: float
    solver_status: str
    max_ufh_comfort_violation_c: float = 0.0
    max_dhw_comfort_violation_c: float = 0.0
    max_legionella_violation_c: float = 0.0
    used_fallback: bool = False

    @property
    def first_ufh_control_kw(self) -> float:
        return float(self.ufh_control_sequence_kw[0]) if self.ufh_control_sequence_kw.size else 0.0

    @property
    def first_dhw_control_kw(self) -> float:
        return float(self.dhw_control_sequence_kw[0]) if self.dhw_control_sequence_kw.size else 0.0


@dataclass(frozen=True, slots=True)
class MPCSolution:
    """UFH-only view of the unified MPC result.

    Attributes
    ----------
    control_sequence_kw:      Optimal P_UFH sequence [kW], length N.
    predicted_states_c:       Predicted [T_r, T_b] trajectory, shape (N+1, 2).
    objective_value:          Value of the cost function J.
    solver_status:            Status string from the convex solver.
    max_comfort_violation_c:  Largest soft-constraint violation in °C (0 = feasible).
    used_fallback:            True when the greedy fallback was used.
    """

    control_sequence_kw: np.ndarray
    predicted_states_c: np.ndarray
    objective_value: float
    solver_status: str
    max_comfort_violation_c: float = 0.0
    used_fallback: bool = False

    @property
    def first_control_kw(self) -> float:
        return float(self.control_sequence_kw[0]) if self.control_sequence_kw.size else 0.0


# ---------------------------------------------------------------------------
# Canonical unified controller
# ---------------------------------------------------------------------------


class MPCController:
    """Receding-horizon MPC controller for UFH or UFH + DHW combined.

    This is the single authoritative solver implementation. The same convex-QP
    and greedy-fallback logic cover both UFH-only and combined operation.

    Parameters
    ----------
    ufh_model:   Discrete UFH thermal model.
    params:      ``MPCParameters`` when DHW is disabled; ``CombinedMPCParameters``
                 when DHW is active.
    dhw_model:   Optional DHW stratification model. ``None`` → UFH-only mode.
    solver:      CVXPY solver name (default: ``"OSQP"``).
    """

    def __init__(
        self,
        ufh_model: ThermalModel,
        params: _AnyMPCParams,
        dhw_model: DHWModel | None = None,
        solver: str = "OSQP",
    ) -> None:
        if dhw_model is None and isinstance(params, CombinedMPCParameters):
            raise ValueError(
                "CombinedMPCParameters requires a dhw_model. "
                "Use MPCParameters for UFH-only operation."
            )
        if dhw_model is not None and not isinstance(params, CombinedMPCParameters):
            raise ValueError(
                "A DHW model requires CombinedMPCParameters. "
                "For UFH-only operation pass dhw_model=None."
            )
        self.ufh_model = ufh_model
        self.dhw_model = dhw_model
        self.params = params
        self.solver = solver

    @property
    def _dhw_enabled(self) -> bool:
        return self.dhw_model is not None

    @property
    def _p_ufh(self) -> MPCParameters:
        return self.params.ufh if isinstance(self.params, CombinedMPCParameters) else self.params  # type: ignore[return-value]

    @property
    def _p_dhw(self) -> DHWMPCParameters | None:
        return self.params.dhw if isinstance(self.params, CombinedMPCParameters) else None

    @property
    def _p_hp_max(self) -> float:
        if isinstance(self.params, CombinedMPCParameters):
            return self.params.P_hp_max
        return self._p_ufh.P_max

    def solve(
        self,
        initial_ufh_state_c: np.ndarray,
        ufh_forecast: ForecastHorizon,
        initial_dhw_state_c: np.ndarray | None = None,
        dhw_forecast: DHWForecastHorizon | None = None,
        previous_p_ufh_kw: float = 0.0,
        previous_p_dhw_kw: float = 0.0,
    ) -> CombinedMPCSolution:
        """Compute the optimal control sequence for the coming horizon.

        Always returns a valid ``CombinedMPCSolution``. Invalid dimensions fail
        fast with ``ValueError``.
        """
        x_ufh0 = np.asarray(initial_ufh_state_c, dtype=float)
        if x_ufh0.shape != (2,):
            raise ValueError("initial_ufh_state_c must be [T_r, T_b].")
        N = self._p_ufh.horizon_steps
        if ufh_forecast.horizon_steps != N:
            raise ValueError("ufh_forecast.horizon_steps must equal MPCParameters.horizon_steps.")

        if self._dhw_enabled:
            if initial_dhw_state_c is None or dhw_forecast is None:
                raise ValueError("initial_dhw_state_c and dhw_forecast are required when DHW is active.")
            x_dhw0 = np.asarray(initial_dhw_state_c, dtype=float)
            if x_dhw0.shape != (2,):
                raise ValueError("initial_dhw_state_c must be [T_top, T_bot].")
            if dhw_forecast.horizon_steps != N:
                raise ValueError("dhw_forecast.horizon_steps must equal MPCParameters.horizon_steps.")
            x0 = np.concatenate([x_ufh0, x_dhw0])
        else:
            x0 = x_ufh0

        if _CVXPY_AVAILABLE:
            try:
                return self._solve_convex(
                    x0,
                    ufh_forecast,
                    dhw_forecast,
                    float(previous_p_ufh_kw),
                    float(previous_p_dhw_kw),
                )
            except Exception:  # noqa: BLE001
                pass

        return self._solve_greedy(
            x0,
            ufh_forecast,
            dhw_forecast,
            float(previous_p_ufh_kw),
            float(previous_p_dhw_kw),
        )

    def _build_matrices(
        self,
        ufh_forecast: ForecastHorizon,
        dhw_forecast: DHWForecastHorizon | None,
    ) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
        """Build time-varying state-space matrices for all horizon steps."""
        A_ufh, B_ufh, E_ufh = self.ufh_model.state_matrices()
        N = self._p_ufh.horizon_steps
        ufh_d = ufh_forecast.disturbance_matrix(self.ufh_model.parameters)

        if not self._dhw_enabled:
            return [A_ufh] * N, B_ufh, [E_ufh] * N, ufh_d

        assert self.dhw_model is not None
        assert dhw_forecast is not None
        _, B_dhw, _ = self.dhw_model.state_matrices(v_tap_m3_per_h=0.0)
        B_mat = np.block([[B_ufh, np.zeros((2, 1))], [np.zeros((2, 1)), B_dhw]])
        dhw_d = dhw_forecast.disturbance_matrix()
        D_tot = np.hstack([ufh_d, dhw_d])

        A_list: list[np.ndarray] = []
        E_list: list[np.ndarray] = []
        for k in range(N):
            v_tap_k = float(dhw_forecast.v_tap_m3_per_h[k])
            A_dhw_k, _, E_dhw_k = self.dhw_model.state_matrices(v_tap_k)
            A_list.append(np.block([[A_ufh, np.zeros((2, 2))], [np.zeros((2, 2)), A_dhw_k]]))
            E_list.append(np.block([[E_ufh, np.zeros((2, 2))], [np.zeros((2, 3)), E_dhw_k]]))

        return A_list, B_mat, E_list, D_tot

    def _solve_convex(
        self,
        x0: np.ndarray,
        ufh_forecast: ForecastHorizon,
        dhw_forecast: DHWForecastHorizon | None,
        prev_u_ufh: float,
        prev_u_dhw: float,
    ) -> CombinedMPCSolution:
        assert cp is not None

        p_ufh = self._p_ufh
        p_dhw = self._p_dhw
        N = p_ufh.horizon_steps
        dt = self.ufh_model.parameters.dt_hours
        refs = ufh_forecast.room_temperature_ref_c
        prices = ufh_forecast.price_eur_per_kwh
        pv = ufh_forecast.pv_kw
        rho_ufh = p_ufh.rho_factor * max(p_ufh.Q_c, 1.0)
        has_pv = bool(np.any(pv > 0.0))

        A_list, B_mat, E_list, D_tot = self._build_matrices(ufh_forecast, dhw_forecast)
        n_states = A_list[0].shape[0]

        x = cp.Variable((n_states, N + 1))
        u_ufh = cp.Variable(N)
        u_dhw = cp.Variable(N) if self._dhw_enabled else None
        P_import = cp.Variable(N, nonneg=True) if has_pv else None
        s_lo_ufh = cp.Variable(N, nonneg=True)
        s_hi_ufh = cp.Variable(N, nonneg=True)
        s_dhw = cp.Variable(N, nonneg=True) if self._dhw_enabled else None
        s_leg = cp.Variable(N, nonneg=True) if self._dhw_enabled else None

        constraints: list = [x[:, 0] == x0]
        cost_terms = []
        leg_req = dhw_forecast.legionella_required if dhw_forecast is not None else None

        for k in range(N):
            u_k = cp.hstack([u_ufh[k:k + 1], u_dhw[k:k + 1]]) if self._dhw_enabled else u_ufh[k:k + 1]  # type: ignore[index]
            constraints.append(x[:, k + 1] == A_list[k] @ x[:, k] + B_mat @ u_k + E_list[k] @ D_tot[k])

            P_hp_k = u_ufh[k] + (u_dhw[k] if self._dhw_enabled else 0.0)  # type: ignore[index]
            if has_pv:
                assert P_import is not None
                constraints.append(P_import[k] >= P_hp_k - float(pv[k]))

            constraints.extend([u_ufh[k] >= 0.0, u_ufh[k] <= p_ufh.P_max])
            prev_ufh = prev_u_ufh if k == 0 else u_ufh[k - 1]
            constraints.append(cp.abs(u_ufh[k] - prev_ufh) <= p_ufh.delta_P_max)

            if self._dhw_enabled:
                assert u_dhw is not None and p_dhw is not None
                constraints.extend([u_dhw[k] >= 0.0, u_dhw[k] <= p_dhw.P_dhw_max])  # type: ignore[index]
                prev_dhw = prev_u_dhw if k == 0 else u_dhw[k - 1]  # type: ignore[index]
                constraints.append(cp.abs(u_dhw[k] - prev_dhw) <= p_dhw.delta_P_dhw_max)  # type: ignore[index]
                constraints.append(u_ufh[k] + u_dhw[k] <= self._p_hp_max)  # type: ignore[index]

            constraints.extend([
                x[0, k + 1] >= p_ufh.T_min - s_lo_ufh[k],
                x[0, k + 1] <= p_ufh.T_max + s_hi_ufh[k],
            ])

            if self._dhw_enabled:
                assert s_dhw is not None and s_leg is not None and p_dhw is not None
                constraints.append(x[2, k + 1] >= p_dhw.T_dhw_min - s_dhw[k])
                if leg_req is not None and leg_req[k]:
                    constraints.append(x[2, k + 1] >= p_dhw.T_legionella - s_leg[k])

            energy_cost_k = prices[k] * P_import[k] * dt if has_pv else prices[k] * P_hp_k * dt  # type: ignore[index]
            cost_k = (
                p_ufh.Q_c * cp.square(x[0, k] - refs[k])
                + energy_cost_k
                + p_ufh.R_c * cp.square(u_ufh[k])
                + rho_ufh * (cp.square(s_lo_ufh[k]) + cp.square(s_hi_ufh[k]))
            )
            if self._dhw_enabled:
                assert s_dhw is not None and s_leg is not None and p_dhw is not None
                cost_k = cost_k + p_dhw.comfort_rho_factor * cp.square(s_dhw[k])
                cost_k = cost_k + p_dhw.legionella_rho_factor * cp.square(s_leg[k])
            cost_terms.append(cost_k)

        obj = cp.Minimize(cp.sum(cost_terms) + p_ufh.Q_N * cp.square(x[0, N] - refs[N]))
        problem = cp.Problem(obj, constraints)

        solve_kw: dict[str, object] = {"solver": self.solver, "warm_start": True, "verbose": False}
        if self.solver.upper() == "OSQP":
            solve_kw.update({"eps_abs": 1e-7, "eps_rel": 1e-7, "polishing": True})
        problem.solve(**solve_kw)

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"Solver returned status '{problem.status}'.")
        if u_ufh.value is None or x.value is None:
            raise RuntimeError("Solver returned no variable values.")

        return self._package_solution(
            x_val=np.asarray(x.value, dtype=float).T,
            u_ufh_val=np.asarray(u_ufh.value, dtype=float).reshape(N),
            u_dhw_val=(np.asarray(u_dhw.value, dtype=float).reshape(N)
                       if (self._dhw_enabled and u_dhw is not None and u_dhw.value is not None)
                       else np.zeros(N)),
            objective=float(problem.value),
            status=str(problem.status),
            ufh_forecast=ufh_forecast,
            dhw_forecast=dhw_forecast,
            used_fallback=False,
        )

    def _solve_greedy(
        self,
        x0: np.ndarray,
        ufh_forecast: ForecastHorizon,
        dhw_forecast: DHWForecastHorizon | None,
        prev_u_ufh: float,
        prev_u_dhw: float,
    ) -> CombinedMPCSolution:
        p_ufh = self._p_ufh
        p_dhw = self._p_dhw
        g = p_ufh.greedy
        N = p_ufh.horizon_steps
        dt = self.ufh_model.parameters.dt_hours
        refs = ufh_forecast.room_temperature_ref_c
        prices = ufh_forecast.price_eur_per_kwh
        pv = ufh_forecast.pv_kw
        rho_ufh = p_ufh.rho_factor * max(p_ufh.Q_c, 1.0)
        leg_req = dhw_forecast.legionella_required if dhw_forecast is not None else None

        A_list, B_mat, E_list, D_tot = self._build_matrices(ufh_forecast, dhw_forecast)

        x = x0.copy()
        xs = [x.copy()]
        us_ufh: list[float] = []
        us_dhw: list[float] = []
        u_prev_ufh, u_prev_dhw = prev_u_ufh, prev_u_dhw

        for k in range(N):
            lo_ufh = max(0.0, u_prev_ufh - p_ufh.delta_P_max)
            hi_ufh = min(p_ufh.P_max, u_prev_ufh + p_ufh.delta_P_max)
            n_cand = min(g.max_candidates, max(
                g.min_candidates,
                int((hi_ufh - lo_ufh) / max(p_ufh.delta_P_max / g.grid_divisor, g.min_grid_step_kw)) + 1,
            ))
            cands_ufh = np.linspace(lo_ufh, hi_ufh, n_cand)

            if self._dhw_enabled and p_dhw is not None:
                lo_dhw = max(0.0, u_prev_dhw - p_dhw.delta_P_dhw_max)
                hi_dhw = min(p_dhw.P_dhw_max, u_prev_dhw + p_dhw.delta_P_dhw_max)
                cands_dhw = np.linspace(lo_dhw, hi_dhw, n_cand)
            else:
                cands_dhw = np.array([0.0])

            best_score = np.inf
            best_u_ufh, best_u_dhw, best_xn = cands_ufh[0], 0.0, None

            for c_ufh in cands_ufh:
                for c_dhw in cands_dhw:
                    if c_ufh + c_dhw > self._p_hp_max + 1e-9:
                        continue
                    u_vec = np.array([c_ufh, c_dhw]) if self._dhw_enabled else np.array([c_ufh])
                    xn = A_list[k] @ x + B_mat @ u_vec + E_list[k] @ D_tot[k]
                    p_import_k = max(0.0, c_ufh + c_dhw - float(pv[k]))

                    s_lo_ufh = max(p_ufh.T_min - xn[0], 0.0)
                    s_hi_ufh = max(xn[0] - p_ufh.T_max, 0.0)
                    score = (
                        p_ufh.Q_c * (x[0] - refs[k]) ** 2
                        + prices[k] * p_import_k * dt
                        + p_ufh.R_c * c_ufh ** 2
                        + rho_ufh * (s_lo_ufh ** 2 + s_hi_ufh ** 2)
                        + g.lookahead_weight * p_ufh.Q_N * max(refs[k + 1] - xn[0], 0.0) ** 2
                    )
                    if self._dhw_enabled and p_dhw is not None:
                        s_dhw_k = max(p_dhw.T_dhw_min - xn[2], 0.0)
                        s_leg_k = max(p_dhw.T_legionella - xn[2], 0.0) if (leg_req is not None and leg_req[k]) else 0.0
                        score += (
                            p_dhw.comfort_rho_factor * s_dhw_k ** 2
                            + p_dhw.legionella_rho_factor * s_leg_k ** 2
                        )
                    if k == N - 1:
                        score += p_ufh.Q_N * (xn[0] - refs[N]) ** 2

                    if score < best_score:
                        best_score, best_u_ufh, best_u_dhw, best_xn = score, float(c_ufh), float(c_dhw), xn

            assert best_xn is not None
            us_ufh.append(best_u_ufh)
            us_dhw.append(best_u_dhw)
            x = best_xn
            xs.append(x.copy())
            u_prev_ufh, u_prev_dhw = best_u_ufh, best_u_dhw

        xs_arr = np.array(xs, dtype=float)
        us_ufh_arr = np.array(us_ufh, dtype=float)
        us_dhw_arr = np.array(us_dhw, dtype=float)

        objective = 0.0
        for k in range(N):
            p_import_k = max(0.0, us_ufh_arr[k] + us_dhw_arr[k] - float(pv[k]))
            objective += (
                p_ufh.Q_c * (xs_arr[k, 0] - refs[k]) ** 2
                + prices[k] * p_import_k * dt
                + p_ufh.R_c * us_ufh_arr[k] ** 2
            )
        objective += p_ufh.Q_N * (xs_arr[N, 0] - refs[N]) ** 2

        return self._package_solution(
            x_val=xs_arr,
            u_ufh_val=us_ufh_arr,
            u_dhw_val=us_dhw_arr,
            objective=objective,
            status="greedy-fallback",
            ufh_forecast=ufh_forecast,
            dhw_forecast=dhw_forecast,
            used_fallback=True,
        )

    def _package_solution(
        self,
        x_val: np.ndarray,
        u_ufh_val: np.ndarray,
        u_dhw_val: np.ndarray,
        objective: float,
        status: str,
        ufh_forecast: ForecastHorizon,
        dhw_forecast: DHWForecastHorizon | None,
        used_fallback: bool,
    ) -> CombinedMPCSolution:
        p_ufh = self._p_ufh
        p_dhw = self._p_dhw
        t_r_pred = x_val[1:, 0]
        ufh_viol = float(np.max(
            np.maximum(p_ufh.T_min - t_r_pred, 0.0).tolist()
            + np.maximum(t_r_pred - p_ufh.T_max, 0.0).tolist()
        ))

        dhw_viol, leg_viol = 0.0, 0.0
        if self._dhw_enabled and p_dhw is not None and dhw_forecast is not None:
            t_top_pred = x_val[1:, 2]
            dhw_viol = float(np.max(np.maximum(p_dhw.T_dhw_min - t_top_pred, 0.0)))
            leg_viol = float(np.max(
                np.where(
                    dhw_forecast.legionella_required,
                    np.maximum(p_dhw.T_legionella - t_top_pred, 0.0),
                    0.0,
                )
            ))

        return CombinedMPCSolution(
            ufh_control_sequence_kw=u_ufh_val,
            dhw_control_sequence_kw=u_dhw_val,
            predicted_states_c=x_val,
            objective_value=objective,
            solver_status=status,
            max_ufh_comfort_violation_c=ufh_viol,
            max_dhw_comfort_violation_c=dhw_viol,
            max_legionella_violation_c=leg_viol,
            used_fallback=used_fallback,
        )


# ---------------------------------------------------------------------------
# Backward-compatible UFH-only adapter and aliases
# ---------------------------------------------------------------------------


class UFHMPCController:
    """UFH-only convenience wrapper over the unified ``MPCController``."""

    def __init__(
        self,
        model: ThermalModel,
        params: MPCParameters,
        solver: str = "OSQP",
    ) -> None:
        self.model = model
        self.params = params
        self._delegate = MPCController(ufh_model=model, params=params, dhw_model=None, solver=solver)

    def solve(
        self,
        initial_state_c: np.ndarray,
        forecast: ForecastHorizon,
        previous_power_kw: float = 0.0,
    ) -> MPCSolution:
        x0 = np.asarray(initial_state_c, dtype=float)
        if x0.shape != (2,):
            raise ValueError("initial_state_c must be [T_r, T_b].")
        if forecast.horizon_steps != self.params.horizon_steps:
            raise ValueError("Forecast horizon must equal MPCParameters.horizon_steps.")

        sol = self._delegate.solve(
            initial_ufh_state_c=x0,
            ufh_forecast=forecast,
            previous_p_ufh_kw=float(previous_power_kw),
        )

        return MPCSolution(
            control_sequence_kw=sol.ufh_control_sequence_kw,
            predicted_states_c=sol.predicted_states_c,
            objective_value=sol.objective_value,
            solver_status=sol.solver_status,
            max_comfort_violation_c=sol.max_ufh_comfort_violation_c,
            used_fallback=sol.used_fallback,
        )


CombinedMPCController = MPCController

