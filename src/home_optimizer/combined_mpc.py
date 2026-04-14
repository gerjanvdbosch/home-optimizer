"""Combined UFH + DHW Model Predictive Controller (§13–§14 of spec).

Combined state vector  x_tot = [T_r, T_b, T_top, T_bot]ᵀ   (4 states)
Combined control       u_tot = [P_UFH, P_dhw]ᵀ              (2 inputs)

Block-diagonal state-space (§13):
  x_tot[k+1] = A_tot[k] x_tot[k] + B_tot u_tot[k] + E_tot[k] d_tot[k]

  A_tot[k] = block_diag(A_UFH, A_dhw[k])      — A_dhw[k] is time-varying
  B_tot    = block_diag(B_UFH, B_dhw)          — constant
  E_tot[k] = block_diag(E_UFH, E_dhw[k])      — E_dhw[k] is time-varying

  d_tot[k] = [T_out, Q_solar, Q_int, T_amb, T_mains]ᵀ  (5-vector)

MPC cost function (§14.1):
  J = Σ_{k=0}^{N-1} [ Q_c·(T_r[k]−T_ref[k])²          UFH comfort
                      + p[k]·P_UFH[k]·Δt                UFH energy cost [€]
                      + R_c·P_UFH[k]²                   UFH regularisation
                      + p[k]·P_dhw[k]·Δt ]              DHW energy cost [€]
    + Q_N·(T_r[N]−T_ref[N])²                            terminal comfort

Constraints (§14.2):
  Hard:   0 ≤ P_UFH[k] ≤ P_UFH_max
          |P_UFH[k] − P_UFH[k-1]| ≤ ΔP_UFH_max
          0 ≤ P_dhw[k] ≤ P_dhw_max
          |P_dhw[k] − P_dhw[k-1]| ≤ ΔP_dhw_max
          P_UFH[k] + P_dhw[k] ≤ P_hp_max          (shared heat pump)
  Soft:   T_r[k+1] ≥ T_min − s_lo_ufh[k]          UFH comfort lower bound
          T_r[k+1] ≤ T_max + s_hi_ufh[k]           UFH comfort upper bound
          T_top[k+1] ≥ T_dhw_min − s_dhw[k]        DHW tap comfort
          T_top[k+1] ≥ T_leg − s_leg[k]            Legionella (if required[k])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dhw_model import DHWModel
from .thermal_model import ThermalModel
from .types import (
    CombinedMPCParameters,
    DHWForecastHorizon,
    ForecastHorizon,
)

try:
    import cvxpy as cp

    _CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    cp = None  # type: ignore[assignment]
    _CVXPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CombinedMPCSolution:
    """Result of a single combined UFH+DHW MPC solve call.

    Attributes
    ----------
    ufh_control_sequence_kw:    Optimal P_UFH sequence [kW], length N.
    dhw_control_sequence_kw:    Optimal P_dhw sequence [kW], length N.
    predicted_states_c:         Predicted [T_r, T_b, T_top, T_bot] trajectory,
                                shape (N+1, 4).
    objective_value:            Value of the combined cost function J.
    solver_status:              Status string from the solver.
    max_ufh_comfort_violation_c: Largest UFH room-temp soft-constraint violation [K].
    max_dhw_comfort_violation_c: Largest DHW tap-temp soft-constraint violation [K].
    max_legionella_violation_c:  Largest legionella soft-constraint violation [K].
    used_fallback:              True when the greedy fallback was used.
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


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class CombinedMPCController:
    """Receding-horizon MPC controller for a combined UFH + DHW system.

    The two thermal subsystems are thermally decoupled; they share only the
    total heat-pump power budget P_hp_max (§13, §14.2).

    Parameters
    ----------
    ufh_model:    Discrete UFH thermal model.
    dhw_model:    Discrete DHW stratification model.
    params:       Combined MPC parameters (UFH + DHW + shared heat pump).
    solver:       CVXPY solver name (default: "OSQP").
    """

    def __init__(
        self,
        ufh_model: ThermalModel,
        dhw_model: DHWModel,
        params: CombinedMPCParameters,
        solver: str = "OSQP",
    ) -> None:
        self.ufh_model = ufh_model
        self.dhw_model = dhw_model
        self.params = params
        self.solver = solver

    def solve(
        self,
        initial_ufh_state_c: np.ndarray,
        initial_dhw_state_c: np.ndarray,
        ufh_forecast: ForecastHorizon,
        dhw_forecast: DHWForecastHorizon,
        previous_p_ufh_kw: float = 0.0,
        previous_p_dhw_kw: float = 0.0,
    ) -> CombinedMPCSolution:
        """Compute the optimal combined control sequence for the coming horizon.

        Always returns a valid CombinedMPCSolution; comfort and legionella
        violations are reported in the result object.

        Parameters
        ----------
        initial_ufh_state_c:  Initial [T_r, T_b] [°C].
        initial_dhw_state_c:  Initial [T_top, T_bot] [°C].
        ufh_forecast:         UFH disturbances + price + reference over N steps.
        dhw_forecast:         DHW disturbances + legionella mask over N steps.
        previous_p_ufh_kw:    UFH power at step k=-1 (for ramp-rate constraint).
        previous_p_dhw_kw:    DHW power at step k=-1 (for ramp-rate constraint).
        """
        x_ufh0 = np.asarray(initial_ufh_state_c, dtype=float)
        x_dhw0 = np.asarray(initial_dhw_state_c, dtype=float)
        if x_ufh0.shape != (2,):
            raise ValueError("initial_ufh_state_c must be [T_r, T_b].")
        if x_dhw0.shape != (2,):
            raise ValueError("initial_dhw_state_c must be [T_top, T_bot].")
        N = self.params.ufh.horizon_steps
        if ufh_forecast.horizon_steps != N or dhw_forecast.horizon_steps != N:
            raise ValueError("Both forecasts must have horizon_steps matching CombinedMPCParameters.ufh.horizon_steps.")

        x0 = np.concatenate([x_ufh0, x_dhw0])  # (4,)

        if _CVXPY_AVAILABLE:
            try:
                return self._solve_convex(x0, ufh_forecast, dhw_forecast,
                                          float(previous_p_ufh_kw), float(previous_p_dhw_kw))
            except Exception:  # noqa: BLE001
                pass

        return self._solve_greedy(x0, ufh_forecast, dhw_forecast,
                                  float(previous_p_ufh_kw), float(previous_p_dhw_kw))

    # ------------------------------------------------------------------
    # Internal helpers: precompute time-varying block-diagonal matrices
    # ------------------------------------------------------------------

    def _build_matrices(
        self,
        ufh_forecast: ForecastHorizon,
        dhw_forecast: DHWForecastHorizon,
    ) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
        """Precompute A_tot[k], B_tot, E_tot[k], d_tot[k] for k = 0…N-1.

        Returns
        -------
        A_list : list of (4,4) arrays, one per step.
        B_tot  : constant (4,2) array.
        E_list : list of (4,5) arrays, one per step.
        D_tot  : (N,5) disturbance matrix [T_out, Q_solar, Q_int, T_amb, T_mains].
        """
        A_ufh, B_ufh, E_ufh = self.ufh_model.state_matrices()
        _, B_dhw, _ = self.dhw_model.state_matrices(v_tap_m3_per_h=0.0)

        # Constant block-diagonal B_tot  (4×2)
        B_tot = np.block([
            [B_ufh, np.zeros((2, 1))],
            [np.zeros((2, 1)), B_dhw],
        ])

        ufh_d = ufh_forecast.disturbance_matrix(self.ufh_model.parameters)  # (N,3)
        dhw_d = dhw_forecast.disturbance_matrix()  # (N,2)  [T_amb, T_mains]
        D_tot = np.hstack([ufh_d, dhw_d])  # (N,5)

        A_list: list[np.ndarray] = []
        E_list: list[np.ndarray] = []
        N = ufh_forecast.horizon_steps
        for k in range(N):
            v_tap_k = float(dhw_forecast.v_tap_m3_per_h[k])
            A_dhw_k, _, E_dhw_k = self.dhw_model.state_matrices(v_tap_k)

            A_tot_k = np.block([
                [A_ufh, np.zeros((2, 2))],
                [np.zeros((2, 2)), A_dhw_k],
            ])
            E_tot_k = np.block([
                [E_ufh, np.zeros((2, 2))],
                [np.zeros((2, 3)), E_dhw_k],
            ])
            A_list.append(A_tot_k)
            E_list.append(E_tot_k)

        return A_list, B_tot, E_list, D_tot

    # ------------------------------------------------------------------
    # Convex QP via cvxpy
    # ------------------------------------------------------------------

    def _solve_convex(
        self,
        x0: np.ndarray,
        ufh_forecast: ForecastHorizon,
        dhw_forecast: DHWForecastHorizon,
        prev_u_ufh: float,
        prev_u_dhw: float,
    ) -> CombinedMPCSolution:
        assert cp is not None

        p_ufh = self.params.ufh
        p_dhw = self.params.dhw
        N = p_ufh.horizon_steps
        dt = self.ufh_model.parameters.dt_hours
        refs = ufh_forecast.room_temperature_ref_c
        prices = ufh_forecast.price_eur_per_kwh
        leg_req = dhw_forecast.legionella_required

        rho_ufh = p_ufh.rho_factor * max(p_ufh.Q_c, 1.0)
        rho_dhw = p_dhw.comfort_rho_factor
        rho_leg = p_dhw.legionella_rho_factor

        A_list, B_tot, E_list, D_tot = self._build_matrices(ufh_forecast, dhw_forecast)

        # Decision variables
        x = cp.Variable((4, N + 1))
        u_ufh = cp.Variable(N)
        u_dhw = cp.Variable(N)
        s_lo_ufh = cp.Variable(N, nonneg=True)
        s_hi_ufh = cp.Variable(N, nonneg=True)
        s_dhw = cp.Variable(N, nonneg=True)
        s_leg = cp.Variable(N, nonneg=True)

        constraints: list = [x[:, 0] == x0]
        cost_terms = []

        for k in range(N):
            u_k = cp.vstack([u_ufh[k:k+1], u_dhw[k:k+1]])  # (2,1)
            # Dynamics (hard)
            constraints.append(
                x[:, k + 1] == A_list[k] @ x[:, k] + B_tot @ u_k[:, 0] + E_list[k] @ D_tot[k]
            )
            # UFH power bounds (hard)
            constraints.extend([u_ufh[k] >= 0.0, u_ufh[k] <= p_ufh.P_max])
            # DHW power bounds (hard)
            constraints.extend([u_dhw[k] >= 0.0, u_dhw[k] <= p_dhw.P_dhw_max])
            # Shared heat-pump budget (hard)
            constraints.append(u_ufh[k] + u_dhw[k] <= self.params.P_hp_max)
            # UFH ramp-rate (hard)
            prev_ufh = prev_u_ufh if k == 0 else u_ufh[k - 1]
            constraints.append(cp.abs(u_ufh[k] - prev_ufh) <= p_ufh.delta_P_max)
            # DHW ramp-rate (hard)
            prev_dhw = prev_u_dhw if k == 0 else u_dhw[k - 1]
            constraints.append(cp.abs(u_dhw[k] - prev_dhw) <= p_dhw.delta_P_dhw_max)
            # UFH soft comfort bounds
            constraints.extend([
                x[0, k + 1] >= p_ufh.T_min - s_lo_ufh[k],
                x[0, k + 1] <= p_ufh.T_max + s_hi_ufh[k],
            ])
            # DHW tap comfort (soft)
            constraints.append(x[2, k + 1] >= p_dhw.T_dhw_min - s_dhw[k])
            # Legionella (stiff soft — only where required)
            if leg_req[k]:
                constraints.append(x[2, k + 1] >= p_dhw.T_legionella - s_leg[k])

            # Stage cost
            cost_terms.append(
                p_ufh.Q_c * cp.square(x[0, k] - refs[k])
                + prices[k] * u_ufh[k] * dt
                + p_ufh.R_c * cp.square(u_ufh[k])
                + prices[k] * u_dhw[k] * dt
                + rho_ufh * (cp.square(s_lo_ufh[k]) + cp.square(s_hi_ufh[k]))
                + rho_dhw * cp.square(s_dhw[k])
                + rho_leg * cp.square(s_leg[k])
            )

        obj = cp.Minimize(
            cp.sum(cost_terms) + p_ufh.Q_N * cp.square(x[0, N] - refs[N])
        )
        problem = cp.Problem(obj, constraints)

        solve_kw: dict[str, object] = {
            "solver": self.solver,
            "warm_start": True,
            "verbose": False,
        }
        if self.solver.upper() == "OSQP":
            solve_kw.update({"eps_abs": 1e-7, "eps_rel": 1e-7, "polishing": True})

        problem.solve(**solve_kw)

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"Solver returned status '{problem.status}'.")
        if u_ufh.value is None or u_dhw.value is None or x.value is None:
            raise RuntimeError("Solver returned no variable values.")

        x_val = np.asarray(x.value, dtype=float)
        u_ufh_val = np.asarray(u_ufh.value, dtype=float).reshape(N)
        u_dhw_val = np.asarray(u_dhw.value, dtype=float).reshape(N)
        T_r_pred = x_val[0, 1:]
        T_top_pred = x_val[2, 1:]

        ufh_viol = float(np.max(
            np.maximum(p_ufh.T_min - T_r_pred, 0.0).tolist()
            + np.maximum(T_r_pred - p_ufh.T_max, 0.0).tolist()
        ))
        dhw_viol = float(np.max(np.maximum(p_dhw.T_dhw_min - T_top_pred, 0.0)))
        leg_viol = float(np.max(
            np.where(leg_req, np.maximum(p_dhw.T_legionella - T_top_pred, 0.0), 0.0)
        ))

        return CombinedMPCSolution(
            ufh_control_sequence_kw=u_ufh_val,
            dhw_control_sequence_kw=u_dhw_val,
            predicted_states_c=x_val.T,
            objective_value=float(problem.value),
            solver_status=str(problem.status),
            max_ufh_comfort_violation_c=ufh_viol,
            max_dhw_comfort_violation_c=dhw_viol,
            max_legionella_violation_c=leg_viol,
            used_fallback=False,
        )

    # ------------------------------------------------------------------
    # Greedy joint-grid fallback  (no cvxpy dependency)
    # ------------------------------------------------------------------

    def _solve_greedy(
        self,
        x0: np.ndarray,
        ufh_forecast: ForecastHorizon,
        dhw_forecast: DHWForecastHorizon,
        prev_u_ufh: float,
        prev_u_dhw: float,
    ) -> CombinedMPCSolution:
        p_ufh = self.params.ufh
        p_dhw = self.params.dhw
        g = p_ufh.greedy  # shared greedy config
        N = p_ufh.horizon_steps
        dt = self.ufh_model.parameters.dt_hours
        refs = ufh_forecast.room_temperature_ref_c
        prices = ufh_forecast.price_eur_per_kwh
        leg_req = dhw_forecast.legionella_required

        rho_ufh = p_ufh.rho_factor * max(p_ufh.Q_c, 1.0)
        rho_dhw = p_dhw.comfort_rho_factor
        rho_leg = p_dhw.legionella_rho_factor

        A_list, B_tot, E_list, D_tot = self._build_matrices(ufh_forecast, dhw_forecast)

        x = x0.copy()
        xs = [x.copy()]
        us_ufh: list[float] = []
        us_dhw: list[float] = []
        u_prev_ufh, u_prev_dhw = prev_u_ufh, prev_u_dhw

        for k in range(N):
            lo_ufh = max(0.0, u_prev_ufh - p_ufh.delta_P_max)
            hi_ufh = min(p_ufh.P_max, u_prev_ufh + p_ufh.delta_P_max)
            lo_dhw = max(0.0, u_prev_dhw - p_dhw.delta_P_dhw_max)
            hi_dhw = min(p_dhw.P_dhw_max, u_prev_dhw + p_dhw.delta_P_dhw_max)

            n_cand = min(g.max_candidates, max(
                g.min_candidates,
                int((hi_ufh - lo_ufh) / max(p_ufh.delta_P_max / g.grid_divisor, g.min_grid_step_kw)) + 1,
            ))
            cands_ufh = np.linspace(lo_ufh, hi_ufh, n_cand)
            cands_dhw = np.linspace(lo_dhw, hi_dhw, n_cand)

            best_score = np.inf
            best_u_ufh, best_u_dhw, best_xn = cands_ufh[0], cands_dhw[0], None

            for c_ufh in cands_ufh:
                for c_dhw in cands_dhw:
                    # Hard shared-power constraint
                    if c_ufh + c_dhw > self.params.P_hp_max + 1e-9:
                        continue
                    u_vec = np.array([c_ufh, c_dhw])
                    xn = A_list[k] @ x + B_tot @ u_vec + E_list[k] @ D_tot[k]

                    # Soft constraint violations
                    s_lo_ufh = max(p_ufh.T_min - xn[0], 0.0)
                    s_hi_ufh = max(xn[0] - p_ufh.T_max, 0.0)
                    s_dhw = max(p_dhw.T_dhw_min - xn[2], 0.0)
                    s_leg_k = max(p_dhw.T_legionella - xn[2], 0.0) if leg_req[k] else 0.0

                    score = (
                        p_ufh.Q_c * (x[0] - refs[k]) ** 2
                        + prices[k] * c_ufh * dt
                        + p_ufh.R_c * c_ufh ** 2
                        + prices[k] * c_dhw * dt
                        + rho_ufh * (s_lo_ufh ** 2 + s_hi_ufh ** 2)
                        + rho_dhw * s_dhw ** 2
                        + rho_leg * s_leg_k ** 2
                        + g.lookahead_weight * p_ufh.Q_N * max(refs[k + 1] - xn[0], 0.0) ** 2
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

        xs_arr = np.array(xs, dtype=float)  # (N+1, 4)
        us_ufh_arr = np.array(us_ufh, dtype=float)
        us_dhw_arr = np.array(us_dhw, dtype=float)
        T_r_pred = xs_arr[1:, 0]
        T_top_pred = xs_arr[1:, 2]
        ufh_viol = float(np.max(
            np.maximum(p_ufh.T_min - T_r_pred, 0.0).tolist()
            + np.maximum(T_r_pred - p_ufh.T_max, 0.0).tolist()
        ))
        dhw_viol = float(np.max(np.maximum(p_dhw.T_dhw_min - T_top_pred, 0.0)))
        leg_viol = float(np.max(
            np.where(leg_req, np.maximum(p_dhw.T_legionella - T_top_pred, 0.0), 0.0)
        ))

        # Compute objective value for the greedy solution
        J = 0.0
        for k in range(N):
            T_r_k = xs_arr[k, 0]
            J += (
                p_ufh.Q_c * (T_r_k - refs[k]) ** 2
                + prices[k] * us_ufh_arr[k] * dt
                + p_ufh.R_c * us_ufh_arr[k] ** 2
                + prices[k] * us_dhw_arr[k] * dt
            )
        J += p_ufh.Q_N * (xs_arr[N, 0] - refs[N]) ** 2

        return CombinedMPCSolution(
            ufh_control_sequence_kw=us_ufh_arr,
            dhw_control_sequence_kw=us_dhw_arr,
            predicted_states_c=xs_arr,
            objective_value=J,
            solver_status="greedy-fallback",
            max_ufh_comfort_violation_c=ufh_viol,
            max_dhw_comfort_violation_c=dhw_viol,
            max_legionella_violation_c=leg_viol,
            used_fallback=True,
        )

