"""Model Predictive Controller for underfloor heating.

Objective (Section 7 of spec):
  J = Σ_{k=0}^{N-1} [ Q_c·(T_r[k] − T_ref[k])²
                      + p[k]·P_UFH[k]·Δt          (energy cost [€])
                      + R_c·P_UFH[k]² ]            (power regularisation)
    + Q_N·(T_r[N] − T_ref[N])²                     (terminal comfort)

Hard constraints:
  0 ≤ P_UFH[k] ≤ P_max
  T_min ≤ T_r[k+1] ≤ T_max
  |P_UFH[k] − P_UFH[k-1]| ≤ ΔP_max              (ramp-rate)

Solved with cvxpy + OSQP (convex QP).  No soft-constraint fallback is used:
physical infeasibility raises a ValueError so that callers can adapt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .thermal_model import ThermalModel
from .types import ForecastHorizon, MPCParameters

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
class MPCSolution:
    """Result of a single MPC solve call.

    Attributes
    ----------
    control_sequence_kw:  Optimal P_UFH sequence [kW], length N.
    predicted_states_c:   Predicted [T_r, T_b] trajectory, shape (N+1, 2).
    objective_value:      Value of the cost function J [mixed units].
    solver_status:        Status string from the convex solver.
    used_fallback:        True when the greedy fallback was used instead.
    """

    control_sequence_kw: np.ndarray
    predicted_states_c: np.ndarray
    objective_value: float
    solver_status: str
    used_fallback: bool = False

    @property
    def first_control_kw(self) -> float:
        """First element of the control sequence (applied at the current step)."""
        return float(self.control_sequence_kw[0]) if self.control_sequence_kw.size else 0.0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class UFHMPCController:
    """Receding-horizon MPC controller for UFH systems.

    Parameters
    ----------
    model:   Discrete thermal model.
    params:  MPC settings (horizon, weights, constraints).
    solver:  cvxpy solver name (default: "OSQP").
    """

    def __init__(
        self,
        model: ThermalModel,
        params: MPCParameters,
        solver: str = "OSQP",
    ) -> None:
        self.model = model
        self.params = params
        self.solver = solver

    def solve(
        self,
        initial_state_c: np.ndarray,
        forecast: ForecastHorizon,
        previous_power_kw: float = 0.0,
    ) -> MPCSolution:
        """Compute the optimal control sequence for the coming horizon.

        Parameters
        ----------
        initial_state_c:   Current state estimate [T_r, T_b] [°C].
        forecast:          Disturbance and price forecast over N steps.
        previous_power_kw: Power applied at the previous step (for ramp-rate).

        Returns
        -------
        MPCSolution with the optimal (or fallback) control sequence.

        Raises
        ------
        ValueError  if the problem is physically infeasible and no fallback
                    can satisfy the hard constraints.
        """
        x0 = np.asarray(initial_state_c, dtype=float)
        if x0.shape != (2,):
            raise ValueError("initial_state_c must be [T_r, T_b].")
        if forecast.horizon_steps != self.params.horizon_steps:
            raise ValueError("Forecast horizon must equal MPCParameters.horizon_steps.")

        if _CVXPY_AVAILABLE:
            try:
                return self._solve_convex(x0, forecast, float(previous_power_kw))
            except cp.SolverError:
                pass  # backend unavailable – try greedy

        return self._solve_greedy(x0, forecast, float(previous_power_kw))

    # ------------------------------------------------------------------
    # Convex QP via cvxpy
    # ------------------------------------------------------------------

    def _solve_convex(
        self,
        x0: np.ndarray,
        forecast: ForecastHorizon,
        prev_u: float,
    ) -> MPCSolution:
        assert cp is not None

        p = self.params
        A, B, E = self.model.state_matrices()
        B_vec = B[:, 0]
        D = forecast.disturbance_matrix(self.model.parameters)  # (N, 3)
        dt = self.model.parameters.dt_hours
        N = p.horizon_steps
        refs = forecast.room_temperature_ref_c
        prices = forecast.price_eur_per_kwh

        # Decision variables
        x = cp.Variable((2, N + 1))
        u = cp.Variable(N)

        constraints: list[cp.Constraint] = [x[:, 0] == x0]
        cost_terms = []

        for k in range(N):
            # Dynamics
            constraints.append(x[:, k + 1] == A @ x[:, k] + B_vec * u[k] + E @ D[k])
            # Input bounds
            constraints.extend([u[k] >= 0.0, u[k] <= p.P_max])
            # Comfort bounds (on the predicted next state)
            constraints.extend([x[0, k + 1] >= p.T_min, x[0, k + 1] <= p.T_max])
            # Ramp-rate
            prev = prev_u if k == 0 else u[k - 1]
            constraints.append(cp.abs(u[k] - prev) <= p.delta_P_max)
            # Stage cost
            cost_terms.append(
                p.Q_c * cp.square(x[0, k] - refs[k])
                + prices[k] * u[k] * dt
                + p.R_c * cp.square(u[k])
            )

        # Terminal cost
        obj = cp.Minimize(cp.sum(cost_terms) + p.Q_N * cp.square(x[0, N] - refs[N]))
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
            raise ValueError(
                f"MPC solve failed with status '{problem.status}'.  "
                "The problem may be physically infeasible given the current "
                "comfort bounds, ramp-rate limits, and forecast."
            )
        if u.value is None or x.value is None:
            raise ValueError("Solver returned no solution.")

        return MPCSolution(
            control_sequence_kw=np.asarray(u.value, dtype=float).reshape(N),
            predicted_states_c=np.asarray(x.value, dtype=float).T,
            objective_value=float(problem.value),
            solver_status=str(problem.status),
            used_fallback=False,
        )

    # ------------------------------------------------------------------
    # Greedy 1-step-ahead fallback (no cvxpy dependency)
    # ------------------------------------------------------------------

    def _solve_greedy(
        self,
        x0: np.ndarray,
        forecast: ForecastHorizon,
        prev_u: float,
    ) -> MPCSolution:
        """Greedy receding-horizon with a dense candidate grid."""
        p = self.params
        A, B, E = self.model.state_matrices()
        B_vec = B[:, 0]
        D = forecast.disturbance_matrix(self.model.parameters)
        dt = self.model.parameters.dt_hours
        refs = forecast.room_temperature_ref_c
        prices = forecast.price_eur_per_kwh
        N = p.horizon_steps

        x = x0.copy()
        xs = [x.copy()]
        us: list[float] = []
        u_prev = prev_u

        for k in range(N):
            lo = max(0.0, u_prev - p.delta_P_max)
            hi = min(p.P_max, u_prev + p.delta_P_max)
            n_cand = max(21, int((hi - lo) / max(p.delta_P_max / 10.0, 0.01)) + 1)
            candidates = np.linspace(lo, hi, min(n_cand, 51))

            best_u, best_xnext, best_score = candidates[0], None, np.inf
            for c in candidates:
                xn = A @ x + B_vec * c + E @ D[k]
                viol = max(p.T_min - xn[0], 0.0) ** 2 + max(xn[0] - p.T_max, 0.0) ** 2
                score = (
                    p.Q_c * (x[0] - refs[k]) ** 2
                    + prices[k] * c * dt
                    + p.R_c * c**2
                    + 1e6 * viol
                    + 5.0 * p.Q_N * max(refs[k + 1] - xn[0], 0.0) ** 2
                )
                if k == N - 1:
                    score += p.Q_N * (xn[0] - refs[N]) ** 2
                if score < best_score:
                    best_score, best_u, best_xnext = score, float(c), xn

            assert best_xnext is not None
            us.append(best_u)
            x = best_xnext
            xs.append(x.copy())
            u_prev = best_u

        us_arr = np.array(us, dtype=float)
        xs_arr = np.array(xs, dtype=float)
        obj = self._eval_objective(xs_arr, us_arr, forecast)
        return MPCSolution(
            control_sequence_kw=us_arr,
            predicted_states_c=xs_arr,
            objective_value=obj,
            solver_status="greedy-fallback",
            used_fallback=True,
        )

    # ------------------------------------------------------------------
    # Objective evaluation helper (post-hoc, unit tests)
    # ------------------------------------------------------------------

    def _eval_objective(
        self,
        xs: np.ndarray,
        us: np.ndarray,
        forecast: ForecastHorizon,
    ) -> float:
        p = self.params
        dt = self.model.parameters.dt_hours
        refs = forecast.room_temperature_ref_c
        prices = forecast.price_eur_per_kwh
        J = sum(
            p.Q_c * (xs[k, 0] - refs[k]) ** 2 + prices[k] * us[k] * dt + p.R_c * us[k] ** 2
            for k in range(len(us))
        )
        return float(J) + p.Q_N * (xs[-1, 0] - refs[-1]) ** 2
