"""Receding-horizon Model Predictive Controller for UFH and UFH + DHW operation.

Overview
--------
This module is the **single canonical MPC entry point** for the project.  It
implements a receding-horizon optimiser that decides, at each control step,
the optimal thermal power sequence for the combined heat-pump system over a
look-ahead horizon of N steps.

Supported modes
---------------
* **UFH-only** : ``MPCController(ufh_model, params=MPCParameters, dhw_model=None)``
* **Combined** : ``MPCController(ufh_model, params=CombinedMPCParameters, dhw_model=DHWModel(...))``

Mathematical Formulation (§13–§14)
------------------------------------
The combined system state is the block-diagonal concatenation of the UFH and
DHW subsystem states:

    x_tot[k] = [T_r, T_b, T_top, T_bot]^T   (4 states when DHW enabled)
    u_tot[k] = [P_UFH, P_dhw]^T              (thermal powers [kW])

State dynamics (Forward-Euler, time-varying due to DHW tap-flow):

    x[k+1] = A[k] x[k] + B u[k] + E[k] d[k]

where d[k] = [T_out, Q_solar, Q_int, T_amb, T_mains]^T are known disturbances.

Cost function J (§14.2) minimised over horizon N:

    J = Σ_{k=0}^{N-1} [
          Q_c · (T_r[k] - T_ref[k])²          ← comfort deviation [K²]
        + p[k] · (P_UFH[k]/COP_UFH[k]
                + P_dhw[k]/COP_dhw[k]) · Δt   ← electrical energy cost [€]
        + R_c · P_UFH[k]²                      ← regularisation (damps spikes)
        + M · (ε_UFH[k]² + ε_dhw[k]²)         ← soft-constraint penalty
    ] + Q_N · (T_r[N] - T_ref[N])²             ← terminal comfort weight

Key design choices:
* **Thermal power as decision variable**: P_UFH and P_dhw drive the state
  equations (heat balance).  Electrical power = thermal / COP appears only
  in the cost function and the shared electrical power constraint.
* **Time-varying COP**: Both COP arrays may vary over the horizon (e.g., as a
  function of outdoor temperature via a Carnot / heating-curve model).  The
  MPC accepts pre-computed arrays from ``ForecastHorizon.cop_ufh_k`` and
  ``DHWForecastHorizon.cop_dhw_k``.
* **Soft constraints**: Room temperature and DHW temperature bounds are
  enforced via slack variables to prevent QP infeasibility (§14.3).
* **Shared electrical budget** (combined mode only):
      P_UFH[k]/COP_UFH[k] + P_dhw[k]/COP_dhw[k]  ≤  P_hp_max_elec
* **Legionella**: Managed by an external State Machine that injects a
  hard lower bound on T_top in the relevant horizon window.

Solver architecture
-------------------
1. **Primary**: CVXPY convex QP via OSQP backend (``_solve_convex``).
2. **Fallback**: Greedy one-step-ahead heuristic (``_solve_greedy``) — used
   automatically when CVXPY raises or is unavailable.

Units: power [kW], energy [kWh], temperature [°C], time step [h], cost [€].
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
class MPCSolution:
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


# ---------------------------------------------------------------------------
# Canonical unified controller
# ---------------------------------------------------------------------------


class MPCController:
    """Receding-horizon MPC controller for UFH or UFH + DHW combined.

    This is the **single authoritative solver implementation**.  The same
    convex-QP and greedy-fallback logic cover both UFH-only and combined
    operation.

    Operating principle
    -------------------
    At each control step the controller solves a finite-horizon optimisation
    problem that balances three competing objectives:

    1. **Comfort**: keep the room temperature T_r close to the setpoint T_ref.
    2. **Cost**: minimise the actual electricity bill (thermal power / COP × price).
    3. **Constraints**: respect actuator limits, ramp-rates, comfort bands, and
       the shared electrical budget of the heat pump.

    The controller then applies only the **first** element of the optimal
    sequence (receding-horizon principle) and re-solves at the next step with
    an updated state estimate from the Kalman filter.

    COP handling (§14.1)
    --------------------
    The MPC decision variables are always **thermal** power [kW].  Electrical
    power is computed as ``P_elec = P_thermal / COP`` and appears exclusively
    in the cost function and the shared electrical-budget constraint.

    The COP may be:

    * **Scalar** (constant over horizon): set via ``MPCParameters.cop_ufh``
      and ``DHWMPCParameters.cop_dhw``.
    * **Time-varying** (per time step): provide ``ForecastHorizon.cop_ufh_k``
      and / or ``DHWForecastHorizon.cop_dhw_k`` arrays of length N.  These
      are typically computed by ``HeatPumpCOPModel`` from the outdoor-temperature
      forecast via the Carnot formula and a heating curve (stooklijn).

    Parameters
    ----------
    ufh_model:
        Discrete UFH two-zone thermal model (§5).
    params:
        ``MPCParameters`` for UFH-only; ``CombinedMPCParameters`` for DHW.
    dhw_model:
        Optional DHW two-node stratification model (§11).  ``None`` → UFH-only.
    solver:
        CVXPY solver identifier.  Default: ``"OSQP"`` (fast, sparse QP).

    Raises
    ------
    ValueError
        If ``dhw_model`` and ``params`` types are inconsistent.
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
    def _p_hp_max_elec(self) -> float:
        """Maximum **electrical** power budget for the shared heat pump [kW].

        In combined mode this is ``CombinedMPCParameters.P_hp_max_elec``.
        In UFH-only mode there is no shared constraint; the property returns
        the maximum UFH electrical power (P_max / cop_ufh) for internal use only.
        """
        if isinstance(self.params, CombinedMPCParameters):
            return self.params.P_hp_max_elec
        # UFH-only: no shared constraint; return the UFH electrical ceiling
        return self._p_ufh.P_max / self._p_ufh.cop_ufh

    def _resolve_cop_ufh(self, forecast: ForecastHorizon) -> np.ndarray:
        """Return the UFH COP as an array of length N [dimensionless].

        Priority:
        1. ``forecast.cop_ufh_k`` (time-varying) if provided.
        2. Scalar ``MPCParameters.cop_ufh`` expanded to length N.

        The cop_max upper-bound check is enforced here against the scalar from
        MPCParameters (the forecast __post_init__ already verified > 1).

        Args:
            forecast: UFH forecast horizon, optionally containing cop_ufh_k.

        Returns:
            Array of COP values, shape (N,) [dimensionless, > 1].

        Raises:
            ValueError: If any time-varying value exceeds cop_max.
        """
        p_ufh = self._p_ufh
        N = p_ufh.horizon_steps
        if forecast.cop_ufh_k is not None:
            cop = np.asarray(forecast.cop_ufh_k, dtype=float)
            if np.any(cop > p_ufh.cop_max):
                raise ValueError(
                    f"cop_ufh_k contains values > cop_max={p_ufh.cop_max}."
                )
            return cop
        # Expand scalar COP to a constant array (already validated > 1 in MPCParameters)
        return np.full(N, p_ufh.cop_ufh)

    def _resolve_cop_dhw(self, forecast: DHWForecastHorizon) -> np.ndarray:
        """Return the DHW COP as an array of length N [dimensionless].

        Priority:
        1. ``forecast.cop_dhw_k`` (time-varying) if provided.
        2. Scalar ``DHWMPCParameters.cop_dhw`` expanded to length N.

        Args:
            forecast: DHW forecast horizon, optionally containing cop_dhw_k.

        Returns:
            Array of COP values, shape (N,) [dimensionless, > 1].

        Raises:
            ValueError: If any time-varying value exceeds cop_max, or if DHW is
                        not enabled.
        """
        p_dhw = self._p_dhw
        assert p_dhw is not None, "DHW must be enabled to resolve DHW COP."
        N = self._p_ufh.horizon_steps
        if forecast.cop_dhw_k is not None:
            cop = np.asarray(forecast.cop_dhw_k, dtype=float)
            if np.any(cop > p_dhw.cop_max):
                raise ValueError(
                    f"cop_dhw_k contains values > cop_max={p_dhw.cop_max}."
                )
            return cop
        return np.full(N, p_dhw.cop_dhw)

    def solve(
        self,
        initial_ufh_state_c: np.ndarray,
        ufh_forecast: ForecastHorizon,
        initial_dhw_state_c: np.ndarray | None = None,
        dhw_forecast: DHWForecastHorizon | None = None,
        previous_p_ufh_kw: float = 0.0,
        previous_p_dhw_kw: float = 0.0,
    ) -> MPCSolution:
        """Compute the optimal control sequence for the coming horizon.

        Always returns a valid ``MPCSolution``. Invalid dimensions fail
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
    ) -> MPCSolution:
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

        # §14.1 Resolve COP arrays over the horizon [dimensionless].
        # Thermal power is the decision variable; electrical = thermal / COP.
        cop_ufh = self._resolve_cop_ufh(ufh_forecast)
        cop_dhw = self._resolve_cop_dhw(dhw_forecast) if self._dhw_enabled and dhw_forecast is not None else None

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

            # §14.1: Electrical power = thermal / COP.  COP values are known floats,
            # so these are affine CVXPY expressions (linear in the decision variables).
            inv_cop_ufh_k = 1.0 / float(cop_ufh[k])
            P_ufh_elec_k = u_ufh[k] * inv_cop_ufh_k  # [kW elec]

            if self._dhw_enabled:
                assert u_dhw is not None and cop_dhw is not None
                inv_cop_dhw_k = 1.0 / float(cop_dhw[k])
                P_dhw_elec_k = u_dhw[k] * inv_cop_dhw_k  # type: ignore[index]  # [kW elec]
                P_elec_k = P_ufh_elec_k + P_dhw_elec_k
            else:
                P_elec_k = P_ufh_elec_k

            if has_pv:
                assert P_import is not None
                # Net grid import = electrical HP demand minus on-site PV generation.
                constraints.append(P_import[k] >= P_elec_k - float(pv[k]))

            # UFH actuator bounds (thermal) and ramp-rate
            constraints.extend([u_ufh[k] >= 0.0, u_ufh[k] <= p_ufh.P_max])
            prev_ufh = prev_u_ufh if k == 0 else u_ufh[k - 1]
            constraints.append(cp.abs(u_ufh[k] - prev_ufh) <= p_ufh.delta_P_max)

            if self._dhw_enabled:
                assert u_dhw is not None and p_dhw is not None
                # DHW actuator bounds (thermal) and ramp-rate
                constraints.extend([u_dhw[k] >= 0.0, u_dhw[k] <= p_dhw.P_dhw_max])  # type: ignore[index]
                prev_dhw = prev_u_dhw if k == 0 else u_dhw[k - 1]  # type: ignore[index]
                constraints.append(cp.abs(u_dhw[k] - prev_dhw) <= p_dhw.delta_P_dhw_max)  # type: ignore[index]
                # §14: Shared WP electrical budget constraint:
                #   P_UFH/COP_UFH + P_dhw/COP_dhw ≤ P_hp_max_elec
                constraints.append(P_elec_k <= self._p_hp_max_elec)

            # UFH comfort (soft constraints via slack variables)
            constraints.extend([
                x[0, k + 1] >= p_ufh.T_min - s_lo_ufh[k],
                x[0, k + 1] <= p_ufh.T_max + s_hi_ufh[k],
            ])

            if self._dhw_enabled:
                assert s_dhw is not None and s_leg is not None and p_dhw is not None
                constraints.append(x[2, k + 1] >= p_dhw.T_dhw_min - s_dhw[k])
                if leg_req is not None and leg_req[k]:
                    constraints.append(x[2, k + 1] >= p_dhw.T_legionella - s_leg[k])

            # §14.2 Cost terms:
            #   - Electrical energy cost: p[k] * (P_UFH/COP_UFH + P_dhw/COP_dhw) * dt  [€]
            #   - Comfort quadratic penalty on room temperature deviation
            #   - Regularisation on UFH thermal power (damps spikes)
            #   - Soft-constraint penalties
            energy_cost_k = prices[k] * P_import[k] * dt if has_pv else prices[k] * P_elec_k * dt  # type: ignore[index]
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
    ) -> MPCSolution:
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

        # §14.1 Resolve COP arrays for the horizon
        cop_ufh = self._resolve_cop_ufh(ufh_forecast)
        cop_dhw = (
            self._resolve_cop_dhw(dhw_forecast)
            if self._dhw_enabled and dhw_forecast is not None
            else None
        )

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

            inv_cop_ufh_k = 1.0 / float(cop_ufh[k])
            inv_cop_dhw_k = 1.0 / float(cop_dhw[k]) if cop_dhw is not None else 0.0

            for c_ufh in cands_ufh:
                for c_dhw in cands_dhw:
                    # §14 Shared WP constraint (electrical):
                    #   P_UFH/COP_UFH + P_dhw/COP_dhw ≤ P_hp_max_elec
                    p_elec_k = c_ufh * inv_cop_ufh_k + c_dhw * inv_cop_dhw_k
                    if self._dhw_enabled and p_elec_k > self._p_hp_max_elec + 1e-9:
                        continue
                    u_vec = np.array([c_ufh, c_dhw]) if self._dhw_enabled else np.array([c_ufh])
                    xn = A_list[k] @ x + B_mat @ u_vec + E_list[k] @ D_tot[k]

                    # Net grid import after PV offset (electrical power basis)
                    p_import_k = max(0.0, p_elec_k - float(pv[k]))

                    s_lo_ufh = max(p_ufh.T_min - xn[0], 0.0)
                    s_hi_ufh = max(xn[0] - p_ufh.T_max, 0.0)
                    score = (
                        p_ufh.Q_c * (x[0] - refs[k]) ** 2
                        + prices[k] * p_import_k * dt   # electrical energy cost [€]
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
            # Reconstruct electrical import for objective accounting
            p_elec_k = float(us_ufh_arr[k]) / float(cop_ufh[k])
            if cop_dhw is not None:
                p_elec_k += float(us_dhw_arr[k]) / float(cop_dhw[k])
            p_import_k = max(0.0, p_elec_k - float(pv[k]))
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
    ) -> MPCSolution:
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

        return MPCSolution(
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


