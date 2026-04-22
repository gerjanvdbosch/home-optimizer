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
* **Soft constraints**: Room temperature bounds and a DHW top-layer target band
  are enforced via slack variables to prevent QP infeasibility (§14.3).
* **Shared electrical budget** (combined mode only):
      P_UFH[k]/COP_UFH[k] + P_dhw[k]/COP_dhw[k]  ≤  P_hp_max_elec
* **Legionella**: Managed by an external State Machine that injects a
  hard lower bound on T_top in the relevant horizon window.

Solver architecture
-------------------
The MPC has exactly one canonical execution path: a convex QP formulated in
CVXPY. The configured backend is tried first (OSQP by default). When that
backend stalls or returns a non-optimal status, the controller may retry with
other installed CVXPY solvers, but it never switches to a heuristic control law.
If no convex solver reaches an optimal status, the controller raises.

Units: power [kW], energy [kWh], temperature [°C], time step [h], cost [€].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from .problem_builder import MpcProblemBuilder
from .supervisors import HeatPumpTopologySupervisor, LegionellaSupervisor
from ..domain.dhw.model import DHWModel
from ..domain.ufh.model import ThermalModel
from ..types.control import CombinedMPCParameters, DHWMPCParameters, MPCParameters
from ..types.forecast import DHWForecastHorizon, ForecastHorizon

try:
    import cvxpy as cp

    _CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    cp = None  # type: ignore[assignment]
    _CVXPY_AVAILABLE = False

_AnyMPCParams = Union[MPCParameters, CombinedMPCParameters]
_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
_CONVEX_SOLVER_RETRY_ORDER = ("CLARABEL", "HIGHS", "SCS")


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
    used_fallback:               Retained for result-schema stability; always
                                 ``False`` in the canonical CVXPY-only architecture.
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
        # First receding-horizon action for UFH [kW].
        # Return 0.0 if the sequence is empty so callers can inspect a partial
        # solution object without an IndexError.
        return float(self.ufh_control_sequence_kw[0]) if self.ufh_control_sequence_kw.size else 0.0

    @property
    def first_dhw_control_kw(self) -> float:
        # First receding-horizon action for DHW [kW].
        # In UFH-only mode the DHW sequence is all zeros, so 0.0 is the correct
        # fail-soft return value when no explicit element exists.
        return float(self.dhw_control_sequence_kw[0]) if self.dhw_control_sequence_kw.size else 0.0


# ---------------------------------------------------------------------------
# Canonical unified controller
# ---------------------------------------------------------------------------


class MPCController:
    """Receding-horizon MPC controller for UFH or UFH + DHW combined.

    This is the **single authoritative solver implementation**.  Both UFH-only
    and combined operation are solved through the same convex CVXPY model.

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
        # Fail fast on inconsistent architecture choices:
        # - combined parameters require a DHW model
        # - a DHW model requires the combined parameter block
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
        # True only when the 4-state combined UFH+DHW model is active.
        return self.dhw_model is not None

    @property
    def _p_ufh(self) -> MPCParameters:
        # Normalise parameter access so the rest of the solver can use one UFH path
        # in both 2-state (UFH-only) and 4-state (combined) operation.
        return self.params.ufh if isinstance(self.params, CombinedMPCParameters) else self.params  # type: ignore[return-value]

    @property
    def _p_dhw(self) -> DHWMPCParameters | None:
        # DHW parameters exist only in combined mode; otherwise there is no DHW
        # actuator, no top-layer comfort constraint and no legionella logic.
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
                raise ValueError(f"cop_ufh_k contains values > cop_max={p_ufh.cop_max}.")
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
                raise ValueError(f"cop_dhw_k contains values > cop_max={p_dhw.cop_max}.")
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
        # Current UFH posterior state estimate [T_r, T_b] in °C.
        # Shape must be (2,) because the UFH model is fixed to two thermal states.
        x_ufh0 = np.asarray(initial_ufh_state_c, dtype=float)
        if x_ufh0.shape != (2,):
            raise ValueError("initial_ufh_state_c must be [T_r, T_b].")
        N = self._p_ufh.horizon_steps
        if ufh_forecast.horizon_steps != N:
            raise ValueError("ufh_forecast.horizon_steps must equal MPCParameters.horizon_steps.")

        if self._dhw_enabled:
            # Combined mode extends the state to x0 = [T_r, T_b, T_top, T_bot] [°C],
            # so both the DHW state and the DHW forecast horizon are mandatory.
            if initial_dhw_state_c is None or dhw_forecast is None:
                raise ValueError(
                    "initial_dhw_state_c and dhw_forecast are required when DHW is active."
                )
            x_dhw0 = np.asarray(initial_dhw_state_c, dtype=float)
            if x_dhw0.shape != (2,):
                raise ValueError("initial_dhw_state_c must be [T_top, T_bot].")
            if dhw_forecast.horizon_steps != N:
                raise ValueError(
                    "dhw_forecast.horizon_steps must equal MPCParameters.horizon_steps."
                )
            x0 = np.concatenate([x_ufh0, x_dhw0])
        else:
            x0 = x_ufh0

        if not _CVXPY_AVAILABLE:
            raise RuntimeError(
                "CVXPY is required for MPCController.solve(); the project architecture "
                "does not provide a heuristic fallback solver."
            )

        return self._solve_convex(
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
        # UFH-only base matrices:
        # - A_ufh: 2x2
        # - B_ufh: 2x1
        # - E_ufh: 2x3 for [T_out, Q_solar, Q_int]
        A_ufh, B_ufh, E_ufh = self.ufh_model.state_matrices()
        N = self._p_ufh.horizon_steps
        ufh_d = ufh_forecast.disturbance_matrix(self.ufh_model.parameters)

        if not self._dhw_enabled:
            # UFH-only mode is time-invariant, so the same A and E can be reused at
            # every horizon step. The disturbance matrix still varies by time step.
            return [A_ufh] * N, B_ufh, [E_ufh] * N, ufh_d

        assert self.dhw_model is not None
        assert dhw_forecast is not None
        _, B_dhw, _ = self.dhw_model.state_matrices(v_tap_m3_per_h=0.0)
        # Combined input matrix B_tot is block diagonal because UFH and DHW are
        # thermally decoupled; they interact only via shared heat-pump constraints.
        B_mat = np.block([[B_ufh, np.zeros((2, 1))], [np.zeros((2, 1)), B_dhw]])
        dhw_d = dhw_forecast.disturbance_matrix()
        # Disturbance vector per step becomes:
        # [T_out, Q_solar, Q_int, T_amb, T_mains]
        D_tot = np.hstack([ufh_d, dhw_d])

        A_list: list[np.ndarray] = []
        E_list: list[np.ndarray] = []
        for k in range(N):
            v_tap_k = float(dhw_forecast.v_tap_m3_per_h[k])
            A_dhw_k, _, E_dhw_k = self.dhw_model.state_matrices(v_tap_k)
            # DHW is LTV: A_dhw[k] and E_dhw[k] must be rebuilt with the forecast
            # tap flow at each step. The combined system remains block diagonal.
            A_list.append(np.block([[A_ufh, np.zeros((2, 2))], [np.zeros((2, 2)), A_dhw_k]]))
            E_list.append(np.block([[E_ufh, np.zeros((2, 2))], [np.zeros((2, 3)), E_dhw_k]]))

        return A_list, B_mat, E_list, D_tot

    def _convex_solver_candidates(self) -> list[str]:
        """Return the ordered list of installed convex solvers to try.

        The configured solver remains first choice. Additional candidates stay
        within the CVXPY ecosystem; they are robustness retries, not heuristic
        fallbacks.
        """
        assert cp is not None
        installed = {name.upper() for name in cp.installed_solvers()}
        ordered: list[str] = []
        for solver_name in (self.solver.upper(), *_CONVEX_SOLVER_RETRY_ORDER):
            if solver_name in installed and solver_name not in ordered:
                ordered.append(solver_name)
        if not ordered:
            raise RuntimeError("No supported CVXPY solvers are installed for the MPC problem.")
        return ordered

    @staticmethod
    def _solver_options(solver_name: str) -> dict[str, object]:
        """Return backend-specific CVXPY solve options for a convex MPC solve."""
        options: dict[str, object] = {"solver": solver_name, "warm_start": True, "verbose": False}
        if solver_name == "OSQP":
            options.update(
                {
                    "eps_abs": 1e-7,
                    "eps_rel": 1e-7,
                    "polishing": True,
                    "max_iter": 200_000,
                }
            )
        elif solver_name == "SCS":
            options.update({"eps": 1e-6, "max_iters": 50_000})
        return options

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

        # Decision variables remain thermal powers [kW]. Electrical power is derived
        # as P_thermal / COP, which keeps the problem convex because COP is known.
        # §14.1 Resolve COP arrays over the horizon [dimensionless].
        # Thermal power is the decision variable; electrical = thermal / COP.
        cop_ufh = self._resolve_cop_ufh(ufh_forecast)
        cop_dhw = (
            self._resolve_cop_dhw(dhw_forecast)
            if self._dhw_enabled and dhw_forecast is not None
            else None
        )

        A_list, B_mat, E_list, D_tot = self._build_matrices(ufh_forecast, dhw_forecast)
        builder = MpcProblemBuilder(
            ufh_parameters=p_ufh,
            dhw_parameters=p_dhw,
            topology_supervisor=HeatPumpTopologySupervisor(
                ufh_parameters=p_ufh,
                dhw_parameters=p_dhw,
                shared_hp_max_elec_kw=self._p_hp_max_elec,
                heat_pump_topology=(
                    self.params.heat_pump_topology
                    if isinstance(self.params, CombinedMPCParameters)
                    else "shared"
                ),
            ),
            legionella_supervisor=LegionellaSupervisor(p_dhw) if p_dhw is not None else None,
            dhw_enabled=self._dhw_enabled,
        )
        build = builder.build(
            x0=x0,
            ufh_forecast=ufh_forecast,
            dhw_forecast=dhw_forecast,
            prev_u_ufh=prev_u_ufh,
            prev_u_dhw=prev_u_dhw,
            a_list=A_list,
            b_matrix=B_mat,
            e_list=E_list,
            disturbance_matrix=D_tot,
            cop_ufh=cop_ufh,
            cop_dhw=cop_dhw,
            dt_hours=dt,
        )
        problem = build.problem
        variables = build.variables

        attempt_log: list[str] = []
        last_exception: Exception | None = None
        for solver_name in self._convex_solver_candidates():
            try:
                problem.solve(**self._solver_options(solver_name))
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                attempt_log.append(f"{solver_name}: {type(exc).__name__}({exc})")
                continue
            attempt_log.append(f"{solver_name}: {problem.status}")
            if str(problem.status).lower() in _OPTIMAL_STATUSES:
                break
        else:
            attempts = "; ".join(attempt_log)
            raise RuntimeError(
                "No convex solver reached an optimal MPC solution. "
                f"Attempts: {attempts}"
            ) from last_exception

        if variables.u_ufh.value is None or variables.x.value is None:
            raise RuntimeError("Solver returned no variable values.")

        return self._package_solution(
            x_val=np.asarray(variables.x.value, dtype=float).T,
            u_ufh_val=np.asarray(variables.u_ufh.value, dtype=float).reshape(N),
            u_dhw_val=(
                np.asarray(variables.u_dhw.value, dtype=float).reshape(N)
                if (
                    self._dhw_enabled
                    and variables.u_dhw is not None
                    and variables.u_dhw.value is not None
                )
                else np.zeros(N)
            ),
            objective=float(problem.value),
            status=str(problem.status),
            ufh_forecast=ufh_forecast,
            dhw_forecast=dhw_forecast,
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
    ) -> MPCSolution:
        p_ufh = self._p_ufh
        p_dhw = self._p_dhw
        # Evaluate comfort violations on predicted future temperatures only.
        # x_val[0] is the fixed estimated current state, not something the current
        # optimisation step can still influence.
        t_r_pred = x_val[1:, 0]
        ufh_viol = float(
            np.max(
                np.maximum(p_ufh.T_min - t_r_pred, 0.0).tolist()
                + np.maximum(t_r_pred - p_ufh.T_max, 0.0).tolist()
            )
        )

        dhw_viol, leg_viol = 0.0, 0.0
        if self._dhw_enabled and p_dhw is not None and dhw_forecast is not None:
            t_top_pred = x_val[1:, 2]
            t_bot_pred = x_val[1:, 3]
            dhw_viol = float(np.max(np.maximum(p_dhw.T_dhw_min - t_top_pred, 0.0)))
            leg_viol = float(
                np.max(
                    np.where(
                        dhw_forecast.legionella_required,
                        np.maximum(
                            np.maximum(p_dhw.T_legionella - t_top_pred, 0.0),
                            np.maximum(p_dhw.T_legionella - t_bot_pred, 0.0),
                        ),
                        0.0,
                    )
                )
            )

        return MPCSolution(
            ufh_control_sequence_kw=u_ufh_val,
            dhw_control_sequence_kw=u_dhw_val,
            predicted_states_c=x_val,
            objective_value=objective,
            solver_status=status,
            max_ufh_comfort_violation_c=ufh_viol,
            max_dhw_comfort_violation_c=dhw_viol,
            max_legionella_violation_c=leg_viol,
            used_fallback=False,
        )
