"""2-state discrete thermal model for a DHW stratification tank (§7–§11 of spec).

State vector    x_dhw = [T_top, T_bot]ᵀ
Control input   u_dhw = P_dhw  [kW]   (heat-pump output to bottom layer — assumption A5)
Disturbances    d_dhw = [T_amb, T_mains]ᵀ

The state-transition matrix A_dhw[k] is **time-varying** because it depends on the
tap-water flow rate V_tap[k] at each time step (bilinear term, linearised as LTV).

Physical equations (§9):
  C_top · dT_top/dt = −(T_top−T_bot)/R_strat − λ·V_tap·T_top − (T_top−T_amb)/R_loss
  C_bot · dT_bot/dt = +(T_top−T_bot)/R_strat + P_dhw + λ·V_tap·T_mains − (T_bot−T_amb)/R_loss

Discrete state-space (forward-Euler, step Δt [h], §10–§11):
──────────────────────────────────────────────────────────────

  Auxiliary scalars (constant):
    a_strat = Δt / (C_top · R_strat)
    b_strat = Δt / (C_bot · R_strat)
    a_loss  = Δt / (C_top · R_loss)
    b_loss  = Δt / (C_bot · R_loss)

  Time-varying scalars (depend on V_tap[k]):
    a_tap[k] = Δt / C_top · λ · V_tap[k]
    b_tap[k] = Δt / C_bot · λ · V_tap[k]

  A_dhw[k] = [[1 − a_strat − a_loss − a_tap[k],  a_strat],
               [b_strat,                          1 − b_strat − b_loss]]

  B_dhw    = [[0       ],     (P_dhw goes to bottom layer — assumption A5)
               [Δt/C_bot]]

  E_dhw[k] = [[a_loss,  0      ],
               [b_loss,  b_tap[k]]]

  d_dhw = [T_amb, T_mains]ᵀ

Energy-balance verification (§9.5):
  d/dt(C_top·T_top + C_bot·T_bot) = P_dhw − λ·V_tap·(T_top−T_mains)
                                    − (T_top−T_amb)/R_loss − (T_bot−T_amb)/R_loss  ✓

Measurement:  y = C_obs · x,  C_obs = [1, 0]  (only T_top is sensed)
Observability: rank([C_obs; C_obs·A_dhw[k]]) = 2  iff  a_strat ≠ 0  (always true, §11).

Derived quantity (not a state, §8.1, assumption A6):
  T_dhw = (C_top · T_top + C_bot · T_bot) / (C_top + C_bot)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import DHWParameters

#: Observation matrix C_obs = [1, 0] — only T_top is measured.
MEASUREMENT_MATRIX_DHW: np.ndarray = np.array([[1.0, 0.0]], dtype=float)


@dataclass(slots=True)
class DHWModel:
    """Discrete grey-box 2-node stratification model for a DHW tank.

    Validates Euler stability on construction (based on constant time constants;
    the stability margin may tighten further at high V_tap — verify separately).
    """

    parameters: DHWParameters

    def __post_init__(self) -> None:
        self.parameters.assert_euler_stable()

    # ------------------------------------------------------------------
    # Constant auxiliary scalars
    # ------------------------------------------------------------------

    def _constant_scalars(self) -> tuple[float, float, float, float]:
        """Return (a_strat, b_strat, a_loss, b_loss) — all Δt-dependent, V_tap-independent."""
        p = self.parameters
        dt = p.dt_hours
        a_strat = dt / (p.C_top * p.R_strat)
        b_strat = dt / (p.C_bot * p.R_strat)
        a_loss = dt / (p.C_top * p.R_loss)
        b_loss = dt / (p.C_bot * p.R_loss)
        return a_strat, b_strat, a_loss, b_loss

    # ------------------------------------------------------------------
    # Time-varying state-space matrices
    # ------------------------------------------------------------------

    def state_matrices(
        self,
        v_tap_m3_per_h: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (A_dhw, B_dhw, E_dhw) for a given tap flow rate V_tap [m³/h].

        A_dhw and E_dhw are time-varying (depend on V_tap[k]).
        B_dhw is constant (P_dhw enters the bottom layer only — assumption A5).
        """
        p = self.parameters
        dt = p.dt_hours
        a_strat, b_strat, a_loss, b_loss = self._constant_scalars()

        # Tap-flow-dependent scalars (bilinear term, linearised via known V_tap[k])
        a_tap = dt / p.C_top * p.lambda_water * v_tap_m3_per_h
        b_tap = dt / p.C_bot * p.lambda_water * v_tap_m3_per_h

        A = np.array(
            [
                [1.0 - a_strat - a_loss - a_tap, a_strat],
                [b_strat, 1.0 - b_strat - b_loss],
            ],
            dtype=float,
        )
        B = np.array([[0.0], [dt / p.C_bot]], dtype=float)
        E = np.array(
            [
                [a_loss, 0.0],
                [b_loss, b_tap],
            ],
            dtype=float,
        )
        return A, B, E

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def continuous_derivative(
        self,
        state: np.ndarray,
        control_kw: float,
        v_tap_m3_per_h: float,
        t_mains_c: float,
        t_amb_c: float,
    ) -> np.ndarray:
        """Evaluate dx/dt = f(x, u, d) from the continuous physics (§9.4)."""
        x = np.asarray(state, dtype=float)
        if x.shape != (2,):
            raise ValueError("state must be [T_top, T_bot].")
        T_top, T_bot = x
        p = self.parameters

        q_strat = (T_top - T_bot) / p.R_strat
        q_loss_top = (T_top - t_amb_c) / p.R_loss
        q_loss_bot = (T_bot - t_amb_c) / p.R_loss
        tap_top = p.lambda_water * v_tap_m3_per_h * T_top  # heat leaving top
        tap_bot = p.lambda_water * v_tap_m3_per_h * t_mains_c  # cold water entering bot

        dT_top = (-q_strat - tap_top - q_loss_top) / p.C_top
        dT_bot = (q_strat + control_kw + tap_bot - q_loss_bot) / p.C_bot

        return np.array([dT_top, dT_bot], dtype=float)

    def step(
        self,
        state: np.ndarray,
        control_kw: float,
        v_tap_m3_per_h: float,
        t_mains_c: float,
        t_amb_c: float,
    ) -> np.ndarray:
        """Forward-Euler step using the matrix form: x[k+1] = A[k] x[k] + B u[k] + E[k] d[k]."""
        x = np.asarray(state, dtype=float)
        if x.shape != (2,):
            raise ValueError("state must be [T_top, T_bot].")
        d = np.array([t_amb_c, t_mains_c], dtype=float)
        A, B, E = self.state_matrices(v_tap_m3_per_h)
        return A @ x + B[:, 0] * control_kw + E @ d

    # ------------------------------------------------------------------
    # Derived quantity (assumption A6 — not an independent state)
    # ------------------------------------------------------------------

    def t_dhw_mean(self, state: np.ndarray) -> float:
        """Weighted mean tank temperature [°C] (assumption A6, §8.1).

        T_dhw = (C_top · T_top + C_bot · T_bot) / (C_top + C_bot)

        This is a *derived* signal, never an independent state variable.
        """
        p = self.parameters
        x = np.asarray(state, dtype=float)
        return float((p.C_top * x[0] + p.C_bot * x[1]) / (p.C_top + p.C_bot))

    # ------------------------------------------------------------------
    # Observability (§11)
    # ------------------------------------------------------------------

    def observability_matrix(self, v_tap_m3_per_h: float = 0.0) -> np.ndarray:
        """O = [C_obs; C_obs·A_dhw[k]]  (2×2)."""
        A, _, _ = self.state_matrices(v_tap_m3_per_h)
        return np.vstack([MEASUREMENT_MATRIX_DHW, MEASUREMENT_MATRIX_DHW @ A])

    def observability_rank(self, v_tap_m3_per_h: float = 0.0) -> int:
        """rank(O) must equal 2 for full observability (requires a_strat ≠ 0)."""
        return int(np.linalg.matrix_rank(self.observability_matrix(v_tap_m3_per_h)))
