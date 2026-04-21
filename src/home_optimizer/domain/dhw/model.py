"""2-state discrete thermal model for a DHW stratification tank (§7–§11 of spec).

State vector    x_dhw = [T_top, T_bot]ᵀ
Control input   u_dhw = P_dhw  [kW]   (heat-pump output to bottom layer — assumption A5)
Disturbances    d_dhw = [T_amb, T_mains]ᵀ

The state-transition matrix A_dhw[k] is **time-varying** because it depends on the
tap-water flow rate V_tap[k] at each time step (bilinear term, linearised as LTV).

Physical equations (§9):
  C_top · dT_top/dt = −(T_top−T_bot)/R_strat − λ·V_tap·T_top − (T_top−T_amb)/R_loss
  C_bot · dT_bot/dt = +(T_top−T_bot)/R_strat + P_dhw + λ·V_tap·T_mains − (T_bot−T_amb)/R_loss

Discrete state-space (exact ZOH, step Δt [h], §10–§11):
──────────────────────────────────────────────────────────

  Auxiliary scalars (constant):
    a_strat = Δt / (C_top · R_strat)
    b_strat = Δt / (C_bot · R_strat)
    a_loss  = Δt / (C_top · R_loss)
    b_loss  = Δt / (C_bot · R_loss)

  Time-varying scalars (depend on V_tap[k]):
    a_tap[k] = Δt / C_top · λ · V_tap[k]
    b_tap[k] = Δt / C_bot · λ · V_tap[k]

  The continuous-time affine system is first assembled from the physics and then
  discretised exactly under a zero-order-hold assumption with constant
  ``P_dhw``, ``T_amb`` and ``T_mains`` over the interval. This follows the
  project requirement from §10.2 to prefer ZOH whenever DHW dynamics are too fast
  for forward Euler at the runtime MPC step.

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
from scipy.linalg import expm

from ...types.constants import ABSOLUTE_ZERO_C
from ...types.physical import DHWParameters

#: Observation matrix C_obs = [1, 0] — only T_top is measured.
MEASUREMENT_MATRIX_DHW: np.ndarray = np.array([[1.0, 0.0]], dtype=float)


def _assert_temperature_above_absolute_zero(*, name: str, temperature_c: float) -> None:
    """Raise when a temperature falls below absolute zero [°C]."""
    if temperature_c < ABSOLUTE_ZERO_C:
        raise ValueError(
            f"{name}={temperature_c:.3f} °C is below absolute zero ({ABSOLUTE_ZERO_C:.2f} °C)."
        )


@dataclass(slots=True)
class DHWModel:
    """Discrete grey-box 2-node stratification model for a DHW tank.

    The runtime MPC and Kalman paths use exact zero-order-hold discretisation of
    the linearised DHW subsystem. This avoids the conditional-stability problems of
    forward Euler for large MPC steps while preserving the same public state-space
    interface ``(A, B, E)``.
    """

    parameters: DHWParameters

    # ------------------------------------------------------------------
    # Constant auxiliary scalars
    # ------------------------------------------------------------------

    def _continuous_matrices(
        self,
        v_tap_m3_per_h: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return continuous-time ``(F, G_u, G_d)`` matrices for the DHW physics.

        Args:
            v_tap_m3_per_h: Piecewise-constant tap-flow rate over the interval [m³/h].

        Returns:
            Tuple with continuous-time state matrix ``F`` [1/h], input matrix
            ``G_u`` [K/(kWh)], and disturbance matrix ``G_d`` [1/h].
        """
        p = self.parameters
        strat_top_per_h = 1.0 / (p.C_top * p.R_strat)
        strat_bot_per_h = 1.0 / (p.C_bot * p.R_strat)
        loss_top_per_h = 1.0 / (p.C_top * p.R_loss)
        loss_bot_per_h = 1.0 / (p.C_bot * p.R_loss)
        tap_top_per_h = p.lambda_water * v_tap_m3_per_h / p.C_top
        tap_bot_per_h = p.lambda_water * v_tap_m3_per_h / p.C_bot

        F = np.array(
            [
                [-(strat_top_per_h + loss_top_per_h + tap_top_per_h), strat_top_per_h],
                [strat_bot_per_h, -(strat_bot_per_h + loss_bot_per_h)],
            ],
            dtype=float,
        )
        G_u = np.array([[0.0], [1.0 / p.C_bot]], dtype=float)
        G_d = np.array(
            [
                [loss_top_per_h, 0.0],
                [loss_bot_per_h, tap_bot_per_h],
            ],
            dtype=float,
        )
        return F, G_u, G_d

    # ------------------------------------------------------------------
    # Time-varying state-space matrices
    # ------------------------------------------------------------------

    def state_matrices(
        self,
        v_tap_m3_per_h: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact discrete ``(A_dhw, B_dhw, E_dhw)`` for a given tap flow.

        A_dhw and E_dhw are time-varying (depend on ``V_tap[k]``). ``B_dhw`` is
        still the bottom-layer actuation channel from assumption A5, but all three
        matrices are obtained from one exact zero-order-hold discretisation of the
        continuous affine DHW model over ``dt_hours``.
        """
        if v_tap_m3_per_h < 0.0:
            raise ValueError("v_tap_m3_per_h must be non-negative.")
        p = self.parameters
        F, G_u, G_d = self._continuous_matrices(v_tap_m3_per_h)
        augmented = np.zeros((5, 5), dtype=float)
        augmented[:2, :2] = F
        augmented[:2, 2:3] = G_u
        augmented[:2, 3:] = G_d
        discretised = expm(augmented * p.dt_hours)
        A = discretised[:2, :2]
        B = discretised[:2, 2:3]
        E = discretised[:2, 3:]
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
        if v_tap_m3_per_h < 0.0:
            raise ValueError("v_tap_m3_per_h must be non-negative.")
        T_top, T_bot = x
        _assert_temperature_above_absolute_zero(name="T_top", temperature_c=float(T_top))
        _assert_temperature_above_absolute_zero(name="T_bot", temperature_c=float(T_bot))
        _assert_temperature_above_absolute_zero(name="t_mains_c", temperature_c=float(t_mains_c))
        _assert_temperature_above_absolute_zero(name="t_amb_c", temperature_c=float(t_amb_c))
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
        """Exact ZOH step using ``x[k+1] = A[k] x[k] + B u[k] + E[k] d[k]``."""
        x = np.asarray(state, dtype=float)
        if x.shape != (2,):
            raise ValueError("state must be [T_top, T_bot].")
        _assert_temperature_above_absolute_zero(name="T_top", temperature_c=float(x[0]))
        _assert_temperature_above_absolute_zero(name="T_bot", temperature_c=float(x[1]))
        _assert_temperature_above_absolute_zero(name="t_mains_c", temperature_c=float(t_mains_c))
        _assert_temperature_above_absolute_zero(name="t_amb_c", temperature_c=float(t_amb_c))
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
