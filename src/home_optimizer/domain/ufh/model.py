"""2-state discrete thermal model for a house with underfloor heating.

State vector   x = [T_r, T_b]ᵀ
Control input  u = P_UFH  [kW]
Disturbances   d = [T_out, Q_solar, Q_int]ᵀ

Discrete state-space (exact ZOH default, step Δt [h])
──────────────────────────────────────────────────────
x[k+1] = A x[k] + B u[k] + E d[k]

Auxiliary scalars:
  a_br = Δt / (C_r · R_br)
  a_ro = Δt / (C_r · R_ro)
  b_br = Δt / (C_b · R_br)

A = [[1 - a_br - a_ro,  a_br ],
     [b_br,             1 - b_br]]

B = [[0         ],
     [Δt / C_b  ]]

E = [[a_ro,  α·Δt/C_r,      Δt/C_r ],
     [0,     (1-α)·Δt/C_b,  0      ]]

Measurement:  y = C x,  C = [1, 0]  (only T_r is sensed)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..state_space import (
    ContinuousLinearModel,
    DiscreteLinearModel,
    DiscretizationConfig,
    Discretizer,
    controllability_matrix,
    observability_matrix,
)
from ...types.constants import W_PER_KW
from ...types.physical import ThermalParameters

# Measurement matrix C = [1, 0]  (T_r is observable, T_b is hidden)
MEASUREMENT_MATRIX: np.ndarray = np.array([[1.0, 0.0]], dtype=float)


def solar_gain_kw(
    gti_w_per_m2: float | np.ndarray,
    glass_area_m2: float,
    transmittance: float,
) -> float | np.ndarray:
    """Convert irradiance to solar heat gain through glazing.

    Q_solar [kW] = A_glass [m²] · GTI [W/m²] · η / 1000
    """
    if glass_area_m2 <= 0.0:
        raise ValueError("glass_area_m2 must be strictly positive.")
    if not 0.0 <= transmittance <= 1.0:
        raise ValueError("transmittance must be in [0, 1].")
    arr = np.asarray(gti_w_per_m2, dtype=float)
    if np.any(arr < 0.0):
        raise ValueError("gti_w_per_m2 cannot be negative.")
    gain = glass_area_m2 * arr * transmittance / W_PER_KW
    return float(gain) if arr.ndim == 0 else gain


@dataclass(slots=True)
class ThermalModel:
    """Discrete grey-box thermal model of a house with UFH.

    The runtime model derives discrete matrices from the continuous physics
    through the shared discretiser. Per the bindende specification, exact ZOH
    is the default production discretisation. Forward Euler remains available
    only as an explicit reference/fallback path.
    """

    parameters: ThermalParameters

    # ------------------------------------------------------------------
    # Continuous-time system matrices (for reference / ZOH conversion)
    # ------------------------------------------------------------------

    def continuous_model(self) -> ContinuousLinearModel:
        """Return the continuous-time UFH state-space model.

        State vector ``x = [T_r, T_b]`` [°C], control ``u = [P_ufh]`` [kW], and
        disturbances ``d = [T_out, Q_solar, Q_int_eff]``.
        """
        p = self.parameters
        inv_CrRbr = 1.0 / (p.C_r * p.R_br)
        inv_CrRro = 1.0 / (p.C_r * p.R_ro)
        inv_CbRbr = 1.0 / (p.C_b * p.R_br)

        return ContinuousLinearModel(
            A=np.array(
                [
                    [-(inv_CrRbr + inv_CrRro), inv_CrRbr],
                    [inv_CbRbr, -inv_CbRbr],
                ],
                dtype=float,
            ),
            B=np.array([[0.0], [1.0 / p.C_b]], dtype=float),
            E=np.array(
                [
                    [inv_CrRro, p.alpha / p.C_r, 1.0 / p.C_r],
                    [0.0, (1.0 - p.alpha) / p.C_b, 0.0],
                ],
                dtype=float,
            ),
            C=MEASUREMENT_MATRIX,
        )

    def continuous_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(A_c, B_c, E_c)`` of the continuous-time UFH model."""
        model = self.continuous_model()
        return model.A, model.B, model.E

    # ------------------------------------------------------------------
    # Discrete-time state-space matrices
    # ------------------------------------------------------------------

    def discrete_model(self) -> DiscreteLinearModel:
        """Return the default UFH discrete model assembled through exact ZOH."""
        return Discretizer.discretize(
            continuous_model=self.continuous_model(),
            config=DiscretizationConfig(
                method="exact_zoh",
                dt_hours=self.parameters.dt_hours,
            ),
        )

    def forward_euler_discrete_model(self) -> DiscreteLinearModel:
        """Return the explicit forward-Euler UFH discretisation reference."""
        self.parameters.assert_euler_stable()
        return Discretizer.discretize(
            continuous_model=self.continuous_model(),
            config=DiscretizationConfig(
                method="forward_euler",
                dt_hours=self.parameters.dt_hours,
            ),
        )

    def state_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the default exact-ZOH discrete UFH matrices ``(A, B, E)``."""
        discrete_model = self.discrete_model()
        return discrete_model.A, discrete_model.B, discrete_model.E

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def continuous_derivative(
        self,
        state: np.ndarray,
        control_kw: float,
        outdoor_temperature_c: float,
        solar_gain_kw_value: float,
        internal_gain_kw: float,
    ) -> np.ndarray:
        """Evaluate dx/dt = f(x, u, d) from the continuous physics."""
        state = np.asarray(state, dtype=float)
        if state.shape != (2,):
            raise ValueError("state must be [T_r, T_b].")
        T_r, T_b = state
        p = self.parameters

        dT_r = (
            (T_b - T_r) / p.R_br
            - (T_r - outdoor_temperature_c) / p.R_ro
            + p.alpha * solar_gain_kw_value
            + internal_gain_kw
        ) / p.C_r

        dT_b = (control_kw - (T_b - T_r) / p.R_br + (1.0 - p.alpha) * solar_gain_kw_value) / p.C_b

        return np.array([dT_r, dT_b], dtype=float)

    def step(
        self,
        state: np.ndarray,
        control_kw: float,
        outdoor_temperature_c: float,
        solar_gain_kw_value: float,
        internal_gain_kw: float,
    ) -> np.ndarray:
        """Discrete UFH step using the default exact-ZOH matrices."""
        disturbance = np.array(
            [outdoor_temperature_c, solar_gain_kw_value, internal_gain_kw],
            dtype=float,
        )
        return self.step_with_disturbance_vector(state, control_kw, disturbance)

    def step_with_disturbance_vector(
        self,
        state: np.ndarray,
        control_kw: float,
        disturbance: np.ndarray,
    ) -> np.ndarray:
        """Matrix form: x[k+1] = A x[k] + B u[k] + E d[k]."""
        d = np.asarray(disturbance, dtype=float)
        if d.shape != (3,):
            raise ValueError("disturbance must be [T_out, Q_solar, Q_int].")
        A, B, E = self.state_matrices()
        return A @ np.asarray(state, dtype=float) + B[:, 0] * control_kw + E @ d

    # ------------------------------------------------------------------
    # Observability and controllability
    # ------------------------------------------------------------------

    def observability_matrix(self) -> np.ndarray:
        """O = [C; C·A]  (2×2)."""
        return observability_matrix(self.discrete_model())

    def observability_rank(self) -> int:
        return int(np.linalg.matrix_rank(self.observability_matrix()))

    def controllability_matrix(self) -> np.ndarray:
        """Mc = [B, A·B]  (2×2)."""
        return controllability_matrix(self.discrete_model())

    def controllability_rank(self) -> int:
        return int(np.linalg.matrix_rank(self.controllability_matrix()))
