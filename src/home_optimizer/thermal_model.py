from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import ThermalParameters

MEASUREMENT_MATRIX = np.array([[1.0, 0.0]], dtype=float)


def solar_gain_kw(
    gti_w_per_m2: float | np.ndarray,
    glass_area_m2: float,
    transmittance: float,
) -> float | np.ndarray:
    if glass_area_m2 <= 0.0:
        raise ValueError("glass_area_m2 must be strictly positive.")
    if not 0.0 <= transmittance <= 1.0:
        raise ValueError("transmittance must lie in the closed interval [0, 1].")

    gti_array = np.asarray(gti_w_per_m2, dtype=float)
    if np.any(gti_array < 0.0):
        raise ValueError("gti_w_per_m2 cannot be negative.")

    gains_kw = glass_area_m2 * gti_array * transmittance / 1000.0
    if np.isscalar(gti_w_per_m2):
        return float(gains_kw)
    return gains_kw


@dataclass(slots=True)
class ThermalModel:
    parameters: ThermalParameters

    def __post_init__(self) -> None:
        self.parameters.assert_euler_stable()

    def continuous_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.parameters
        A_c = np.array(
            [
                [-(1.0 / (p.C_r * p.R_br) + 1.0 / (p.C_r * p.R_ro)), 1.0 / (p.C_r * p.R_br)],
                [1.0 / (p.C_b * p.R_br), -1.0 / (p.C_b * p.R_br)],
            ],
            dtype=float,
        )
        B_c = np.array([[0.0], [1.0 / p.C_b]], dtype=float)
        E_c = np.array(
            [
                [1.0 / (p.C_r * p.R_ro), p.alpha / p.C_r, 1.0 / p.C_r],
                [0.0, (1.0 - p.alpha) / p.C_b, 0.0],
            ],
            dtype=float,
        )
        return A_c, B_c, E_c

    def state_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.parameters
        a_br = p.dt_hours / (p.C_r * p.R_br)
        a_ro = p.dt_hours / (p.C_r * p.R_ro)
        b_br = p.dt_hours / (p.C_b * p.R_br)

        A = np.array(
            [[1.0 - a_br - a_ro, a_br], [b_br, 1.0 - b_br]],
            dtype=float,
        )
        B = np.array([[0.0], [p.dt_hours / p.C_b]], dtype=float)
        E = np.array(
            [
                [a_ro, p.alpha * p.dt_hours / p.C_r, p.dt_hours / p.C_r],
                [0.0, (1.0 - p.alpha) * p.dt_hours / p.C_b, 0.0],
            ],
            dtype=float,
        )
        return A, B, E

    def continuous_derivative(
        self,
        state_c: np.ndarray,
        control_kw: float,
        outdoor_temperature_c: float,
        solar_gain_kw_input: float,
        internal_gain_kw: float,
    ) -> np.ndarray:
        state_c = np.asarray(state_c, dtype=float)
        if state_c.shape != (2,):
            raise ValueError("state_c must be a 2-vector ordered as [T_r, T_b].")

        T_r, T_b = state_c
        p = self.parameters

        dT_r_dt = (
            (T_b - T_r) / p.R_br
            - (T_r - outdoor_temperature_c) / p.R_ro
            + p.alpha * solar_gain_kw_input
            + internal_gain_kw
        ) / p.C_r
        dT_b_dt = (
            control_kw - (T_b - T_r) / p.R_br + (1.0 - p.alpha) * solar_gain_kw_input
        ) / p.C_b
        return np.array([dT_r_dt, dT_b_dt], dtype=float)

    def discrete_step(
        self,
        state_c: np.ndarray,
        control_kw: float,
        outdoor_temperature_c: float,
        solar_gain_kw_input: float,
        internal_gain_kw: float,
    ) -> np.ndarray:
        derivative = self.continuous_derivative(
            state_c=state_c,
            control_kw=control_kw,
            outdoor_temperature_c=outdoor_temperature_c,
            solar_gain_kw_input=solar_gain_kw_input,
            internal_gain_kw=internal_gain_kw,
        )
        return np.asarray(state_c, dtype=float) + self.parameters.dt_hours * derivative

    def step_with_disturbance_vector(
        self,
        state_c: np.ndarray,
        control_kw: float,
        disturbance_vector: np.ndarray,
    ) -> np.ndarray:
        disturbance_vector = np.asarray(disturbance_vector, dtype=float)
        if disturbance_vector.shape != (3,):
            raise ValueError(
                "disturbance_vector must be a 3-vector ordered as [T_out, Q_solar, Q_int]."
            )
        A, B, E = self.state_matrices()
        return A @ np.asarray(state_c, dtype=float) + B[:, 0] * control_kw + E @ disturbance_vector

    def observability_matrix(self) -> np.ndarray:
        A, _, _ = self.state_matrices()
        return np.vstack([MEASUREMENT_MATRIX, MEASUREMENT_MATRIX @ A])

    def observability_rank(self) -> int:
        return int(np.linalg.matrix_rank(self.observability_matrix()))

    def controllability_matrix(self) -> np.ndarray:
        A, B, _ = self.state_matrices()
        return np.column_stack([B, A @ B])

    def controllability_rank(self) -> int:
        return int(np.linalg.matrix_rank(self.controllability_matrix()))
