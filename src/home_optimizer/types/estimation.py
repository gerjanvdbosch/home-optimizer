"""Noise parameter dataclasses for linear and extended Kalman filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class KalmanNoiseParameters:
    """Noise covariance parameters for a 2-state Kalman filter."""

    process_covariance: np.ndarray
    measurement_variance: float

    def __post_init__(self) -> None:
        q = np.asarray(self.process_covariance, dtype=float)
        if q.shape == (2,):
            q = np.diag(q)
        if q.shape != (2, 2):
            raise ValueError("process_covariance must be 2×2 or a length-2 diagonal.")
        if not np.allclose(q, q.T):
            raise ValueError("process_covariance must be symmetric.")
        if np.min(np.linalg.eigvalsh(q)) < -1e-10:
            raise ValueError("process_covariance must be positive semi-definite.")
        if self.measurement_variance <= 0.0:
            raise ValueError("measurement_variance must be strictly positive.")
        object.__setattr__(self, "process_covariance", q)


@dataclass(frozen=True, slots=True)
class EKFNoiseParameters:
    """Noise covariance parameters for the DHW Extended Kalman Filter."""

    process_cov_temperatures: np.ndarray
    process_var_vtap: float
    measurement_var_t_top: float
    measurement_var_t_bot: float

    def __post_init__(self) -> None:
        q = np.asarray(self.process_cov_temperatures, dtype=float)
        if q.shape == (2,):
            q = np.diag(q)
        if q.shape != (2, 2):
            raise ValueError("process_cov_temperatures must be 2×2 or a length-2 diagonal.")
        if not np.allclose(q, q.T):
            raise ValueError("process_cov_temperatures must be symmetric.")
        if np.min(np.linalg.eigvalsh(q)) < -1e-10:
            raise ValueError("process_cov_temperatures must be positive semi-definite.")
        if self.process_var_vtap <= 0.0:
            raise ValueError("process_var_vtap must be strictly positive.")
        if self.measurement_var_t_top <= 0.0:
            raise ValueError("measurement_var_t_top must be strictly positive.")
        if self.measurement_var_t_bot <= 0.0:
            raise ValueError("measurement_var_t_bot must be strictly positive.")
        object.__setattr__(self, "process_cov_temperatures", q)

    @property
    def Q_aug(self) -> np.ndarray:
        """3×3 block-diagonal process noise covariance Q_aug = diag(Q_n_dhw, Q_n_Vtap)."""
        q = np.zeros((3, 3), dtype=float)
        q[:2, :2] = self.process_cov_temperatures
        q[2, 2] = self.process_var_vtap
        return q

    @property
    def R_n(self) -> np.ndarray:
        """2×2 diagonal measurement noise covariance diag(σ²_T_top, σ²_T_bot) [K²]."""
        return np.diag(np.array([self.measurement_var_t_top, self.measurement_var_t_bot], dtype=float))


__all__ = ["EKFNoiseParameters", "KalmanNoiseParameters"]

