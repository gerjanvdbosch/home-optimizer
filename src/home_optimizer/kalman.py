from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .thermal_model import MEASUREMENT_MATRIX, ThermalModel
from .types import KalmanNoiseParameters


@dataclass(frozen=True, slots=True)
class KalmanEstimate:
    mean_c: np.ndarray
    covariance: np.ndarray


class UFHKalmanFilter:
    def __init__(
        self,
        model: ThermalModel,
        noise: KalmanNoiseParameters,
        initial_state_c: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        self.model = model
        self.noise = noise
        self.state_estimate = np.asarray(initial_state_c, dtype=float).copy()
        self.covariance = np.asarray(initial_covariance, dtype=float).copy()

        if self.state_estimate.shape != (2,):
            raise ValueError("initial_state_c must be a 2-vector ordered as [T_r, T_b].")
        if self.covariance.shape != (2, 2):
            raise ValueError("initial_covariance must be a 2x2 covariance matrix.")
        if not np.allclose(self.covariance, self.covariance.T):
            raise ValueError("initial_covariance must be symmetric.")

    @property
    def estimate(self) -> KalmanEstimate:
        return KalmanEstimate(self.state_estimate.copy(), self.covariance.copy())

    def predict(self, control_kw: float, disturbance_vector: np.ndarray) -> KalmanEstimate:
        disturbance_vector = np.asarray(disturbance_vector, dtype=float)
        if disturbance_vector.shape != (3,):
            raise ValueError(
                "disturbance_vector must be a 3-vector ordered as [T_out, Q_solar, Q_int]."
            )

        A, B, E = self.model.state_matrices()
        self.state_estimate = (
            A @ self.state_estimate + B[:, 0] * control_kw + E @ disturbance_vector
        )
        self.covariance = A @ self.covariance @ A.T + self.noise.process_covariance
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return self.estimate

    def update(
        self, room_temperature_measurement_c: float
    ) -> tuple[KalmanEstimate, float, np.ndarray]:
        C = MEASUREMENT_MATRIX
        measurement = np.array([room_temperature_measurement_c], dtype=float)
        innovation = measurement - C @ self.state_estimate
        innovation_covariance = C @ self.covariance @ C.T + np.array(
            [[self.noise.measurement_variance]], dtype=float
        )
        kalman_gain = self.covariance @ C.T @ np.linalg.inv(innovation_covariance)

        self.state_estimate = self.state_estimate + (kalman_gain @ innovation).reshape(-1)

        identity = np.eye(2, dtype=float)
        measurement_covariance = np.array([[self.noise.measurement_variance]], dtype=float)
        covariance = (identity - kalman_gain @ C) @ self.covariance @ (
            identity - kalman_gain @ C
        ).T + kalman_gain @ measurement_covariance @ kalman_gain.T
        self.covariance = 0.5 * (covariance + covariance.T)
        return self.estimate, float(innovation[0]), kalman_gain

    def step(
        self,
        control_kw: float,
        disturbance_vector: np.ndarray,
        room_temperature_measurement_c: float,
    ) -> tuple[KalmanEstimate, float, np.ndarray]:
        self.predict(control_kw=control_kw, disturbance_vector=disturbance_vector)
        return self.update(room_temperature_measurement_c=room_temperature_measurement_c)
