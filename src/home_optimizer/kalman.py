"""Linear Kalman filter for estimating T_r (room) and T_b (floor) temperatures.

Only the room temperature T_r is measured (C = [1, 0]).
The floor temperature T_b is a hidden state estimated from past observations.

Predict:
  x̂⁻[k] = A x̂[k-1] + B u[k-1] + E d[k-1]
  P⁻[k]  = A P[k-1] Aᵀ + Q_n

Update (measurement y[k] = T_r[k]):
  K[k]   = P⁻[k] Cᵀ (C P⁻[k] Cᵀ + R_n)⁻¹
  x̂[k]   = x̂⁻[k] + K[k] (y[k] − C x̂⁻[k])
  P[k]   = (I − K[k] C) P⁻[k] (I − K[k] C)ᵀ + K[k] R_n K[k]ᵀ  (Joseph form)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dhw_model import MEASUREMENT_MATRIX_DHW, DHWModel
from .thermal_model import MEASUREMENT_MATRIX, ThermalModel
from .types import KalmanNoiseParameters


@dataclass(frozen=True, slots=True)
class KalmanEstimate:
    """Snapshot of the Kalman state estimate.

    Attributes
    ----------
    mean_c:       State estimate [T_r, T_b] in °C.
    covariance:   2×2 error covariance matrix [K²].
    """

    mean_c: np.ndarray
    covariance: np.ndarray


class UFHKalmanFilter:
    """Discrete-time linear Kalman filter for the 2-state UFH thermal model.

    Parameters
    ----------
    model:                Discrete thermal model (provides A, B, E).
    noise:                Process noise Q_n and measurement noise R_n.
    initial_state_c:      Initial state estimate [T_r, T_b] [°C].
    initial_covariance:   Initial 2×2 error covariance [K²].
    """

    def __init__(
        self,
        model: ThermalModel,
        noise: KalmanNoiseParameters,
        initial_state_c: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        self.model = model
        self.noise = noise
        self._x = np.asarray(initial_state_c, dtype=float).copy()
        self._P = np.asarray(initial_covariance, dtype=float).copy()

        if self._x.shape != (2,):
            raise ValueError("initial_state_c must be [T_r, T_b].")
        if self._P.shape != (2, 2):
            raise ValueError("initial_covariance must be 2×2.")
        if not np.allclose(self._P, self._P.T):
            raise ValueError("initial_covariance must be symmetric.")

    @property
    def estimate(self) -> KalmanEstimate:
        """Current (post-update) state estimate."""
        return KalmanEstimate(self._x.copy(), self._P.copy())

    # ------------------------------------------------------------------
    # Predict step
    # ------------------------------------------------------------------

    def predict(self, control_kw: float, disturbance: np.ndarray) -> KalmanEstimate:
        """Propagate the state estimate one time step forward.

        Parameters
        ----------
        control_kw:   UFH power applied at the previous step [kW].
        disturbance:  [T_out, Q_solar, Q_int] at the previous step.
        """
        d = np.asarray(disturbance, dtype=float)
        if d.shape != (3,):
            raise ValueError("disturbance must be [T_out, Q_solar, Q_int].")

        A, B, E = self.model.state_matrices()
        self._x = A @ self._x + B[:, 0] * control_kw + E @ d
        self._P = A @ self._P @ A.T + self.noise.process_covariance
        # Enforce symmetry to prevent numerical drift
        self._P = 0.5 * (self._P + self._P.T)
        return self.estimate

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def update(self, room_temp_measurement_c: float) -> tuple[KalmanEstimate, float, np.ndarray]:
        """Incorporate a new room-temperature measurement.

        Returns
        -------
        estimate:    Updated state estimate.
        innovation:  Scalar measurement residual y − C x̂⁻  [K].
        gain:        2×1 Kalman gain vector.
        """
        C = MEASUREMENT_MATRIX  # shape (1, 2)
        R_n = np.array([[self.noise.measurement_variance]], dtype=float)

        y = np.array([room_temp_measurement_c], dtype=float)
        innovation = y - C @ self._x  # shape (1,)
        S = C @ self._P @ C.T + R_n  # shape (1, 1)
        K = self._P @ C.T @ np.linalg.inv(S)  # shape (2, 1)

        self._x = self._x + K[:, 0] * float(innovation[0])

        # Joseph form for numerical stability: P = (I-KC)P(I-KC)ᵀ + K R_n Kᵀ
        I_KC = np.eye(2) - K @ C
        self._P = I_KC @ self._P @ I_KC.T + K @ R_n @ K.T
        self._P = 0.5 * (self._P + self._P.T)

        return self.estimate, float(innovation[0]), K

    # ------------------------------------------------------------------
    # Combined predict + update
    # ------------------------------------------------------------------

    def step(
        self,
        control_kw: float,
        disturbance: np.ndarray,
        room_temp_measurement_c: float,
    ) -> tuple[KalmanEstimate, float, np.ndarray]:
        """Predict then update in a single call."""
        self.predict(control_kw=control_kw, disturbance=disturbance)
        return self.update(room_temp_measurement_c=room_temp_measurement_c)


# ---------------------------------------------------------------------------
# DHW Kalman Filter  (§12 of spec)
# ---------------------------------------------------------------------------


class DHWKalmanFilter:
    """Discrete-time linear Kalman filter for the 2-node DHW stratification model.

    The state-transition matrix A_dhw[k] is **time-varying** — the current tap-flow
    V_tap[k] must be supplied at every predict step (§12 note).

    Only T_top is measured (C_obs = [1, 0]).  T_bot is a hidden state estimated
    from the T_top trajectory.

    Predict (uses A_dhw[k-1] evaluated at the previous V_tap):
      x̂⁻[k] = A_dhw[k-1] x̂[k-1] + B_dhw u[k-1] + E_dhw[k-1] d[k-1]
      P⁻[k]  = A_dhw[k-1] P[k-1] A_dhw[k-1]ᵀ + Q_n

    Update (Joseph form for numerical stability):
      K[k]   = P⁻[k] C_obsᵀ (C_obs P⁻[k] C_obsᵀ + R_n)⁻¹
      x̂[k]   = x̂⁻[k] + K[k] (y[k] − C_obs x̂⁻[k])
      P[k]   = (I − K C_obs) P⁻[k] (I − K C_obs)ᵀ + K R_n Kᵀ

    Parameters
    ----------
    model:               DHW stratification model (provides state_matrices).
    noise:               Process noise Q_n (2×2) and measurement noise R_n (scalar).
    initial_state_c:     Initial state estimate [T_top, T_bot] [°C].
    initial_covariance:  Initial 2×2 error covariance [K²].
    """

    def __init__(
        self,
        model: DHWModel,
        noise: KalmanNoiseParameters,
        initial_state_c: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        self.model = model
        self.noise = noise
        self._x = np.asarray(initial_state_c, dtype=float).copy()
        self._P = np.asarray(initial_covariance, dtype=float).copy()

        if self._x.shape != (2,):
            raise ValueError("initial_state_c must be [T_top, T_bot].")
        if self._P.shape != (2, 2):
            raise ValueError("initial_covariance must be 2×2.")
        if not np.allclose(self._P, self._P.T):
            raise ValueError("initial_covariance must be symmetric.")

    @property
    def estimate(self) -> KalmanEstimate:
        """Current (post-update) state estimate."""
        return KalmanEstimate(self._x.copy(), self._P.copy())

    # ------------------------------------------------------------------
    # Predict step  (time-varying A_dhw[k-1])
    # ------------------------------------------------------------------

    def predict(
        self,
        control_kw: float,
        v_tap_m3_per_h: float,
        t_mains_c: float,
        t_amb_c: float,
    ) -> KalmanEstimate:
        """Propagate the state estimate one time step forward.

        Parameters
        ----------
        control_kw:       DHW thermal power applied at the previous step [kW].
        v_tap_m3_per_h:   Tap-water flow rate at the *previous* step [m³/h].
        t_mains_c:        Mains water temperature at the previous step [°C].
        t_amb_c:          Ambient temperature at the previous step [°C].
        """
        d = np.array([t_amb_c, t_mains_c], dtype=float)
        # Use A_dhw[k-1]: time-varying matrix evaluated at the previous V_tap
        A, B, E = self.model.state_matrices(v_tap_m3_per_h)
        self._x = A @ self._x + B[:, 0] * control_kw + E @ d
        self._P = A @ self._P @ A.T + self.noise.process_covariance
        # Enforce symmetry to prevent numerical drift
        self._P = 0.5 * (self._P + self._P.T)
        return self.estimate

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def update(self, t_top_measurement_c: float) -> tuple[KalmanEstimate, float, np.ndarray]:
        """Incorporate a new top-layer temperature measurement.

        Returns
        -------
        estimate:    Updated state estimate [T_top, T_bot].
        innovation:  Scalar measurement residual y − C_obs x̂⁻  [K].
        gain:        2×1 Kalman gain vector.
        """
        C = MEASUREMENT_MATRIX_DHW  # shape (1, 2)
        R_n = np.array([[self.noise.measurement_variance]], dtype=float)

        y = np.array([t_top_measurement_c], dtype=float)
        innovation = y - C @ self._x  # shape (1,)
        S = C @ self._P @ C.T + R_n  # shape (1, 1)
        K = self._P @ C.T @ np.linalg.inv(S)  # shape (2, 1)

        self._x = self._x + K[:, 0] * float(innovation[0])

        # Joseph form for numerical stability: P = (I-KC)P(I-KC)ᵀ + K R_n Kᵀ
        I_KC = np.eye(2) - K @ C
        self._P = I_KC @ self._P @ I_KC.T + K @ R_n @ K.T
        self._P = 0.5 * (self._P + self._P.T)

        return self.estimate, float(innovation[0]), K

    # ------------------------------------------------------------------
    # Combined predict + update
    # ------------------------------------------------------------------

    def step(
        self,
        control_kw: float,
        v_tap_m3_per_h: float,
        t_mains_c: float,
        t_amb_c: float,
        t_top_measurement_c: float,
    ) -> tuple[KalmanEstimate, float, np.ndarray]:
        """Predict then update in a single call."""
        self.predict(
            control_kw=control_kw,
            v_tap_m3_per_h=v_tap_m3_per_h,
            t_mains_c=t_mains_c,
            t_amb_c=t_amb_c,
        )
        return self.update(t_top_measurement_c=t_top_measurement_c)

