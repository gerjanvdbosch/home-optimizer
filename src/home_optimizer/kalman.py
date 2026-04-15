"""Generic discrete-time Kalman filtering for the Home Optimizer thermal models.

This module intentionally centralises the shared Kalman mathematics for the UFH
and DHW subsystems to satisfy the DRY architecture requirement from the project
specification.  The generic :class:`LinearKalmanFilter` implements the common
linear-algebra core:

Predict:
  x̂⁻[k] = A[k-1] x̂[k-1] + B[k-1] u[k-1] + E[k-1] d[k-1]
  P⁻[k]  = A[k-1] P[k-1] A[k-1]ᵀ + Q_n

Update (Joseph form for numerical stability):
  K[k]   = P⁻[k] Cᵀ (C P⁻[k] Cᵀ + R_n)⁻¹
  x̂[k]  = x̂⁻[k] + K[k] (y[k] − C x̂⁻[k])
  P[k]   = (I − K[k] C) P⁻[k] (I − K[k] C)ᵀ + K[k] R_n K[k]ᵀ

UFH and DHW remain available as thin domain wrappers that only translate their
physical inputs to the generic state-space form (A, B, E, d, C).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .dhw_model import MEASUREMENT_MATRIX_DHW, DHWModel
from .thermal_model import MEASUREMENT_MATRIX, ThermalModel
from .types import KalmanNoiseParameters

STATE_DIMENSION: Final[int] = 2
SCALAR_MEASUREMENT_DIMENSION: Final[int] = 1


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


class LinearKalmanFilter:
    """Generic discrete-time linear Kalman filter for 2-state thermal subsystems.

    The filter is intentionally agnostic to subsystem physics: callers supply the
    discrete state-space matrices and the disturbance vector for each predict step.
    This class implements the shared Joseph-form covariance update from §6 and §12
    of the project specification.

    Parameters
    ----------
    measurement_matrix:
        Observation matrix C with shape (1, 2). Unitless mapping from state [°C]
        to the measured scalar temperature [°C].
    noise:
        Kalman process noise covariance Q_n [K²] and measurement variance R_n [K²].
    initial_state_c:
        Initial posterior state estimate [°C] with shape (2,).
    initial_covariance:
        Initial posterior covariance matrix [K²] with shape (2, 2).
    state_labels:
        Human-readable state names used only in fail-fast validation messages.
    """

    def __init__(
        self,
        measurement_matrix: np.ndarray,
        noise: KalmanNoiseParameters,
        initial_state_c: np.ndarray,
        initial_covariance: np.ndarray,
        state_labels: tuple[str, str],
    ) -> None:
        self.noise = noise
        self.state_labels = state_labels
        self._C = np.asarray(measurement_matrix, dtype=float).copy()
        self._x = np.asarray(initial_state_c, dtype=float).copy()
        self._P = np.asarray(initial_covariance, dtype=float).copy()

        expected_state = f"[{state_labels[0]}, {state_labels[1]}]"
        if self._C.shape != (SCALAR_MEASUREMENT_DIMENSION, STATE_DIMENSION):
            raise ValueError("measurement_matrix must have shape (1, 2) for a scalar temperature sensor.")
        if self._x.shape != (STATE_DIMENSION,):
            raise ValueError(f"initial_state_c must be {expected_state}.")
        if self._P.shape != (STATE_DIMENSION, STATE_DIMENSION):
            raise ValueError("initial_covariance must be 2×2.")
        if not np.allclose(self._P, self._P.T):
            raise ValueError("initial_covariance must be symmetric.")

    @property
    def estimate(self) -> KalmanEstimate:
        """Return the current posterior state estimate.

        Returns
        -------
        KalmanEstimate
            Snapshot of the state mean [°C] and covariance [K²]. Copies are
            returned to prevent accidental external mutation of filter state.
        """
        return KalmanEstimate(self._x.copy(), self._P.copy())

    def _symmetrise_covariance(self) -> None:
        """Project covariance onto the symmetric matrix subspace.

        Floating-point round-off can introduce tiny anti-symmetric terms after
        repeated matrix multiplications. Symmetrising preserves the intended
        physical meaning of covariance without altering the model equations.
        """
        self._P = 0.5 * (self._P + self._P.T)

    @staticmethod
    def _validate_state_space_matrices(
        A: np.ndarray,
        B: np.ndarray,
        E: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate and normalise state-space matrices for a 2-state subsystem.

        Parameters
        ----------
        A, B, E:
            Discrete-time state-space matrices for x[k+1] = A x[k] + B u[k] + E d[k].

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Float arrays with validated dimensions.
        """
        A_arr = np.asarray(A, dtype=float)
        B_arr = np.asarray(B, dtype=float)
        E_arr = np.asarray(E, dtype=float)

        if A_arr.shape != (STATE_DIMENSION, STATE_DIMENSION):
            raise ValueError("A must be a 2×2 matrix.")
        if B_arr.shape != (STATE_DIMENSION, SCALAR_MEASUREMENT_DIMENSION):
            raise ValueError("B must be a 2×1 matrix for a scalar thermal power input.")
        if E_arr.shape[0] != STATE_DIMENSION:
            raise ValueError("E must have 2 rows to match the state dimension.")
        return A_arr, B_arr, E_arr

    def predict_from_matrices(
        self,
        control_kw: float,
        disturbance: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        E: np.ndarray,
    ) -> KalmanEstimate:
        """Run the generic predict step for supplied state-space matrices.

        Parameters
        ----------
        control_kw:
            Applied thermal power u[k-1] [kW].
        disturbance:
            Disturbance vector d[k-1]. The physical meaning and units depend on the
            subsystem wrapper, but its length must match the number of columns in E.
        A, B, E:
            Discrete-time state-space matrices evaluated for the previous time step.

        Returns
        -------
        KalmanEstimate
            Predicted prior estimate after the time update.
        """
        A_arr, B_arr, E_arr = self._validate_state_space_matrices(A=A, B=B, E=E)
        d = np.asarray(disturbance, dtype=float)
        expected_disturbance_shape = (E_arr.shape[1],)
        if d.shape != expected_disturbance_shape:
            raise ValueError(
                "disturbance must have shape "
                f"{expected_disturbance_shape} to match the supplied E matrix."
            )

        self._x = A_arr @ self._x + B_arr[:, 0] * control_kw + E_arr @ d
        self._P = A_arr @ self._P @ A_arr.T + self.noise.process_covariance
        self._symmetrise_covariance()
        return self.estimate

    def update(self, measurement_c: float) -> tuple[KalmanEstimate, float, np.ndarray]:
        """Incorporate a scalar temperature measurement using Joseph form.

        Parameters
        ----------
        measurement_c:
            Measured scalar temperature y[k] [°C].

        Returns
        -------
        tuple[KalmanEstimate, float, np.ndarray]
            Updated estimate, scalar innovation y − Cx̂⁻ [K], and the 2×1 Kalman
            gain vector.
        """
        R_n = np.array([[self.noise.measurement_variance]], dtype=float)
        y = np.array([measurement_c], dtype=float)
        innovation = y - self._C @ self._x  # scalar temperature residual [K]
        S = self._C @ self._P @ self._C.T + R_n
        innovation_variance = float(S[0, 0])
        if innovation_variance <= 0.0:
            raise ValueError("Innovation variance must remain strictly positive.")
        K = (self._P @ self._C.T) / innovation_variance

        self._x = self._x + K[:, 0] * float(innovation[0])

        # Implements the Joseph-form posterior covariance update from §6 / §12.
        I_KC = np.eye(STATE_DIMENSION) - K @ self._C
        self._P = I_KC @ self._P @ I_KC.T + K @ R_n @ K.T
        self._symmetrise_covariance()

        return self.estimate, float(innovation[0]), K

    def step_from_matrices(
        self,
        control_kw: float,
        disturbance: np.ndarray,
        measurement_c: float,
        A: np.ndarray,
        B: np.ndarray,
        E: np.ndarray,
    ) -> tuple[KalmanEstimate, float, np.ndarray]:
        """Run a complete predict-then-update cycle for supplied matrices.

        Parameters
        ----------
        control_kw:
            Applied thermal power u[k-1] [kW].
        disturbance:
            Disturbance vector d[k-1].
        measurement_c:
            Scalar measured temperature y[k] [°C].
        A, B, E:
            Discrete-time state-space matrices for the previous step.
        """
        self.predict_from_matrices(
            control_kw=control_kw,
            disturbance=disturbance,
            A=A,
            B=B,
            E=E,
        )
        return self.update(measurement_c=measurement_c)


class UFHKalmanFilter:
    """UFH-specific wrapper around the generic 2-state Kalman core.

    This wrapper preserves UFH-specific terminology (`T_r`, `T_b`, and the
    disturbance vector `[T_out, Q_solar, Q_int]`) while delegating all Kalman
    algebra to :class:`LinearKalmanFilter`.
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
        self._filter = LinearKalmanFilter(
            measurement_matrix=MEASUREMENT_MATRIX,
            noise=noise,
            initial_state_c=initial_state_c,
            initial_covariance=initial_covariance,
            state_labels=("T_r", "T_b"),
        )

    @property
    def estimate(self) -> KalmanEstimate:
        """Current (post-update) UFH state estimate [T_r, T_b] [°C]."""
        return self._filter.estimate

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
        return self._filter.predict_from_matrices(
            control_kw=control_kw,
            disturbance=d,
            A=A,
            B=B,
            E=E,
        )

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
        return self._filter.update(measurement_c=room_temp_measurement_c)

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
        d = np.asarray(disturbance, dtype=float)
        if d.shape != (3,):
            raise ValueError("disturbance must be [T_out, Q_solar, Q_int].")
        A, B, E = self.model.state_matrices()
        return self._filter.step_from_matrices(
            control_kw=control_kw,
            disturbance=d,
            measurement_c=room_temp_measurement_c,
            A=A,
            B=B,
            E=E,
        )


# ---------------------------------------------------------------------------
# DHW Kalman Filter  (§12 of spec)
# ---------------------------------------------------------------------------


class DHWKalmanFilter:
    """DHW-specific wrapper around the generic 2-state Kalman core.

    The state-transition matrix A_dhw[k] is **time-varying** — the current tap-flow
    V_tap[k] must be supplied at every predict step (§12 note).

    Only T_top is measured (C_obs = [1, 0]).  T_bot is a hidden state estimated
    from the T_top trajectory.

    Only the DHW-specific physics remain here: the time-varying matrix lookup and
    the disturbance-vector construction d = [T_amb, T_mains].
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
        self._filter = LinearKalmanFilter(
            measurement_matrix=MEASUREMENT_MATRIX_DHW,
            noise=noise,
            initial_state_c=initial_state_c,
            initial_covariance=initial_covariance,
            state_labels=("T_top", "T_bot"),
        )

    @property
    def estimate(self) -> KalmanEstimate:
        """Current (post-update) DHW state estimate [T_top, T_bot] [°C]."""
        return self._filter.estimate

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
        return self._filter.predict_from_matrices(
            control_kw=control_kw,
            disturbance=d,
            A=A,
            B=B,
            E=E,
        )

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
        return self._filter.update(measurement_c=t_top_measurement_c)

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
        d = np.array([t_amb_c, t_mains_c], dtype=float)
        A, B, E = self.model.state_matrices(v_tap_m3_per_h)
        return self._filter.step_from_matrices(
            control_kw=control_kw,
            disturbance=d,
            measurement_c=t_top_measurement_c,
            A=A,
            B=B,
            E=E,
        )

