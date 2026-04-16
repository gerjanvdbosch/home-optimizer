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
from .types import EKFNoiseParameters, KalmanNoiseParameters

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
            raise ValueError(
                "measurement_matrix must have shape (1, 2) for a scalar temperature sensor."
            )
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


# ---------------------------------------------------------------------------
# Extended Kalman Filter — DHW augmented state (§12 of spec)
# ---------------------------------------------------------------------------

#: Observation matrix C_obs ∈ ℝ^{2×3} — both T_top and T_bot are measured,
#: V_tap is NOT directly observable.  Implements §12.2.
_C_OBS_AUG: np.ndarray = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=float,
)

#: Augmented state dimension n_aug = 3 (T_top, T_bot, V_tap).
_AUG_STATE_DIM: Final[int] = 3

#: Measurement dimension for the EKF (both temperature sensors).
_AUG_MEAS_DIM: Final[int] = 2


@dataclass(frozen=True, slots=True)
class EKFEstimate:
    """Snapshot of the DHW EKF augmented state estimate.

    Attributes
    ----------
    mean:       Augmented state [T_top (°C), T_bot (°C), V_tap (m³/h)].
    covariance: 3×3 error covariance matrix [mixed units: K² and (m³/h)²].
    v_tap_m3_per_h:  Clamped non-negative tap-flow estimate [m³/h] (§12, Step 4).
    """

    mean: np.ndarray
    covariance: np.ndarray

    @property
    def t_top_c(self) -> float:
        """Top-layer temperature estimate [°C]."""
        return float(self.mean[0])

    @property
    def t_bot_c(self) -> float:
        """Bottom-layer temperature estimate [°C]."""
        return float(self.mean[1])

    @property
    def v_tap_m3_per_h(self) -> float:
        """Tap-flow estimate after physical clamp ≥ 0 [m³/h] (§12 Step 4)."""
        return float(max(0.0, self.mean[2]))


class DHWExtendedKalmanFilter:
    """Extended Kalman Filter for the DHW tank with augmented state.

    Estimates the 3-dimensional augmented state
        x_aug[k] = [T_top[k], T_bot[k], V_tap[k]]ᵀ

    from two temperature measurements y[k] = [T_top_meas, T_bot_meas]ᵀ
    and a known control input P_dhw (thermal power to the bottom layer).

    Because V_tap is NOT directly observed (no flow meter — assumption A7),
    the tap terms in the DHW physics create a bilinear (non-linear)
    state-transition function.  The EKF linearises this function at every
    step by computing the Jacobian F[k] = ∂f/∂x_aug|_{x̂[k]} (§12.4).

    Architecture DRY note (§ArchReq):
        This class is a *domain-specific* EKF and does NOT subclass
        LinearKalmanFilter because the core Kalman algebra must operate on
        3-dimensional augmented state with a 2-dimensional measurement.  The
        Joseph-form update is re-implemented here for the generalised
        dimensions while keeping the mathematics identical to §6/§12.

    Parameters
    ----------
    model:
        DHWModel instance supplying physical parameters (λ, C_top, C_bot, …).
    noise:
        EKFNoiseParameters with Q_aug (3×3) and R_n (2×2).
    initial_state:
        Initial augmented state estimate [T_top, T_bot, V_tap] [°C, °C, m³/h].
    initial_covariance:
        Initial 3×3 covariance matrix.  High uncertainty on V_tap is recommended
        at start-up (e.g. diag([1, 1, 1e-4])).
    """

    def __init__(
        self,
        model: DHWModel,
        noise: EKFNoiseParameters,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        self._model = model
        self._noise = noise

        x0 = np.asarray(initial_state, dtype=float).copy()
        P0 = np.asarray(initial_covariance, dtype=float).copy()

        if x0.shape != (_AUG_STATE_DIM,):
            raise ValueError("initial_state must have shape (3,): [T_top, T_bot, V_tap].")
        if P0.shape != (_AUG_STATE_DIM, _AUG_STATE_DIM):
            raise ValueError("initial_covariance must be 3×3.")
        if not np.allclose(P0, P0.T):
            raise ValueError("initial_covariance must be symmetric.")
        if np.min(np.linalg.eigvalsh(P0)) < -1e-10:
            raise ValueError("initial_covariance must be positive semi-definite.")
        if x0[2] < 0.0:
            raise ValueError("Initial V_tap estimate must be ≥ 0 (physical constraint).")

        self._x = x0
        self._P = P0

    # ------------------------------------------------------------------
    # Public read-only access
    # ------------------------------------------------------------------

    @property
    def estimate(self) -> EKFEstimate:
        """Return the current posterior augmented state estimate (copies)."""
        return EKFEstimate(self._x.copy(), self._P.copy())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _symmetrise(self) -> None:
        """Project covariance onto symmetric subspace (numerical stability)."""
        self._P = 0.5 * (self._P + self._P.T)

    def _propagate(
        self,
        control_kw: float,
        t_mains_c: float,
        t_amb_c: float,
    ) -> np.ndarray:
        """Evaluate nonlinear propagation f(x_aug, u, d) from §12.3.

        Parameters
        ----------
        control_kw:  DHW thermal power P_dhw [kW].
        t_mains_c:   Cold mains temperature T_mains [°C].
        t_amb_c:     Ambient temperature T_amb [°C].

        Returns
        -------
        x_next : np.ndarray
            Predicted augmented state [T_top, T_bot, V_tap], shape (3,).
        """
        p = self._model.parameters
        dt = p.dt_hours
        T_top, T_bot, V_tap = self._x

        # Constant scalars (§10.1)
        a_strat = dt / (p.C_top * p.R_strat)
        b_strat = dt / (p.C_bot * p.R_strat)
        a_loss = dt / (p.C_top * p.R_loss)
        b_loss = dt / (p.C_bot * p.R_loss)

        # Nonlinear propagation (§12.3) — bilinear tap terms
        T_top_next = T_top + dt / p.C_top * (
            -(T_top - T_bot) / p.R_strat
            - p.lambda_water * V_tap * T_top
            - (T_top - t_amb_c) / p.R_loss
        )
        T_bot_next = T_bot + dt / p.C_bot * (
            (T_top - T_bot) / p.R_strat
            + control_kw
            + p.lambda_water * V_tap * t_mains_c
            - (T_bot - t_amb_c) / p.R_loss
        )
        # Random-walk model for V_tap (§12.2): V_tap[k+1] = V_tap[k]
        V_tap_next = V_tap

        # Suppress unused-variable warning: a_strat/b_strat/a_loss/b_loss
        # are computed here for clarity but their roles appear in the
        # Jacobian helper.  Reference them to avoid linting errors.
        _ = (a_strat, b_strat, a_loss, b_loss)

        return np.array([T_top_next, T_bot_next, V_tap_next], dtype=float)

    def _jacobian(self, t_mains_c: float) -> np.ndarray:
        """Compute Jacobian F[k] = ∂f/∂x_aug evaluated at current x̂ (§12.4).

        Parameters
        ----------
        t_mains_c:  Cold mains temperature T_mains [°C] at the current step.

        Returns
        -------
        F : np.ndarray, shape (3, 3)
            Linearised state-transition matrix around the current estimate.

        Notes
        -----
        The third column of F[k] is the sensitivity of the temperature
        states to V_tap.  This column drives observability of V_tap:
        when T_top ≈ T_mains the sensitivity is near zero and the EKF
        cannot estimate V_tap well (§12.5 edge case).
        """
        p = self._model.parameters
        dt = p.dt_hours
        T_top_hat = self._x[0]
        V_tap_hat = self._x[2]

        # Constant scalars (same as in §11 and §12.4)
        a_strat = dt / (p.C_top * p.R_strat)
        b_strat = dt / (p.C_bot * p.R_strat)
        a_loss = dt / (p.C_top * p.R_loss)
        b_loss = dt / (p.C_bot * p.R_loss)
        a_tap_hat = dt / p.C_top * p.lambda_water * V_tap_hat

        # ∂f_T_top/∂V_tap = -Δt/C_top · λ · T̂_top  (third column, row 0)
        df_Ttop_dVtap = -dt / p.C_top * p.lambda_water * T_top_hat
        # ∂f_T_bot/∂V_tap = +Δt/C_bot · λ · T_mains  (third column, row 1)
        df_Tbot_dVtap = dt / p.C_bot * p.lambda_water * t_mains_c

        # Jacobian matrix (§12.4)
        F = np.array(
            [
                [1.0 - a_strat - a_loss - a_tap_hat, a_strat, df_Ttop_dVtap],
                [b_strat, 1.0 - b_strat - b_loss, df_Tbot_dVtap],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return F

    # ------------------------------------------------------------------
    # EKF predict step (§12.6 Step 1)
    # ------------------------------------------------------------------

    def predict(
        self,
        control_kw: float,
        t_mains_c: float,
        t_amb_c: float,
    ) -> EKFEstimate:
        """EKF prediction step: propagate state and covariance forward.

        Implements §12.6 Step 1:
            x̂⁻[k] = f(x̂[k-1], u[k-1], d[k-1])
            P⁻[k]  = F[k-1] P[k-1] F[k-1]ᵀ + Q_aug

        Parameters
        ----------
        control_kw:  DHW thermal power P_dhw applied at the previous step [kW].
        t_mains_c:   Mains temperature at the previous step [°C].
        t_amb_c:     Ambient temperature at the previous step [°C].

        Returns
        -------
        EKFEstimate
            Prior (predicted) augmented state estimate.
        """
        # Jacobian evaluated at current (prior) estimate before propagating
        F = self._jacobian(t_mains_c)
        # Nonlinear propagation (§12.3)
        self._x = self._propagate(control_kw, t_mains_c, t_amb_c)
        # Covariance propagation (§12.6 Step 1)
        self._P = F @ self._P @ F.T + self._noise.Q_aug
        self._symmetrise()
        return self.estimate

    # ------------------------------------------------------------------
    # EKF update step (§12.6 Steps 2–4)
    # ------------------------------------------------------------------

    def update(self, t_top_meas_c: float, t_bot_meas_c: float) -> EKFEstimate:
        """EKF update step: incorporate both temperature measurements.

        Implements §12.6 Steps 2–4 using the Joseph form for numerical
        stability.  After the update, V_tap is clamped to ≥ 0 (Step 4).

        Parameters
        ----------
        t_top_meas_c:  Measured top-layer temperature [°C].
        t_bot_meas_c:  Measured bottom-layer temperature [°C].

        Returns
        -------
        EKFEstimate
            Posterior augmented state estimate with V_tap ≥ 0 (clamped).
        """
        C = _C_OBS_AUG  # 2×3
        R_n = self._noise.R_n  # 2×2
        y = np.array([t_top_meas_c, t_bot_meas_c], dtype=float)
        innovation = y - C @ self._x  # 2-vector

        # Innovation covariance S = C P⁻ Cᵀ + R_n  (2×2)
        S = C @ self._P @ C.T + R_n
        S_inv = np.linalg.inv(S)

        # Kalman gain K ∈ ℝ^{3×2}  (§12.6 Step 2)
        K = self._P @ C.T @ S_inv

        # State update  (§12.6 Step 3)
        self._x = self._x + K @ innovation

        # Covariance update — Joseph form (§12.6 Step 3)
        I_KC = np.eye(_AUG_STATE_DIM) - K @ C
        self._P = I_KC @ self._P @ I_KC.T + K @ R_n @ K.T
        self._symmetrise()

        # §12.6 Step 4 — Fail-Fast physical clamp: V_tap ≥ 0.
        # A negative flow rate is physically impossible.  The clamp is a hard
        # projection onto the feasible set, NOT a workaround.
        if self._x[2] < 0.0:
            self._x[2] = 0.0

        return self.estimate

    # ------------------------------------------------------------------
    # Combined predict + update
    # ------------------------------------------------------------------

    def step(
        self,
        control_kw: float,
        t_mains_c: float,
        t_amb_c: float,
        t_top_meas_c: float,
        t_bot_meas_c: float,
    ) -> EKFEstimate:
        """Run a complete EKF predict-then-update cycle.

        Parameters
        ----------
        control_kw:    DHW thermal power P_dhw at the previous step [kW].
        t_mains_c:     Mains temperature at the previous step [°C].
        t_amb_c:       Ambient temperature at the previous step [°C].
        t_top_meas_c:  Measured top-layer temperature y_top[k] [°C].
        t_bot_meas_c:  Measured bottom-layer temperature y_bot[k] [°C].

        Returns
        -------
        EKFEstimate
            Posterior augmented state estimate after full predict+update.
        """
        self.predict(control_kw, t_mains_c, t_amb_c)
        return self.update(t_top_meas_c, t_bot_meas_c)

    # ------------------------------------------------------------------
    # Observability (§12.5)
    # ------------------------------------------------------------------

    def observability_matrix(self, t_mains_c: float) -> np.ndarray:
        """Augmented observability matrix O_aug ∈ ℝ^{4×3} (§12.5).

        O_aug = [C_obs; C_obs · F[k]]

        rank(O_aug) = 3  iff  a_strat ≠ 0  AND  T̂_top ≠ T_mains.

        Parameters
        ----------
        t_mains_c:  Mains temperature used to evaluate the Jacobian [°C].

        Returns
        -------
        O_aug : np.ndarray, shape (4, 3)
        """
        F = self._jacobian(t_mains_c)
        C = _C_OBS_AUG
        return np.vstack([C, C @ F])
