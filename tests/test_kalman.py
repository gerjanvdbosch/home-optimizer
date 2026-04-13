"""Unit tests for the Kalman filter: convergence and numerical properties."""

from __future__ import annotations

import numpy as np

from home_optimizer.kalman import UFHKalmanFilter
from home_optimizer.thermal_model import ThermalModel
from home_optimizer.types import KalmanNoiseParameters, ThermalParameters


def _make_model() -> ThermalModel:
    return ThermalModel(
        ThermalParameters(
            dt_hours=1.0,
            C_r=3.0,
            C_b=18.0,
            R_br=2.5,
            R_ro=4.0,
            alpha=0.35,
            eta=0.62,
            A_glass=12.0,
        )
    )


def test_kalman_recovers_room_temperature_exactly() -> None:
    """T_r is directly observed, so the filter must track it with near-zero error."""
    model = _make_model()
    kf = UFHKalmanFilter(
        model=model,
        noise=KalmanNoiseParameters(
            process_covariance=np.diag([1e-4, 1e-4]),
            measurement_variance=1e-6,
        ),
        initial_state_c=np.array([18.0, 18.0]),
        initial_covariance=np.diag([4.0, 4.0]),
    )
    true_state = np.array([20.0, 23.5])
    d = np.array([4.0, 0.5, 0.3])
    u = 1.25
    estimate = kf.estimate  # initialise before loop

    for _ in range(24):
        true_state = model.step_with_disturbance_vector(true_state, u, d)
        estimate, _, _ = kf.step(
            control_kw=u,
            disturbance=d,
            room_temp_measurement_c=float(true_state[0]),
        )

    # T_r is observable → near-zero residual after convergence
    assert abs(estimate.mean_c[0] - true_state[0]) < 1e-3


def test_kalman_converges_on_hidden_floor_temperature() -> None:
    """T_b is hidden but observable via the T_r trajectory; filter must converge."""
    model = _make_model()
    kf = UFHKalmanFilter(
        model=model,
        noise=KalmanNoiseParameters(
            process_covariance=np.diag([1e-4, 1e-4]),
            measurement_variance=1e-6,
        ),
        initial_state_c=np.array([18.0, 18.0]),
        initial_covariance=np.diag([4.0, 4.0]),
    )
    true_state = np.array([20.0, 23.5])
    d = np.array([4.0, 0.5, 0.3])
    u = 1.25
    estimate = kf.estimate

    for _ in range(24):
        true_state = model.step_with_disturbance_vector(true_state, u, d)
        estimate, _, _ = kf.step(
            control_kw=u,
            disturbance=d,
            room_temp_measurement_c=float(true_state[0]),
        )

    # T_b convergence is slower; tolerance is physically reasonable (< 0.15 K)
    assert abs(estimate.mean_c[1] - true_state[1]) < 0.15


def test_kalman_covariance_remains_positive_definite() -> None:
    """Error covariance must remain PSD throughout the filter run."""
    model = _make_model()
    kf = UFHKalmanFilter(
        model=model,
        noise=KalmanNoiseParameters(
            process_covariance=np.diag([0.1, 0.2]),
            measurement_variance=0.01,
        ),
        initial_state_c=np.array([21.0, 23.0]),
        initial_covariance=np.eye(2),
    )
    d = np.array([8.0, 0.3, 0.25])
    for _ in range(48):
        kf.predict(1.0, d)
        kf.update(21.0 + np.random.default_rng(0).normal(0, 0.1))
        eigvals = np.linalg.eigvalsh(kf.estimate.covariance)
        assert np.all(eigvals >= -1e-10), "Covariance must remain positive semi-definite."
