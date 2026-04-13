import numpy as np

from home_optimizer.kalman import UFHKalmanFilter
from home_optimizer.thermal_model import ThermalModel
from home_optimizer.types import KalmanNoiseParameters, ThermalParameters


def test_kalman_filter_recovers_hidden_floor_temperature() -> None:
    parameters = ThermalParameters(
        dt_hours=1.0,
        C_r=3.0,
        C_b=18.0,
        R_br=2.5,
        R_ro=4.0,
        alpha=0.35,
        eta=0.62,
        A_glass=12.0,
    )
    model = ThermalModel(parameters)

    filter_ = UFHKalmanFilter(
        model=model,
        noise=KalmanNoiseParameters(
            process_covariance=np.diag([1e-4, 1e-4]),
            measurement_variance=1e-5,
        ),
        initial_state_c=np.array([18.0, 18.0], dtype=float),
        initial_covariance=np.diag([4.0, 4.0]),
    )

    true_state = np.array([20.0, 23.5], dtype=float)
    disturbance = np.array([4.0, 0.5, 0.3], dtype=float)
    control_kw = 1.25
    estimate = filter_.estimate

    for _ in range(24):
        true_state = model.step_with_disturbance_vector(true_state, control_kw, disturbance)
        estimate, _, _ = filter_.step(
            control_kw=control_kw,
            disturbance_vector=disturbance,
            room_temperature_measurement_c=float(true_state[0]),
        )

    assert abs(estimate.mean_c[0] - true_state[0]) < 1e-3
    assert abs(estimate.mean_c[1] - true_state[1]) < 0.15
