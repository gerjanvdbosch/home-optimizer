from __future__ import annotations

import numpy as np

from .kalman import UFHKalmanFilter
from .mpc import UFHMPCController
from .thermal_model import ThermalModel
from .types import ForecastHorizon, KalmanNoiseParameters, MPCParameters, ThermalParameters


def build_demo_configuration() -> tuple[
    ThermalModel,
    MPCParameters,
    ForecastHorizon,
    np.ndarray,
    float,
    KalmanNoiseParameters,
]:
    thermal_parameters = ThermalParameters(
        dt_hours=1.0,
        C_r=3.0,
        C_b=18.0,
        R_br=2.5,
        R_ro=4.0,
        alpha=0.35,
        eta=0.62,
        A_glass=12.0,
    )
    model = ThermalModel(thermal_parameters)

    mpc_parameters = MPCParameters(
        horizon_steps=12,
        Q_c=8.0,
        R_c=0.05,
        Q_N=12.0,
        P_max=4.0,
        delta_P_max=1.0,
        T_min=19.0,
        T_max=22.5,
    )
    forecast = ForecastHorizon(
        outdoor_temperature_c=np.array([10, 10, 9, 9, 8, 8, 9, 10, 11, 11, 10, 10], dtype=float),
        gti_w_per_m2=np.array([0, 0, 40, 90, 150, 220, 220, 160, 80, 20, 0, 0], dtype=float),
        internal_gains_kw=np.full(12, 0.35, dtype=float),
        price_eur_per_kwh=np.array(
            [0.34, 0.30, 0.27, 0.21, 0.18, 0.17, 0.18, 0.24, 0.32, 0.38, 0.41, 0.36],
            dtype=float,
        ),
        room_temperature_ref_c=np.array(
            [20.0, 20.0, 20.5, 20.5, 21.0, 21.0, 21.0, 21.0, 21.0, 20.5, 20.5, 20.0, 20.0],
            dtype=float,
        ),
    )
    initial_state_c = np.array([20.8, 24.0], dtype=float)
    previous_power_kw = 0.5
    kalman_noise = KalmanNoiseParameters(
        process_covariance=np.diag([0.01, 0.015]),
        measurement_variance=0.05**2,
    )
    return model, mpc_parameters, forecast, initial_state_c, previous_power_kw, kalman_noise


def main() -> None:
    model, mpc_parameters, forecast, initial_state_c, previous_power_kw, kalman_noise = (
        build_demo_configuration()
    )
    controller = UFHMPCController(model=model, parameters=mpc_parameters)
    solution = controller.solve(
        initial_state_c=initial_state_c,
        forecast=forecast,
        previous_power_kw=previous_power_kw,
    )

    print(f"Observability rank: {model.observability_rank()}")
    print(f"Controllability rank: {model.controllability_rank()}")
    print(f"MPC status: {solution.solver_status}")
    print(f"Fallback used: {solution.used_fallback}")
    print(f"First UFH power command: {solution.first_control_kw:.2f} kW")
    print(
        "Predicted room temperatures [°C]:",
        np.round(solution.predicted_states_c[:, 0], 2).tolist(),
    )

    kalman_filter = UFHKalmanFilter(
        model=model,
        noise=kalman_noise,
        initial_state_c=np.array([19.0, 20.0], dtype=float),
        initial_covariance=np.diag([0.25, 1.0]),
    )
    first_disturbance = forecast.disturbance_matrix(model.parameters)[0]
    first_measurement_c = 19.6
    estimate, innovation, _ = kalman_filter.step(
        control_kw=solution.first_control_kw,
        disturbance_vector=first_disturbance,
        room_temperature_measurement_c=first_measurement_c,
    )
    print(f"Kalman innovation: {innovation:.3f} °C")
    print("Estimated state [T_r, T_b] [°C]:", np.round(estimate.mean_c, 3).tolist())


if __name__ == "__main__":
    main()
