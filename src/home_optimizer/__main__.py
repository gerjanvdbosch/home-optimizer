"""Demo entry-point: run a single MPC step and one Kalman update.

Usage
-----
  python -m home_optimizer
"""

from __future__ import annotations

import numpy as np

from .kalman import UFHKalmanFilter
from .mpc import UFHMPCController
from .thermal_model import ThermalModel
from .types import ForecastHorizon, KalmanNoiseParameters, MPCParameters, ThermalParameters


def build_demo() -> tuple[
    ThermalModel,
    MPCParameters,
    ForecastHorizon,
    np.ndarray,
    float,
    KalmanNoiseParameters,
]:
    """Construct a representative winter-day demo configuration.

    The forecast uses mild outdoor conditions (≥10 °C) and a warm floor
    initial state so the MPC problem is physically feasible under the
    hard comfort bounds (T_min=19 °C, T_max=22.5 °C).
    """
    params = ThermalParameters(
        dt_hours=1.0,
        C_r=3.0,  # kWh/K  – room air + furniture
        C_b=18.0,  # kWh/K  – concrete floor slab
        R_br=2.5,  # K/kW   – floor → room
        R_ro=4.0,  # K/kW   – room → outside
        alpha=0.35,
        eta=0.62,
        A_glass=12.0,
    )
    model = ThermalModel(params)

    mpc_params = MPCParameters(
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
    # Initial state: room slightly below setpoint, warm slab
    initial_state_c = np.array([20.8, 24.0], dtype=float)
    previous_power_kw = 0.5

    kalman_noise = KalmanNoiseParameters(
        process_covariance=np.diag([0.01, 0.015]),
        measurement_variance=0.05**2,
    )
    return model, mpc_params, forecast, initial_state_c, previous_power_kw, kalman_noise


def main() -> None:
    model, mpc_params, forecast, x0, u_prev, kalman_noise = build_demo()

    # ── Model checks ──────────────────────────────────────────────────
    obs_rank = model.observability_rank()
    ctrl_rank = model.controllability_rank()
    print(f"Model observability rank : {obs_rank} / 2  ({'✓' if obs_rank == 2 else '✗'})")
    print(f"Model controllability rank: {ctrl_rank} / 2  ({'✓' if ctrl_rank == 2 else '✗'})")

    # ── MPC solve ─────────────────────────────────────────────────────
    controller = UFHMPCController(model=model, params=mpc_params)
    solution = controller.solve(initial_state_c=x0, forecast=forecast, previous_power_kw=u_prev)

    print(f"\nMPC solver status : {solution.solver_status}")
    print(f"Fallback used     : {solution.used_fallback}")
    print(f"First P_UFH [kW]  : {solution.first_control_kw:.3f}")
    t_r_traj = np.round(solution.predicted_states_c[:, 0], 2).tolist()
    print(f"T_r trajectory [°C]: {t_r_traj}")

    # ── Kalman update ─────────────────────────────────────────────────
    kf = UFHKalmanFilter(
        model=model,
        noise=kalman_noise,
        initial_state_c=np.array([19.5, 22.0], dtype=float),
        initial_covariance=np.diag([0.25, 1.0]),
    )
    d0 = forecast.disturbance_matrix(model.parameters)[0]
    estimate, innovation, K = kf.step(
        control_kw=solution.first_control_kw,
        disturbance=d0,
        room_temp_measurement_c=20.4,
    )
    print(f"\nKalman innovation : {innovation:+.3f} °C")
    print(f"Estimated [T_r, T_b]: {np.round(estimate.mean_c, 3).tolist()} °C")
    print(f"Kalman gain         : {np.round(K[:, 0], 4).tolist()}")


if __name__ == "__main__":
    main()
