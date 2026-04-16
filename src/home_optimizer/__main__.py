"""Demo entry-point: run a single MPC step and one Kalman update.

Usage
-----
  python -m home_optimizer
"""

from __future__ import annotations

import numpy as np

from .cop_model import HeatPumpCOPModel, HeatPumpCOPParameters
from .kalman import UFHKalmanFilter
from .mpc import MPCController
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
    """Construct a representative demo for a 2023 Dutch terraced house with UFH.

    The defaults reflect a reasonably well insulated between-house dwelling with
    a heat pump, south-facing glazing, and a pre-warmed floor slab.

    The COP is no longer a magic-number scalar; it is derived from the physical
    Carnot model (§14.1).  The ``HeatPumpCOPModel`` computes a time-varying
    ``cop_ufh_k`` array from the outdoor-temperature forecast using the heating
    curve (stooklijn), so colder hours automatically attract a lower COP.
    """
    params = ThermalParameters(
        dt_hours=1.0,
        C_r=6.0,  # kWh/K  – room air + light interior mass
        C_b=10.0,  # kWh/K  – thermally active UFH floor zone
        R_br=1.0,  # K/kW   – floor → room
        R_ro=10.0,  # K/kW   – room → outside
        alpha=0.25,
        eta=0.55,
        A_glass=7.5,
    )
    model = ThermalModel(params)

    # ── Outdoor temperature forecast ──────────────────────────────────────
    t_out = np.array([6, 6, 5, 5, 4, 4, 5, 6, 7, 7, 6, 6], dtype=float)

    # ── Carnot COP model — no magic numbers; all parameters named (§14.1) ─
    # These parameters should come from a config file in production use.
    cop_params = HeatPumpCOPParameters(
        eta_carnot=0.45,  # Carnot efficiency factor η [-]
        delta_T_cond=5.0,  # condensing approach temperature Δ_cond [K]
        delta_T_evap=5.0,  # evaporating approach temperature Δ_evap [K]
        T_supply_min=28.0,  # minimum UFH supply temperature [°C]
        T_ref_outdoor=18.0,  # heating-curve balance point [°C]
        heating_curve_slope=1.0,  # heating-curve slope [K/K]
        cop_min=1.5,  # physical floor on COP (Fail-Fast lower bound)
        cop_max=7.0,  # Fail-Fast upper bound on COP
    )
    cop_model = HeatPumpCOPModel(cop_params)

    # §14.1 – pre-compute time-varying COP for the full forecast horizon.
    # Colder outdoor hours → higher T_supply (heating curve) AND lower T_evap
    # → double COP penalty, modelled correctly by cop_model.cop_ufh().
    cop_ufh_k = cop_model.cop_ufh(t_out)

    # Scalar COP for MPCParameters (Fail-Fast bounds check at construction time).
    # Use the minimum COP in the horizon as a conservative representative value
    # so that MPCParameters validation is always satisfied; the MPC itself uses
    # the time-varying array from the forecast.
    cop_ufh_scalar = float(np.min(cop_ufh_k))

    mpc_params = MPCParameters(
        horizon_steps=12,
        Q_c=8.0,
        R_c=0.05,
        Q_N=12.0,
        P_max=4.5,
        delta_P_max=1.0,
        T_min=19.0,
        T_max=22.5,
        # §14.1: scalar derived from the Carnot model (not a magic number)
        cop_ufh=cop_ufh_scalar,
        cop_max=cop_params.cop_max,
    )
    forecast = ForecastHorizon(
        outdoor_temperature_c=t_out,
        gti_w_per_m2=np.array([0, 0, 20, 80, 160, 260, 300, 220, 120, 40, 0, 0], dtype=float),
        internal_gains_kw=np.full(12, 0.30, dtype=float),
        price_eur_per_kwh=np.array(
            [0.34, 0.30, 0.27, 0.21, 0.18, 0.17, 0.18, 0.24, 0.32, 0.38, 0.41, 0.36],
            dtype=float,
        ),
        room_temperature_ref_c=np.array(
            [20.0, 20.0, 20.5, 20.5, 20.5, 20.5, 20.5, 20.5, 20.5, 20.5, 20.5, 20.0, 20.0],
            dtype=float,
        ),
        # §14.1 – time-varying COP from the physical Carnot model
        cop_ufh_k=cop_ufh_k,
    )
    # Initial state: room near setpoint, slab already a bit warmer than the air
    initial_state_c = np.array([20.5, 22.5], dtype=float)
    previous_power_kw = 0.8

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

    # ── COP profile (from physical Carnot model) ───────────────────────
    assert forecast.cop_ufh_k is not None  # always set by build_demo()
    cop_arr = forecast.cop_ufh_k
    print(
        f"\nCOP_UFH profile (Carnot): min={cop_arr.min():.3f}  "
        f"max={cop_arr.max():.3f}  mean={cop_arr.mean():.3f}"
    )
    print(f"  per step: {np.round(cop_arr, 3).tolist()}")

    # ── MPC solve ─────────────────────────────────────────────────────
    controller = MPCController(ufh_model=model, params=mpc_params)
    solution = controller.solve(
        initial_ufh_state_c=x0,
        ufh_forecast=forecast,
        previous_p_ufh_kw=u_prev,
    )

    print(f"\nMPC solver status : {solution.solver_status}")
    print(f"Fallback used     : {solution.used_fallback}")
    print(f"First P_UFH [kW]  : {solution.first_ufh_control_kw:.3f}")
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
        control_kw=solution.first_ufh_control_kw,
        disturbance=d0,
        room_temp_measurement_c=20.4,
    )
    print(f"\nKalman innovation : {innovation:+.3f} °C")
    print(f"Estimated [T_r, T_b]: {np.round(estimate.mean_c, 3).tolist()} °C")
    print(f"Kalman gain         : {np.round(K[:, 0], 4).tolist()}")


if __name__ == "__main__":
    main()
