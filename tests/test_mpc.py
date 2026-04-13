import numpy as np
import pytest

from home_optimizer.mpc import UFHMPCController
from home_optimizer.thermal_model import ThermalModel
from home_optimizer.types import ForecastHorizon, MPCParameters, ThermalParameters


def test_mpc_solution_respects_power_temperature_and_ramp_constraints() -> None:
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
    mpc_parameters = MPCParameters(
        horizon_steps=8,
        Q_c=10.0,
        R_c=0.05,
        Q_N=15.0,
        P_max=4.0,
        delta_P_max=1.0,
        T_min=19.0,
        T_max=22.5,
    )
    forecast = ForecastHorizon(
        outdoor_temperature_c=np.full(8, 10.0, dtype=float),
        gti_w_per_m2=np.array([0.0, 0.0, 50.0, 150.0, 150.0, 50.0, 0.0, 0.0], dtype=float),
        internal_gains_kw=np.full(8, 0.25, dtype=float),
        price_eur_per_kwh=np.array([0.38, 0.30, 0.22, 0.18, 0.20, 0.28, 0.40, 0.45], dtype=float),
        room_temperature_ref_c=np.full(9, 21.0, dtype=float),
    )

    controller = UFHMPCController(
        model=ThermalModel(thermal_parameters),
        parameters=mpc_parameters,
    )
    previous_power_kw = 0.5
    solution = controller.solve(
        initial_state_c=np.array([20.8, 24.0], dtype=float),
        forecast=forecast,
        previous_power_kw=previous_power_kw,
    )

    assert not solution.used_fallback
    assert solution.control_sequence_kw.shape == (mpc_parameters.horizon_steps,)
    assert solution.predicted_states_c.shape == (mpc_parameters.horizon_steps + 1, 2)
    assert np.all(solution.control_sequence_kw >= -1e-8)
    assert np.all(solution.control_sequence_kw <= mpc_parameters.P_max + 1e-8)
    ramp_deltas = np.diff(np.concatenate([[previous_power_kw], solution.control_sequence_kw]))
    assert np.all(np.abs(ramp_deltas) <= mpc_parameters.delta_P_max + 1e-5)
    assert np.all(solution.predicted_states_c[1:, 0] >= mpc_parameters.T_min - 1e-6)
    assert np.all(solution.predicted_states_c[1:, 0] <= mpc_parameters.T_max + 1e-6)
    assert solution.first_control_kw > 0.0


def test_mpc_raises_when_hard_comfort_constraints_are_physically_infeasible() -> None:
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
    mpc_parameters = MPCParameters(
        horizon_steps=4,
        Q_c=10.0,
        R_c=0.05,
        Q_N=15.0,
        P_max=4.0,
        delta_P_max=1.0,
        T_min=19.0,
        T_max=22.5,
    )
    forecast = ForecastHorizon(
        outdoor_temperature_c=np.full(4, 2.0, dtype=float),
        gti_w_per_m2=np.zeros(4, dtype=float),
        internal_gains_kw=np.full(4, 0.25, dtype=float),
        price_eur_per_kwh=np.full(4, 0.30, dtype=float),
        room_temperature_ref_c=np.full(5, 21.0, dtype=float),
    )

    controller = UFHMPCController(
        model=ThermalModel(thermal_parameters),
        parameters=mpc_parameters,
    )

    with pytest.raises(ValueError, match="infeasible"):
        controller.solve(
            initial_state_c=np.array([20.0, 21.0], dtype=float),
            forecast=forecast,
            previous_power_kw=0.5,
        )
