"""Unit tests for the MPC controller: feasibility, constraints, and properties."""

from __future__ import annotations

import numpy as np
import pytest

from home_optimizer.mpc import UFHMPCController
from home_optimizer.thermal_model import ThermalModel
from home_optimizer.types import ForecastHorizon, MPCParameters, ThermalParameters

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

THERMAL_PARAMS = ThermalParameters(
    dt_hours=1.0,
    C_r=3.0,
    C_b=18.0,
    R_br=2.5,
    R_ro=4.0,
    alpha=0.35,
    eta=0.62,
    A_glass=12.0,
)
MPC_PARAMS = MPCParameters(
    horizon_steps=8,
    Q_c=10.0,
    R_c=0.05,
    Q_N=15.0,
    P_max=4.0,
    delta_P_max=1.0,
    T_min=19.0,
    T_max=22.5,
)


def _feasible_forecast(n: int = 8) -> ForecastHorizon:
    """A mild winter-day forecast that keeps T_r ≥ 19 °C when P_UFH > 0."""
    return ForecastHorizon(
        outdoor_temperature_c=np.full(n, 10.0),
        gti_w_per_m2=np.array([0, 0, 50, 150, 150, 50, 0, 0], dtype=float)[:n],
        internal_gains_kw=np.full(n, 0.25),
        price_eur_per_kwh=np.array([0.38, 0.30, 0.22, 0.18, 0.20, 0.28, 0.40, 0.45])[:n],
        room_temperature_ref_c=np.full(n + 1, 21.0),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mpc_solution_respects_all_hard_constraints() -> None:
    """Convex MPC must satisfy power, comfort, and ramp-rate constraints."""
    model = ThermalModel(THERMAL_PARAMS)
    controller = UFHMPCController(model=model, params=MPC_PARAMS)
    prev_u = 0.5
    # Warm slab ensures the first few steps stay above T_min = 19 °C
    sol = controller.solve(
        initial_state_c=np.array([20.8, 24.0]),
        forecast=_feasible_forecast(),
        previous_power_kw=prev_u,
    )

    assert not sol.used_fallback, "Convex solver must be used for a feasible problem."
    assert sol.control_sequence_kw.shape == (MPC_PARAMS.horizon_steps,)
    assert sol.predicted_states_c.shape == (MPC_PARAMS.horizon_steps + 1, 2)

    # Power bounds
    assert np.all(sol.control_sequence_kw >= -1e-5)
    assert np.all(sol.control_sequence_kw <= MPC_PARAMS.P_max + 1e-5)

    # Ramp-rate  (generous tolerance for floating-point residuals from OSQP)
    deltas = np.diff(np.concatenate([[prev_u], sol.control_sequence_kw]))
    assert np.all(np.abs(deltas) <= MPC_PARAMS.delta_P_max + 1e-5)

    # Comfort bounds on predicted room temperatures
    assert np.all(sol.predicted_states_c[1:, 0] >= MPC_PARAMS.T_min - 1e-5)
    assert np.all(sol.predicted_states_c[1:, 0] <= MPC_PARAMS.T_max + 1e-5)

    # Controller must request some heating
    assert sol.first_control_kw > 0.0


def test_mpc_prefers_cheap_hours() -> None:
    """With equal comfort, the MPC should concentrate heating in the cheapest slot."""
    model = ThermalModel(THERMAL_PARAMS)
    controller = UFHMPCController(model=model, params=MPC_PARAMS)
    # One cheap step (index 3) surrounded by expensive steps
    prices = np.array([0.40, 0.38, 0.35, 0.10, 0.35, 0.38, 0.40, 0.40])
    forecast = ForecastHorizon(
        outdoor_temperature_c=np.full(8, 10.0),
        gti_w_per_m2=np.zeros(8),
        internal_gains_kw=np.full(8, 0.25),
        price_eur_per_kwh=prices,
        room_temperature_ref_c=np.full(9, 21.0),
    )
    sol = controller.solve(
        initial_state_c=np.array([20.8, 24.0]),
        forecast=forecast,
        previous_power_kw=0.5,
    )
    # The cheap slot should attract higher power than the peak-price slots
    cheap_idx = 3
    avg_expensive = np.mean(np.delete(sol.control_sequence_kw, cheap_idx))
    assert sol.control_sequence_kw[cheap_idx] >= avg_expensive - 1e-3


def test_mpc_raises_for_infeasible_comfort_bounds() -> None:
    """MPC must raise ValueError (not silently violate) when T_min is unreachable."""
    model = ThermalModel(THERMAL_PARAMS)
    tight_params = MPCParameters(
        horizon_steps=4,
        Q_c=10.0,
        R_c=0.05,
        Q_N=15.0,
        P_max=4.0,
        delta_P_max=1.0,
        T_min=19.0,  # cold outdoor + small floor buffer → infeasible
        T_max=22.5,
    )
    controller = UFHMPCController(model=model, params=tight_params)
    # Very cold outdoor, no solar, small slab buffer → T_r will drop below T_min
    with pytest.raises(ValueError, match="infeasible"):
        controller.solve(
            initial_state_c=np.array([20.0, 21.0]),
            forecast=ForecastHorizon(
                outdoor_temperature_c=np.full(4, 2.0),
                gti_w_per_m2=np.zeros(4),
                internal_gains_kw=np.full(4, 0.25),
                price_eur_per_kwh=np.full(4, 0.30),
                room_temperature_ref_c=np.full(5, 21.0),
            ),
            previous_power_kw=0.5,
        )
