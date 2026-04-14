"""Unit tests for the MPC controller: feasibility, constraints, and properties."""

from __future__ import annotations

import numpy as np
import pytest

from home_optimizer.dhw_model import DHWModel
from home_optimizer.mpc import MPCController
from home_optimizer.thermal_model import ThermalModel
from home_optimizer.types import (
    CombinedMPCParameters,
    DHWForecastHorizon,
    DHWMPCParameters,
    DHWParameters,
    ForecastHorizon,
    MPCParameters,
    ThermalParameters,
)

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
DHW_PARAMS = DHWParameters(
    dt_hours=1.0,
    C_top=0.5814,
    C_bot=0.5814,
    R_strat=10.0,
    R_loss=50.0,
)
DHW_MPC_PARAMS = DHWMPCParameters(
    P_dhw_max=3.0,
    delta_P_dhw_max=1.0,
    T_dhw_min=50.0,
    T_legionella=60.0,
    legionella_period_steps=168,
    legionella_duration_steps=1,
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
    """Convex MPC must satisfy power and ramp-rate constraints; no comfort violation expected."""
    model = ThermalModel(THERMAL_PARAMS)
    controller = MPCController(ufh_model=model, params=MPC_PARAMS)
    prev_u = 0.5
    sol = controller.solve(
        initial_ufh_state_c=np.array([20.8, 24.0]),
        ufh_forecast=_feasible_forecast(),
        previous_p_ufh_kw=prev_u,
    )

    assert not sol.used_fallback, "Convex solver must be used for a feasible problem."
    assert sol.ufh_control_sequence_kw.shape == (MPC_PARAMS.horizon_steps,)
    assert sol.predicted_states_c.shape == (MPC_PARAMS.horizon_steps + 1, 2)

    # Power bounds (hard)
    assert np.all(sol.ufh_control_sequence_kw >= -1e-5)
    assert np.all(sol.ufh_control_sequence_kw <= MPC_PARAMS.P_max + 1e-5)

    # Ramp-rate (hard, generous tolerance for OSQP floating-point residuals)
    deltas = np.diff(np.concatenate([[prev_u], sol.ufh_control_sequence_kw]))
    assert np.all(np.abs(deltas) <= MPC_PARAMS.delta_P_max + 1e-5)

    # No comfort violation expected in the mild scenario
    assert sol.max_ufh_comfort_violation_c < 0.01
    assert sol.first_ufh_control_kw > 0.0


def test_mpc_prefers_cheap_hours() -> None:
    """With equal comfort, the MPC should concentrate heating in the cheapest slot."""
    model = ThermalModel(THERMAL_PARAMS)
    controller = MPCController(ufh_model=model, params=MPC_PARAMS)
    prices = np.array([0.40, 0.38, 0.35, 0.10, 0.35, 0.38, 0.40, 0.40])
    forecast = ForecastHorizon(
        outdoor_temperature_c=np.full(8, 10.0),
        gti_w_per_m2=np.zeros(8),
        internal_gains_kw=np.full(8, 0.25),
        price_eur_per_kwh=prices,
        room_temperature_ref_c=np.full(9, 21.0),
    )
    sol = controller.solve(
        initial_ufh_state_c=np.array([20.8, 24.0]),
        ufh_forecast=forecast,
        previous_p_ufh_kw=0.5,
    )
    cheap_idx = 3
    avg_expensive = np.mean(np.delete(sol.ufh_control_sequence_kw, cheap_idx))
    assert sol.ufh_control_sequence_kw[cheap_idx] >= avg_expensive - 1e-3


def test_mpc_always_returns_solution_when_physics_prevent_comfort() -> None:
    """MPC must never raise; it should return best-effort with comfort violation reported."""
    model = ThermalModel(THERMAL_PARAMS)
    tight_params = MPCParameters(
        horizon_steps=4,
        Q_c=10.0,
        R_c=0.05,
        Q_N=15.0,
        P_max=4.0,
        delta_P_max=1.0,
        T_min=19.0,  # physics prevent this when outdoor=2°C and slab is cold
        T_max=22.5,
    )
    controller = MPCController(ufh_model=model, params=tight_params)
    # Scenario that was previously infeasible
    sol = controller.solve(
        initial_ufh_state_c=np.array([20.0, 21.0]),
        ufh_forecast=ForecastHorizon(
            outdoor_temperature_c=np.full(4, 2.0),
            gti_w_per_m2=np.zeros(4),
            internal_gains_kw=np.full(4, 0.25),
            price_eur_per_kwh=np.full(4, 0.30),
            room_temperature_ref_c=np.full(5, 21.0),
        ),
        previous_p_ufh_kw=0.5,
    )
    # Must always return a valid control sequence
    assert sol.ufh_control_sequence_kw.shape == (4,)
    assert sol.predicted_states_c.shape == (5, 2)
    # Power bounds still enforced as hard constraints
    assert np.all(sol.ufh_control_sequence_kw >= -1e-6)
    assert np.all(sol.ufh_control_sequence_kw <= tight_params.P_max + 1e-6)
    # Physics forced a violation – controller must report it
    assert sol.max_ufh_comfort_violation_c > 0.0
    # Controller must respond by requesting maximum heating
    assert sol.first_ufh_control_kw > 0.0


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate:UserWarning")
def test_unified_controller_supports_combined_ufh_and_dhw() -> None:
    """The canonical MPCController must solve the combined UFH + DHW problem."""
    ufh_model = ThermalModel(THERMAL_PARAMS)
    dhw_model = DHWModel(DHW_PARAMS)
    controller = MPCController(
        ufh_model=ufh_model,
        dhw_model=dhw_model,
        params=CombinedMPCParameters(
            ufh=MPC_PARAMS,
            dhw=DHW_MPC_PARAMS,
            P_hp_max=6.0,
        ),
    )
    ufh_forecast = ForecastHorizon(
        outdoor_temperature_c=np.full(MPC_PARAMS.horizon_steps, 8.0),
        gti_w_per_m2=np.zeros(MPC_PARAMS.horizon_steps),
        internal_gains_kw=np.full(MPC_PARAMS.horizon_steps, 0.25),
        price_eur_per_kwh=np.full(MPC_PARAMS.horizon_steps, 0.30),
        room_temperature_ref_c=np.full(MPC_PARAMS.horizon_steps + 1, 21.0),
        pv_kw=np.full(MPC_PARAMS.horizon_steps, 0.5),
    )
    dhw_forecast = DHWForecastHorizon(
        v_tap_m3_per_h=np.full(MPC_PARAMS.horizon_steps, 0.01),
        t_mains_c=np.full(MPC_PARAMS.horizon_steps, 10.0),
        t_amb_c=np.full(MPC_PARAMS.horizon_steps, 20.0),
        legionella_required=np.zeros(MPC_PARAMS.horizon_steps, dtype=bool),
    )

    sol = controller.solve(
        initial_ufh_state_c=np.array([20.8, 24.0]),
        ufh_forecast=ufh_forecast,
        initial_dhw_state_c=np.array([55.0, 45.0]),
        dhw_forecast=dhw_forecast,
        previous_p_ufh_kw=0.5,
    )

    assert sol.ufh_control_sequence_kw.shape == (MPC_PARAMS.horizon_steps,)
    assert sol.dhw_control_sequence_kw.shape == (MPC_PARAMS.horizon_steps,)
    assert sol.predicted_states_c.shape == (MPC_PARAMS.horizon_steps + 1, 4)
    assert np.all(sol.ufh_control_sequence_kw >= -1e-6)
    assert np.all(sol.dhw_control_sequence_kw >= -1e-6)
    assert np.all(sol.ufh_control_sequence_kw + sol.dhw_control_sequence_kw <= 6.0 + 1e-5)

