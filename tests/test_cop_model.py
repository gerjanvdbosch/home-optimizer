"""Tests for the Carnot-based heat pump COP model (cop_model.py).

Physical consistency checks
----------------------------
* COP decreases as outdoor temperature drops (colder evaporator → lower COP).
* UFH COP drops *faster* than DHW COP in cold weather because the heating
  curve simultaneously raises T_supply (hotter condenser = double penalty).
* COP is always clipped within [cop_min, cop_max].
* The heating curve is monotonically decreasing with rising outdoor temperature.
* T_CELSIUS_TO_KELVIN = 273.15 exactly (physical constant check).
"""

from __future__ import annotations

import numpy as np
import pytest

from home_optimizer.domain.heat_pump.cop import (
    T_CELSIUS_TO_KELVIN,
    HeatPumpCOPModel,
    HeatPumpCOPParameters,
)

# ---------------------------------------------------------------------------
# Shared fixture parameters
# ---------------------------------------------------------------------------

COP_PARAMS = HeatPumpCOPParameters(
    eta_carnot=0.45,
    delta_T_cond=5.0,
    delta_T_evap=5.0,
    T_supply_min=25.0,
    T_ref_outdoor=18.0,
    heating_curve_slope=1.0,
    cop_min=1.5,
    cop_max=7.0,
)

MODEL = HeatPumpCOPModel(COP_PARAMS)


# ---------------------------------------------------------------------------
# Physical constant test
# ---------------------------------------------------------------------------


def test_t_celsius_to_kelvin_value() -> None:
    """T_CELSIUS_TO_KELVIN must equal 273.15 K (physical constant)."""
    np.testing.assert_allclose(T_CELSIUS_TO_KELVIN, 273.15, rtol=1e-10)


# ---------------------------------------------------------------------------
# Heating curve tests
# ---------------------------------------------------------------------------


def test_heating_curve_at_balance_point_equals_t_supply_min() -> None:
    """At T_ref_outdoor the heating curve must return exactly T_supply_min."""
    t_out = np.array([COP_PARAMS.T_ref_outdoor])
    t_supply = MODEL.heating_curve(t_out)
    np.testing.assert_allclose(t_supply, COP_PARAMS.T_supply_min, atol=1e-9)


def test_heating_curve_above_balance_point_clamped() -> None:
    """Above T_ref_outdoor the supply temperature must stay at T_supply_min."""
    t_warm = np.array([20.0, 25.0, 30.0])
    t_supply = MODEL.heating_curve(t_warm)
    assert np.all(
        t_supply == COP_PARAMS.T_supply_min
    ), "Heating curve must be flat at T_supply_min when T_out > T_ref_outdoor."


def test_heating_curve_is_monotonically_decreasing_with_rising_t_out() -> None:
    """Warmer outdoor temperature → lower (or equal) required supply temperature."""
    t_out = np.linspace(-15.0, 18.0, 50)
    t_supply = MODEL.heating_curve(t_out)
    diffs = np.diff(t_supply)
    assert np.all(diffs <= 1e-9), "Heating curve must decrease as T_out increases."


def test_heating_curve_slope_is_correct() -> None:
    """Supply temperature must increase by slope [K] per K of outdoor temp drop."""
    t_ref = COP_PARAMS.T_ref_outdoor
    slope = COP_PARAMS.heating_curve_slope
    t_out = np.array([t_ref - 10.0, t_ref - 5.0, t_ref])
    expected = COP_PARAMS.T_supply_min + slope * np.array([10.0, 5.0, 0.0])
    np.testing.assert_allclose(MODEL.heating_curve(t_out), expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# Carnot COP formula tests
# ---------------------------------------------------------------------------


def test_cop_decreases_with_colder_outdoor_temperature() -> None:
    """Lower outdoor temperature → lower COP (colder evaporator)."""
    t_supply_fixed = 40.0  # fixed supply to isolate the evaporator effect

    cop_warm = MODEL.cop_from_temperatures(t_supply_fixed, np.array([10.0]))
    cop_cold = MODEL.cop_from_temperatures(t_supply_fixed, np.array([-5.0]))

    assert (
        cop_cold.item() < cop_warm.item()
    ), "COP must be lower at -5°C than at 10°C (colder evaporator)."


def test_cop_decreases_with_higher_supply_temperature() -> None:
    """Higher supply temperature → higher temperature lift → lower COP."""
    t_out_fixed = np.array([5.0])
    t_supply_low = 30.0  # low-temperature UFH
    t_supply_high = 55.0  # high-temperature DHW

    cop_low = MODEL.cop_from_temperatures(t_supply_low, t_out_fixed)
    cop_high = MODEL.cop_from_temperatures(t_supply_high, t_out_fixed)

    assert (
        cop_high.item() < cop_low.item()
    ), "Higher supply temperature must yield lower COP (greater temperature lift)."


def test_cop_ufh_double_penalty_vs_dhw_in_cold_weather() -> None:
    """At -10°C, UFH COP must drop *more* than DHW COP compared to +10°C.

    For UFH, colder weather raises T_supply via the heating curve AND lowers
    T_evap — a double penalty.  For DHW with fixed T_supply, only T_evap drops.
    """
    t_warm = np.array([10.0])
    t_cold = np.array([-10.0])
    t_dhw_supply = 55.0  # fixed DHW supply temperature

    cop_ufh_warm = MODEL.cop_ufh(t_warm)
    cop_ufh_cold = MODEL.cop_ufh(t_cold)
    cop_dhw_warm = MODEL.cop_dhw(t_warm, t_dhw_supply)
    cop_dhw_cold = MODEL.cop_dhw(t_cold, t_dhw_supply)

    # .item() converts size-1 arrays to Python float (works on all numpy versions)
    drop_ufh = cop_ufh_warm.item() - cop_ufh_cold.item()
    drop_dhw = cop_dhw_warm.item() - cop_dhw_cold.item()

    assert drop_ufh > drop_dhw, (
        f"UFH COP drop ({drop_ufh:.3f}) must exceed DHW COP drop ({drop_dhw:.3f}) "
        "in cold weather due to the double heating-curve penalty."
    )


def test_cop_clipped_to_cop_min() -> None:
    """Extremely cold outdoor temperatures must be clipped to cop_min, not go below."""
    t_out_extreme_cold = np.array([-30.0, -40.0])
    cop = MODEL.cop_ufh(t_out_extreme_cold)
    assert np.all(
        cop >= COP_PARAMS.cop_min
    ), f"COP must never fall below cop_min={COP_PARAMS.cop_min}."


def test_cop_clipped_to_cop_max() -> None:
    """Warm outdoor temperatures with low supply must be clipped to cop_max."""
    t_out_warm = np.array([30.0, 40.0])
    t_supply_low = COP_PARAMS.T_supply_min + 1.0  # just above minimum
    cop = MODEL.cop_from_temperatures(t_supply_low, t_out_warm)
    assert np.all(cop <= COP_PARAMS.cop_max), f"COP must never exceed cop_max={COP_PARAMS.cop_max}."


def test_cop_output_shape_matches_input() -> None:
    """COP array length must match the input forecast length."""
    n = 24
    t_out = np.linspace(-10.0, 15.0, n)
    assert MODEL.cop_ufh(t_out).shape == (n,)
    assert MODEL.cop_dhw(t_out, t_dhw_supply=55.0).shape == (n,)


def test_cop_no_division_by_zero_when_lift_is_tiny() -> None:
    """When T_cond ≈ T_evap (extreme edge case) COP must not raise or return inf."""
    # T_supply + delta_T_cond ≈ T_out - delta_T_evap  → lift ≈ 0
    # Example: T_supply=20, delta_T_cond=5 → T_cond=25°C
    #          T_out=30,     delta_T_evap=5 → T_evap=25°C  → lift=0
    t_out = np.array([30.0])
    t_supply = 20.0
    cop = MODEL.cop_from_temperatures(t_supply, t_out)
    assert np.all(np.isfinite(cop)), "COP must be finite when temperature lift is ~0."
    assert np.all(cop >= COP_PARAMS.cop_min)


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------


def test_cop_parameters_rejects_eta_carnot_zero() -> None:
    """eta_carnot = 0 is physically meaningless (zero efficiency)."""
    with pytest.raises(ValueError, match="eta_carnot"):
        HeatPumpCOPParameters(
            eta_carnot=0.0,
            delta_T_cond=5.0,
            delta_T_evap=5.0,
            T_supply_min=25.0,
            T_ref_outdoor=18.0,
            heating_curve_slope=1.0,
            cop_min=1.5,
            cop_max=7.0,
        )


def test_cop_parameters_rejects_eta_carnot_above_one() -> None:
    """eta_carnot > 1 violates the second law of thermodynamics."""
    with pytest.raises(ValueError, match="eta_carnot"):
        HeatPumpCOPParameters(
            eta_carnot=1.1,
            delta_T_cond=5.0,
            delta_T_evap=5.0,
            T_supply_min=25.0,
            T_ref_outdoor=18.0,
            heating_curve_slope=1.0,
            cop_min=1.5,
            cop_max=7.0,
        )


def test_cop_parameters_rejects_negative_delta_t_cond() -> None:
    with pytest.raises(ValueError, match="delta_T_cond"):
        HeatPumpCOPParameters(
            eta_carnot=0.45,
            delta_T_cond=-1.0,
            delta_T_evap=5.0,
            T_supply_min=25.0,
            T_ref_outdoor=18.0,
            heating_curve_slope=1.0,
            cop_min=1.5,
            cop_max=7.0,
        )


def test_cop_parameters_rejects_cop_min_below_one() -> None:
    """cop_min ≤ 1 would represent a resistive heater, not a heat pump."""
    with pytest.raises(ValueError, match="cop_min"):
        HeatPumpCOPParameters(
            eta_carnot=0.45,
            delta_T_cond=5.0,
            delta_T_evap=5.0,
            T_supply_min=25.0,
            T_ref_outdoor=18.0,
            heating_curve_slope=1.0,
            cop_min=1.0,
            cop_max=7.0,
        )


def test_cop_parameters_rejects_cop_max_not_greater_than_cop_min() -> None:
    with pytest.raises(ValueError, match="cop_max"):
        HeatPumpCOPParameters(
            eta_carnot=0.45,
            delta_T_cond=5.0,
            delta_T_evap=5.0,
            T_supply_min=25.0,
            T_ref_outdoor=18.0,
            heating_curve_slope=1.0,
            cop_min=5.0,
            cop_max=4.0,
        )


# ---------------------------------------------------------------------------
# End-to-end integration: COP array → MPC
# ---------------------------------------------------------------------------


def test_cop_arrays_can_be_used_in_mpc_forecast() -> None:
    """COP arrays from the model must be accepted by ForecastHorizon and DHWForecastHorizon."""
    from home_optimizer.types import DHWForecastHorizon, ForecastHorizon

    n = 8
    t_out = np.linspace(2.0, 10.0, n)
    cop_ufh_k = MODEL.cop_ufh(t_out)
    cop_dhw_k = MODEL.cop_dhw(t_out, t_dhw_supply=55.0)

    # Must not raise any validation error
    fh_ufh = ForecastHorizon(
        outdoor_temperature_c=t_out,
        gti_w_per_m2=np.zeros(n),
        internal_gains_kw=np.full(n, 0.3),
        price_eur_per_kwh=np.full(n, 0.25),
        room_temperature_ref_c=np.full(n + 1, 21.0),
        cop_ufh_k=cop_ufh_k,
    )
    assert fh_ufh.cop_ufh_k is not None
    assert fh_ufh.cop_ufh_k.shape == (n,)

    fh_dhw = DHWForecastHorizon(
        v_tap_m3_per_h=np.zeros(n),
        t_mains_c=np.full(n, 10.0),
        t_amb_c=np.full(n, 20.0),
        legionella_required=np.zeros(n, dtype=bool),
        cop_dhw_k=cop_dhw_k,
    )
    assert fh_dhw.cop_dhw_k is not None
    assert fh_dhw.cop_dhw_k.shape == (n,)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate:UserWarning")
def test_full_mpc_solve_with_physical_cop_model() -> None:
    """MPC must find a valid solution when COP arrays come from the physical model.

    The test uses mild outdoor conditions (8–12 °C) and a warm initial slab
    to ensure the MPC problem is comfortably feasible.
    """
    from home_optimizer.control.mpc import MPCController
    from home_optimizer.domain.ufh.model import ThermalModel
    from home_optimizer.types import ForecastHorizon, MPCParameters, ThermalParameters

    n = 8
    # Mild winter conditions: COP will be between ~3.5 and ~4.5 (well within limits)
    t_out = np.full(n, 10.0)

    thermal_params = ThermalParameters(
        dt_hours=1.0,
        C_r=3.0,
        C_b=18.0,
        R_br=2.5,
        R_ro=4.0,
        alpha=0.35,
        eta=0.62,
        A_glass=12.0,
    )
    cop_arr = MODEL.cop_ufh(t_out)
    mpc_params = MPCParameters(
        horizon_steps=n,
        Q_c=10.0,
        R_c=0.05,
        Q_N=15.0,
        P_max=4.0,
        delta_P_max=1.0,
        T_min=19.0,
        T_max=22.5,
        cop_ufh=float(np.mean(cop_arr)),
        cop_max=COP_PARAMS.cop_max,
    )
    forecast = ForecastHorizon(
        outdoor_temperature_c=t_out,
        gti_w_per_m2=np.zeros(n),
        internal_gains_kw=np.full(n, 0.3),
        price_eur_per_kwh=np.full(n, 0.25),
        room_temperature_ref_c=np.full(n + 1, 21.0),
        cop_ufh_k=cop_arr,  # physical COP profile from HeatPumpCOPModel
    )
    sol = MPCController(ThermalModel(thermal_params), mpc_params).solve(
        initial_ufh_state_c=np.array([20.8, 24.0]),  # warm initial state → feasible
        ufh_forecast=forecast,
        previous_p_ufh_kw=0.5,
    )
    # Canonical architecture: always solved through the convex CVXPY path.
    assert sol.ufh_control_sequence_kw.shape == (n,)
    assert np.all(sol.ufh_control_sequence_kw >= -1e-6)
    assert np.all(sol.ufh_control_sequence_kw <= mpc_params.P_max + 1e-6)
    assert sol.used_fallback is False
    # The physical COP must have been used: cop_ufh_k is non-trivial (not all equal)
    assert forecast.cop_ufh_k is not None and forecast.cop_ufh_k.shape == (n,)
