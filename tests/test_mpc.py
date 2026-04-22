"""Unit tests for the MPC controller: feasibility, constraints, and properties."""

from __future__ import annotations

import numpy as np
import pytest

import home_optimizer.control.mpc as mpc_module
from home_optimizer.application.optimizer import Optimizer, RunRequest, validate_run_request_physics
from home_optimizer.control.mpc import MPCController
from home_optimizer.domain.dhw.model import DHWModel
from home_optimizer.domain.heat_pump.cop import HeatPumpCOPModel, HeatPumpCOPParameters
from home_optimizer.domain.ufh.model import ThermalModel
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
# Shared COP model fixture — used by tests that exercise the physical model
# ---------------------------------------------------------------------------

#: Physical Carnot COP parameters representative of a Dutch ASHP installation.
COP_PARAMS = HeatPumpCOPParameters(
    eta_carnot=0.45,
    delta_T_cond=5.0,
    delta_T_evap=5.0,
    T_supply_min=28.0,
    T_ref_outdoor=18.0,
    heating_curve_slope=1.0,
    cop_min=1.5,
    cop_max=7.0,
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
    # §14.1: scalar COP for UFH heat-pump mode
    cop_ufh=3.5,
    cop_max=7.0,
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
    # §14.1: scalar COP for DHW heat-pump mode (higher lift → lower COP than UFH)
    cop_dhw=3.0,
    cop_max=7.0,
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
    """Soft constraints keep the convex MPC feasible even when comfort cannot be met."""
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
        cop_ufh=3.5,
        cop_max=7.0,
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


def test_mpc_fail_fast_when_cvxpy_backend_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The controller must raise explicitly instead of switching to a hidden fallback."""
    model = ThermalModel(THERMAL_PARAMS)
    controller = MPCController(ufh_model=model, params=MPC_PARAMS)
    monkeypatch.setattr(mpc_module, "_CVXPY_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="CVXPY is required"):
        controller.solve(
            initial_ufh_state_c=np.array([20.8, 24.0]),
            ufh_forecast=_feasible_forecast(),
            previous_p_ufh_kw=0.5,
        )


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate:UserWarning")
def test_unified_controller_supports_combined_ufh_and_dhw() -> None:
    """The canonical MPCController must solve the combined UFH + DHW problem."""
    ufh_model = ThermalModel(THERMAL_PARAMS)
    dhw_model = DHWModel(DHW_PARAMS)
    # P_hp_max_elec: maximum electrical budget for the shared heat pump.
    # With COP_UFH=3.5 and P_UFH_max=4.0 → max UFH elec ≈ 1.14 kW.
    # With COP_DHW=3.0 and P_dhw_max=3.0 → max DHW elec = 1.0 kW.
    # Setting 3.0 kW elec is non-binding, ensuring the problem is always feasible.
    controller = MPCController(
        ufh_model=ufh_model,
        dhw_model=dhw_model,
        params=CombinedMPCParameters(
            ufh=MPC_PARAMS,
            dhw=DHW_MPC_PARAMS,
            P_hp_max_elec=3.0,
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
    # §14: Verify the shared electrical budget constraint:
    #   P_UFH / COP_UFH + P_dhw / COP_dhw ≤ P_hp_max_elec
    elec_combined = (
        sol.ufh_control_sequence_kw / MPC_PARAMS.cop_ufh
        + sol.dhw_control_sequence_kw / DHW_MPC_PARAMS.cop_dhw
    )
    assert np.all(
        elec_combined <= 3.0 + 1e-5
    ), f"Electrical budget violated: max={elec_combined.max():.4f} kW > 3.0 kW"


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate:UserWarning")
def test_optimizer_does_not_preheat_dhw_above_minimum_just_because_pv_is_available() -> None:
    """Sunny surplus hours must not trigger DHW heating while T_top is still above the ``dhw_T_min``.

    This regression covers the user-facing failure mode where the convex MPC used
    free PV hours to opportunistically charge the tank even though the top layer
    was still comfortably above the minimum tap temperature. The updated DHW
    target-band penalty should keep the first control action at zero until the
    top layer actually approaches the lower comfort bound.
    """

    request = RunRequest.model_validate(
        {
            "horizon_hours": 4,
            "pv_enabled": True,
            "pv_peak_power_kw": 10.0,
            "gti_pv_forecast": [1000.0, 1000.0, 1000.0, 1000.0],
            "gti_window_forecast": [0.0, 0.0, 0.0, 0.0],
            "t_out_forecast": [15.0, 15.0, 15.0, 15.0],
            "price_buy_forecast": [0.22, 0.22, 0.22, 0.22],
            "price_sell_forecast": [0.07, 0.07, 0.07, 0.07],
            "dhw_enabled": True,
            "dhw_T_top_init": 55.0,
            "dhw_T_bot_init": 50.0,
            "dhw_T_min": 50.0,
            "dhw_v_tap_forecast": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = Optimizer().solve(request)

    assert result.solution.first_dhw_control_kw <= 1e-4
    assert result.solution.predicted_states_c[1, 2] > request.dhw_T_min


def test_optimizer_request_validation_accepts_dhw_high_tap_flow_with_exact_zoh() -> None:
    """Runtime request validation must accept DHW tap flows when exact ZOH discretisation is used."""
    req = RunRequest.model_validate(
        {
            "dt_hours": 1.0,
            "dhw_enabled": True,
            "dhw_C_top": 0.05,
            "dhw_C_bot": 0.05,
            "dhw_R_strat": 100.0,
            "dhw_R_loss": 200.0,
            "dhw_v_tap_forecast": [0.3] * 24,
        }
    )

    validate_run_request_physics(req)


def test_optimizer_request_validation_rejects_missing_dhw_tap_forecast() -> None:
    """DHW solve-context assembly must fail fast when the tap-flow forecast is absent."""
    req = RunRequest.model_validate(
        {
            "dt_hours": 1.0,
            "dhw_enabled": True,
        }
    )

    with pytest.raises(ValueError, match="dhw_v_tap_forecast is required"):
        Optimizer().solve(req)


# ---------------------------------------------------------------------------
# COP Fail-Fast validation tests (§14.1)
# ---------------------------------------------------------------------------


def test_cop_validation_ufh_rejects_cop_below_or_equal_one() -> None:
    """MPCParameters must raise ValueError when cop_ufh ≤ 1 (physically impossible)."""
    with pytest.raises(ValueError, match="cop_ufh"):
        MPCParameters(
            horizon_steps=4,
            Q_c=1.0,
            R_c=0.01,
            Q_N=1.0,
            P_max=4.0,
            delta_P_max=1.0,
            T_min=19.0,
            T_max=23.0,
            cop_ufh=1.0,  # exactly 1 is invalid (must be strictly > 1)
            cop_max=7.0,
        )


def test_cop_validation_ufh_rejects_cop_of_zero() -> None:
    """MPCParameters must raise ValueError when cop_ufh ≤ 0."""
    with pytest.raises(ValueError, match="cop_ufh"):
        MPCParameters(
            horizon_steps=4,
            Q_c=1.0,
            R_c=0.01,
            Q_N=1.0,
            P_max=4.0,
            delta_P_max=1.0,
            T_min=19.0,
            T_max=23.0,
            cop_ufh=0.0,
            cop_max=7.0,
        )


def test_cop_validation_ufh_rejects_cop_above_max() -> None:
    """MPCParameters must raise ValueError when cop_ufh > cop_max."""
    with pytest.raises(ValueError, match="cop_max"):
        MPCParameters(
            horizon_steps=4,
            Q_c=1.0,
            R_c=0.01,
            Q_N=1.0,
            P_max=4.0,
            delta_P_max=1.0,
            T_min=19.0,
            T_max=23.0,
            cop_ufh=8.0,  # exceeds cop_max
            cop_max=7.0,
        )


def test_cop_validation_dhw_rejects_cop_below_or_equal_one() -> None:
    """DHWMPCParameters must raise ValueError when cop_dhw ≤ 1."""
    with pytest.raises(ValueError, match="cop_dhw"):
        DHWMPCParameters(
            P_dhw_max=3.0,
            delta_P_dhw_max=1.0,
            T_dhw_min=50.0,
            T_legionella=60.0,
            legionella_period_steps=168,
            legionella_duration_steps=1,
            cop_dhw=0.5,  # physically impossible
            cop_max=7.0,
        )


def test_cop_validation_dhw_rejects_cop_above_max() -> None:
    """DHWMPCParameters must raise ValueError when cop_dhw > cop_max."""
    with pytest.raises(ValueError, match="cop_max"):
        DHWMPCParameters(
            P_dhw_max=3.0,
            delta_P_dhw_max=1.0,
            T_dhw_min=50.0,
            T_legionella=60.0,
            legionella_period_steps=168,
            legionella_duration_steps=1,
            cop_dhw=9.0,
            cop_max=7.0,
        )


def test_cop_time_varying_forecast_rejects_cop_below_one() -> None:
    """ForecastHorizon must raise ValueError if any cop_ufh_k value is ≤ 1."""
    n = 4
    with pytest.raises(ValueError, match="cop_ufh_k"):
        ForecastHorizon(
            outdoor_temperature_c=np.full(n, 10.0),
            gti_w_per_m2=np.zeros(n),
            internal_gains_kw=np.full(n, 0.3),
            price_eur_per_kwh=np.full(n, 0.25),
            room_temperature_ref_c=np.full(n + 1, 21.0),
            cop_ufh_k=np.array([3.5, 3.2, 0.9, 3.0]),  # 0.9 is invalid
        )


def test_cop_time_varying_forecast_valid() -> None:
    """A valid time-varying cop_ufh_k array must be accepted without error."""
    n = 4
    fh = ForecastHorizon(
        outdoor_temperature_c=np.full(n, 5.0),
        gti_w_per_m2=np.zeros(n),
        internal_gains_kw=np.full(n, 0.3),
        price_eur_per_kwh=np.full(n, 0.25),
        room_temperature_ref_c=np.full(n + 1, 21.0),
        cop_ufh_k=np.array([4.0, 3.8, 3.5, 3.2]),  # valid: all > 1
    )
    assert fh.cop_ufh_k is not None
    assert fh.cop_ufh_k.shape == (n,)


def test_mpc_cost_uses_electrical_power() -> None:
    """With COP > 1, the MPC cost must be strictly lower than with COP=1 assumption.

    A higher COP means the same thermal output costs less electricity.
    The MPC objective value must decrease as the COP increases (cheaper electricity).
    """
    model = ThermalModel(THERMAL_PARAMS)
    forecast = _feasible_forecast(n=8)

    def _solve_with_cop(cop: float) -> float:
        params = MPCParameters(
            horizon_steps=8,
            Q_c=10.0,
            R_c=0.05,
            Q_N=15.0,
            P_max=4.0,
            delta_P_max=1.0,
            T_min=19.0,
            T_max=22.5,
            cop_ufh=cop,
            cop_max=8.0,
        )
        sol = MPCController(ufh_model=model, params=params).solve(
            initial_ufh_state_c=np.array([20.8, 24.0]),
            ufh_forecast=forecast,
            previous_p_ufh_kw=0.5,
        )
        return sol.objective_value

    cost_low_cop = _solve_with_cop(cop=2.0)  # low COP → expensive electricity
    cost_high_cop = _solve_with_cop(cop=5.0)  # high COP → cheap electricity
    assert (
        cost_high_cop < cost_low_cop
    ), "Higher COP must yield lower objective (cheaper electrical energy)."


def test_optimizer_build_ufh_forecast_scales_solar_gain_with_shutter() -> None:
    """Optimizer UFH forecasts must scale solar disturbance with shutter openness."""
    horizon_steps = 4
    run_request = RunRequest.model_validate(
        {
            "horizon_hours": horizon_steps,
            "outdoor_temperature_c": 8.0,
            "t_out_forecast": [8.0] * horizon_steps,
            "gti_window_forecast": [600.0] * horizon_steps,
            "gti_pv_forecast": [0.0] * horizon_steps,
            "shutter_living_room_pct": 25.0,
            "pv_enabled": False,
        }
    )
    forecast = Optimizer._build_ufh_forecast(
        run_request,
        start_hour=0,
        cop_model=HeatPumpCOPModel(COP_PARAMS),
    )

    expected_shutter = np.full(horizon_steps, 25.0)
    np.testing.assert_allclose(forecast.shutter_pct, expected_shutter)

    fully_open_gain = THERMAL_PARAMS.A_glass * 600.0 * THERMAL_PARAMS.eta / 1000.0
    np.testing.assert_allclose(
        forecast.solar_gains_kw(THERMAL_PARAMS),
        np.full(horizon_steps, fully_open_gain * 0.25),
    )


def test_optimizer_build_ufh_forecast_prefers_explicit_shutter_forecast() -> None:
    """A real shutter forecast array must override the scalar live/manual fallback."""
    horizon_steps = 4
    run_request = RunRequest.model_validate(
        {
            "horizon_hours": horizon_steps,
            "outdoor_temperature_c": 8.0,
            "t_out_forecast": [8.0] * horizon_steps,
            "gti_window_forecast": [600.0] * horizon_steps,
            "gti_pv_forecast": [0.0] * horizon_steps,
            "shutter_living_room_pct": 100.0,
            "shutter_forecast": [100.0, 50.0, 25.0, 0.0],
            "pv_enabled": False,
        }
    )

    forecast = Optimizer._build_ufh_forecast(
        run_request,
        start_hour=0,
        cop_model=HeatPumpCOPModel(COP_PARAMS),
    )

    np.testing.assert_allclose(forecast.shutter_pct, np.array([100.0, 50.0, 25.0, 0.0]))

    fully_open_gain = THERMAL_PARAMS.A_glass * 600.0 * THERMAL_PARAMS.eta / 1000.0
    np.testing.assert_allclose(
        forecast.solar_gains_kw(THERMAL_PARAMS),
        np.array([fully_open_gain, fully_open_gain * 0.5, fully_open_gain * 0.25, 0.0]),
    )


def test_optimizer_build_ufh_forecast_includes_feed_in_prices_for_dual_tariff() -> None:
    """UFH forecast construction must propagate feed-in prices so PV surplus has an opportunity cost."""
    horizon_steps = 4
    run_request = RunRequest.model_validate(
        {
            "horizon_hours": horizon_steps,
            "outdoor_temperature_c": 8.0,
            "t_out_forecast": [8.0] * horizon_steps,
            "gti_window_forecast": [0.0] * horizon_steps,
            "gti_pv_forecast": [0.0] * horizon_steps,
            "price_config": {
                "mode": "dual",
                "high_rate_eur_per_kwh": 0.30,
                "low_rate_eur_per_kwh": 0.20,
                "feed_in_rate_eur_per_kwh": 0.07,
                "low_tariff_hours": [0, 1, 2, 3],
            },
        }
    )

    forecast = Optimizer._build_ufh_forecast(
        run_request,
        start_hour=0,
        cop_model=HeatPumpCOPModel(COP_PARAMS),
    )

    np.testing.assert_allclose(forecast.feed_in_price_eur_per_kwh, np.full(horizon_steps, 0.07))


def test_optimizer_build_ufh_forecast_rejects_short_shutter_forecast() -> None:
    """Explicit shutter forecasts must cover the full MPC horizon."""
    horizon_steps = 4
    run_request = RunRequest.model_validate(
        {
            "horizon_hours": horizon_steps,
            "outdoor_temperature_c": 8.0,
            "t_out_forecast": [8.0] * horizon_steps,
            "gti_window_forecast": [600.0] * horizon_steps,
            "gti_pv_forecast": [0.0] * horizon_steps,
            "shutter_forecast": [100.0, 50.0],
            "pv_enabled": False,
        }
    )

    with pytest.raises(ValueError, match="shutter_forecast must provide at least 4 values"):
        Optimizer._build_ufh_forecast(
            run_request,
            start_hour=0,
            cop_model=HeatPumpCOPModel(COP_PARAMS),
        )


def test_optimizer_build_dhw_forecast_prefers_explicit_tap_flow_forecast() -> None:
    """A caller-supplied DHW tap-flow forecast must override the scalar fallback disturbance."""
    horizon_steps = 4
    run_request = RunRequest.model_validate(
        {
            "horizon_hours": horizon_steps,
            "outdoor_temperature_c": 8.0,
            "t_out_forecast": [8.0] * horizon_steps,
            "dhw_v_tap_forecast": [0.0, 0.0, 0.03, 0.04],
        }
    )

    forecast = Optimizer._build_dhw_forecast(
        run_request,
        horizon_steps,
        HeatPumpCOPModel(COP_PARAMS),
    )

    np.testing.assert_allclose(forecast.v_tap_m3_per_h, np.array([0.0, 0.0, 0.03, 0.04]))


def test_optimizer_build_dhw_forecast_rejects_short_tap_flow_forecast() -> None:
    """Explicit DHW tap-flow forecasts must cover the full MPC horizon."""
    run_request = RunRequest.model_validate(
        {
            "horizon_hours": 4,
            "outdoor_temperature_c": 8.0,
            "t_out_forecast": [8.0, 8.0, 8.0, 8.0],
            "dhw_v_tap_forecast": [0.02, 0.01],
        }
    )

    with pytest.raises(ValueError, match="dhw_v_tap_forecast must provide at least 4 values"):
        Optimizer._build_dhw_forecast(
            run_request,
            4,
            HeatPumpCOPModel(COP_PARAMS),
        )


# ---------------------------------------------------------------------------
# Combined UFH + DHW with physical Carnot COP arrays (§14.1 end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate:UserWarning")
def test_combined_mpc_with_carnot_cop_arrays() -> None:
    """Combined UFH + DHW MPC with time-varying Carnot COP arrays (§14.1).

    This test exercises the full physical COP path:
    1. Compute cop_ufh_k and cop_dhw_k from ``HeatPumpCOPModel``.
    2. Inject them into the forecast objects.
    3. Solve the combined MPC.
    4. Verify the shared electrical budget constraint using the *actual* COP
       arrays that the MPC used — not scalar stand-ins.

    Physical expectations:
    - DHW COP is lower than UFH COP at the same outdoor temperature, because
      the DHW supply temperature (≈ T_dhw_min = 50 °C) is higher than the
      UFH supply temperature from the heating curve (~34 °C at 8 °C outdoor).
    - The electrical constraint P_UFH/COP_UFH + P_dhw/COP_dhw ≤ P_hp_max_elec
      must be satisfied for every time step.
    """
    cop_model = HeatPumpCOPModel(COP_PARAMS)

    n = MPC_PARAMS.horizon_steps
    t_out = np.full(n, 8.0)  # mild winter conditions [°C]

    # §14.1 – pre-compute time-varying COP arrays before passing to the MPC.
    cop_ufh_k = cop_model.cop_ufh(t_out)
    cop_dhw_k = cop_model.cop_dhw(t_out, t_dhw_supply=DHW_MPC_PARAMS.T_dhw_min)

    # UFH COP > DHW COP: heating curve raises T_supply for UFH, but UFH supply is
    # still much lower than T_dhw_min.  Both should be well above cop_min.
    assert np.all(
        cop_ufh_k > cop_dhw_k
    ), "DHW requires higher T_supply → higher lift → lower COP than UFH at same T_out."
    assert np.all(cop_ufh_k >= COP_PARAMS.cop_min)
    assert np.all(cop_dhw_k >= COP_PARAMS.cop_min)

    # Derive scalar COP values for MPCParameters (Fail-Fast bounds check).
    # Use np.min to always stay within [cop_min, cop_max] regardless of horizon.
    cop_ufh_scalar = float(np.min(cop_ufh_k))
    cop_dhw_scalar = float(np.min(cop_dhw_k))

    mpc_params_with_cop = MPCParameters(
        horizon_steps=MPC_PARAMS.horizon_steps,
        Q_c=MPC_PARAMS.Q_c,
        R_c=MPC_PARAMS.R_c,
        Q_N=MPC_PARAMS.Q_N,
        P_max=MPC_PARAMS.P_max,
        delta_P_max=MPC_PARAMS.delta_P_max,
        T_min=MPC_PARAMS.T_min,
        T_max=MPC_PARAMS.T_max,
        cop_ufh=cop_ufh_scalar,  # from Carnot model — no magic number
        cop_max=COP_PARAMS.cop_max,
    )
    dhw_mpc_params_with_cop = DHWMPCParameters(
        P_dhw_max=DHW_MPC_PARAMS.P_dhw_max,
        delta_P_dhw_max=DHW_MPC_PARAMS.delta_P_dhw_max,
        T_dhw_min=DHW_MPC_PARAMS.T_dhw_min,
        T_legionella=DHW_MPC_PARAMS.T_legionella,
        legionella_period_steps=DHW_MPC_PARAMS.legionella_period_steps,
        legionella_duration_steps=DHW_MPC_PARAMS.legionella_duration_steps,
        cop_dhw=cop_dhw_scalar,  # from Carnot model — no magic number
        cop_max=COP_PARAMS.cop_max,
    )

    # P_hp_max_elec: generous budget so the constraint does not cut into comfort.
    # At T_out=8°C: COP_UFH ≈ 3.5, COP_DHW ≈ 2.5.
    # P_UFH_max_elec ≈ 4.5/3.5 ≈ 1.3 kW; P_dhw_max_elec ≈ 3.0/2.5 ≈ 1.2 kW → ~2.5 total.
    # Setting 4.0 kW keeps the budget non-binding; feasibility is guaranteed.
    p_hp_max_elec = 4.0  # [kW elec]

    controller = MPCController(
        ufh_model=ThermalModel(THERMAL_PARAMS),
        dhw_model=DHWModel(DHW_PARAMS),
        params=CombinedMPCParameters(
            ufh=mpc_params_with_cop,
            dhw=dhw_mpc_params_with_cop,
            P_hp_max_elec=p_hp_max_elec,
        ),
    )

    ufh_forecast = ForecastHorizon(
        outdoor_temperature_c=t_out,
        gti_w_per_m2=np.zeros(n),
        internal_gains_kw=np.full(n, 0.25),
        price_eur_per_kwh=np.full(n, 0.30),
        room_temperature_ref_c=np.full(n + 1, 21.0),
        cop_ufh_k=cop_ufh_k,  # §14.1: time-varying COP from physical model
    )
    dhw_forecast = DHWForecastHorizon(
        v_tap_m3_per_h=np.full(n, 0.01),
        t_mains_c=np.full(n, 10.0),
        t_amb_c=np.full(n, 20.0),
        legionella_required=np.zeros(n, dtype=bool),
        cop_dhw_k=cop_dhw_k,  # §14.1: time-varying COP from physical model
    )

    sol = controller.solve(
        initial_ufh_state_c=np.array([20.8, 24.0]),
        ufh_forecast=ufh_forecast,
        initial_dhw_state_c=np.array([55.0, 45.0]),
        dhw_forecast=dhw_forecast,
        previous_p_ufh_kw=0.5,
    )

    # ── Shape and sign checks ─────────────────────────────────────────
    assert sol.ufh_control_sequence_kw.shape == (n,)
    assert sol.dhw_control_sequence_kw.shape == (n,)
    assert sol.predicted_states_c.shape == (n + 1, 4)
    assert np.all(sol.ufh_control_sequence_kw >= -1e-6)
    assert np.all(sol.dhw_control_sequence_kw >= -1e-6)

    # ── §14 Shared electrical budget constraint ───────────────────────
    # Verify using the *actual* time-varying COP arrays used by the MPC,
    # not the scalar stand-ins from MPCParameters.
    elec_ufh = sol.ufh_control_sequence_kw / cop_ufh_k  # [kW elec], shape (N,)
    elec_dhw = sol.dhw_control_sequence_kw / cop_dhw_k  # [kW elec], shape (N,)
    elec_combined = elec_ufh + elec_dhw  # total electrical demand
    assert np.all(elec_combined <= p_hp_max_elec + 1e-4), (
        f"Shared electrical budget violated: max={elec_combined.max():.4f} kW "
        f"> {p_hp_max_elec} kW (using time-varying Carnot COP arrays)"
    )
