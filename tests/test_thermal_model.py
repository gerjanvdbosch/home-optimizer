"""Unit tests for the thermal model: equations, matrices, and system properties."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from home_optimizer.domain.ufh.model import ThermalModel, solar_gain_kw
from home_optimizer.types import ForecastHorizon, ThermalParameters


@pytest.fixture
def params() -> ThermalParameters:
    return ThermalParameters(
        dt_hours=1.0,
        C_r=3.0,
        C_b=18.0,
        R_br=2.5,
        R_ro=4.0,
        alpha=0.35,
        eta=0.62,
        A_glass=12.0,
    )


# ── solar_gain_kw ─────────────────────────────────────────────────────────────


def test_solar_gain_formula() -> None:
    # Q = A * GTI * η / 1000 = 10 * 800 * 0.6 / 1000 = 4.8 kW
    assert solar_gain_kw(800.0, glass_area_m2=10.0, transmittance=0.6) == pytest.approx(4.8)


def test_solar_gain_zero_irradiance() -> None:
    assert solar_gain_kw(0.0, 12.0, 0.62) == pytest.approx(0.0)


def test_solar_gain_array_input() -> None:
    gti = np.array([0.0, 400.0, 800.0])
    result = solar_gain_kw(gti, glass_area_m2=10.0, transmittance=0.5)
    np.testing.assert_allclose(result, [0.0, 2.0, 4.0])


def test_forecast_solar_gain_scales_with_shutter_position(params: ThermalParameters) -> None:
    """ForecastHorizon must scale Q_solar with the living-room shutter opening [%]."""
    forecast = ForecastHorizon(
        outdoor_temperature_c=np.full(3, 8.0),
        gti_w_per_m2=np.array([400.0, 400.0, 400.0]),
        internal_gains_kw=np.full(3, 0.3),
        price_eur_per_kwh=np.full(3, 0.25),
        room_temperature_ref_c=np.full(4, 21.0),
        shutter_pct=np.array([100.0, 50.0, 0.0]),
    )

    gains = forecast.solar_gains_kw(params)
    fully_open_gain = params.A_glass * forecast.gti_w_per_m2[0] * params.eta / 1000.0

    np.testing.assert_allclose(gains, [fully_open_gain, fully_open_gain * 0.5, 0.0])


# ── state matrices ────────────────────────────────────────────────────────────


def test_state_matrix_A_matches_specification(params: ThermalParameters) -> None:
    model = ThermalModel(params)
    A, _, _ = model.state_matrices()
    inv_CrRbr = 1.0 / (params.C_r * params.R_br)
    inv_CrRro = 1.0 / (params.C_r * params.R_ro)
    inv_CbRbr = 1.0 / (params.C_b * params.R_br)
    a_cont = np.array(
        [
            [-(inv_CrRbr + inv_CrRro), inv_CrRbr],
            [inv_CbRbr, -inv_CbRbr],
        ]
    )
    expected = expm(a_cont * params.dt_hours)
    np.testing.assert_allclose(A, expected)


def test_state_matrix_B_matches_specification(params: ThermalParameters) -> None:
    model = ThermalModel(params)
    _, B, _ = model.state_matrices()
    inv_CrRbr = 1.0 / (params.C_r * params.R_br)
    inv_CrRro = 1.0 / (params.C_r * params.R_ro)
    inv_CbRbr = 1.0 / (params.C_b * params.R_br)
    a_cont = np.array(
        [
            [-(inv_CrRbr + inv_CrRro), inv_CrRbr],
            [inv_CbRbr, -inv_CbRbr],
        ]
    )
    b_cont = np.array([[0.0], [1.0 / params.C_b]])
    augmented = np.zeros((3, 3), dtype=float)
    augmented[:2, :2] = a_cont
    augmented[:2, 2:3] = b_cont
    expected = expm(augmented * params.dt_hours)[:2, 2:3]
    np.testing.assert_allclose(B, expected)


def test_state_matrix_E_matches_specification(params: ThermalParameters) -> None:
    model = ThermalModel(params)
    _, _, E = model.state_matrices()
    inv_CrRbr = 1.0 / (params.C_r * params.R_br)
    inv_CrRro = 1.0 / (params.C_r * params.R_ro)
    inv_CbRbr = 1.0 / (params.C_b * params.R_br)
    a_cont = np.array(
        [
            [-(inv_CrRbr + inv_CrRro), inv_CrRbr],
            [inv_CbRbr, -inv_CbRbr],
        ]
    )
    e_cont = np.array(
        [
            [inv_CrRro, params.alpha / params.C_r, 1.0 / params.C_r],
            [0.0, (1 - params.alpha) / params.C_b, 0.0],
        ]
    )
    augmented = np.zeros((5, 5), dtype=float)
    augmented[:2, :2] = a_cont
    augmented[:2, 2:] = e_cont
    expected = expm(augmented * params.dt_hours)[:2, 2:]
    np.testing.assert_allclose(E, expected)


# ── simulation consistency ────────────────────────────────────────────────────


def test_step_and_matrix_form_agree(params: ThermalParameters) -> None:
    model = ThermalModel(params)
    x = np.array([20.5, 22.0])
    u, d = 1.4, np.array([5.0, 0.85, 0.3])

    direct = model.step(x, u, d[0], d[1], d[2])
    matrix = model.step_with_disturbance_vector(x, u, d)
    np.testing.assert_allclose(direct, matrix)


def test_room_can_warm_without_solar_for_realistic_ufh_house() -> None:
    params = ThermalParameters(
        dt_hours=1.0,
        C_r=6.0,
        C_b=10.0,
        R_br=1.0,
        R_ro=10.0,
        alpha=0.25,
        eta=0.55,
        A_glass=7.5,
    )
    model = ThermalModel(params)
    x = np.array([20.5, 22.5])

    x_next = model.step(
        state=x,
        control_kw=1.5,
        outdoor_temperature_c=6.0,
        solar_gain_kw_value=0.0,
        internal_gain_kw=0.3,
    )

    assert (
        x_next[0] > x[0]
    ), "A warmer slab plus UFH should be able to raise room temperature without solar gains."


# ── observability and controllability ────────────────────────────────────────


def test_system_is_fully_observable_and_controllable(params: ThermalParameters) -> None:
    model = ThermalModel(params)
    assert model.observability_rank() == 2, "System must be fully observable (rank 2)"
    assert model.controllability_rank() == 2, "System must be fully controllable (rank 2)"


# ── energy balance (§3 / §16.3 required test) ────────────────────────────────


def test_ufh_energy_balance(params: ThermalParameters) -> None:
    """UFH continuous energy balance must match the normatieve stored-energy derivative."""
    model = ThermalModel(params)
    p = params

    x = np.array([19.0, 23.0], dtype=float)
    control_kw = 1.5
    t_out = 5.0
    q_solar = 0.8
    q_int = 0.3

    for _ in range(10):
        dxdt = model.continuous_derivative(
            state=x,
            control_kw=control_kw,
            outdoor_temperature_c=t_out,
            solar_gain_kw_value=q_solar,
            internal_gain_kw=q_int,
        )
        lhs = p.C_r * dxdt[0] + p.C_b * dxdt[1]
        expected_power = control_kw - (x[0] - t_out) / p.R_ro + q_solar + q_int

        np.testing.assert_allclose(
            lhs,
            expected_power,
            rtol=1e-10,
            atol=1e-10,
            err_msg="UFH continuous energy balance violated.",
        )
        x = model.step(
            state=x,
            control_kw=control_kw,
            outdoor_temperature_c=t_out,
            solar_gain_kw_value=q_solar,
            internal_gain_kw=q_int,
        )


def test_forward_euler_reference_path_remains_available(params: ThermalParameters) -> None:
    """UFH should keep an explicit Euler reference path, but not as runtime default."""
    model = ThermalModel(params)
    discrete = model.forward_euler_discrete_model()
    dt = params.dt_hours
    a_br = dt / (params.C_r * params.R_br)
    a_ro = dt / (params.C_r * params.R_ro)
    b_br = dt / (params.C_b * params.R_br)
    expected = np.array([[1 - a_br - a_ro, a_br], [b_br, 1 - b_br]])
    np.testing.assert_allclose(discrete.A, expected)


# ── lambda constant (§8.4 / §16.3 required test) ──────────────────────────────


def test_lambda_constant() -> None:
    """λ = ρ·c_p computed from physical constants equals 1.1628 kWh/(m³·K).

    Verifies §8.4: λ = 1000 kg/m³ × 4186 J/(kg·K) / 3_600_000 J/kWh.
    Tolerance: rtol=1e-4.
    """
    rho_kg_per_m3 = 1000.0  # water density [kg/m³]
    cp_j_per_kg_k = 4186.0  # specific heat capacity of water [J/(kg·K)]
    j_per_kwh = 3_600_000.0  # joules per kWh

    lambda_calc = rho_kg_per_m3 * cp_j_per_kg_k / j_per_kwh

    np.testing.assert_allclose(
        lambda_calc,
        1.1628,
        rtol=1e-4,
        err_msg="λ = ρ·c_p does not equal 1.1628 kWh/(m³·K) within tolerance.",
    )
