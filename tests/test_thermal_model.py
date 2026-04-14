"""Unit tests for the thermal model: equations, matrices, and system properties."""

from __future__ import annotations

import numpy as np
import pytest

from home_optimizer.thermal_model import ThermalModel, solar_gain_kw
from home_optimizer.types import ThermalParameters


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


# ── state matrices ────────────────────────────────────────────────────────────


def test_state_matrix_A_matches_specification(params: ThermalParameters) -> None:
    model = ThermalModel(params)
    A, _, _ = model.state_matrices()
    dt = params.dt_hours
    a_br = dt / (params.C_r * params.R_br)
    a_ro = dt / (params.C_r * params.R_ro)
    b_br = dt / (params.C_b * params.R_br)
    expected = np.array([[1 - a_br - a_ro, a_br], [b_br, 1 - b_br]])
    np.testing.assert_allclose(A, expected)


def test_state_matrix_B_matches_specification(params: ThermalParameters) -> None:
    model = ThermalModel(params)
    _, B, _ = model.state_matrices()
    expected = np.array([[0.0], [params.dt_hours / params.C_b]])
    np.testing.assert_allclose(B, expected)


def test_state_matrix_E_matches_specification(params: ThermalParameters) -> None:
    model = ThermalModel(params)
    _, _, E = model.state_matrices()
    dt = params.dt_hours
    a_ro = dt / (params.C_r * params.R_ro)
    expected = np.array(
        [
            [a_ro, params.alpha * dt / params.C_r, dt / params.C_r],
            [0.0, (1 - params.alpha) * dt / params.C_b, 0.0],
        ]
    )
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
        internal_gain_kw=0.30,
    )

    assert x_next[0] > x[0], "A warmer slab plus UFH should be able to raise room temperature without solar gains."


# ── observability and controllability ────────────────────────────────────────


def test_system_is_fully_observable_and_controllable(params: ThermalParameters) -> None:
    model = ThermalModel(params)
    assert model.observability_rank() == 2, "System must be fully observable (rank 2)"
    assert model.controllability_rank() == 2, "System must be fully controllable (rank 2)"
