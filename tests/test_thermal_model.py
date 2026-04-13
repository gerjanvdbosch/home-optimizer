import numpy as np
import pytest

from home_optimizer.thermal_model import ThermalModel, solar_gain_kw
from home_optimizer.types import ThermalParameters


@pytest.fixture
def thermal_parameters() -> ThermalParameters:
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


def test_solar_gain_kw_matches_physical_formula() -> None:
    assert solar_gain_kw(800.0, glass_area_m2=10.0, transmittance=0.6) == pytest.approx(4.8)


def test_state_matrices_match_forward_euler_derivation(
    thermal_parameters: ThermalParameters,
) -> None:
    model = ThermalModel(thermal_parameters)
    A, B, E = model.state_matrices()

    a_br = thermal_parameters.dt_hours / (thermal_parameters.C_r * thermal_parameters.R_br)
    a_ro = thermal_parameters.dt_hours / (thermal_parameters.C_r * thermal_parameters.R_ro)
    b_br = thermal_parameters.dt_hours / (thermal_parameters.C_b * thermal_parameters.R_br)

    np.testing.assert_allclose(
        A,
        np.array(
            [[1.0 - a_br - a_ro, a_br], [b_br, 1.0 - b_br]],
            dtype=float,
        ),
    )
    np.testing.assert_allclose(
        B,
        np.array([[0.0], [thermal_parameters.dt_hours / thermal_parameters.C_b]]),
    )
    np.testing.assert_allclose(
        E,
        np.array(
            [
                [
                    a_ro,
                    thermal_parameters.alpha * thermal_parameters.dt_hours / thermal_parameters.C_r,
                    thermal_parameters.dt_hours / thermal_parameters.C_r,
                ],
                [
                    0.0,
                    (1.0 - thermal_parameters.alpha)
                    * thermal_parameters.dt_hours
                    / thermal_parameters.C_b,
                    0.0,
                ],
            ],
            dtype=float,
        ),
    )


def test_discrete_step_matches_state_space_update(thermal_parameters: ThermalParameters) -> None:
    model = ThermalModel(thermal_parameters)
    state = np.array([20.5, 22.0], dtype=float)
    control_kw = 1.4
    disturbance = np.array([5.0, 0.85, 0.3], dtype=float)

    direct = model.discrete_step(
        state_c=state,
        control_kw=control_kw,
        outdoor_temperature_c=disturbance[0],
        solar_gain_kw_input=disturbance[1],
        internal_gain_kw=disturbance[2],
    )
    matrix_form = model.step_with_disturbance_vector(state, control_kw, disturbance)
    np.testing.assert_allclose(direct, matrix_form)


def test_observability_and_controllability_are_full_rank(
    thermal_parameters: ThermalParameters,
) -> None:
    model = ThermalModel(thermal_parameters)
    assert model.observability_rank() == 2
    assert model.controllability_rank() == 2
