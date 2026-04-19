"""Unit tests for the DHW stratification model and DHW Kalman filter."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from home_optimizer.dhw_model import MEASUREMENT_MATRIX_DHW, DHWModel
from home_optimizer.kalman import DHWKalmanFilter
from home_optimizer.types import DHWParameters, KalmanNoiseParameters

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def params() -> DHWParameters:
    """Physically realistic 1000 L buffer vessel: C = λ·0.5 m³ ≈ 0.58 kWh/K per layer.

    With R_strat=10 K/kW: time constant = 0.58·10 = 5.8 h → stability bound = 1.16 h,
    which is > Δt=1.0 h.  Reflects a large DHW buffer, not a small domestic boiler.
    """
    return DHWParameters(
        dt_hours=1.0,
        C_top=0.5814,  # kWh/K  (λ · 0.5 m³; half of 1000 L)
        C_bot=0.5814,
        R_strat=10.0,  # K/kW
        R_loss=50.0,  # K/kW
        lambda_water=1.1628,
    )


@pytest.fixture
def model(params: DHWParameters) -> DHWModel:
    return DHWModel(params)


# ---------------------------------------------------------------------------
# Matrix specification tests (§11)
# ---------------------------------------------------------------------------


def test_A_matrix_matches_specification(model: DHWModel, params: DHWParameters) -> None:
    """A_dhw must equal the exact ZOH discretisation of the continuous DHW physics."""
    v_tap = 0.0
    A, _, _ = model.state_matrices(v_tap)
    f_cont = np.array(
        [
            [-(1.0 / (params.C_top * params.R_strat) + 1.0 / (params.C_top * params.R_loss)), 1.0 / (params.C_top * params.R_strat)],
            [1.0 / (params.C_bot * params.R_strat), -(1.0 / (params.C_bot * params.R_strat) + 1.0 / (params.C_bot * params.R_loss))],
        ]
    )
    expected = expm(f_cont * params.dt_hours)
    np.testing.assert_allclose(A, expected)


def test_B_matrix_is_constant_and_correct(model: DHWModel, params: DHWParameters) -> None:
    """B_dhw must be the exact ZOH actuation map with bottom-layer-only heating."""
    for v_tap in (0.0, 0.01, 0.05):
        _, B, _ = model.state_matrices(v_tap)
        f_cont = np.array(
            [
                [-(1.0 / (params.C_top * params.R_strat) + 1.0 / (params.C_top * params.R_loss) + params.lambda_water * v_tap / params.C_top), 1.0 / (params.C_top * params.R_strat)],
                [1.0 / (params.C_bot * params.R_strat), -(1.0 / (params.C_bot * params.R_strat) + 1.0 / (params.C_bot * params.R_loss))],
            ]
        )
        g_u = np.array([[0.0], [1.0 / params.C_bot]])
        augmented = np.zeros((3, 3), dtype=float)
        augmented[:2, :2] = f_cont
        augmented[:2, 2:3] = g_u
        expected = expm(augmented * params.dt_hours)[:2, 2:3]
        np.testing.assert_allclose(B, expected)


def test_E_matrix_matches_specification(model: DHWModel, params: DHWParameters) -> None:
    """E_dhw[k] must equal the exact ZOH disturbance map for [T_amb, T_mains]."""
    v_tap = 0.02
    _, _, E = model.state_matrices(v_tap)
    f_cont = np.array(
        [
            [-(1.0 / (params.C_top * params.R_strat) + 1.0 / (params.C_top * params.R_loss) + params.lambda_water * v_tap / params.C_top), 1.0 / (params.C_top * params.R_strat)],
            [1.0 / (params.C_bot * params.R_strat), -(1.0 / (params.C_bot * params.R_strat) + 1.0 / (params.C_bot * params.R_loss))],
        ]
    )
    g_d = np.array(
        [
            [1.0 / (params.C_top * params.R_loss), 0.0],
            [1.0 / (params.C_bot * params.R_loss), params.lambda_water * v_tap / params.C_bot],
        ]
    )
    augmented = np.zeros((4, 4), dtype=float)
    augmented[:2, :2] = f_cont
    augmented[:2, 2:] = g_d
    expected = expm(augmented * params.dt_hours)[:2, 2:]
    np.testing.assert_allclose(E, expected)


def test_A_matrix_varies_with_v_tap(model: DHWModel) -> None:
    """A_dhw[k] must change when V_tap changes (LTV system)."""
    A0, _, _ = model.state_matrices(0.0)
    A1, _, _ = model.state_matrices(0.05)
    assert not np.allclose(A0, A1), "A_dhw must be time-varying w.r.t. V_tap."


# ---------------------------------------------------------------------------
# Energy balance verification (§9.5)
# ---------------------------------------------------------------------------


def test_dhw_energy_balance(model: DHWModel, params: DHWParameters) -> None:
    """Verify §9.5: d/dt(C_top·T_top + C_bot·T_bot) = P_dhw − Q_tap − Q_loss."""
    T_top, T_bot = 55.0, 40.0
    state = np.array([T_top, T_bot])
    P_dhw = 2.0
    v_tap = 0.02
    t_mains = 10.0
    t_amb = 20.0

    dxdt = model.continuous_derivative(state, P_dhw, v_tap, t_mains, t_amb)

    # LHS: d/dt(C_top·T_top + C_bot·T_bot)
    lhs = params.C_top * dxdt[0] + params.C_bot * dxdt[1]

    # RHS from §9.5
    q_tap = params.lambda_water * v_tap * (T_top - t_mains)
    q_loss = (T_top - t_amb) / params.R_loss + (T_bot - t_amb) / params.R_loss
    rhs = P_dhw - q_tap - q_loss

    np.testing.assert_allclose(lhs, rhs, atol=1e-10, err_msg="DHW energy balance violated.")


# ---------------------------------------------------------------------------
# Tap-stream split (§9.1 requirement)
# ---------------------------------------------------------------------------


def test_tap_stream_split_correct(model: DHWModel, params: DHWParameters) -> None:
    """Top layer loses λ·V̇·T_top; bottom layer gains λ·V̇·T_mains (§9.1)."""
    T_top, T_bot = 55.0, 40.0
    state = np.array([T_top, T_bot])
    v_tap = 0.01
    t_mains = 10.0
    dxdt = model.continuous_derivative(state, 0.0, v_tap, t_mains, 20.0)

    tap_top_expected = -params.lambda_water * v_tap * T_top / params.C_top
    tap_bot_expected = params.lambda_water * v_tap * t_mains / params.C_bot

    # Isolate just the tap contribution by comparing with v_tap=0 run
    dxdt_no_tap = model.continuous_derivative(state, 0.0, 0.0, t_mains, 20.0)
    tap_contribution_top = dxdt[0] - dxdt_no_tap[0]
    tap_contribution_bot = dxdt[1] - dxdt_no_tap[1]

    np.testing.assert_allclose(tap_contribution_top, tap_top_expected, rtol=1e-10)
    np.testing.assert_allclose(tap_contribution_bot, tap_bot_expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Simulation consistency
# ---------------------------------------------------------------------------


def test_step_and_continuous_derivative_agree(model: DHWModel, params: DHWParameters) -> None:
    """For a small dt, the exact ZOH step must match Euler to first order."""
    small_dt_params = DHWParameters(
        dt_hours=5.0 / 60.0,
        C_top=params.C_top,
        C_bot=params.C_bot,
        R_strat=params.R_strat,
        R_loss=params.R_loss,
        lambda_water=params.lambda_water,
    )
    small_dt_model = DHWModel(small_dt_params)
    state = np.array([55.0, 42.0])
    P_dhw, v_tap, t_mains, t_amb = 1.5, 0.02, 10.0, 18.0

    # Matrix form
    x_mat = small_dt_model.step(state, P_dhw, v_tap, t_mains, t_amb)
    # Euler via continuous derivative
    dxdt = small_dt_model.continuous_derivative(state, P_dhw, v_tap, t_mains, t_amb)
    x_euler = state + small_dt_params.dt_hours * dxdt

    np.testing.assert_allclose(x_mat, x_euler, atol=5e-2)


def test_t_dhw_mean_is_weighted_average(model: DHWModel, params: DHWParameters) -> None:
    """T_dhw = (C_top·T_top + C_bot·T_bot)/(C_top+C_bot) — not an independent state."""
    state = np.array([60.0, 40.0])
    expected = (params.C_top * 60.0 + params.C_bot * 40.0) / (params.C_top + params.C_bot)
    assert model.t_dhw_mean(state) == pytest.approx(expected)


def test_heating_raises_bottom_layer(model: DHWModel) -> None:
    """P_dhw enters the bottom layer (B[0,0]=0, B[1,0]>0 — assumption A5)."""
    state = np.array([50.0, 30.0])
    x_next = model.step(state, control_kw=3.0, v_tap_m3_per_h=0.0, t_mains_c=10.0, t_amb_c=20.0)
    assert x_next[1] > state[1], "Heating should raise T_bot."


def test_tap_cools_top_layer(model: DHWModel) -> None:
    """Tapping hot water should reduce the top-layer temperature."""
    state = np.array([60.0, 50.0])
    x_next = model.step(state, control_kw=0.0, v_tap_m3_per_h=0.1, t_mains_c=10.0, t_amb_c=20.0)
    assert x_next[0] < state[0], "Tapping should cool T_top."


# ---------------------------------------------------------------------------
# Observability (§11)
# ---------------------------------------------------------------------------


def test_system_is_fully_observable(model: DHWModel) -> None:
    """rank(O) must equal 2 as long as R_strat > 0 and C_top > 0."""
    assert model.observability_rank() == 2


def test_measurement_matrix_is_correct() -> None:
    """C_obs = [1, 0] — only T_top is measured."""
    np.testing.assert_array_equal(MEASUREMENT_MATRIX_DHW, [[1.0, 0.0]])


# ---------------------------------------------------------------------------
# Euler stability
# ---------------------------------------------------------------------------


def test_exact_zoh_runtime_supports_large_dt_that_would_break_euler() -> None:
    """DHWModel must remain usable at large dt because runtime discretisation is exact ZOH."""
    model = DHWModel(
        DHWParameters(
            dt_hours=5.0,
            C_top=0.5814,
            C_bot=0.5814,
            R_strat=10.0,
            R_loss=50.0,
        )
    )
    a_mat, b_mat, e_mat = model.state_matrices(v_tap_m3_per_h=0.0)
    assert np.all(np.isfinite(a_mat))
    assert np.all(np.isfinite(b_mat))
    assert np.all(np.isfinite(e_mat))


def test_exact_zoh_runtime_supports_high_tap_flow_without_euler_rejection() -> None:
    """Exact ZOH runtime discretisation must not reject high tap flow purely on Euler grounds."""
    params = DHWParameters(
        dt_hours=1.0,
        C_top=0.5814,
        C_bot=0.5814,
        R_strat=10.0,
        R_loss=50.0,
    )
    model = DHWModel(params)
    a_mat, b_mat, e_mat = model.state_matrices(v_tap_m3_per_h=0.3)
    assert np.all(np.isfinite(a_mat))
    assert np.all(np.isfinite(b_mat))
    assert np.all(np.isfinite(e_mat))


def test_negative_tap_flow_is_rejected_fail_fast(model: DHWModel) -> None:
    """Negative tap flow is physically impossible and must never reach the solver."""
    with pytest.raises(ValueError, match="non-negative"):
        model.state_matrices(v_tap_m3_per_h=-0.01)


# ---------------------------------------------------------------------------
# DHW Kalman filter (§12)
# ---------------------------------------------------------------------------


def _make_kf(model: DHWModel) -> DHWKalmanFilter:
    return DHWKalmanFilter(
        model=model,
        noise=KalmanNoiseParameters(
            process_covariance=np.diag([1e-4, 1e-4]),
            measurement_variance=1e-6,
        ),
        initial_state_c=np.array([45.0, 35.0]),
        initial_covariance=np.diag([9.0, 9.0]),
    )


def test_dhw_kalman_tracks_t_top_exactly(model: DHWModel) -> None:
    """T_top is directly observed; filter must track it with near-zero residual."""
    kf = _make_kf(model)
    true_state = np.array([55.0, 42.0])

    for _ in range(24):
        true_state = model.step(true_state, 1.0, 0.0, 10.0, 20.0)
        kf.step(
            control_kw=1.0,
            v_tap_m3_per_h=0.0,
            t_mains_c=10.0,
            t_amb_c=20.0,
            t_top_measurement_c=float(true_state[0]),
        )

    assert abs(kf.estimate.mean_c[0] - true_state[0]) < 1e-3


def test_dhw_kalman_converges_on_hidden_t_bot(model: DHWModel) -> None:
    """T_bot is hidden but observable via T_top trajectory; filter must converge."""
    kf = _make_kf(model)
    true_state = np.array([55.0, 42.0])

    for _ in range(48):
        true_state = model.step(true_state, 1.0, 0.0, 10.0, 20.0)
        kf.step(1.0, 0.0, 10.0, 20.0, float(true_state[0]))

    assert abs(kf.estimate.mean_c[1] - true_state[1]) < 0.5


def test_dhw_kalman_covariance_remains_positive_definite(model: DHWModel) -> None:
    """Error covariance must remain PSD throughout the filter run (Joseph form)."""
    kf = _make_kf(model)
    rng = np.random.default_rng(42)

    for _ in range(48):
        kf.predict(0.5, 0.0, 10.0, 20.0)
        kf.update(55.0 + rng.normal(0, 0.1))
        eigvals = np.linalg.eigvalsh(kf.estimate.covariance)
        assert np.all(eigvals >= -1e-10), "Covariance must remain positive semi-definite."


def test_dhw_kalman_uses_time_varying_A(model: DHWModel) -> None:
    """Filter predict with V_tap > 0 must differ from V_tap = 0 (LTV check)."""
    kf0 = _make_kf(model)
    kf1 = _make_kf(model)
    kf0.predict(0.0, 0.0, 10.0, 20.0)
    kf1.predict(0.0, 0.05, 10.0, 20.0)
    assert not np.allclose(
        kf0.estimate.mean_c, kf1.estimate.mean_c
    ), "Predict must use time-varying A_dhw[k] (different result for V_tap=0.05)."
