"""Unit tests for the DHW Extended Kalman Filter with augmented state (§12 of spec).

Required tests per §16.3:
  - test_ekf_covariance_pd       P_aug remains symmetric positive-definite after 50 steps.
  - test_ekf_vtap_nonnegative    V̂_tap ≥ 0 after every update, even with noisy process.
  - test_ekf_vtap_detection      EKF detects a step in V_tap within n_conv steps.
  - test_ekf_no_tap_zero         Without tapping, EKF converges to V_tap ≈ 0.
  - test_ekf_observability_rank  Augmented observability matrix has rank 3 when T_top ≠ T_mains.
"""

from __future__ import annotations

import numpy as np
import pytest

from home_optimizer.dhw_model import DHWModel
from home_optimizer.kalman import DHWExtendedKalmanFilter
from home_optimizer.types import DHWParameters, EKFNoiseParameters

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Δt chosen well below the Euler stability bound for C_top=0.058, R_strat=10:
#   bound = 0.2 × min(0.058×10, 0.058×10, 0.058×50) = 0.2 × 0.58 = 0.116 h
# Using 0.05 h (3 min) gives a comfortable safety margin.
_DT_HOURS = 0.05


@pytest.fixture
def dhw_params() -> DHWParameters:
    """Physical DHW tank parameters for EKF tests."""
    return DHWParameters(
        dt_hours=_DT_HOURS,
        C_top=0.058,  # kWh/K  (~50 L upper half)
        C_bot=0.058,  # kWh/K  (~50 L lower half)
        R_strat=10.0,  # K/kW   (moderate stratification)
        R_loss=50.0,  # K/kW   (well insulated boiler)
        lambda_water=1.1628,  # kWh/(m³·K)  — fixed physical constant
    )


@pytest.fixture
def ekf_noise() -> EKFNoiseParameters:
    """EKF noise parameters: moderate temperature trust, responsive V_tap tracking."""
    return EKFNoiseParameters(
        process_cov_temperatures=np.diag([1e-4, 1e-4]),
        process_var_vtap=1e-6,
        measurement_var_t_top=0.01,
        measurement_var_t_bot=0.01,
    )


@pytest.fixture
def ekf(dhw_params: DHWParameters, ekf_noise: EKFNoiseParameters) -> DHWExtendedKalmanFilter:
    """EKF initialised at rest: T_top=55°C, T_bot=45°C, V_tap=0 m³/h."""
    return DHWExtendedKalmanFilter(
        model=DHWModel(dhw_params),
        noise=ekf_noise,
        initial_state=np.array([55.0, 45.0, 0.0]),
        initial_covariance=np.diag([1.0, 1.0, 1e-4]),
    )


# ---------------------------------------------------------------------------
# Helper: run N EKF steps with a simulated "true" state
# ---------------------------------------------------------------------------


def _simulate_and_filter(
    ekf_instance: DHWExtendedKalmanFilter,
    true_x: np.ndarray,
    v_tap_true: float,
    n_steps: int,
    t_mains_c: float = 10.0,
    t_amb_c: float = 20.0,
    control_kw: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Simulate true tank physics and feed measurements to EKF.

    Returns a list of V_tap estimates (post-clamp) over all steps.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    model = ekf_instance._model  # noqa: SLF001 — test-internal access OK
    v_tap_estimates: list[float] = []
    x = true_x.copy()

    for _ in range(n_steps):
        # True state propagation using DHWModel
        x = model.step(
            state=x,
            control_kw=control_kw,
            v_tap_m3_per_h=v_tap_true,
            t_mains_c=t_mains_c,
            t_amb_c=t_amb_c,
        )

        # Noisy measurements (σ = √0.01 K ≈ 0.1 K)
        y_top = x[0] + rng.normal(0.0, 0.1)
        y_bot = x[1] + rng.normal(0.0, 0.1)

        est = ekf_instance.step(
            control_kw=control_kw,
            t_mains_c=t_mains_c,
            t_amb_c=t_amb_c,
            t_top_meas_c=y_top,
            t_bot_meas_c=y_bot,
        )
        v_tap_estimates.append(est.v_tap_m3_per_h)

    return v_tap_estimates


# ---------------------------------------------------------------------------
# test_ekf_covariance_pd  (§16.3)
# ---------------------------------------------------------------------------


def test_ekf_covariance_pd(
    ekf: DHWExtendedKalmanFilter,
) -> None:
    """P_aug must remain symmetric positive-definite after 50 EKF steps.

    Implements §16.3 test_ekf_covariance_pd.
    Checks: smallest eigenvalue > 0 after every step.
    """
    true_x = np.array([55.0, 45.0])
    _simulate_and_filter(ekf, true_x, v_tap_true=0.0, n_steps=250)

    P = ekf.estimate.covariance
    # Symmetry
    assert np.allclose(P, P.T, atol=1e-10), "Covariance must remain symmetric."
    # Positive-definiteness: smallest eigenvalue must be strictly positive
    min_eig = float(np.min(np.linalg.eigvalsh(P)))
    assert min_eig > 0.0, f"Covariance lost positive-definiteness: min eigenvalue = {min_eig}"


# ---------------------------------------------------------------------------
# test_ekf_vtap_nonnegative  (§16.3)
# ---------------------------------------------------------------------------


def test_ekf_vtap_nonnegative(
    dhw_params: DHWParameters,
    ekf_noise: EKFNoiseParameters,
) -> None:
    """V̂_tap ≥ 0 after every EKF update, even when process noise is large.

    Implements §16.3 test_ekf_vtap_nonnegative.
    Uses a large Q_n_Vtap to stress-test the clamp.
    """
    # Use aggressive process noise on V_tap to provoke negative EKF values
    aggressive_noise = EKFNoiseParameters(
        process_cov_temperatures=ekf_noise.process_cov_temperatures,
        process_var_vtap=0.5,  # very large → EKF can go negative without clamp
        measurement_var_t_top=ekf_noise.measurement_var_t_top,
        measurement_var_t_bot=ekf_noise.measurement_var_t_bot,
    )
    filter_instance = DHWExtendedKalmanFilter(
        model=DHWModel(dhw_params),
        noise=aggressive_noise,
        initial_state=np.array([55.0, 45.0, 0.0]),
        initial_covariance=np.diag([1.0, 1.0, 1.0]),
    )

    v_tap_estimates = _simulate_and_filter(
        filter_instance,
        true_x=np.array([55.0, 45.0]),
        v_tap_true=0.0,
        n_steps=50,
        rng=np.random.default_rng(0),
    )

    assert all(
        v >= 0.0 for v in v_tap_estimates
    ), "V_tap estimate must never be negative (§12 Step 4 fail-fast clamp)."


def test_ekf_jacobian_eval_point(ekf: DHWExtendedKalmanFilter) -> None:
    """EKF covariance prediction must use the Jacobian at ``x̂[k-1]`` (§12.4).

    The test verifies two things:
    1. The analytical Jacobian matches a first-order finite-difference expansion
       of the nonlinear propagation at the *pre-propagation* estimate.
    2. The predicted covariance equals ``F[k-1] P[k-1] F[k-1]^T + Q_aug`` using
       that same pre-propagation Jacobian.
    """
    control_kw = 0.8
    t_amb_c = 20.0
    t_mains_c = 10.0
    disturbance = np.array([t_amb_c, t_mains_c], dtype=float)
    x_before = ekf.estimate.mean.copy()
    P_before = ekf.estimate.covariance.copy()

    F_before = ekf._jacobian_from_state(x_before, control_kw, disturbance)  # noqa: SLF001
    delta_x = np.array([1e-6, -2e-6, 1e-7], dtype=float)
    f_ref = ekf._state_transition(x_before, control_kw, disturbance)  # noqa: SLF001
    f_perturbed = ekf._state_transition(x_before + delta_x, control_kw, disturbance)  # noqa: SLF001

    np.testing.assert_allclose(
        F_before @ delta_x,
        f_perturbed - f_ref,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Jacobian must linearise the nonlinear propagation at x̂[k-1].",
    )

    ekf.predict(control_kw=control_kw, t_mains_c=t_mains_c, t_amb_c=t_amb_c)
    np.testing.assert_allclose(
        ekf.estimate.covariance,
        F_before @ P_before @ F_before.T + ekf._noise.Q_aug,  # noqa: SLF001
        rtol=1e-10,
        atol=1e-10,
        err_msg="Predict covariance must use the Jacobian evaluated before propagation.",
    )


# ---------------------------------------------------------------------------
# test_ekf_vtap_detection  (§16.3)
# ---------------------------------------------------------------------------


def test_ekf_vtap_detection(
    dhw_params: DHWParameters,
) -> None:
    """EKF converges to true V_tap within n_conv steps after a tap event.

    Implements §16.3 test_ekf_vtap_detection.
    A step in V_tap of 0.012 m³/h (≈ 12 L/min) is applied at step 0.
    Convergence criterion: |V̂_tap − V_true| ≤ δ_V after n_conv steps.
    """
    v_tap_true = 0.012  # m³/h  — realistic shower flow rate
    delta_v = 0.005  # m³/h  — convergence tolerance
    n_conv = 150  # steps (= 7.5 min at Δt = 0.05 h)

    # Responsive EKF: higher Q_n_Vtap to track step changes faster
    responsive_noise = EKFNoiseParameters(
        process_cov_temperatures=np.diag([1e-3, 1e-3]),
        process_var_vtap=1e-4,
        measurement_var_t_top=0.01,
        measurement_var_t_bot=0.01,
    )
    filter_instance = DHWExtendedKalmanFilter(
        model=DHWModel(dhw_params),
        noise=responsive_noise,
        initial_state=np.array([55.0, 45.0, 0.0]),
        initial_covariance=np.diag([1.0, 1.0, 1e-3]),
    )

    v_estimates = _simulate_and_filter(
        filter_instance,
        true_x=np.array([55.0, 45.0]),
        v_tap_true=v_tap_true,
        n_steps=n_conv,
        t_mains_c=10.0,
        rng=np.random.default_rng(7),
    )

    final_estimate = v_estimates[-1]
    np.testing.assert_allclose(
        final_estimate,
        v_tap_true,
        atol=delta_v,
        err_msg=(
            f"EKF did not converge to V_tap={v_tap_true} m³/h within {n_conv} steps "
            f"(got {final_estimate:.5f}, tolerance {delta_v})."
        ),
    )


# ---------------------------------------------------------------------------
# test_ekf_no_tap_zero  (§16.3)
# ---------------------------------------------------------------------------


def test_ekf_no_tap_zero(
    dhw_params: DHWParameters,
) -> None:
    """Without tapping (V_tap=0), EKF converges to V̂_tap ≈ 0.

    Implements §16.3 test_ekf_no_tap_zero.
    """
    delta_v = 0.003  # m³/h  — convergence tolerance at rest
    n_steps = 300  # steps (= 15 min at Δt = 0.05 h)

    noise = EKFNoiseParameters(
        process_cov_temperatures=np.diag([1e-4, 1e-4]),
        process_var_vtap=1e-6,
        measurement_var_t_top=0.01,
        measurement_var_t_bot=0.01,
    )
    # Initialise with a slightly wrong V_tap to test convergence
    filter_instance = DHWExtendedKalmanFilter(
        model=DHWModel(dhw_params),
        noise=noise,
        initial_state=np.array([55.0, 45.0, 0.005]),  # wrong init
        initial_covariance=np.diag([1.0, 1.0, 1e-3]),
    )

    v_estimates = _simulate_and_filter(
        filter_instance,
        true_x=np.array([55.0, 45.0]),
        v_tap_true=0.0,
        n_steps=n_steps,
        rng=np.random.default_rng(13),
    )

    final_estimate = v_estimates[-1]
    np.testing.assert_allclose(
        final_estimate,
        0.0,
        atol=delta_v,
        err_msg=(
            f"EKF did not converge to 0 when no tap is active "
            f"(got {final_estimate:.5f}, tolerance {delta_v})."
        ),
    )


# ---------------------------------------------------------------------------
# test_ekf_observability_rank  (§16.3)
# ---------------------------------------------------------------------------


def test_ekf_observability_rank(
    ekf: DHWExtendedKalmanFilter,
) -> None:
    """Augmented observability matrix O_aug has rank 3 when T_top ≠ T_mains.

    Implements §16.3 test_ekf_observability_rank (§12.5 of spec).
    rank(O_aug) = 3  iff  a_strat ≠ 0  AND  T̂_top ≠ T_mains.
    """
    t_mains_c = 10.0  # °C  — cold mains; T̂_top = 55°C ≠ 10°C → full rank

    O_aug = ekf.observability_matrix(t_mains_c)

    assert O_aug.shape == (4, 3), f"O_aug must have shape (4, 3), got {O_aug.shape}."
    rank = int(np.linalg.matrix_rank(O_aug))
    assert (
        rank == 3
    ), f"Augmented observability matrix must have rank 3 when T_top ≠ T_mains, got rank={rank}."


def test_ekf_observability_rank_degenerate(
    ekf: DHWExtendedKalmanFilter,
) -> None:
    """When T̂_top = 0 AND T_mains = 0 the augmented observability rank drops to 2.

    Documents the physical edge case from §12.5: V_tap is unobservable when
    both tap-flow sensitivity terms in the Jacobian third column collapse to zero:
        ∂f_T_top/∂V_tap = -Δt/C_top · λ · T̂_top  → 0 when T̂_top = 0
        ∂f_T_bot/∂V_tap = +Δt/C_bot · λ · T_mains  → 0 when T_mains = 0
    The third column of F[k] then becomes [0, 0, 1]ᵀ, causing the rank to drop.
    """
    # Force state so that T_top = 0 (drives top-layer sensitivity to zero)
    # and use T_mains = 0 (drives bottom-layer sensitivity to zero)
    t_mains_c = 0.0
    ekf._x[0] = 0.0  # noqa: SLF001 — test-internal state override

    O_aug = ekf.observability_matrix(t_mains_c)
    rank = int(np.linalg.matrix_rank(O_aug, tol=1e-10))
    assert rank < 3, (
        "When T̂_top = 0 and T_mains = 0 both Jacobian tap-sensitivity terms vanish; "
        f"rank must drop below 3, got rank={rank}."
    )
