"""Unit tests for the shared linear state-space and discretisation primitives."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from home_optimizer.domain.state_space import (
    ContinuousLinearModel,
    DiscretizationConfig,
    Discretizer,
    controllability_matrix,
    numerical_rank,
    observability_is_well_conditioned,
    observability_matrix,
    observability_min_singular_value,
)


def test_forward_euler_matches_closed_form_reference() -> None:
    """Forward Euler must implement A_d=I+dt*A_c, B_d=dt*B_c, E_d=dt*E_c."""
    continuous = ContinuousLinearModel(
        A=np.array([[-0.3, 0.1], [0.2, -0.4]], dtype=float),
        B=np.array([[0.0], [0.5]], dtype=float),
        E=np.array([[0.7, 0.2], [0.1, 0.6]], dtype=float),
        C=np.array([[1.0, 0.0]], dtype=float),
    )
    dt_hours = 0.25

    discrete = Discretizer.discretize(
        continuous_model=continuous,
        config=DiscretizationConfig(method="forward_euler", dt_hours=dt_hours),
    )

    np.testing.assert_allclose(discrete.A, np.eye(2) + dt_hours * continuous.A)
    np.testing.assert_allclose(discrete.B, dt_hours * continuous.B)
    np.testing.assert_allclose(discrete.E, dt_hours * continuous.E)


def test_exact_zoh_matches_augmented_matrix_exponential() -> None:
    """Exact ZOH must use the augmented-matrix exponential from the specification."""
    continuous = ContinuousLinearModel(
        A=np.array([[-0.5, 0.1], [0.2, -0.3]], dtype=float),
        B=np.array([[0.0], [0.8]], dtype=float),
        E=np.array([[0.4], [0.2]], dtype=float),
        C=np.array([[1.0, 0.0]], dtype=float),
    )
    dt_hours = 0.5

    discrete = Discretizer.discretize(
        continuous_model=continuous,
        config=DiscretizationConfig(method="exact_zoh", dt_hours=dt_hours),
    )

    augmented = np.zeros((4, 4), dtype=float)
    augmented[:2, :2] = continuous.A
    augmented[:2, 2:3] = continuous.B
    augmented[:2, 3:] = continuous.E
    expected = expm(augmented * dt_hours)

    np.testing.assert_allclose(discrete.A, expected[:2, :2])
    np.testing.assert_allclose(discrete.B, expected[:2, 2:3])
    np.testing.assert_allclose(discrete.E, expected[:2, 3:])


def test_observability_and_controllability_helpers_match_linear_algebra() -> None:
    """The shared helpers must return [C; C A] and [B, A B] respectively."""
    continuous = ContinuousLinearModel(
        A=np.array([[-0.2, 0.3], [0.4, -0.1]], dtype=float),
        B=np.array([[0.0], [1.0]], dtype=float),
        E=np.array([[0.5], [0.2]], dtype=float),
        C=np.array([[1.0, 0.0]], dtype=float),
    )
    discrete = Discretizer.forward_euler(continuous_model=continuous, dt_hours=1.0)

    np.testing.assert_allclose(
        observability_matrix(discrete),
        np.vstack([discrete.C, discrete.C @ discrete.A]),
    )
    np.testing.assert_allclose(
        controllability_matrix(discrete),
        np.column_stack([discrete.B, discrete.A @ discrete.B]),
    )


def test_numerical_rank_and_conditioning_helpers_use_svd() -> None:
    """Observability diagnostics must expose SVD-based rank and conditioning helpers."""
    continuous = ContinuousLinearModel(
        A=np.array([[-0.2, 0.3], [0.4, -0.1]], dtype=float),
        B=np.array([[0.0], [1.0]], dtype=float),
        E=np.array([[0.5], [0.2]], dtype=float),
        C=np.array([[1.0, 0.0]], dtype=float),
    )
    discrete = Discretizer.exact_zoh(continuous_model=continuous, dt_hours=1.0)
    obs = observability_matrix(discrete)

    assert numerical_rank(obs) == 2
    assert observability_min_singular_value(discrete) > 0.0
    assert observability_is_well_conditioned(discrete)
