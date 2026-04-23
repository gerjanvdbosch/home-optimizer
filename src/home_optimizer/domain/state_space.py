"""Generic linear state-space primitives shared by UFH, DHW, filtering, and MPC.

This module centralises the linear-algebra building blocks required by the
project specification:

* ``ContinuousLinearModel`` stores ``A_c``, ``B_c``, ``E_c``, and ``C``.
* ``DiscreteLinearModel`` stores ``A_d``, ``B_d``, ``E_d``, and ``C``.
* ``Discretizer`` derives discrete models from continuous physics using either
  exact zero-order hold or forward Euler.

Units
-----
The module is intentionally unit-agnostic at runtime. Physical units are imposed
by the caller, but the intended Home Optimizer usage is:

* states in ``°C``
* control inputs in ``kW``
* disturbances in subsystem-specific physical units
* continuous matrices in ``1/h`` or the corresponding input/disturbance gains
* discrete matrices for the step ``x[k+1] = A_d x[k] + B_d u[k] + E_d d[k]``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import expm

from ..types.constants import DEFAULT_NUMERICAL_VALIDATION_CONFIG, NumericalValidationConfig

DiscretizationMethod = Literal["exact_zoh", "forward_euler"]


def _as_float_matrix(*, name: str, matrix: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Return a copied float matrix with optional exact shape validation."""
    matrix_arr = np.asarray(matrix, dtype=float).copy()
    if matrix_arr.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 matrix.")
    if shape is not None and matrix_arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix_arr.shape}.")
    return matrix_arr


@dataclass(frozen=True, slots=True)
class ContinuousLinearModel:
    """Continuous-time linear subsystem ``dx/dt = A_c x + B_c u + E_c d``.

    Attributes
    ----------
    A:
        State matrix ``A_c`` with shape ``(n_x, n_x)`` [1/h].
    B:
        Control matrix ``B_c`` with shape ``(n_x, n_u)``.
    E:
        Disturbance matrix ``E_c`` with shape ``(n_x, n_d)``.
    C:
        Optional measurement matrix with shape ``(n_y, n_x)``.
    """

    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    C: np.ndarray | None = None

    def __post_init__(self) -> None:
        A = _as_float_matrix(name="A", matrix=self.A)
        state_dimension = A.shape[0]
        if A.shape[1] != state_dimension:
            raise ValueError("A must be square.")
        B = _as_float_matrix(name="B", matrix=self.B)
        E = _as_float_matrix(name="E", matrix=self.E)
        if B.shape[0] != state_dimension:
            raise ValueError("B row dimension must match the state dimension.")
        if E.shape[0] != state_dimension:
            raise ValueError("E row dimension must match the state dimension.")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "E", E)
        if self.C is not None:
            C = _as_float_matrix(name="C", matrix=self.C)
            if C.shape[1] != state_dimension:
                raise ValueError("C column dimension must match the state dimension.")
            object.__setattr__(self, "C", C)


@dataclass(frozen=True, slots=True)
class DiscreteLinearModel:
    """Discrete-time linear subsystem ``x[k+1] = A_d x[k] + B_d u[k] + E_d d[k]``."""

    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    C: np.ndarray | None = None

    def __post_init__(self) -> None:
        continuous = ContinuousLinearModel(A=self.A, B=self.B, E=self.E, C=self.C)
        object.__setattr__(self, "A", continuous.A)
        object.__setattr__(self, "B", continuous.B)
        object.__setattr__(self, "E", continuous.E)
        object.__setattr__(self, "C", continuous.C)


@dataclass(frozen=True, slots=True)
class DiscretizationConfig:
    """Validated discretisation settings for piecewise-constant inputs over one step.

    Attributes
    ----------
    method:
        Discretisation scheme. Supported values are ``"exact_zoh"`` and
        ``"forward_euler"``.
    dt_hours:
        Sample time ``Δt`` [h], strictly positive.
    """

    method: DiscretizationMethod
    dt_hours: float

    def __post_init__(self) -> None:
        if self.method not in ("exact_zoh", "forward_euler"):
            raise ValueError(f"Unsupported discretization method: {self.method}.")
        if self.dt_hours <= 0.0:
            raise ValueError("dt_hours must be strictly positive.")


class Discretizer:
    """Shared discretisation backend for the Home Optimizer linear models."""

    @staticmethod
    def discretize(
        continuous_model: ContinuousLinearModel,
        config: DiscretizationConfig,
    ) -> DiscreteLinearModel:
        """Derive a discrete model from continuous physics using the selected scheme."""
        if config.method == "exact_zoh":
            return Discretizer.exact_zoh(continuous_model=continuous_model, dt_hours=config.dt_hours)
        return Discretizer.forward_euler(
            continuous_model=continuous_model,
            dt_hours=config.dt_hours,
        )

    @staticmethod
    def exact_zoh(
        *,
        continuous_model: ContinuousLinearModel,
        dt_hours: float,
    ) -> DiscreteLinearModel:
        """Exact zero-order-hold discretisation via the augmented matrix exponential."""
        if dt_hours <= 0.0:
            raise ValueError("dt_hours must be strictly positive.")
        n_x = continuous_model.A.shape[0]
        n_u = continuous_model.B.shape[1]
        n_d = continuous_model.E.shape[1]
        augmented = np.zeros((n_x + n_u + n_d, n_x + n_u + n_d), dtype=float)
        augmented[:n_x, :n_x] = continuous_model.A
        augmented[:n_x, n_x : n_x + n_u] = continuous_model.B
        augmented[:n_x, n_x + n_u :] = continuous_model.E
        discretised = expm(augmented * dt_hours)
        return DiscreteLinearModel(
            A=discretised[:n_x, :n_x],
            B=discretised[:n_x, n_x : n_x + n_u],
            E=discretised[:n_x, n_x + n_u :],
            C=continuous_model.C,
        )

    @staticmethod
    def forward_euler(
        *,
        continuous_model: ContinuousLinearModel,
        dt_hours: float,
    ) -> DiscreteLinearModel:
        """Forward-Euler discretisation of a continuous linear subsystem."""
        if dt_hours <= 0.0:
            raise ValueError("dt_hours must be strictly positive.")
        identity = np.eye(continuous_model.A.shape[0], dtype=float)
        return DiscreteLinearModel(
            A=identity + dt_hours * continuous_model.A,
            B=dt_hours * continuous_model.B,
            E=dt_hours * continuous_model.E,
            C=continuous_model.C,
        )


def observability_matrix(discrete_model: DiscreteLinearModel) -> np.ndarray:
    """Return the 2-block observability matrix ``[C; C A_d]`` for a discrete model."""
    if discrete_model.C is None:
        raise ValueError("A measurement matrix C is required to build the observability matrix.")
    return np.vstack([discrete_model.C, discrete_model.C @ discrete_model.A])


def numerical_rank(
    matrix: np.ndarray,
    *,
    rtol: float = DEFAULT_NUMERICAL_VALIDATION_CONFIG.observability_rank_tolerance,
) -> int:
    """Return the SVD-based numerical rank of a matrix.

    The cutoff follows the standard relative rule ``s_i > rtol * s_max``.
    """
    matrix_arr = _as_float_matrix(name="matrix", matrix=matrix)
    singular_values = np.linalg.svd(matrix_arr, compute_uv=False)
    if singular_values.size == 0:
        return 0
    threshold = rtol * singular_values[0]
    return int(np.sum(singular_values > threshold))


def observability_min_singular_value(discrete_model: DiscreteLinearModel) -> float:
    """Return the smallest singular value of the observability matrix."""
    singular_values = np.linalg.svd(observability_matrix(discrete_model), compute_uv=False)
    return float(singular_values[-1])


def observability_condition_number(discrete_model: DiscreteLinearModel) -> float:
    """Return the singular-value condition number of the observability matrix."""
    singular_values = np.linalg.svd(observability_matrix(discrete_model), compute_uv=False)
    sigma_min = float(singular_values[-1])
    if sigma_min == 0.0:
        return float("inf")
    return float(singular_values[0] / sigma_min)


def observability_is_well_conditioned(
    discrete_model: DiscreteLinearModel,
    *,
    config: NumericalValidationConfig = DEFAULT_NUMERICAL_VALIDATION_CONFIG,
) -> bool:
    """Return whether the observability matrix clears the configured conditioning policy."""
    if config.observability_condition_policy == "min_singular_value":
        return observability_min_singular_value(discrete_model) >= config.observability_condition_min_sv
    return observability_condition_number(discrete_model) <= config.observability_condition_max


def controllability_matrix(discrete_model: DiscreteLinearModel) -> np.ndarray:
    """Return the 2-block controllability matrix ``[B, A_d B]`` for a discrete model."""
    return np.column_stack([discrete_model.B, discrete_model.A @ discrete_model.B])


__all__ = [
    "ContinuousLinearModel",
    "DiscreteLinearModel",
    "DiscretizationConfig",
    "DiscretizationMethod",
    "Discretizer",
    "controllability_matrix",
    "numerical_rank",
    "observability_condition_number",
    "observability_is_well_conditioned",
    "observability_matrix",
    "observability_min_singular_value",
]
