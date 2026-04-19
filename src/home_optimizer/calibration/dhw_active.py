"""Active DHW stratification calibration from no-draw charging telemetry.

This second DHW calibration stage fits the next identifiable parameter after the
standby-loss stage: the effective stratification resistance ``R_strat``.

Why only ``R_strat``?
---------------------
With the current telemetry we have:

* measured ``T_top`` and ``T_bot``
* measured/derived DHW thermal charging power ``P_dhw``
* estimated ``T_mains`` and measured ``T_amb``
* no tap-flow meter

Therefore the safest physically consistent stage is to:

1. select DHW charging windows that look like **no-draw** intervals by applying
   the total-energy balance,
2. hold ``C_top``, ``C_bot``, ``R_loss`` and ``lambda_water`` fixed from the
   injected reference parameter object, and
3. fit only ``R_strat`` by replaying the 2-state DHW model with ``V_tap = 0``.

The objective is the concatenated one-step residual vector on both measured
states:

    [T_top_pred - T_top_meas,
     T_bot_pred - T_bot_meas]

Using the measured start state for each interval is physically valid here
because both temperatures are observed directly; unlike UFH, there is no hidden
thermal state that requires a Kalman replay in this offline stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.optimize import least_squares

from .models import (
    DHWActiveCalibrationDataset,
    DHWActiveCalibrationResult,
    DHWActiveCalibrationSettings,
)
from ..dhw_model import DHWModel
from ..types import DHWParameters

_INVALID_PARAMETER_RESIDUAL_C: float = 1_000.0
_MIN_REGULARIZATION_SCALE: float = 1e-9


@dataclass(frozen=True, slots=True)
class _ActiveCalibrationArrays:
    """Vectorised numerical view of the active DHW replay dataset."""

    dt_hours: np.ndarray
    t_top_start_c: np.ndarray
    t_top_end_c: np.ndarray
    t_bot_start_c: np.ndarray
    t_bot_end_c: np.ndarray
    p_dhw_mean_kw: np.ndarray
    t_mains_c: np.ndarray
    t_amb_c: np.ndarray


@dataclass(frozen=True, slots=True)
class _ReplayDiagnostics:
    """Summary diagnostics from one DHW replay under a candidate parameter set."""

    t_top_residuals_c: np.ndarray
    t_bot_residuals_c: np.ndarray

    @property
    def rmse_t_top_c(self) -> float:
        """Root-mean-square top-layer residual [°C]."""
        return sqrt(float(np.mean(np.square(self.t_top_residuals_c))))

    @property
    def rmse_t_bot_c(self) -> float:
        """Root-mean-square bottom-layer residual [°C]."""
        return sqrt(float(np.mean(np.square(self.t_bot_residuals_c))))

    @property
    def max_abs_residual_c(self) -> float:
        """Maximum absolute temperature residual across both layers [°C]."""
        return float(
            np.max(
                np.abs(
                    np.concatenate([self.t_top_residuals_c, self.t_bot_residuals_c])
                )
            )
        )


def _as_arrays(dataset: DHWActiveCalibrationDataset) -> _ActiveCalibrationArrays:
    """Convert immutable replay samples into NumPy arrays for optimisation."""
    samples = dataset.samples
    return _ActiveCalibrationArrays(
        dt_hours=np.array([sample.dt_hours for sample in samples], dtype=float),
        t_top_start_c=np.array([sample.t_top_start_c for sample in samples], dtype=float),
        t_top_end_c=np.array([sample.t_top_end_c for sample in samples], dtype=float),
        t_bot_start_c=np.array([sample.t_bot_start_c for sample in samples], dtype=float),
        t_bot_end_c=np.array([sample.t_bot_end_c for sample in samples], dtype=float),
        p_dhw_mean_kw=np.array([sample.p_dhw_mean_kw for sample in samples], dtype=float),
        t_mains_c=np.array([sample.t_mains_c for sample in samples], dtype=float),
        t_amb_c=np.array([sample.t_amb_c for sample in samples], dtype=float),
    )


def _initial_theta(settings: DHWActiveCalibrationSettings) -> np.ndarray:
    """Return the initial optimisation vector from the reference parameter set."""
    return np.array([settings.reference_parameters.R_strat], dtype=float)


def dhw_active_r_strat_bounds(settings: DHWActiveCalibrationSettings) -> tuple[float, float]:
    """Return the explicit active-DHW optimiser box for ``R_strat`` [K/kW].

    The active-DHW fit identifies a grey-box stratification parameter, not a
    first-principles material property. Therefore the admissible interval is
    carried explicitly in physical units via ``DHWActiveCalibrationSettings`` and
    shared between the least-squares solver and the automatic-calibration
    diagnostics.
    """
    lower_bound = settings.min_r_strat_k_per_kw
    upper_bound = settings.max_r_strat_k_per_kw
    if lower_bound <= 0.0:
        raise ValueError("R_strat lower bound must remain strictly positive.")
    if lower_bound >= upper_bound:
        raise ValueError("R_strat lower bound must be < upper bound.")
    return lower_bound, upper_bound


def _theta_bounds(settings: DHWActiveCalibrationSettings) -> tuple[np.ndarray, np.ndarray]:
    """Construct the optimiser bounds for the scalar ``R_strat`` parameter."""
    lower_bound, upper_bound = dhw_active_r_strat_bounds(settings)
    return np.array([lower_bound], dtype=float), np.array([upper_bound], dtype=float)


def _parameters_from_theta(theta: np.ndarray, settings: DHWActiveCalibrationSettings) -> DHWParameters:
    """Map the optimisation vector back to a full DHW parameter object."""
    theta_arr = np.asarray(theta, dtype=float)
    if theta_arr.shape != (1,):
        raise ValueError("theta must have shape (1,) for active DHW calibration.")
    reference = settings.reference_parameters
    return DHWParameters(
        dt_hours=reference.dt_hours,
        C_top=reference.C_top,
        C_bot=reference.C_bot,
        R_strat=float(theta_arr[0]),
        R_loss=reference.R_loss,
        lambda_water=reference.lambda_water,
    )


def _regularization_residual(theta: np.ndarray, settings: DHWActiveCalibrationSettings) -> np.ndarray:
    """Return optional prior residuals anchored to the reference ``R_strat`` value."""
    if settings.regularization_weight <= 0.0:
        return np.empty(0, dtype=float)
    theta_arr = np.asarray(theta, dtype=float)
    theta_prior = _initial_theta(settings)
    scales = np.maximum(theta_prior * settings.regularization_scale_ratio, _MIN_REGULARIZATION_SCALE)
    return np.sqrt(settings.regularization_weight) * (theta_arr - theta_prior) / scales


def _replay_with_parameters(
    arrays: _ActiveCalibrationArrays,
    parameters: DHWParameters,
) -> _ReplayDiagnostics:
    """Replay one-step DHW transitions with ``V_tap = 0`` and collect residuals.

    Each persisted telemetry interval carries its own ``dt_hours``. The replay
    therefore reconstructs the DHW model per sample so mild bucket jitter does
    not force the dataset builder to reject otherwise informative no-draw runs.
    """
    t_top_residuals: list[float] = []
    t_bot_residuals: list[float] = []
    for index in range(arrays.t_top_end_c.size):
        sample_parameters = DHWParameters(
            dt_hours=float(arrays.dt_hours[index]),
            C_top=parameters.C_top,
            C_bot=parameters.C_bot,
            R_strat=parameters.R_strat,
            R_loss=parameters.R_loss,
            lambda_water=parameters.lambda_water,
        )
        model = DHWModel(sample_parameters)
        predicted_state = model.step(
            state=np.array([arrays.t_top_start_c[index], arrays.t_bot_start_c[index]], dtype=float),
            control_kw=float(arrays.p_dhw_mean_kw[index]),
            v_tap_m3_per_h=0.0,
            t_mains_c=float(arrays.t_mains_c[index]),
            t_amb_c=float(arrays.t_amb_c[index]),
        )
        t_top_residuals.append(float(predicted_state[0] - arrays.t_top_end_c[index]))
        t_bot_residuals.append(float(predicted_state[1] - arrays.t_bot_end_c[index]))
    return _ReplayDiagnostics(
        t_top_residuals_c=np.array(t_top_residuals, dtype=float),
        t_bot_residuals_c=np.array(t_bot_residuals, dtype=float),
    )


def calibrate_dhw_active_stratification(
    dataset: DHWActiveCalibrationDataset,
    settings: DHWActiveCalibrationSettings,
) -> DHWActiveCalibrationResult:
    """Fit ``R_strat`` from active DHW charging telemetry under no-draw assumptions.

    Args:
        dataset: Validated active no-draw replay dataset built from consecutive DHW buckets.
        settings: Reference parameters, optimiser bounds, and filtering assumptions.

    Returns:
        Fitted DHW parameter object and replay diagnostics.
    """
    if dataset.sample_count < settings.min_sample_count:
        raise ValueError(
            f"dataset must contain at least {settings.min_sample_count} samples; "
            f"received {dataset.sample_count}."
        )

    arrays = _as_arrays(dataset)
    initial_theta = _initial_theta(settings)
    lower_bounds, upper_bounds = _theta_bounds(settings)

    def residuals(theta: np.ndarray) -> np.ndarray:
        try:
            parameters = _parameters_from_theta(theta, settings)
            diagnostics = _replay_with_parameters(arrays, parameters)
            return np.concatenate(
                [
                    diagnostics.t_top_residuals_c,
                    diagnostics.t_bot_residuals_c,
                    _regularization_residual(theta, settings),
                ]
            )
        except ValueError:
            invalid = np.full(2 * arrays.t_top_end_c.size, _INVALID_PARAMETER_RESIDUAL_C, dtype=float)
            return np.concatenate([invalid, _regularization_residual(theta, settings)])

    result = least_squares(
        residuals,
        x0=initial_theta,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
    )
    if not result.success:
        raise RuntimeError(f"DHW active stratification calibration failed: {result.message}")

    fitted_parameters = _parameters_from_theta(result.x, settings)
    diagnostics = _replay_with_parameters(arrays, fitted_parameters)
    return DHWActiveCalibrationResult(
        fitted_parameters=fitted_parameters,
        rmse_t_top_c=diagnostics.rmse_t_top_c,
        rmse_t_bot_c=diagnostics.rmse_t_bot_c,
        max_abs_residual_c=diagnostics.max_abs_residual_c,
        sample_count=dataset.sample_count,
        segment_count=dataset.segment_count,
        dataset_start_utc=dataset.start_utc,
        dataset_end_utc=dataset.end_utc,
        optimizer_status=str(result.message),
        optimizer_cost=float(result.cost),
    )

