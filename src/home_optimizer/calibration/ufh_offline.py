"""Offline UFH envelope calibration from historical low-solar off-mode telemetry.

This first-stage fitter intentionally estimates only the parameter that is
well-conditioned in passive cool-down windows:

    dT_r/dt = -(T_r - T_out)/tau_house

The goal is not yet to recover the full 2-state UFH parameter set
``(C_r, C_b, R_br, R_ro)``.  Without active heating excitation those parameters
are not structurally identifiable from room-temperature cool-down alone.
Instead, this module learns the physically meaningful envelope time constant:

* ``tau_house = C_eff · R_ro`` — passive room-envelope time constant [h]
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.optimize import least_squares

from .models import UFHCalibrationDataset, UFHOffCalibrationResult, UFHOffCalibrationSettings


@dataclass(frozen=True, slots=True)
class _CalibrationArrays:
    """Vectorised numerical view of the calibration dataset."""

    dt_hours: np.ndarray
    room_temperature_start_c: np.ndarray
    room_temperature_end_c: np.ndarray
    outdoor_temperature_mean_c: np.ndarray


def _as_arrays(dataset: UFHCalibrationDataset) -> _CalibrationArrays:
    """Convert immutable sample objects into NumPy arrays for optimisation."""
    samples = dataset.samples
    return _CalibrationArrays(
        dt_hours=np.array([sample.dt_hours for sample in samples], dtype=float),
        room_temperature_start_c=np.array(
            [sample.room_temperature_start_c for sample in samples], dtype=float
        ),
        room_temperature_end_c=np.array([sample.room_temperature_end_c for sample in samples], dtype=float),
        outdoor_temperature_mean_c=np.array(
            [sample.outdoor_temperature_mean_c for sample in samples], dtype=float
        ),
    )


def _predict_room_temperature_end(
    arrays: _CalibrationArrays,
    *,
    tau_house_hours: float,
) -> np.ndarray:
    """Evaluate the forward-Euler one-step envelope model on a batch of samples."""
    return arrays.room_temperature_start_c + arrays.dt_hours / tau_house_hours * (
        -(arrays.room_temperature_start_c - arrays.outdoor_temperature_mean_c)
    )


def calibrate_ufh_off_envelope(
    dataset: UFHCalibrationDataset,
    settings: UFHOffCalibrationSettings,
) -> UFHOffCalibrationResult:
    """Fit the passive envelope time constant from low-solar off-mode telemetry.

    Args:
        dataset: Validated off-mode transition samples.
        settings: Bounds, initial guesses, and sample-count requirements.

    Returns:
        Calibrated effective envelope parameters and fit diagnostics.
    """
    if dataset.sample_count < settings.min_sample_count:
        raise ValueError(
            f"dataset must contain at least {settings.min_sample_count} samples; "
            f"received {dataset.sample_count}."
        )

    arrays = _as_arrays(dataset)

    def residuals(theta: np.ndarray) -> np.ndarray:
        tau_house_hours = theta[0]
        predictions = _predict_room_temperature_end(
            arrays,
            tau_house_hours=float(tau_house_hours),
        )
        return predictions - arrays.room_temperature_end_c

    initial_theta = np.array([settings.initial_tau_hours], dtype=float)
    lower_bounds = np.array([settings.min_tau_hours], dtype=float)
    upper_bounds = np.array([settings.max_tau_hours], dtype=float)

    result = least_squares(
        residuals,
        x0=initial_theta,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
    )
    if not result.success:
        raise RuntimeError(f"UFH off-envelope calibration failed: {result.message}")

    fitted_residuals = residuals(result.x)
    rmse_room_temperature_c = sqrt(float(np.mean(np.square(fitted_residuals))))
    max_abs_residual_c = float(np.max(np.abs(fitted_residuals)))
    tau_house_hours = float(result.x[0])
    suggested_r_ro_k_per_kw = (
        tau_house_hours / settings.reference_c_eff_kwh_per_k
        if settings.reference_c_eff_kwh_per_k is not None
        else None
    )

    return UFHOffCalibrationResult(
        tau_house_hours=tau_house_hours,
        suggested_r_ro_k_per_kw=suggested_r_ro_k_per_kw,
        reference_c_eff_kwh_per_k=settings.reference_c_eff_kwh_per_k,
        rmse_room_temperature_c=rmse_room_temperature_c,
        max_abs_residual_c=max_abs_residual_c,
        sample_count=dataset.sample_count,
        dataset_start_utc=dataset.start_utc,
        dataset_end_utc=dataset.end_utc,
        optimizer_status=str(result.message),
        optimizer_cost=float(result.cost),
    )

