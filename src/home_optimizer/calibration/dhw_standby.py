"""Offline DHW standby-loss calibration from quasi-mixed non-DHW telemetry.

This first DHW calibration stage intentionally fits only the parameter that is
well-conditioned in passive, near-mixed standby windows derived from the full
2-node DHW energy balance (§9.5 of the theory document):

    tau_standby = (C_top + C_bot) · R_loss / 2

Under the conservative assumptions used by the dataset builder:

* no DHW heating input over the interval (``hp_mode_last != dhw``)
* both layers remain close enough that ``T_top ≈ T_bot``
* no explicit tap-flow estimation in this stage

The weighted mean tank temperature therefore follows the identifiable one-state
standby envelope:

    dT_dhw/dt = -(T_dhw - T_amb) / tau_standby

This module fits ``tau_standby`` and derives ``R_loss`` from the injected total
heat capacity ``C_top + C_bot``.  It deliberately does *not* attempt to fit
``R_strat`` or tap-flow behaviour; those require a richer active DHW stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.optimize import least_squares

from .models import (
    DHWStandbyCalibrationDataset,
    DHWStandbyCalibrationResult,
    DHWStandbyCalibrationSettings,
)


@dataclass(frozen=True, slots=True)
class _CalibrationArrays:
    """Vectorised numerical view of the DHW standby calibration dataset."""

    dt_hours: np.ndarray
    t_top_start_c: np.ndarray
    t_top_end_c: np.ndarray
    t_bot_start_c: np.ndarray
    t_bot_end_c: np.ndarray
    boiler_ambient_mean_c: np.ndarray


def _as_arrays(dataset: DHWStandbyCalibrationDataset) -> _CalibrationArrays:
    """Convert immutable standby sample objects into NumPy arrays for optimisation."""
    samples = dataset.samples
    return _CalibrationArrays(
        dt_hours=np.array([sample.dt_hours for sample in samples], dtype=float),
        t_top_start_c=np.array([sample.t_top_start_c for sample in samples], dtype=float),
        t_top_end_c=np.array([sample.t_top_end_c for sample in samples], dtype=float),
        t_bot_start_c=np.array([sample.t_bot_start_c for sample in samples], dtype=float),
        t_bot_end_c=np.array([sample.t_bot_end_c for sample in samples], dtype=float),
        boiler_ambient_mean_c=np.array([sample.boiler_ambient_mean_c for sample in samples], dtype=float),
    )


def _weighted_mean_tank_temperature_c(
    t_top_c: np.ndarray,
    t_bot_c: np.ndarray,
    settings: DHWStandbyCalibrationSettings,
) -> np.ndarray:
    """Return the capacity-weighted mean tank temperature [°C].

    Implements the derived DHW mean temperature definition from §8.1:

        T_dhw = (C_top · T_top + C_bot · T_bot) / (C_top + C_bot)
    """
    c_top = settings.reference_c_top_kwh_per_k
    c_bot = settings.reference_c_bot_kwh_per_k
    return (c_top * t_top_c + c_bot * t_bot_c) / (c_top + c_bot)


def _predict_mean_tank_temperature_end(
    arrays: _CalibrationArrays,
    settings: DHWStandbyCalibrationSettings,
    *,
    tau_standby_hours: float,
) -> np.ndarray:
    """Evaluate the forward-Euler one-step standby envelope on a batch of samples.

    This is the conservative single-state reduction of the full DHW energy balance
    for quasi-mixed no-DHW windows:

        dT_dhw/dt = -(T_dhw - T_amb) / tau_standby
    """
    mean_start_c = _weighted_mean_tank_temperature_c(
        arrays.t_top_start_c,
        arrays.t_bot_start_c,
        settings,
    )
    return mean_start_c + arrays.dt_hours / tau_standby_hours * (
        -(mean_start_c - arrays.boiler_ambient_mean_c)
    )


def calibrate_dhw_standby_loss(
    dataset: DHWStandbyCalibrationDataset,
    settings: DHWStandbyCalibrationSettings,
) -> DHWStandbyCalibrationResult:
    """Fit the identifiable DHW standby time constant from quasi-mixed telemetry.

    Args:
        dataset: Validated standby transition samples.
        settings: Bounds, initial guesses, and injected layer capacities.

    Returns:
        Calibrated standby envelope parameters and fit diagnostics.
    """
    if dataset.sample_count < settings.min_sample_count:
        raise ValueError(
            f"dataset must contain at least {settings.min_sample_count} samples; "
            f"received {dataset.sample_count}."
        )

    arrays = _as_arrays(dataset)
    mean_end_c = _weighted_mean_tank_temperature_c(
        arrays.t_top_end_c,
        arrays.t_bot_end_c,
        settings,
    )

    def residuals(theta: np.ndarray) -> np.ndarray:
        tau_standby_hours = float(theta[0])
        predictions = _predict_mean_tank_temperature_end(
            arrays,
            settings,
            tau_standby_hours=tau_standby_hours,
        )
        return predictions - mean_end_c

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
        raise RuntimeError(f"DHW standby-loss calibration failed: {result.message}")

    fitted_residuals = residuals(result.x)
    rmse_mean_tank_temperature_c = sqrt(float(np.mean(np.square(fitted_residuals))))
    max_abs_residual_c = float(np.max(np.abs(fitted_residuals)))
    tau_standby_hours = float(result.x[0])
    suggested_r_loss_k_per_kw = 2.0 * tau_standby_hours / settings.reference_c_total_kwh_per_k

    return DHWStandbyCalibrationResult(
        tau_standby_hours=tau_standby_hours,
        suggested_r_loss_k_per_kw=suggested_r_loss_k_per_kw,
        reference_c_total_kwh_per_k=settings.reference_c_total_kwh_per_k,
        rmse_mean_tank_temperature_c=rmse_mean_tank_temperature_c,
        max_abs_residual_c=max_abs_residual_c,
        sample_count=dataset.sample_count,
        dataset_start_utc=dataset.start_utc,
        dataset_end_utc=dataset.end_utc,
        optimizer_status=str(result.message),
        optimizer_cost=float(result.cost),
    )

