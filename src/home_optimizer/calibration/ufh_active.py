"""Active UFH RC calibration via Kalman-filter replay.

This second-stage fitter uses the existing 2-state UFH physics and the generic
Kalman architecture to learn the identifiable active-heating parameters from
historical telemetry:

* ``R_br`` [K/kW] — floor-to-room thermal resistance
* ``R_ro`` [K/kW] — room-to-outdoor thermal resistance
* ``C_b``  [kWh/K] — floor / slab thermal capacity
* optionally ``C_r`` [kWh/K] — room-air effective thermal capacity

The optimisation objective is the batch vector of room-temperature innovations
produced by :class:`home_optimizer.kalman.UFHKalmanFilter`.  This keeps the
hidden floor state ``T_b`` inside the Kalman architecture instead of treating it
as a directly observed regression target.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.optimize import least_squares

from .models import UFHActiveCalibrationDataset, UFHActiveCalibrationResult, UFHActiveCalibrationSettings
from ..kalman import UFHKalmanFilter
from ..thermal_model import ThermalModel, solar_gain_kw
from ..types import KalmanNoiseParameters, ThermalParameters

_INVALID_PARAMETER_RESIDUAL_C: float = 1_000.0
_MIN_REGULARIZATION_SCALE: float = 1e-9


@dataclass(frozen=True, slots=True)
class _ActiveCalibrationArrays:
    """Vectorised numerical view of the active UFH replay dataset."""

    room_temperature_start_c: np.ndarray
    room_temperature_end_c: np.ndarray
    outdoor_temperature_mean_c: np.ndarray
    gti_w_per_m2: np.ndarray
    internal_gain_proxy_kw: np.ndarray
    ufh_power_mean_kw: np.ndarray


@dataclass(frozen=True, slots=True)
class _ReplayDiagnostics:
    """Summary diagnostics from one Kalman replay under a candidate parameter set."""

    innovations_c: np.ndarray

    @property
    def rmse_room_temperature_c(self) -> float:
        """Root-mean-square room-temperature innovation [°C]."""
        return sqrt(float(np.mean(np.square(self.innovations_c))))

    @property
    def max_abs_innovation_c(self) -> float:
        """Maximum absolute room-temperature innovation [°C]."""
        return float(np.max(np.abs(self.innovations_c)))


def _as_arrays(dataset: UFHActiveCalibrationDataset) -> _ActiveCalibrationArrays:
    """Convert immutable replay samples into NumPy arrays for optimisation."""
    samples = dataset.samples
    return _ActiveCalibrationArrays(
        room_temperature_start_c=np.array([sample.room_temperature_start_c for sample in samples], dtype=float),
        room_temperature_end_c=np.array([sample.room_temperature_end_c for sample in samples], dtype=float),
        outdoor_temperature_mean_c=np.array(
            [sample.outdoor_temperature_mean_c for sample in samples], dtype=float
        ),
        gti_w_per_m2=np.array([sample.gti_w_per_m2 for sample in samples], dtype=float),
        internal_gain_proxy_kw=np.array([sample.internal_gain_proxy_kw for sample in samples], dtype=float),
        ufh_power_mean_kw=np.array([sample.ufh_power_mean_kw for sample in samples], dtype=float),
    )


def _parameter_names(settings: UFHActiveCalibrationSettings) -> tuple[str, ...]:
    """Return the ordered parameter vector names used by the optimiser."""
    names = ["C_b", "R_br", "R_ro"]
    if settings.fit_c_r:
        names.insert(0, "C_r")
    return tuple(names)


def _initial_theta(settings: UFHActiveCalibrationSettings) -> np.ndarray:
    """Return the initial optimisation vector from the reference parameter set."""
    p = settings.reference_parameters
    values = {"C_r": p.C_r, "C_b": p.C_b, "R_br": p.R_br, "R_ro": p.R_ro}
    return np.array([values[name] for name in _parameter_names(settings)], dtype=float)


def _theta_bounds(settings: UFHActiveCalibrationSettings) -> tuple[np.ndarray, np.ndarray]:
    """Construct physical parameter bounds relative to the reference set."""
    p = settings.reference_parameters
    reference_values = {"C_r": p.C_r, "C_b": p.C_b, "R_br": p.R_br, "R_ro": p.R_ro}
    lower = []
    upper = []
    for name in _parameter_names(settings):
        base_value = reference_values[name]
        lower.append(base_value * settings.min_parameter_ratio)
        upper.append(base_value * settings.max_parameter_ratio)
    lower_arr = np.array(lower, dtype=float)
    upper_arr = np.array(upper, dtype=float)
    if np.any(lower_arr <= 0.0):
        raise ValueError("All calibration parameter lower bounds must remain strictly positive.")
    if np.any(lower_arr >= upper_arr):
        raise ValueError("Each calibration parameter lower bound must be < its upper bound.")
    return lower_arr, upper_arr


def _parameters_from_theta(theta: np.ndarray, settings: UFHActiveCalibrationSettings) -> ThermalParameters:
    """Map the optimisation vector back to a full UFH parameter object."""
    theta_arr = np.asarray(theta, dtype=float)
    expected_shape = (_initial_theta(settings).size,)
    if theta_arr.shape != expected_shape:
        raise ValueError(f"theta must have shape {expected_shape}.")

    reference = settings.reference_parameters
    values = {
        "C_r": reference.C_r,
        "C_b": reference.C_b,
        "R_br": reference.R_br,
        "R_ro": reference.R_ro,
    }
    for name, value in zip(_parameter_names(settings), theta_arr, strict=True):
        values[name] = float(value)

    return ThermalParameters(
        dt_hours=reference.dt_hours,
        C_r=values["C_r"],
        C_b=values["C_b"],
        R_br=values["R_br"],
        R_ro=values["R_ro"],
        alpha=reference.alpha,
        eta=reference.eta,
        A_glass=reference.A_glass,
    )


def _regularization_residual(theta: np.ndarray, settings: UFHActiveCalibrationSettings) -> np.ndarray:
    """Return optional prior residuals anchored to the reference parameter set."""
    if settings.regularization_weight == 0.0:
        return np.empty(0, dtype=float)
    theta_prior = _initial_theta(settings)
    scales = np.maximum(theta_prior * settings.regularization_scale_ratio, _MIN_REGULARIZATION_SCALE)
    return np.sqrt(settings.regularization_weight) * (np.asarray(theta, dtype=float) - theta_prior) / scales


def _replay_with_parameters(
    arrays: _ActiveCalibrationArrays,
    settings: UFHActiveCalibrationSettings,
    parameters: ThermalParameters,
) -> _ReplayDiagnostics:
    """Replay the active UFH trajectory and collect room-temperature innovations.

    This function implements the phase-2 calibration objective: evaluate the
    candidate RC parameter set by replaying the exact Kalman architecture used in
    production for UFH state estimation.
    """
    model = ThermalModel(parameters)
    noise = KalmanNoiseParameters(
        process_covariance=np.diag(
            [settings.process_noise_room_k2, settings.process_noise_floor_k2]
        ),
        measurement_variance=settings.measurement_variance_k2,
    )
    initial_state_c = np.array(
        [
            arrays.room_temperature_start_c[0],
            arrays.room_temperature_start_c[0] + settings.initial_floor_temperature_offset_c,
        ],
        dtype=float,
    )
    initial_covariance = np.diag(
        [settings.initial_room_covariance_k2, settings.initial_floor_covariance_k2]
    )
    filter_ = UFHKalmanFilter(
        model=model,
        noise=noise,
        initial_state_c=initial_state_c,
        initial_covariance=initial_covariance,
    )

    innovations: list[float] = []
    for index in range(arrays.room_temperature_end_c.size):
        q_solar_kw = float(
            solar_gain_kw(
                arrays.gti_w_per_m2[index],
                glass_area_m2=parameters.A_glass,
                transmittance=parameters.eta,
            )
        )
        disturbance = np.array(
            [
                arrays.outdoor_temperature_mean_c[index],
                q_solar_kw,
                arrays.internal_gain_proxy_kw[index],
            ],
            dtype=float,
        )
        _, innovation_c, _ = filter_.step(
            control_kw=float(arrays.ufh_power_mean_kw[index]),
            disturbance=disturbance,
            room_temp_measurement_c=float(arrays.room_temperature_end_c[index]),
        )
        innovations.append(float(innovation_c))
    return _ReplayDiagnostics(innovations_c=np.array(innovations, dtype=float))


def calibrate_ufh_active_rc(
    dataset: UFHActiveCalibrationDataset,
    settings: UFHActiveCalibrationSettings,
) -> UFHActiveCalibrationResult:
    """Fit active UFH RC parameters from historical telemetry via Kalman replay.

    Args:
        dataset: Validated active replay dataset built from consecutive UFH buckets.
        settings: Reference parameters, optimiser bounds, and Kalman-noise settings.

    Returns:
        Fitted UFH thermal parameters and replay diagnostics.

    Raises:
        ValueError: If the dataset is too small for the configured minimum sample count.
        RuntimeError: If the numerical optimiser fails to converge.
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
            diagnostics = _replay_with_parameters(arrays, settings, parameters)
            return np.concatenate(
                [diagnostics.innovations_c, _regularization_residual(theta, settings)]
            )
        except ValueError:
            invalid = np.full(arrays.room_temperature_end_c.size, _INVALID_PARAMETER_RESIDUAL_C, dtype=float)
            return np.concatenate([invalid, _regularization_residual(theta, settings)])

    result = least_squares(
        residuals,
        x0=initial_theta,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
    )
    if not result.success:
        raise RuntimeError(f"UFH active RC calibration failed: {result.message}")

    fitted_parameters = _parameters_from_theta(result.x, settings)
    diagnostics = _replay_with_parameters(arrays, settings, fitted_parameters)
    return UFHActiveCalibrationResult(
        fitted_parameters=fitted_parameters,
        fit_c_r=settings.fit_c_r,
        rmse_room_temperature_c=diagnostics.rmse_room_temperature_c,
        max_abs_innovation_c=diagnostics.max_abs_innovation_c,
        sample_count=dataset.sample_count,
        dataset_start_utc=dataset.start_utc,
        dataset_end_utc=dataset.end_utc,
        optimizer_status=str(result.message),
        optimizer_cost=float(result.cost),
    )

