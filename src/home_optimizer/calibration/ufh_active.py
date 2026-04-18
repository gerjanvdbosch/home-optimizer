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

    segment_index: np.ndarray
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


@dataclass(frozen=True, slots=True)
class _ThetaSpec:
    """Metadata describing the optimisation vector used by the active fitter."""

    parameter_names: tuple[str, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    theta_prior: np.ndarray
    fit_initial_floor_temperature_offset: bool


def _as_arrays(dataset: UFHActiveCalibrationDataset) -> _ActiveCalibrationArrays:
    """Convert immutable replay samples into NumPy arrays for optimisation."""
    samples = dataset.samples
    return _ActiveCalibrationArrays(
        segment_index=np.array([sample.segment_index for sample in samples], dtype=int),
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


def _build_theta_spec(settings: UFHActiveCalibrationSettings) -> _ThetaSpec:
    """Return the optimiser-vector specification for the active UFH fit."""
    reference = settings.reference_parameters
    reference_values = {
        "C_r": reference.C_r,
        "C_b": reference.C_b,
        "R_br": reference.R_br,
        "R_ro": reference.R_ro,
    }
    parameter_names = list(_parameter_names(settings))
    theta_prior = [reference_values[name] for name in parameter_names]
    lower_bounds = [reference_values[name] * settings.min_parameter_ratio for name in parameter_names]
    upper_bounds = [reference_values[name] * settings.max_parameter_ratio for name in parameter_names]

    if settings.fit_initial_floor_temperature_offset:
        parameter_names.append("initial_floor_temperature_offset_c")
        theta_prior.append(settings.initial_floor_temperature_offset_c)
        lower_bounds.append(settings.min_initial_floor_temperature_offset_c)
        upper_bounds.append(settings.max_initial_floor_temperature_offset_c)

    lower_bounds_arr = np.array(lower_bounds, dtype=float)
    upper_bounds_arr = np.array(upper_bounds, dtype=float)
    if np.any(lower_bounds_arr <= 0.0) and not settings.fit_initial_floor_temperature_offset:
        raise ValueError("All physical calibration parameter lower bounds must remain strictly positive.")
    if np.any(lower_bounds_arr[:-1] <= 0.0) and settings.fit_initial_floor_temperature_offset:
        raise ValueError("All physical calibration parameter lower bounds must remain strictly positive.")
    if np.any(lower_bounds_arr >= upper_bounds_arr):
        raise ValueError("Each calibration parameter lower bound must be < its upper bound.")
    return _ThetaSpec(
        parameter_names=tuple(parameter_names),
        lower_bounds=lower_bounds_arr,
        upper_bounds=upper_bounds_arr,
        theta_prior=np.array(theta_prior, dtype=float),
        fit_initial_floor_temperature_offset=settings.fit_initial_floor_temperature_offset,
    )


def _initial_theta(settings: UFHActiveCalibrationSettings) -> np.ndarray:
    """Return the initial optimisation vector from the reference parameter set."""
    return _build_theta_spec(settings).theta_prior.copy()


def _theta_bounds(settings: UFHActiveCalibrationSettings) -> tuple[np.ndarray, np.ndarray]:
    """Construct physical parameter bounds relative to the reference set."""
    theta_spec = _build_theta_spec(settings)
    return theta_spec.lower_bounds.copy(), theta_spec.upper_bounds.copy()


def _parameters_from_theta(theta: np.ndarray, settings: UFHActiveCalibrationSettings) -> ThermalParameters:
    """Map the optimisation vector back to a full UFH parameter object."""
    theta_arr = np.asarray(theta, dtype=float)
    theta_spec = _build_theta_spec(settings)
    expected_shape = (theta_spec.theta_prior.size,)
    if theta_arr.shape != expected_shape:
        raise ValueError(f"theta must have shape {expected_shape}.")

    reference = settings.reference_parameters
    values = {
        "C_r": reference.C_r,
        "C_b": reference.C_b,
        "R_br": reference.R_br,
        "R_ro": reference.R_ro,
    }
    physical_parameter_count = len(_parameter_names(settings))
    for name, value in zip(_parameter_names(settings), theta_arr[:physical_parameter_count], strict=True):
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


def _initial_floor_offset_from_theta(theta: np.ndarray, settings: UFHActiveCalibrationSettings) -> float:
    """Return the nuisance floor offset encoded in the optimisation vector [°C]."""
    if not settings.fit_initial_floor_temperature_offset:
        return settings.initial_floor_temperature_offset_c
    theta_arr = np.asarray(theta, dtype=float)
    return float(theta_arr[-1])


def _regularization_residual(theta: np.ndarray, settings: UFHActiveCalibrationSettings) -> np.ndarray:
    """Return optional prior residuals anchored to the reference parameter set."""
    theta_arr = np.asarray(theta, dtype=float)
    residual_blocks: list[np.ndarray] = []
    physical_parameter_count = len(_parameter_names(settings))

    if settings.regularization_weight > 0.0:
        theta_prior = _initial_theta(settings)[:physical_parameter_count]
        theta_physical = theta_arr[:physical_parameter_count]
        scales = np.maximum(theta_prior * settings.regularization_scale_ratio, _MIN_REGULARIZATION_SCALE)
        residual_blocks.append(
            np.sqrt(settings.regularization_weight) * (theta_physical - theta_prior) / scales
        )

    if settings.fit_initial_floor_temperature_offset and settings.initial_floor_offset_regularization_weight > 0.0:
        offset_residual = np.array(
            [
                np.sqrt(settings.initial_floor_offset_regularization_weight)
                * (theta_arr[-1] - settings.initial_floor_temperature_offset_c)
                / max(settings.initial_floor_offset_scale_c, _MIN_REGULARIZATION_SCALE)
            ],
            dtype=float,
        )
        residual_blocks.append(offset_residual)

    if not residual_blocks:
        return np.empty(0, dtype=float)
    return np.concatenate(residual_blocks)


def _replay_with_parameters(
    arrays: _ActiveCalibrationArrays,
    settings: UFHActiveCalibrationSettings,
    parameters: ThermalParameters,
    *,
    initial_floor_temperature_offset_c: float,
) -> _ReplayDiagnostics:
    """Replay the active UFH trajectory and collect room-temperature innovations.

    This function implements the phase-2 calibration objective: evaluate the
    candidate RC parameter set by replaying the exact Kalman architecture used in
    production for UFH state estimation.
    """
    if initial_floor_temperature_offset_c < settings.min_initial_floor_temperature_offset_c:
        raise ValueError("initial_floor_temperature_offset_c violates its lower physical bound.")
    if initial_floor_temperature_offset_c > settings.max_initial_floor_temperature_offset_c:
        raise ValueError("initial_floor_temperature_offset_c violates its upper physical bound.")

    model = ThermalModel(parameters)
    noise = KalmanNoiseParameters(
        process_covariance=np.diag(
            [settings.process_noise_room_k2, settings.process_noise_floor_k2]
        ),
        measurement_variance=settings.measurement_variance_k2,
    )
    initial_covariance = np.diag(
        [settings.initial_room_covariance_k2, settings.initial_floor_covariance_k2]
    )

    innovations: list[float] = []
    filter_: UFHKalmanFilter | None = None
    previous_segment_index: int | None = None
    for index in range(arrays.room_temperature_end_c.size):
        current_segment_index = int(arrays.segment_index[index])
        if previous_segment_index is None or current_segment_index != previous_segment_index:
            # Implements phase-2.5 replay reset at each contiguous UFH run boundary.
            initial_state_c = np.array(
                [
                    arrays.room_temperature_start_c[index],
                    arrays.room_temperature_start_c[index] + initial_floor_temperature_offset_c,
                ],
                dtype=float,
            )
            filter_ = UFHKalmanFilter(
                model=model,
                noise=noise,
                initial_state_c=initial_state_c,
                initial_covariance=initial_covariance,
            )
            previous_segment_index = current_segment_index

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
        if filter_ is None:
            raise RuntimeError("UFH calibration replay filter was not initialised for the current segment.")
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
            diagnostics = _replay_with_parameters(
                arrays,
                settings,
                parameters,
                initial_floor_temperature_offset_c=_initial_floor_offset_from_theta(theta, settings),
            )
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
    fitted_initial_floor_temperature_offset_c = _initial_floor_offset_from_theta(result.x, settings)
    diagnostics = _replay_with_parameters(
        arrays,
        settings,
        fitted_parameters,
        initial_floor_temperature_offset_c=fitted_initial_floor_temperature_offset_c,
    )
    return UFHActiveCalibrationResult(
        fitted_parameters=fitted_parameters,
        fit_c_r=settings.fit_c_r,
        fit_initial_floor_temperature_offset=settings.fit_initial_floor_temperature_offset,
        fitted_initial_floor_temperature_offset_c=fitted_initial_floor_temperature_offset_c,
        rmse_room_temperature_c=diagnostics.rmse_room_temperature_c,
        max_abs_innovation_c=diagnostics.max_abs_innovation_c,
        sample_count=dataset.sample_count,
        segment_count=dataset.segment_count,
        dataset_start_utc=dataset.start_utc,
        dataset_end_utc=dataset.end_utc,
        optimizer_status=str(result.message),
        optimizer_cost=float(result.cost),
    )

