"""Offline calibration of the Carnot-based heat-pump COP model.

This stage fits the subset of COP-model parameters that are practically
identifiable from the currently persisted telemetry:

* ``T_supply_min`` [°C] — UFH heating-curve intercept
* ``heating_curve_slope`` [K/K] — UFH heating-curve slope
* ``eta_carnot`` [-] — shared Carnot efficiency factor

The remaining COP-model parameters stay fixed in this stage:

* ``T_ref_outdoor`` [°C]
* ``delta_T_cond`` [K]
* ``delta_T_evap`` [K]
* ``cop_min`` [-]
* ``cop_max`` [-]

Why this split?
----------------
The current telemetry can reliably tell us:

* which mode the heat pump was in (UFH / DHW)
* the bucket thermal energy [kWh]
* the bucket electrical energy [kWh]
* the outdoor temperature [°C]
* the commanded/target hydraulic supply temperature [°C]

That is sufficient to fit the UFH heating-curve shape against target-supply
telemetry and then fit one shared Carnot efficiency factor against measured
energy use.  It is *not* sufficient to robustly identify all approach
temperatures and per-mode efficiencies simultaneously without strong collinearity.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.optimize import least_squares

from .models import (
    COPCalibrationDataset,
    COPCalibrationResult,
    COPCalibrationSettings,
)
from ..cop_model import HeatPumpCOPModel, HeatPumpCOPParameters


@dataclass(frozen=True, slots=True)
class _CalibrationArrays:
    """Vectorised numerical view of the offline COP calibration dataset."""

    mode_name: np.ndarray
    outdoor_temperature_mean_c: np.ndarray
    supply_target_temperature_mean_c: np.ndarray
    thermal_energy_kwh: np.ndarray
    electric_energy_kwh: np.ndarray


@dataclass(frozen=True, slots=True)
class _HeatingCurveFit:
    """Intermediate UFH heating-curve fit result."""

    t_supply_min_c: float
    heating_curve_slope: float
    optimizer_status: str
    optimizer_cost: float
    rmse_supply_temperature_c: float


@dataclass(frozen=True, slots=True)
class _EtaFit:
    """Intermediate Carnot-efficiency fit result."""

    eta_carnot: float
    optimizer_status: str
    optimizer_cost: float


def _as_arrays(dataset: COPCalibrationDataset) -> _CalibrationArrays:
    """Convert immutable COP sample objects into NumPy arrays for optimisation."""
    samples = dataset.samples
    return _CalibrationArrays(
        mode_name=np.array([sample.mode_name for sample in samples], dtype=object),
        outdoor_temperature_mean_c=np.array(
            [sample.outdoor_temperature_mean_c for sample in samples],
            dtype=float,
        ),
        supply_target_temperature_mean_c=np.array(
            [sample.supply_target_temperature_mean_c for sample in samples],
            dtype=float,
        ),
        thermal_energy_kwh=np.array([sample.thermal_energy_kwh for sample in samples], dtype=float),
        electric_energy_kwh=np.array([sample.electric_energy_kwh for sample in samples], dtype=float),
    )


def _fit_heating_curve(
    arrays: _CalibrationArrays,
    settings: COPCalibrationSettings,
) -> _HeatingCurveFit:
    """Fit UFH heating-curve parameters against target supply temperature telemetry."""
    ufh_mask = arrays.mode_name == settings.ufh_mode_name
    t_out_ufh = arrays.outdoor_temperature_mean_c[ufh_mask]
    t_supply_target_ufh = arrays.supply_target_temperature_mean_c[ufh_mask]

    def residuals(theta: np.ndarray) -> np.ndarray:
        t_supply_min_c = float(theta[0])
        heating_curve_slope = float(theta[1])
        predicted_supply_c = t_supply_min_c + heating_curve_slope * np.maximum(
            settings.t_ref_outdoor_c - t_out_ufh,
            0.0,
        )
        return predicted_supply_c - t_supply_target_ufh

    initial_theta = np.array(
        [settings.initial_t_supply_min_c, settings.initial_heating_curve_slope],
        dtype=float,
    )
    lower_bounds = np.array(
        [settings.min_t_supply_min_c, settings.min_heating_curve_slope],
        dtype=float,
    )
    upper_bounds = np.array(
        [settings.max_t_supply_min_c, settings.max_heating_curve_slope],
        dtype=float,
    )

    result = least_squares(
        residuals,
        x0=initial_theta,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
    )
    if not result.success:
        raise RuntimeError(f"COP heating-curve calibration failed: {result.message}")

    fitted_residuals = residuals(result.x)
    rmse_supply_temperature_c = sqrt(float(np.mean(np.square(fitted_residuals))))
    return _HeatingCurveFit(
        t_supply_min_c=float(result.x[0]),
        heating_curve_slope=float(result.x[1]),
        optimizer_status=str(result.message),
        optimizer_cost=float(result.cost),
        rmse_supply_temperature_c=rmse_supply_temperature_c,
    )


def _predicted_supply_temperature_c(
    arrays: _CalibrationArrays,
    settings: COPCalibrationSettings,
    *,
    t_supply_min_c: float,
    heating_curve_slope: float,
) -> np.ndarray:
    """Return the per-sample supply temperature used by the COP model [°C].

    * UFH samples use the fitted heating curve.
    * DHW samples use the measured target supply temperature directly.
    """
    predicted_supply_c = arrays.supply_target_temperature_mean_c.copy()
    ufh_mask = arrays.mode_name == settings.ufh_mode_name
    predicted_supply_c[ufh_mask] = t_supply_min_c + heating_curve_slope * np.maximum(
        settings.t_ref_outdoor_c - arrays.outdoor_temperature_mean_c[ufh_mask],
        0.0,
    )
    return predicted_supply_c


def _build_cop_parameters(
    settings: COPCalibrationSettings,
    *,
    eta_carnot: float,
    t_supply_min_c: float,
    heating_curve_slope: float,
) -> HeatPumpCOPParameters:
    """Construct the calibrated COP-parameter object with fixed approach terms."""
    return HeatPumpCOPParameters(
        eta_carnot=eta_carnot,
        delta_T_cond=settings.delta_t_cond_k,
        delta_T_evap=settings.delta_t_evap_k,
        T_supply_min=t_supply_min_c,
        T_ref_outdoor=settings.t_ref_outdoor_c,
        heating_curve_slope=heating_curve_slope,
        cop_min=settings.cop_min,
        cop_max=settings.cop_max,
    )


def _fit_eta_carnot(
    arrays: _CalibrationArrays,
    settings: COPCalibrationSettings,
    *,
    t_supply_min_c: float,
    heating_curve_slope: float,
) -> _EtaFit:
    """Fit the shared Carnot efficiency factor against measured electrical energy."""
    predicted_supply_c = _predicted_supply_temperature_c(
        arrays,
        settings,
        t_supply_min_c=t_supply_min_c,
        heating_curve_slope=heating_curve_slope,
    )

    def residuals(theta: np.ndarray) -> np.ndarray:
        eta_carnot = float(theta[0])
        params = _build_cop_parameters(
            settings,
            eta_carnot=eta_carnot,
            t_supply_min_c=t_supply_min_c,
            heating_curve_slope=heating_curve_slope,
        )
        model = HeatPumpCOPModel(params)
        predicted_cop = model.cop_from_temperatures(
            t_supply=predicted_supply_c,
            t_out=arrays.outdoor_temperature_mean_c,
        )
        predicted_electric_energy_kwh = arrays.thermal_energy_kwh / predicted_cop
        return predicted_electric_energy_kwh - arrays.electric_energy_kwh

    result = least_squares(
        residuals,
        x0=np.array([settings.initial_eta_carnot], dtype=float),
        bounds=(
            np.array([settings.min_eta_carnot], dtype=float),
            np.array([settings.max_eta_carnot], dtype=float),
        ),
        method="trf",
    )
    if not result.success:
        raise RuntimeError(f"COP eta_carnot calibration failed: {result.message}")
    return _EtaFit(
        eta_carnot=float(result.x[0]),
        optimizer_status=str(result.message),
        optimizer_cost=float(result.cost),
    )


def calibrate_cop_model(
    dataset: COPCalibrationDataset,
    settings: COPCalibrationSettings,
) -> COPCalibrationResult:
    """Fit offline COP-model parameters from historical operating buckets.

    Args:
        dataset: Validated UFH/DHW operating buckets with measured energy use.
        settings: Bounds, fixed approach temperatures, and sample-count rules.

    Returns:
        Calibrated COP model parameters and fit diagnostics.
    """
    if dataset.sample_count < settings.min_sample_count:
        raise ValueError(
            f"dataset must contain at least {settings.min_sample_count} samples; "
            f"received {dataset.sample_count}."
        )
    if dataset.ufh_sample_count < settings.min_ufh_curve_sample_count:
        raise ValueError(
            f"dataset must contain at least {settings.min_ufh_curve_sample_count} UFH samples; "
            f"received {dataset.ufh_sample_count}."
        )

    arrays = _as_arrays(dataset)
    heating_curve_fit = _fit_heating_curve(arrays, settings)
    eta_fit = _fit_eta_carnot(
        arrays,
        settings,
        t_supply_min_c=heating_curve_fit.t_supply_min_c,
        heating_curve_slope=heating_curve_fit.heating_curve_slope,
    )
    fitted_parameters = _build_cop_parameters(
        settings,
        eta_carnot=eta_fit.eta_carnot,
        t_supply_min_c=heating_curve_fit.t_supply_min_c,
        heating_curve_slope=heating_curve_fit.heating_curve_slope,
    )
    model = HeatPumpCOPModel(fitted_parameters)
    predicted_supply_c = _predicted_supply_temperature_c(
        arrays,
        settings,
        t_supply_min_c=fitted_parameters.T_supply_min,
        heating_curve_slope=fitted_parameters.heating_curve_slope,
    )
    predicted_cop = model.cop_from_temperatures(
        t_supply=predicted_supply_c,
        t_out=arrays.outdoor_temperature_mean_c,
    )
    predicted_electric_energy_kwh = arrays.thermal_energy_kwh / predicted_cop
    actual_cop = arrays.thermal_energy_kwh / arrays.electric_energy_kwh

    return COPCalibrationResult(
        fitted_parameters=fitted_parameters,
        rmse_supply_temperature_c=heating_curve_fit.rmse_supply_temperature_c,
        rmse_electric_energy_kwh=sqrt(
            float(np.mean(np.square(predicted_electric_energy_kwh - arrays.electric_energy_kwh)))
        ),
        rmse_actual_cop=sqrt(float(np.mean(np.square(predicted_cop - actual_cop)))),
        sample_count=dataset.sample_count,
        ufh_sample_count=dataset.ufh_sample_count,
        dhw_sample_count=dataset.dhw_sample_count,
        dataset_start_utc=dataset.start_utc,
        dataset_end_utc=dataset.end_utc,
        heating_curve_optimizer_status=heating_curve_fit.optimizer_status,
        eta_optimizer_status=eta_fit.optimizer_status,
        heating_curve_optimizer_cost=heating_curve_fit.optimizer_cost,
        eta_optimizer_cost=eta_fit.optimizer_cost,
    )

