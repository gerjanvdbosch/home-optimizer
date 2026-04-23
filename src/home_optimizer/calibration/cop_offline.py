"""Offline calibration of the Carnot-based heat-pump COP model.

This stage fits the subset of COP-model parameters that are practically
identifiable from the currently persisted telemetry:

* ``T_supply_min`` [°C] — UFH heating-curve intercept
* ``T_ref_outdoor`` [°C] — UFH heating-curve balance-point / breakpoint
* ``heating_curve_slope`` [K/K] — UFH heating-curve slope
* ``eta_carnot_ufh`` [-] — UFH Carnot efficiency factor
* ``eta_carnot_dhw`` [-] — DHW Carnot efficiency factor

The remaining COP-model parameters stay fixed in this stage:

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
telemetry and then fit separate UFH/DHW Carnot efficiency factors against measured
energy use.  It is *not* sufficient to robustly identify all approach
temperatures and per-mode efficiencies simultaneously without strong collinearity.

The UFH breakpoint ``T_ref_outdoor`` is fitted only when the retained UFH samples
excite **both** sides of the breakpoint candidate. Without warm-side samples
(``T_out >= T_ref_outdoor``), the heating curve collapses to a single line and
``T_supply_min`` / ``T_ref_outdoor`` become structurally non-identifiable. In
that case the fitter keeps ``T_ref_outdoor`` fixed at the configured reference
value and reports this explicitly in the result metadata.
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
from ..domain.heat_pump.cop import HeatPumpCOPModel, HeatPumpCOPParameters, T_CELSIUS_TO_KELVIN

_MIN_DIAGNOSTIC_TEMPERATURE_LIFT_K: float = float(np.finfo(float).eps)


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
    t_ref_outdoor_c: float
    t_ref_outdoor_was_fitted: bool
    optimizer_status: str
    optimizer_cost: float
    rmse_supply_temperature_c: float


@dataclass(frozen=True, slots=True)
class _EtaFit:
    """Intermediate Carnot-efficiency fit result."""

    eta_carnot: float
    optimizer_status: str
    optimizer_cost: float


def _normalised_energy_weights(thermal_energy_kwh: np.ndarray) -> np.ndarray:
    """Return positive sample weights with mean value 1 based on delivered energy."""
    mean_thermal_energy_kwh = float(np.mean(thermal_energy_kwh))
    if mean_thermal_energy_kwh <= 0.0:
        raise ValueError("mean_thermal_energy_kwh must be strictly positive.")
    return thermal_energy_kwh / mean_thermal_energy_kwh


def _weighted_residuals(residuals: np.ndarray, *, weights: np.ndarray) -> np.ndarray:
    """Scale residuals by sqrt(weights) for weighted least squares."""
    return np.sqrt(weights) * residuals


def _ideal_carnot_cop(
    *,
    t_supply_c: np.ndarray,
    t_out_c: np.ndarray,
    settings: COPCalibrationSettings,
) -> np.ndarray:
    """Return the unclipped ideal Carnot COP used for diagnostic eta estimates."""
    t_cond_k = t_supply_c + settings.delta_t_cond_k + T_CELSIUS_TO_KELVIN
    t_evap_k = t_out_c - settings.delta_t_evap_k + T_CELSIUS_TO_KELVIN
    lift_k = np.maximum(t_cond_k - t_evap_k, _MIN_DIAGNOSTIC_TEMPERATURE_LIFT_K)
    return t_cond_k / lift_k


def _rmse(values: np.ndarray) -> float:
    """Return the root-mean-square of a residual vector."""
    return sqrt(float(np.mean(np.square(values))))


def _diagnostic_eta_carnot(
    actual_cop: np.ndarray,
    ideal_carnot_cop: np.ndarray,
    *,
    weights: np.ndarray,
) -> float:
    """Return the weighted mean η that best explains actual COP diagnostically."""
    eta_samples = actual_cop / ideal_carnot_cop
    return float(np.average(eta_samples, weights=weights))


def _initial_eta_carnot_guess(
    *,
    actual_cop: np.ndarray,
    ideal_carnot_cop: np.ndarray,
    settings: COPCalibrationSettings,
    weights: np.ndarray,
) -> float:
    """Return a fail-fast initial η guess for the clipped COP least-squares fit.

    Why
    ---
    The optimisation objective in :func:`_fit_eta_carnot` uses the same clipped
    COP model as the runtime MPC path. When the initial η lies on a region where
    **all** selected samples clip to ``cop_max``, the residual Jacobian with
    respect to η becomes zero and SciPy can terminate immediately at the
    user-provided initial guess even though a lower, physically better η exists.

    To avoid this flat-plateau failure mode, the initial guess is derived from
    the weighted diagnostic ratio ``actual_cop / ideal_carnot_cop`` and then
    projected strictly below the η threshold where every sample would saturate at
    ``cop_max``.

    Args:
        actual_cop: Measured per-sample COP array [-], shape (N,).
        ideal_carnot_cop: Unclipped ideal Carnot COP array [-], shape (N,).
        settings: Validated COP calibration settings containing η and COP bounds.
        weights: Positive weighted-least-squares sample weights [-], shape (N,).

    Returns:
        Initial η guess [-] for the one-dimensional least-squares optimisation.

    Raises:
        ValueError: If the configured ``[min_eta_carnot, max_eta_carnot]`` range
            is fully clipped for all samples, making η unidentifiable under the
            current ``cop_max`` bound.
    """
    diagnostic_eta = _diagnostic_eta_carnot(
        actual_cop,
        ideal_carnot_cop,
        weights=weights,
    )
    max_eta_before_full_clip = float(np.max(settings.cop_max / ideal_carnot_cop))
    if settings.min_eta_carnot >= max_eta_before_full_clip:
        raise ValueError(
            "eta_carnot is unidentifiable for the selected COP calibration samples: "
            "all admissible eta values clip every sample to cop_max. Increase cop_max "
            "or use calibration samples with a larger temperature lift."
        )
    eta_initial = float(np.clip(diagnostic_eta, settings.min_eta_carnot, settings.max_eta_carnot))
    if eta_initial >= max_eta_before_full_clip:
        eta_initial = float(np.nextafter(max_eta_before_full_clip, settings.min_eta_carnot))
    return eta_initial


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


def _predict_heating_curve_supply_temperature_c(
    *,
    t_out_c: np.ndarray,
    t_supply_min_c: float,
    t_ref_outdoor_c: float,
    heating_curve_slope: float,
) -> np.ndarray:
    """Return the UFH heating-curve supply target [°C] for one outdoor profile.

    Implements the runtime heating curve

        T_supply = T_supply_min + slope · max(T_ref_outdoor - T_out, 0)

    from ``cop_model.py`` so the offline fitter stays algebraically identical to
    the runtime COP pre-calculation.
    """
    return t_supply_min_c + heating_curve_slope * np.maximum(t_ref_outdoor_c - t_out_c, 0.0)


def _has_t_ref_outdoor_excitation(
    t_out_ufh: np.ndarray,
    settings: COPCalibrationSettings,
) -> bool:
    """Return whether the UFH dataset can identify the heating-curve breakpoint.

    ``T_ref_outdoor`` is only structurally observable when the retained UFH data
    include samples on both sides of the breakpoint candidate: warm-side samples
    constrain the flat ``T_supply_min`` branch, while cold-side samples constrain
    the sloped branch.
    """
    warm_side_sample_count = int(np.count_nonzero(t_out_ufh >= settings.t_ref_outdoor_c))
    cold_side_sample_count = int(np.count_nonzero(t_out_ufh < settings.t_ref_outdoor_c))
    return (
        warm_side_sample_count >= settings.min_ufh_t_ref_side_sample_count
        and cold_side_sample_count >= settings.min_ufh_t_ref_side_sample_count
    )


def _fit_heating_curve(
    arrays: _CalibrationArrays,
    settings: COPCalibrationSettings,
) -> _HeatingCurveFit:
    """Fit UFH heating-curve parameters against target supply temperature telemetry."""
    ufh_mask = arrays.mode_name == settings.ufh_mode_name
    t_out_ufh = arrays.outdoor_temperature_mean_c[ufh_mask]
    t_supply_target_ufh = arrays.supply_target_temperature_mean_c[ufh_mask]
    weights_ufh = _normalised_energy_weights(arrays.thermal_energy_kwh[ufh_mask])

    t_ref_outdoor_is_identifiable = settings.fit_t_ref_outdoor and _has_t_ref_outdoor_excitation(
        t_out_ufh,
        settings,
    )

    def residuals(theta: np.ndarray) -> np.ndarray:
        t_supply_min_c = float(theta[0])
        heating_curve_slope = float(theta[1])
        t_ref_outdoor_c = float(theta[2]) if t_ref_outdoor_is_identifiable else settings.t_ref_outdoor_c
        predicted_supply_c = _predict_heating_curve_supply_temperature_c(
            t_out_c=t_out_ufh,
            t_supply_min_c=t_supply_min_c,
            t_ref_outdoor_c=t_ref_outdoor_c,
            heating_curve_slope=heating_curve_slope,
        )
        return _weighted_residuals(predicted_supply_c - t_supply_target_ufh, weights=weights_ufh)

    if t_ref_outdoor_is_identifiable:
        initial_theta = np.array(
            [
                settings.initial_t_supply_min_c,
                settings.initial_heating_curve_slope,
                settings.t_ref_outdoor_c,
            ],
            dtype=float,
        )
        lower_bounds = np.array(
            [
                settings.min_t_supply_min_c,
                settings.min_heating_curve_slope,
                settings.min_t_ref_outdoor_c,
            ],
            dtype=float,
        )
        upper_bounds = np.array(
            [
                settings.max_t_supply_min_c,
                settings.max_heating_curve_slope,
                settings.max_t_ref_outdoor_c,
            ],
            dtype=float,
        )
    else:
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
        loss=settings.heating_curve_loss_name,
        f_scale=settings.heating_curve_loss_scale_c,
    )
    if not result.success:
        raise RuntimeError(f"COP heating-curve calibration failed: {result.message}")

    fitted_t_ref_outdoor_c = float(result.x[2]) if t_ref_outdoor_is_identifiable else settings.t_ref_outdoor_c
    predicted_supply_c = _predict_heating_curve_supply_temperature_c(
        t_out_c=t_out_ufh,
        t_supply_min_c=float(result.x[0]),
        t_ref_outdoor_c=fitted_t_ref_outdoor_c,
        heating_curve_slope=float(result.x[1]),
    )
    rmse_supply_temperature_c = _rmse(predicted_supply_c - t_supply_target_ufh)
    optimizer_status = str(result.message)
    if not t_ref_outdoor_is_identifiable:
        optimizer_status = (
            f"{optimizer_status} | T_ref_outdoor kept fixed at {settings.t_ref_outdoor_c:.3f} °C "
            "because the retained UFH samples do not excite both sides of the breakpoint."
        )
    return _HeatingCurveFit(
        t_supply_min_c=float(result.x[0]),
        heating_curve_slope=float(result.x[1]),
        t_ref_outdoor_c=fitted_t_ref_outdoor_c,
        t_ref_outdoor_was_fitted=t_ref_outdoor_is_identifiable,
        optimizer_status=optimizer_status,
        optimizer_cost=float(result.cost),
        rmse_supply_temperature_c=rmse_supply_temperature_c,
    )


def _predicted_supply_temperature_c(
    arrays: _CalibrationArrays,
    settings: COPCalibrationSettings,
    *,
    t_supply_min_c: float,
    t_ref_outdoor_c: float,
    heating_curve_slope: float,
) -> np.ndarray:
    """Return the per-sample supply temperature used by the COP model [°C].

    * UFH samples use the fitted heating curve.
    * DHW samples use the measured target supply temperature directly.
    """
    predicted_supply_c = arrays.supply_target_temperature_mean_c.copy()
    ufh_mask = arrays.mode_name == settings.ufh_mode_name
    predicted_supply_c[ufh_mask] = _predict_heating_curve_supply_temperature_c(
        t_out_c=arrays.outdoor_temperature_mean_c[ufh_mask],
        t_supply_min_c=t_supply_min_c,
        t_ref_outdoor_c=t_ref_outdoor_c,
        heating_curve_slope=heating_curve_slope,
    )
    return predicted_supply_c


def _build_cop_parameters(
    settings: COPCalibrationSettings,
    *,
    eta_carnot_ufh: float,
    eta_carnot_dhw: float,
    t_supply_min_c: float,
    t_ref_outdoor_c: float,
    heating_curve_slope: float,
) -> HeatPumpCOPParameters:
    """Construct the calibrated COP-parameter object with fixed approach terms."""
    return HeatPumpCOPParameters(
        eta_carnot_ufh=eta_carnot_ufh,
        eta_carnot_dhw=eta_carnot_dhw,
        delta_T_cond=settings.delta_t_cond_k,
        delta_T_evap=settings.delta_t_evap_k,
        T_supply_min=t_supply_min_c,
        T_ref_outdoor=t_ref_outdoor_c,
        heating_curve_slope=heating_curve_slope,
        cop_min=settings.cop_min,
        cop_max=settings.cop_max,
    )


def _fit_eta_carnot_for_mask(
    arrays: _CalibrationArrays,
    settings: COPCalibrationSettings,
    *,
    mode_mask: np.ndarray,
    eta_label: str,
    t_supply_min_c: float,
    t_ref_outdoor_c: float,
    heating_curve_slope: float,
) -> _EtaFit:
    """Fit one mode-specific Carnot efficiency factor against measured electrical energy."""
    predicted_supply_c = _predicted_supply_temperature_c(
        arrays,
        settings,
        t_supply_min_c=t_supply_min_c,
        t_ref_outdoor_c=t_ref_outdoor_c,
        heating_curve_slope=heating_curve_slope,
    )
    predicted_supply_mode_c = predicted_supply_c[mode_mask]
    outdoor_mode_c = arrays.outdoor_temperature_mean_c[mode_mask]
    thermal_mode_kwh = arrays.thermal_energy_kwh[mode_mask]
    electric_mode_kwh = arrays.electric_energy_kwh[mode_mask]
    weights = _normalised_energy_weights(thermal_mode_kwh)
    actual_cop = thermal_mode_kwh / electric_mode_kwh
    ideal_carnot_cop = _ideal_carnot_cop(
        t_supply_c=predicted_supply_mode_c,
        t_out_c=outdoor_mode_c,
        settings=settings,
    )
    initial_eta_carnot = _initial_eta_carnot_guess(
        actual_cop=actual_cop,
        ideal_carnot_cop=ideal_carnot_cop,
        settings=settings,
        weights=weights,
    )

    def residuals(theta: np.ndarray) -> np.ndarray:
        eta_carnot = float(theta[0])
        model = HeatPumpCOPModel(
            _build_cop_parameters(
                settings,
                eta_carnot_ufh=eta_carnot,
                eta_carnot_dhw=eta_carnot,
                t_supply_min_c=t_supply_min_c,
                t_ref_outdoor_c=t_ref_outdoor_c,
                heating_curve_slope=heating_curve_slope,
            )
        )
        predicted_cop = model._cop_from_temperatures_with_eta(
            t_supply=predicted_supply_mode_c,
            t_out=outdoor_mode_c,
            eta_carnot=eta_carnot,
        )
        predicted_electric_energy_kwh = thermal_mode_kwh / predicted_cop
        return _weighted_residuals(
            predicted_electric_energy_kwh - electric_mode_kwh,
            weights=weights,
        )

    result = least_squares(
        residuals,
        x0=np.array([initial_eta_carnot], dtype=float),
        bounds=(
            np.array([settings.min_eta_carnot], dtype=float),
            np.array([settings.max_eta_carnot], dtype=float),
        ),
        method="trf",
        loss=settings.eta_loss_name,
        f_scale=settings.eta_loss_scale_kwh,
    )
    if not result.success:
        raise RuntimeError(f"COP {eta_label} calibration failed: {result.message}")
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
        Calibrated COP model parameters and fit diagnostics, including per-mode
        UFH/DHW RMSE and diagnostic-only Carnot-efficiency estimates.
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
    ufh_mask = arrays.mode_name == settings.ufh_mode_name
    dhw_mask = arrays.mode_name == settings.dhw_mode_name

    eta_fit_ufh = _fit_eta_carnot_for_mask(
        arrays,
        settings,
        mode_mask=ufh_mask,
        eta_label="eta_carnot_ufh",
        t_supply_min_c=heating_curve_fit.t_supply_min_c,
        t_ref_outdoor_c=heating_curve_fit.t_ref_outdoor_c,
        heating_curve_slope=heating_curve_fit.heating_curve_slope,
    )
    if np.any(dhw_mask):
        eta_fit_dhw = _fit_eta_carnot_for_mask(
            arrays,
            settings,
            mode_mask=dhw_mask,
            eta_label="eta_carnot_dhw",
            t_supply_min_c=heating_curve_fit.t_supply_min_c,
            t_ref_outdoor_c=heating_curve_fit.t_ref_outdoor_c,
            heating_curve_slope=heating_curve_fit.heating_curve_slope,
        )
    else:
        eta_fit_dhw = _EtaFit(
            eta_carnot=eta_fit_ufh.eta_carnot,
            optimizer_status="No DHW samples retained; reusing eta_carnot_ufh.",
            optimizer_cost=0.0,
        )
    fitted_parameters = _build_cop_parameters(
        settings,
        eta_carnot_ufh=eta_fit_ufh.eta_carnot,
        eta_carnot_dhw=eta_fit_dhw.eta_carnot,
        t_supply_min_c=heating_curve_fit.t_supply_min_c,
        t_ref_outdoor_c=heating_curve_fit.t_ref_outdoor_c,
        heating_curve_slope=heating_curve_fit.heating_curve_slope,
    )
    model = HeatPumpCOPModel(fitted_parameters)
    predicted_supply_c = _predicted_supply_temperature_c(
        arrays,
        settings,
        t_supply_min_c=fitted_parameters.T_supply_min,
        t_ref_outdoor_c=fitted_parameters.T_ref_outdoor,
        heating_curve_slope=fitted_parameters.heating_curve_slope,
    )
    predicted_cop = np.empty_like(arrays.outdoor_temperature_mean_c, dtype=float)
    predicted_cop[ufh_mask] = model._cop_from_temperatures_with_eta(
        t_supply=predicted_supply_c[ufh_mask],
        t_out=arrays.outdoor_temperature_mean_c[ufh_mask],
        eta_carnot=fitted_parameters.eta_carnot_ufh,
    )
    if np.any(dhw_mask):
        predicted_cop[dhw_mask] = model._cop_from_temperatures_with_eta(
            t_supply=predicted_supply_c[dhw_mask],
            t_out=arrays.outdoor_temperature_mean_c[dhw_mask],
            eta_carnot=fitted_parameters.eta_carnot_dhw,
        )
    predicted_electric_energy_kwh = arrays.thermal_energy_kwh / predicted_cop
    actual_cop = arrays.thermal_energy_kwh / arrays.electric_energy_kwh
    weights = _normalised_energy_weights(arrays.thermal_energy_kwh)

    ufh_electric_residuals = predicted_electric_energy_kwh[ufh_mask] - arrays.electric_energy_kwh[ufh_mask]
    ufh_cop_residuals = predicted_cop[ufh_mask] - actual_cop[ufh_mask]

    dhw_rmse_electric_energy_kwh: float | None = None
    dhw_rmse_actual_cop: float | None = None
    dhw_bias_actual_cop: float | None = None
    diagnostic_eta_carnot_dhw: float | None = None
    if np.any(dhw_mask):
        dhw_electric_residuals = predicted_electric_energy_kwh[dhw_mask] - arrays.electric_energy_kwh[dhw_mask]
        dhw_cop_residuals = predicted_cop[dhw_mask] - actual_cop[dhw_mask]
        dhw_rmse_electric_energy_kwh = _rmse(dhw_electric_residuals)
        dhw_rmse_actual_cop = _rmse(dhw_cop_residuals)
        dhw_bias_actual_cop = float(np.mean(dhw_cop_residuals))
        diagnostic_eta_carnot_dhw = eta_fit_dhw.eta_carnot

    return COPCalibrationResult(
        fitted_parameters=fitted_parameters,
        t_ref_outdoor_was_fitted=heating_curve_fit.t_ref_outdoor_was_fitted,
        rmse_supply_temperature_c=heating_curve_fit.rmse_supply_temperature_c,
        rmse_electric_energy_kwh=_rmse(predicted_electric_energy_kwh - arrays.electric_energy_kwh),
        rmse_actual_cop=_rmse(predicted_cop - actual_cop),
        ufh_rmse_electric_energy_kwh=_rmse(ufh_electric_residuals),
        dhw_rmse_electric_energy_kwh=dhw_rmse_electric_energy_kwh,
        ufh_rmse_actual_cop=_rmse(ufh_cop_residuals),
        dhw_rmse_actual_cop=dhw_rmse_actual_cop,
        ufh_bias_actual_cop=float(np.mean(ufh_cop_residuals)),
        dhw_bias_actual_cop=dhw_bias_actual_cop,
        diagnostic_eta_carnot_ufh=eta_fit_ufh.eta_carnot,
        diagnostic_eta_carnot_dhw=diagnostic_eta_carnot_dhw,
        sample_count=dataset.sample_count,
        ufh_sample_count=dataset.ufh_sample_count,
        dhw_sample_count=dataset.dhw_sample_count,
        dataset_start_utc=dataset.start_utc,
        dataset_end_utc=dataset.end_utc,
        heating_curve_optimizer_status=heating_curve_fit.optimizer_status,
        eta_optimizer_status=f"UFH: {eta_fit_ufh.optimizer_status} | DHW: {eta_fit_dhw.optimizer_status}",
        heating_curve_optimizer_cost=heating_curve_fit.optimizer_cost,
        eta_optimizer_cost=eta_fit_ufh.optimizer_cost + eta_fit_dhw.optimizer_cost,
    )
