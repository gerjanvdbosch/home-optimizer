"""High-level calibration services orchestrating repository access and fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections import Counter
from datetime import datetime, timezone
from math import isfinite
from typing import cast

from sqlalchemy import text

from .dataset import (
    build_cop_calibration_dataset,
    build_dhw_active_calibration_dataset,
    build_dhw_standby_calibration_dataset,
    diagnose_cop_calibration_dataset,
    build_ufh_active_calibration_dataset,
    build_ufh_off_calibration_dataset,
)
from .cop_offline import calibrate_cop_model
from .dhw_active import dhw_active_r_strat_bounds
from .dhw_active import calibrate_dhw_active_stratification
from .models import (
    AutomaticCalibrationSettings,
    COPCalibrationDiagnostics,
    COPCalibrationDataset,
    COPCalibrationResult,
    COPCalibrationSettings,
    DHWActiveCalibrationDataset,
    DHWActiveCalibrationResult,
    DHWActiveCalibrationSettings,
    DHWStandbyCalibrationDataset,
    DHWStandbyCalibrationResult,
    DHWStandbyCalibrationSettings,
    UFHActiveCalibrationDataset,
    UFHActiveCalibrationResult,
    UFHActiveCalibrationSettings,
    UFHCalibrationDataset,
    UFHOffCalibrationResult,
    UFHOffCalibrationSettings,
    DEFAULT_MIN_DHW_R_STRAT_K_PER_KW,
    DEFAULT_MAX_PAIR_DT_HOURS,
)
from .settings_factory import (
    build_cop_calibration_settings,
    build_dhw_active_calibration_settings,
    build_dhw_standby_calibration_settings,
    build_ufh_active_calibration_settings,
)
from .dhw_standby import calibrate_dhw_standby_loss
from .ufh_active import calibrate_ufh_active_rc
from .ufh_offline import calibrate_ufh_off_envelope
from ..application.optimizer import RunRequest, merge_run_request_updates, sanitize_calibration_overrides
from ..telemetry.models import ForecastSnapshot, TelemetryAggregate
from ..telemetry.repository import TelemetryRepository
from ..types import CalibrationParameterOverrides, CalibrationSnapshotPayload, CalibrationStageResult, DHWParameters, ThermalParameters


@dataclass(frozen=True, slots=True)
class _CalibrationAggregateRow:
    """Typed SQL projection row for offline calibration aggregate loading.

    The calibration service selects only the telemetry columns required by the
    offline calibrators, so this lightweight dataclass preserves attribute access
    without pretending the row is a full ORM ``TelemetryAggregate`` instance.
    """

    bucket_start_utc: datetime
    bucket_end_utc: datetime
    hp_mode_last: str
    defrost_active_fraction: float
    booster_heater_active_fraction: float
    room_temperature_last_c: float
    outdoor_temperature_mean_c: float
    hp_supply_temperature_mean_c: float
    hp_supply_target_temperature_mean_c: float
    household_elec_power_mean_kw: float
    hp_thermal_power_mean_kw: float
    hp_electric_energy_delta_kwh: float
    dhw_top_temperature_last_c: float
    dhw_bottom_temperature_last_c: float
    boiler_ambient_temp_mean_c: float
    t_mains_estimated_mean_c: float


@dataclass(frozen=True, slots=True)
class _CalibrationForecastRow:
    """Typed SQL projection row for offline forecast loading."""

    valid_at_utc: datetime
    gti_w_per_m2: float


def _parse_utc(value: object) -> datetime:
    """Parse SQLite/SQLAlchemy timestamp values into timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported timestamp type for calibration loader: {type(value)!r}")


def _load_calibration_aggregates(repository: TelemetryRepository) -> list[_CalibrationAggregateRow]:
    """Load the telemetry columns required by the offline UFH calibrators."""
    statement = text(
        """
        SELECT
            bucket_start_utc,
            bucket_end_utc,
            hp_mode_last,
            defrost_active_fraction,
            booster_heater_active_fraction,
            room_temperature_last_c,
            outdoor_temperature_mean_c,
            hp_supply_temperature_mean_c,
            hp_supply_target_temperature_mean_c,
            household_elec_power_mean_kw,
            hp_thermal_power_mean_kw,
            hp_electric_energy_delta_kwh,
            dhw_top_temperature_last_c,
            dhw_bottom_temperature_last_c,
            boiler_ambient_temp_mean_c,
            t_mains_estimated_mean_c
        FROM telemetry_aggregates
        ORDER BY bucket_end_utc ASC
        """
    )
    with repository.engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()
    return [
        _CalibrationAggregateRow(
            bucket_start_utc=_parse_utc(row["bucket_start_utc"]),
            bucket_end_utc=_parse_utc(row["bucket_end_utc"]),
            hp_mode_last=str(row["hp_mode_last"]),
            defrost_active_fraction=float(row["defrost_active_fraction"]),
            booster_heater_active_fraction=float(row["booster_heater_active_fraction"]),
            room_temperature_last_c=float(row["room_temperature_last_c"]),
            outdoor_temperature_mean_c=float(row["outdoor_temperature_mean_c"]),
            hp_supply_temperature_mean_c=float(row["hp_supply_temperature_mean_c"]),
            hp_supply_target_temperature_mean_c=float(row["hp_supply_target_temperature_mean_c"]),
            household_elec_power_mean_kw=float(row["household_elec_power_mean_kw"]),
            hp_thermal_power_mean_kw=float(row["hp_thermal_power_mean_kw"]),
            hp_electric_energy_delta_kwh=float(row["hp_electric_energy_delta_kwh"]),
            dhw_top_temperature_last_c=float(row["dhw_top_temperature_last_c"]),
            dhw_bottom_temperature_last_c=float(row["dhw_bottom_temperature_last_c"]),
            boiler_ambient_temp_mean_c=float(row["boiler_ambient_temp_mean_c"]),
            t_mains_estimated_mean_c=float(row["t_mains_estimated_mean_c"]),
        )
        for row in rows
    ]


def _load_calibration_forecasts(repository: TelemetryRepository) -> list[_CalibrationForecastRow]:
    """Load only the forecast columns required by the first-stage UFH calibrator."""
    statement = text(
        """
        SELECT valid_at_utc, gti_w_per_m2
        FROM forecast_snapshots
        ORDER BY valid_at_utc ASC
        """
    )
    with repository.engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()
    return [
        _CalibrationForecastRow(
            valid_at_utc=_parse_utc(row["valid_at_utc"]),
            gti_w_per_m2=float(row["gti_w_per_m2"]),
        )
        for row in rows
    ]


def _infer_calibration_replay_dt_hours(repository: TelemetryRepository) -> float:
    """Infer the dominant persisted telemetry replay timestep Δt [h].

    Automatic calibration must replay the offline datasets with the same Δt that
    the manual CLI expects the operator to pass via ``--dt-hours`` /
    ``--dhw-dt-hours``. Reusing the MPC timestep here is physically wrong because
    the dataset builders compare consecutive telemetry-pair spacing against the
    reference model Δt with a strict compatibility tolerance.
    """
    bucket_end_utc = [row.bucket_end_utc for row in _load_calibration_aggregates(repository)]
    candidate_dt_seconds = [
        round((next_bucket_end_utc - previous_bucket_end_utc).total_seconds())
        for previous_bucket_end_utc, next_bucket_end_utc in zip(bucket_end_utc, bucket_end_utc[1:])
        if 0.0
        < (next_bucket_end_utc - previous_bucket_end_utc).total_seconds() / 3600.0
        <= DEFAULT_MAX_PAIR_DT_HOURS
    ]
    if not candidate_dt_seconds:
        raise ValueError(
            "Could not infer a positive persisted telemetry replay timestep <= "
            f"{DEFAULT_MAX_PAIR_DT_HOURS:.3f} h from telemetry_aggregates."
        )
    dominant_dt_seconds = Counter(candidate_dt_seconds).most_common(1)[0][0]
    return dominant_dt_seconds / 3600.0


def build_ufh_off_dataset_from_repository(
    repository: TelemetryRepository,
    settings: UFHOffCalibrationSettings,
) -> UFHCalibrationDataset:
    """Load telemetry history from the repository and build a UFH off-mode dataset."""
    return build_ufh_off_calibration_dataset(
        aggregates=cast(list[TelemetryAggregate], cast(object, _load_calibration_aggregates(repository))),
        forecast_rows=cast(list[ForecastSnapshot], cast(object, _load_calibration_forecasts(repository))),
        settings=settings,
    )


def build_cop_dataset_from_repository(
    repository: TelemetryRepository,
    settings: COPCalibrationSettings,
) -> COPCalibrationDataset:
    """Load telemetry history from the repository and build a COP calibration dataset."""
    return build_cop_calibration_dataset(
        aggregates=cast(list[TelemetryAggregate], cast(object, _load_calibration_aggregates(repository))),
        settings=settings,
    )


def diagnose_cop_dataset_from_repository(
    repository: TelemetryRepository,
    settings: COPCalibrationSettings,
) -> COPCalibrationDiagnostics:
    """Load telemetry history and return COP dataset-filter diagnostics without fitting."""
    return diagnose_cop_calibration_dataset(
        aggregates=cast(list[TelemetryAggregate], cast(object, _load_calibration_aggregates(repository))),
        settings=settings,
    )


def build_dhw_standby_dataset_from_repository(
    repository: TelemetryRepository,
    settings: DHWStandbyCalibrationSettings,
) -> DHWStandbyCalibrationDataset:
    """Load telemetry history from the repository and build a DHW standby dataset."""
    return build_dhw_standby_calibration_dataset(
        aggregates=cast(list[TelemetryAggregate], cast(object, _load_calibration_aggregates(repository))),
        settings=settings,
    )


def build_dhw_active_dataset_from_repository(
    repository: TelemetryRepository,
    settings: DHWActiveCalibrationSettings,
) -> DHWActiveCalibrationDataset:
    """Load telemetry history from the repository and build an active DHW dataset."""
    return build_dhw_active_calibration_dataset(
        aggregates=cast(list[TelemetryAggregate], cast(object, _load_calibration_aggregates(repository))),
        settings=settings,
    )


def build_ufh_active_dataset_from_repository(
    repository: TelemetryRepository,
    settings: UFHActiveCalibrationSettings,
) -> UFHActiveCalibrationDataset:
    """Load telemetry history from the repository and build an active UFH replay dataset."""
    return build_ufh_active_calibration_dataset(
        aggregates=cast(list[TelemetryAggregate], cast(object, _load_calibration_aggregates(repository))),
        forecast_rows=cast(list[ForecastSnapshot], cast(object, _load_calibration_forecasts(repository))),
        settings=settings,
    )


def calibrate_ufh_off_from_repository(
    repository: TelemetryRepository,
    settings: UFHOffCalibrationSettings | None = None,
) -> UFHOffCalibrationResult:
    """Run the first-stage UFH off-mode envelope calibration from persisted telemetry."""
    effective_settings = settings or UFHOffCalibrationSettings()
    dataset = build_ufh_off_dataset_from_repository(repository, effective_settings)
    return calibrate_ufh_off_envelope(dataset, effective_settings)


def calibrate_cop_from_repository(
    repository: TelemetryRepository,
    settings: COPCalibrationSettings,
) -> COPCalibrationResult:
    """Run the offline COP calibration from persisted telemetry."""
    dataset = build_cop_dataset_from_repository(repository, settings)
    return calibrate_cop_model(dataset, settings)


def calibrate_dhw_standby_from_repository(
    repository: TelemetryRepository,
    settings: DHWStandbyCalibrationSettings,
) -> DHWStandbyCalibrationResult:
    """Run the first-stage DHW standby-loss calibration from persisted telemetry."""
    dataset = build_dhw_standby_dataset_from_repository(repository, settings)
    return calibrate_dhw_standby_loss(dataset, settings)


def calibrate_dhw_active_from_repository(
    repository: TelemetryRepository,
    settings: DHWActiveCalibrationSettings,
) -> DHWActiveCalibrationResult:
    """Run the active DHW stratification calibration from persisted telemetry."""
    dataset = build_dhw_active_dataset_from_repository(repository, settings)
    return calibrate_dhw_active_stratification(dataset, settings)


def calibrate_ufh_active_from_repository(
    repository: TelemetryRepository,
    settings: UFHActiveCalibrationSettings,
) -> UFHActiveCalibrationResult:
    """Run the active UFH RC calibration from persisted telemetry history."""
    dataset = build_ufh_active_dataset_from_repository(repository, settings)
    return calibrate_ufh_active_rc(dataset, settings)


def _apply_calibration_overrides(
    base_request: RunRequest,
    overrides: CalibrationParameterOverrides,
) -> RunRequest:
    """Return a field-validated request copy for building calibration references.

    Automatic calibration may need to operate on telemetry replay timesteps that
    differ from the runtime MPC timestep. Therefore this helper intentionally
    performs only Pydantic field validation and does *not* enforce the coupled
    runtime Euler-stability checks. Runtime safety is enforced later, when newly
    fitted stage overrides are considered for persistence via
    ``_merge_runtime_safe_stage_overrides``.
    """
    updates = overrides.as_run_request_updates()
    if not updates:
        return base_request
    return RunRequest.model_validate({**base_request.model_dump(mode="python"), **updates})


def _merge_runtime_safe_stage_overrides(
    base_request: RunRequest,
    current_effective_parameters: CalibrationParameterOverrides,
    stage_overrides: CalibrationParameterOverrides,
) -> CalibrationParameterOverrides:
    """Merge one stage result only when the resulting runtime request stays valid."""
    current_request = _apply_calibration_overrides(base_request, current_effective_parameters)
    merge_run_request_updates(current_request, stage_overrides.as_run_request_updates())
    return current_effective_parameters.merged_with(stage_overrides)


def _stage_failure(
    stage_name: str,
    message: str,
    *,
    diagnostics: dict[str, Any] | None = None,
) -> CalibrationStageResult:
    """Create a compact failed-stage summary for persistence/API observability."""
    return CalibrationStageResult(
        stage_name=stage_name,
        succeeded=False,
        message=message,
        diagnostics={} if diagnostics is None else diagnostics,
    )


def _stage_success(
    stage_name: str,
    message: str,
    *,
    overrides: CalibrationParameterOverrides,
    sample_count: int,
    dataset_start_utc: datetime,
    dataset_end_utc: datetime,
    segment_count: int | None = None,
    optimizer_status: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> CalibrationStageResult:
    """Create a compact successful-stage summary for persistence/API observability."""
    return CalibrationStageResult(
        stage_name=stage_name,
        succeeded=True,
        message=message,
        sample_count=sample_count,
        segment_count=segment_count,
        dataset_start_utc=dataset_start_utc,
        dataset_end_utc=dataset_end_utc,
        optimizer_status=optimizer_status,
        diagnostics={} if diagnostics is None else diagnostics,
        overrides=overrides,
    )


class _CalibrationValidationError(ValueError):
    """Internal validation error carrying structured stage diagnostics."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _build_ufh_active_diagnostics(
    result: UFHActiveCalibrationResult,
    *,
    active_settings: UFHActiveCalibrationSettings,
    automatic_settings: AutomaticCalibrationSettings,
    bound_violations: tuple[str, ...] = (),
    passive_r_ro_k_per_kw: float | None = None,
    r_ro_mismatch_ratio: float | None = None,
) -> dict[str, Any]:
    """Return structured UFH active-fit diagnostics for API/persistence visibility."""
    fit_c_r = bool(getattr(result, "fit_c_r", False))
    fit_eta = bool(getattr(result, "fit_eta", False))
    fit_internal_gains_heat_fraction = bool(getattr(result, "fit_internal_gains_heat_fraction", False))
    fit_room_temperature_bias = bool(getattr(result, "fit_room_temperature_bias", False))
    fitted_parameter_names = ["C_b", "R_br", "R_ro"]
    if fit_c_r:
        fitted_parameter_names.insert(0, "C_r")
    if fit_eta:
        fitted_parameter_names.append("eta")
    bounds = {
        parameter_name: {
            "reference": getattr(active_settings.reference_parameters, parameter_name),
            "fitted": getattr(result.fitted_parameters, parameter_name),
            "lower_bound": (
                active_settings.min_eta
                if parameter_name == "eta"
                else getattr(active_settings.reference_parameters, parameter_name) * active_settings.min_parameter_ratio
            ),
            "upper_bound": (
                active_settings.max_eta
                if parameter_name == "eta"
                else getattr(active_settings.reference_parameters, parameter_name) * active_settings.max_parameter_ratio
            ),
        }
        for parameter_name in fitted_parameter_names
    }
    return {
        "selected_segment_count": result.segment_count,
        "required_min_selected_segments": automatic_settings.ufh_active_min_selected_segments,
        "fit_c_r": fit_c_r,
        "fit_eta": fit_eta,
        "fit_internal_gains_heat_fraction": fit_internal_gains_heat_fraction,
        "fit_room_temperature_bias": fit_room_temperature_bias,
        "fitted_eta": getattr(result.fitted_parameters, "eta", active_settings.reference_parameters.eta),
        "fitted_internal_gains_heat_fraction": getattr(
            result,
            "fitted_internal_gains_heat_fraction",
            active_settings.reference_internal_gains_heat_fraction,
        ),
        "fitted_room_temperature_bias_c": getattr(result, "fitted_room_temperature_bias_c", 0.0),
        "fit_initial_floor_temperature_offset": getattr(result, "fit_initial_floor_temperature_offset", False),
        "fitted_initial_floor_temperature_offset_c": getattr(result, "fitted_initial_floor_temperature_offset_c", 0.0),
        "rmse_room_temperature_c": result.rmse_room_temperature_c,
        "max_abs_innovation_c": result.max_abs_innovation_c,
        "bound_tolerance_ratio": automatic_settings.ufh_active_bound_tolerance_ratio,
        "bound_violations": list(bound_violations),
        "parameter_bounds": bounds,
        "active_r_ro_k_per_kw": result.fitted_parameters.R_ro,
        "passive_r_ro_k_per_kw": passive_r_ro_k_per_kw,
        "r_ro_mismatch_ratio": r_ro_mismatch_ratio,
        "max_r_ro_mismatch_ratio": automatic_settings.ufh_active_max_r_ro_mismatch_ratio,
    }


def _build_dhw_standby_diagnostics(
    result: DHWStandbyCalibrationResult,
    *,
    standby_settings: DHWStandbyCalibrationSettings,
    automatic_settings: AutomaticCalibrationSettings,
) -> dict[str, Any]:
    """Return structured DHW standby-fit diagnostics for API/persistence visibility."""
    return {
        "tau_standby_hours": result.tau_standby_hours,
        "tau_lower_bound_hours": standby_settings.min_tau_hours,
        "tau_upper_bound_hours": standby_settings.max_tau_hours,
        "bound_tolerance_ratio": automatic_settings.dhw_standby_bound_tolerance_ratio,
        "suggested_r_loss_k_per_kw": result.suggested_r_loss_k_per_kw,
        "fit_ambient_temperature_bias": getattr(result, "fit_ambient_temperature_bias", False),
        "fitted_ambient_temperature_bias_c": getattr(result, "fitted_ambient_temperature_bias_c", 0.0),
        "rmse_mean_tank_temperature_c": result.rmse_mean_tank_temperature_c,
        "max_abs_residual_c": result.max_abs_residual_c,
    }


def _build_dhw_active_diagnostics(
    result: DHWActiveCalibrationResult,
    *,
    active_settings: DHWActiveCalibrationSettings,
    automatic_settings: AutomaticCalibrationSettings,
) -> dict[str, Any]:
    """Return structured DHW active-fit diagnostics for API/persistence visibility."""
    lower_bound, upper_bound = dhw_active_r_strat_bounds(active_settings)
    tolerance_ratio = automatic_settings.dhw_active_bound_tolerance_ratio
    hits_lower_bound = result.fitted_parameters.R_strat <= lower_bound * (1.0 + tolerance_ratio)
    lower_bound_represents_nearly_perfect_mixing = lower_bound <= DEFAULT_MIN_DHW_R_STRAT_K_PER_KW * (1.0 + tolerance_ratio)
    return {
        "selected_segment_count": result.segment_count,
        "required_min_selected_segments": automatic_settings.dhw_active_min_selected_segments,
        "fitted_r_strat_k_per_kw": result.fitted_parameters.R_strat,
        "fitted_c_top_kwh_per_k": result.fitted_parameters.C_top,
        "fitted_c_bot_kwh_per_k": result.fitted_parameters.C_bot,
        "fit_capacity_split": getattr(result, "fit_capacity_split", False),
        "fitted_c_top_fraction": getattr(
            result,
            "fitted_c_top_fraction",
            result.fitted_parameters.C_top / (result.fitted_parameters.C_top + result.fitted_parameters.C_bot),
        ),
        "fit_temperature_biases": getattr(result, "fit_temperature_biases", False),
        "fitted_t_top_bias_c": getattr(result, "fitted_t_top_bias_c", 0.0),
        "fitted_t_bot_bias_c": getattr(result, "fitted_t_bot_bias_c", 0.0),
        "r_strat_lower_bound_k_per_kw": lower_bound,
        "r_strat_upper_bound_k_per_kw": upper_bound,
        "bound_tolerance_ratio": tolerance_ratio,
        "hits_lower_bound": hits_lower_bound,
        "lower_bound_represents_nearly_perfect_mixing": lower_bound_represents_nearly_perfect_mixing,
        "near_perfect_mixing_regime": hits_lower_bound and lower_bound_represents_nearly_perfect_mixing,
        "fixed_r_loss_k_per_kw": result.fitted_parameters.R_loss,
        "rmse_t_top_c": result.rmse_t_top_c,
        "rmse_t_bot_c": result.rmse_t_bot_c,
        "max_abs_residual_c": result.max_abs_residual_c,
    }


def _ufh_effective_envelope_capacity_kwh_per_k(parameters: ThermalParameters) -> float:
    """Return the conservative effective UFH envelope capacity proxy ``C_r + C_b`` [kWh/K].

    The passive off-mode stage identifies ``tau_house = C_eff * R_ro``. To compare
    the active two-state UFH fit against that passive estimate, this helper uses the
    total fitted two-state heat capacity as the closest grey-box proxy for
    ``C_eff``. This is intentionally conservative: it avoids comparing the passive
    envelope time constant against room-air capacity alone, which would overstate
    the implied passive ``R_ro``.
    """
    return parameters.C_r + parameters.C_b


def _bound_tolerance_width(*, lower_bound: float, upper_bound: float, tolerance_ratio: float) -> float:
    """Return the absolute near-bound tolerance derived from a bounded interval.

    The automatic calibration gates use a relative tolerance stored in
    :class:`AutomaticCalibrationSettings`. For signed parameters such as sensor
    biases, multiplicative bound checks are not meaningful because the lower bound
    may be negative. This helper converts the ratio to an absolute tolerance based
    on the full interval width.
    """
    if upper_bound <= lower_bound:
        raise ValueError("upper_bound must be strictly greater than lower_bound.")
    if tolerance_ratio < 0.0:
        raise ValueError("tolerance_ratio must be non-negative.")
    return tolerance_ratio * (upper_bound - lower_bound)


def _hits_lower_bound(*, fitted_value: float, lower_bound: float, upper_bound: float, tolerance_ratio: float) -> bool:
    """Return ``True`` when a fitted value is at/near its lower bound."""
    return fitted_value <= lower_bound + _bound_tolerance_width(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tolerance_ratio=tolerance_ratio,
    )


def _hits_upper_bound(*, fitted_value: float, lower_bound: float, upper_bound: float, tolerance_ratio: float) -> bool:
    """Return ``True`` when a fitted value is at/near its upper bound."""
    return fitted_value >= upper_bound - _bound_tolerance_width(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tolerance_ratio=tolerance_ratio,
    )


def _ufh_active_bound_violations(
    result: UFHActiveCalibrationResult,
    settings: UFHActiveCalibrationSettings,
    *,
    tolerance_ratio: float,
) -> tuple[str, ...]:
    """Return any UFH active-fit parameters that effectively sit on optimizer bounds.

    Args:
        result: Fitted active-UFH RC result to validate.
        settings: Active-UFH optimiser settings defining the reference tuple and
            relative min/max parameter bounds.
        tolerance_ratio: Relative tolerance used to classify near-bound solutions [-].

    Returns:
        Tuple with human-readable violation messages. Empty tuple means the fit is
        safely interior to the allowed parameter box.
    """
    fitted_parameter_names = ["C_b", "R_br", "R_ro"]
    if bool(getattr(result, "fit_c_r", False)):
        fitted_parameter_names.insert(0, "C_r")
    if bool(getattr(result, "fit_eta", False)):
        fitted_parameter_names.append("eta")
    violations: list[str] = []
    for parameter_name in fitted_parameter_names:
        fitted_value = getattr(result.fitted_parameters, parameter_name)
        if parameter_name == "eta":
            lower_bound = settings.min_eta
            upper_bound = settings.max_eta
        else:
            reference_value = getattr(settings.reference_parameters, parameter_name)
            lower_bound = reference_value * settings.min_parameter_ratio
            upper_bound = reference_value * settings.max_parameter_ratio
        if _hits_lower_bound(
            fitted_value=fitted_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            tolerance_ratio=tolerance_ratio,
        ):
            violations.append(
                f"{parameter_name}={fitted_value:.6g} is at/near the lower bound {lower_bound:.6g}."
            )
        elif _hits_upper_bound(
            fitted_value=fitted_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            tolerance_ratio=tolerance_ratio,
        ):
            violations.append(
                f"{parameter_name}={fitted_value:.6g} is at/near the upper bound {upper_bound:.6g}."
            )
    if bool(getattr(result, "fit_internal_gains_heat_fraction", False)):
        lower_bound = settings.min_internal_gains_heat_fraction
        upper_bound = settings.max_internal_gains_heat_fraction
        fitted_value = float(getattr(result, "fitted_internal_gains_heat_fraction"))
        if _hits_lower_bound(
            fitted_value=fitted_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            tolerance_ratio=tolerance_ratio,
        ):
            violations.append(
                f"internal_gains_heat_fraction={fitted_value:.6g} is at/near the lower bound {lower_bound:.6g}."
            )
        elif _hits_upper_bound(
            fitted_value=fitted_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            tolerance_ratio=tolerance_ratio,
        ):
            violations.append(
                f"internal_gains_heat_fraction={fitted_value:.6g} is at/near the upper bound {upper_bound:.6g}."
            )
    if bool(getattr(result, "fit_room_temperature_bias", False)):
        lower_bound = settings.min_room_temperature_bias_c
        upper_bound = settings.max_room_temperature_bias_c
        fitted_value = float(getattr(result, "fitted_room_temperature_bias_c"))
        if _hits_lower_bound(
            fitted_value=fitted_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            tolerance_ratio=tolerance_ratio,
        ):
            violations.append(
                f"room_temperature_bias_c={fitted_value:.6g} is at/near the lower bound {lower_bound:.6g}."
            )
        elif _hits_upper_bound(
            fitted_value=fitted_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            tolerance_ratio=tolerance_ratio,
        ):
            violations.append(
                f"room_temperature_bias_c={fitted_value:.6g} is at/near the upper bound {upper_bound:.6g}."
            )
    return tuple(violations)


def _validate_automatic_ufh_active_fit(
    repository: TelemetryRepository,
    result: UFHActiveCalibrationResult,
    *,
    active_settings: UFHActiveCalibrationSettings,
    automatic_settings: AutomaticCalibrationSettings,
) -> dict[str, Any]:
    """Fail fast when an automatic active-UFH RC fit is physically untrustworthy.

    The active replay stage can achieve a small one-step room-temperature RMSE even
    when the RC tuple is under-identified, for example when only one short segment
    is available or when multiple parameters are pushed onto their box constraints.
    This post-fit gate rejects such solutions before they can become runtime MPC
    overrides.

    Validation criteria:

    1. Enough selected active UFH segments are present.
    2. No fitted active-UFH parameter sits on its optimiser bounds.
    3. The active fitted ``R_ro`` remains reasonably consistent with the passive
       off-mode envelope fit derived from the same telemetry history.
    """
    if result.segment_count < automatic_settings.ufh_active_min_selected_segments:
        raise _CalibrationValidationError(
            "Automatic UFH RC fit rejected: insufficient active excitation. "
            f"Selected segments={result.segment_count}, required >= "
            f"{automatic_settings.ufh_active_min_selected_segments}.",
            diagnostics=_build_ufh_active_diagnostics(
                result,
                active_settings=active_settings,
                automatic_settings=automatic_settings,
            ),
        )

    bound_violations = _ufh_active_bound_violations(
        result,
        active_settings,
        tolerance_ratio=automatic_settings.ufh_active_bound_tolerance_ratio,
    )
    if bound_violations:
        raise _CalibrationValidationError(
            "Automatic UFH RC fit rejected: optimiser converged to parameter bounds. "
            + " ".join(bound_violations),
            diagnostics=_build_ufh_active_diagnostics(
                result,
                active_settings=active_settings,
                automatic_settings=automatic_settings,
                bound_violations=bound_violations,
            ),
        )

    passive_result = calibrate_ufh_off_from_repository(
        repository,
        UFHOffCalibrationSettings(
            reference_c_eff_kwh_per_k=_ufh_effective_envelope_capacity_kwh_per_k(result.fitted_parameters)
        ),
    )
    passive_r_ro = passive_result.suggested_r_ro_k_per_kw
    if passive_r_ro is None or not isfinite(passive_r_ro) or passive_r_ro <= 0.0:
        raise _CalibrationValidationError(
            "Automatic UFH RC fit rejected: passive off-mode envelope stage did not produce a finite R_ro.",
            diagnostics=_build_ufh_active_diagnostics(
                result,
                active_settings=active_settings,
                automatic_settings=automatic_settings,
                passive_r_ro_k_per_kw=passive_r_ro,
            ),
        )

    active_r_ro = result.fitted_parameters.R_ro
    mismatch_ratio = max(active_r_ro / passive_r_ro, passive_r_ro / active_r_ro)
    if mismatch_ratio > automatic_settings.ufh_active_max_r_ro_mismatch_ratio:
        raise _CalibrationValidationError(
            "Automatic UFH RC fit rejected: active/passive R_ro mismatch is too large. "
            f"Active R_ro={active_r_ro:.6g} K/kW, passive-derived R_ro={passive_r_ro:.6g} K/kW, "
            f"mismatch ratio={mismatch_ratio:.3f}, allowed <= "
            f"{automatic_settings.ufh_active_max_r_ro_mismatch_ratio:.3f}.",
            diagnostics=_build_ufh_active_diagnostics(
                result,
                active_settings=active_settings,
                automatic_settings=automatic_settings,
                passive_r_ro_k_per_kw=passive_r_ro,
                r_ro_mismatch_ratio=mismatch_ratio,
            ),
        )
    return _build_ufh_active_diagnostics(
        result,
        active_settings=active_settings,
        automatic_settings=automatic_settings,
        passive_r_ro_k_per_kw=passive_r_ro,
        r_ro_mismatch_ratio=mismatch_ratio,
    )


def _validate_automatic_dhw_standby_fit(
    result: DHWStandbyCalibrationResult,
    *,
    standby_settings: DHWStandbyCalibrationSettings,
    automatic_settings: AutomaticCalibrationSettings,
) -> dict[str, Any]:
    """Fail fast when an automatic DHW standby-loss fit converges to its box bounds.

    The standby stage identifies a one-state envelope time constant ``tau_standby``.
    When the fit lands on its optimiser bounds, the implied ``R_loss`` is not
    trustworthy enough to become a runtime MPC override.
    """
    lower_bound = standby_settings.min_tau_hours
    upper_bound = standby_settings.max_tau_hours
    fitted_value = result.tau_standby_hours
    tolerance_ratio = automatic_settings.dhw_standby_bound_tolerance_ratio
    if _hits_lower_bound(
        fitted_value=fitted_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tolerance_ratio=tolerance_ratio,
    ):
        raise _CalibrationValidationError(
            "Automatic DHW standby fit rejected: tau_standby converged to the lower bound. "
            f"tau_standby={fitted_value:.6g} h, lower bound={lower_bound:.6g} h.",
            diagnostics=_build_dhw_standby_diagnostics(
                result,
                standby_settings=standby_settings,
                automatic_settings=automatic_settings,
            ),
        )
    if _hits_upper_bound(
        fitted_value=fitted_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tolerance_ratio=tolerance_ratio,
    ):
        raise _CalibrationValidationError(
            "Automatic DHW standby fit rejected: tau_standby converged to the upper bound. "
            f"tau_standby={fitted_value:.6g} h, upper bound={upper_bound:.6g} h.",
            diagnostics=_build_dhw_standby_diagnostics(
                result,
                standby_settings=standby_settings,
                automatic_settings=automatic_settings,
            ),
        )
    if bool(getattr(result, "fit_ambient_temperature_bias", False)):
        bias_lower_bound = standby_settings.min_ambient_temperature_bias_c
        bias_upper_bound = standby_settings.max_ambient_temperature_bias_c
        fitted_bias_c = float(getattr(result, "fitted_ambient_temperature_bias_c"))
        if _hits_lower_bound(
            fitted_value=fitted_bias_c,
            lower_bound=bias_lower_bound,
            upper_bound=bias_upper_bound,
            tolerance_ratio=tolerance_ratio,
        ) or _hits_upper_bound(
            fitted_value=fitted_bias_c,
            lower_bound=bias_lower_bound,
            upper_bound=bias_upper_bound,
            tolerance_ratio=tolerance_ratio,
        ):
            raise _CalibrationValidationError(
                "Automatic DHW standby fit rejected: ambient sensor bias converged to its bound. "
                f"ambient_bias={fitted_bias_c:.6g} °C, bounds=[{bias_lower_bound:.6g}, {bias_upper_bound:.6g}] °C.",
                diagnostics=_build_dhw_standby_diagnostics(
                    result,
                    standby_settings=standby_settings,
                    automatic_settings=automatic_settings,
                ),
            )
    return _build_dhw_standby_diagnostics(
        result,
        standby_settings=standby_settings,
        automatic_settings=automatic_settings,
    )


def _validate_automatic_dhw_active_fit(
    result: DHWActiveCalibrationResult,
    *,
    active_settings: DHWActiveCalibrationSettings,
    automatic_settings: AutomaticCalibrationSettings,
) -> dict[str, Any]:
    """Fail fast when an automatic active-DHW ``R_strat`` fit is under-identified.

    The active DHW stage should only become a runtime override when enough selected
    no-draw charging segments are available and the fitted ``R_strat`` remains well
    inside the physical optimiser box.
    """
    if result.segment_count < automatic_settings.dhw_active_min_selected_segments:
        raise _CalibrationValidationError(
            "Automatic DHW active fit rejected: insufficient active DHW excitation. "
            f"Selected segments={result.segment_count}, required >= "
            f"{automatic_settings.dhw_active_min_selected_segments}.",
            diagnostics=_build_dhw_active_diagnostics(
                result,
                active_settings=active_settings,
                automatic_settings=automatic_settings,
            ),
        )

    lower_bound, upper_bound = dhw_active_r_strat_bounds(active_settings)
    fitted_value = result.fitted_parameters.R_strat
    tolerance_ratio = automatic_settings.dhw_active_bound_tolerance_ratio
    lower_bound_represents_nearly_perfect_mixing = lower_bound <= DEFAULT_MIN_DHW_R_STRAT_K_PER_KW * (1.0 + tolerance_ratio)
    if _hits_lower_bound(
        fitted_value=fitted_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tolerance_ratio=tolerance_ratio,
    ) and not lower_bound_represents_nearly_perfect_mixing:
        raise _CalibrationValidationError(
            "Automatic DHW active fit rejected: R_strat converged to the lower bound. "
            f"R_strat={fitted_value:.6g} K/kW, lower bound={lower_bound:.6g} K/kW.",
            diagnostics=_build_dhw_active_diagnostics(
                result,
                active_settings=active_settings,
                automatic_settings=automatic_settings,
            ),
        )
    if _hits_upper_bound(
        fitted_value=fitted_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tolerance_ratio=tolerance_ratio,
    ):
        raise _CalibrationValidationError(
            "Automatic DHW active fit rejected: R_strat converged to the upper bound. "
            f"R_strat={fitted_value:.6g} K/kW, upper bound={upper_bound:.6g} K/kW.",
            diagnostics=_build_dhw_active_diagnostics(
                result,
                active_settings=active_settings,
                automatic_settings=automatic_settings,
            ),
        )
    if bool(getattr(result, "fit_capacity_split", False)):
        c_top_fraction = float(getattr(result, "fitted_c_top_fraction"))
        if _hits_lower_bound(
            fitted_value=c_top_fraction,
            lower_bound=active_settings.min_c_top_fraction,
            upper_bound=active_settings.max_c_top_fraction,
            tolerance_ratio=tolerance_ratio,
        ) or _hits_upper_bound(
            fitted_value=c_top_fraction,
            lower_bound=active_settings.min_c_top_fraction,
            upper_bound=active_settings.max_c_top_fraction,
            tolerance_ratio=tolerance_ratio,
        ):
            raise _CalibrationValidationError(
                "Automatic DHW active fit rejected: C_top_fraction converged to its bound. "
                f"C_top_fraction={c_top_fraction:.6g}, bounds=[{active_settings.min_c_top_fraction:.6g}, {active_settings.max_c_top_fraction:.6g}].",
                diagnostics=_build_dhw_active_diagnostics(
                    result,
                    active_settings=active_settings,
                    automatic_settings=automatic_settings,
                ),
            )
    if bool(getattr(result, "fit_temperature_biases", False)):
        bias_lower_bound = active_settings.min_temperature_bias_c
        bias_upper_bound = active_settings.max_temperature_bias_c
        fitted_t_top_bias_c = float(getattr(result, "fitted_t_top_bias_c"))
        fitted_t_bot_bias_c = float(getattr(result, "fitted_t_bot_bias_c"))
        if _hits_lower_bound(
            fitted_value=fitted_t_top_bias_c,
            lower_bound=bias_lower_bound,
            upper_bound=bias_upper_bound,
            tolerance_ratio=tolerance_ratio,
        ) or _hits_upper_bound(
            fitted_value=fitted_t_top_bias_c,
            lower_bound=bias_lower_bound,
            upper_bound=bias_upper_bound,
            tolerance_ratio=tolerance_ratio,
        ):
            raise _CalibrationValidationError(
                "Automatic DHW active fit rejected: T_top sensor bias converged to its bound. "
                f"T_top_bias={fitted_t_top_bias_c:.6g} °C, bounds=[{bias_lower_bound:.6g}, {bias_upper_bound:.6g}] °C.",
                diagnostics=_build_dhw_active_diagnostics(
                    result,
                    active_settings=active_settings,
                    automatic_settings=automatic_settings,
                ),
            )
        if _hits_lower_bound(
            fitted_value=fitted_t_bot_bias_c,
            lower_bound=bias_lower_bound,
            upper_bound=bias_upper_bound,
            tolerance_ratio=tolerance_ratio,
        ) or _hits_upper_bound(
            fitted_value=fitted_t_bot_bias_c,
            lower_bound=bias_lower_bound,
            upper_bound=bias_upper_bound,
            tolerance_ratio=tolerance_ratio,
        ):
            raise _CalibrationValidationError(
                "Automatic DHW active fit rejected: T_bot sensor bias converged to its bound. "
                f"T_bot_bias={fitted_t_bot_bias_c:.6g} °C, bounds=[{bias_lower_bound:.6g}, {bias_upper_bound:.6g}] °C.",
                diagnostics=_build_dhw_active_diagnostics(
                    result,
                    active_settings=active_settings,
                    automatic_settings=automatic_settings,
                ),
            )
    return _build_dhw_active_diagnostics(
        result,
        active_settings=active_settings,
        automatic_settings=automatic_settings,
    )


def build_automatic_calibration_snapshot(
    repository: TelemetryRepository,
    *,
    base_request: RunRequest,
    settings: AutomaticCalibrationSettings,
) -> CalibrationSnapshotPayload | None:
    """Run the enabled offline calibrators and assemble one persisted MPC snapshot.

    The snapshot is intentionally built from the latest successful overrides plus
    any newly fitted stages in the current cycle. This lets one stage fail
    temporarily without discarding previously validated parameters.
    """
    history_start_utc, history_end_utc = repository.get_aggregate_time_bounds()
    if history_start_utc is None or history_end_utc is None:
        return None
    history_hours = (history_end_utc - history_start_utc).total_seconds() / 3600.0
    if history_hours < settings.min_history_hours:
        return None

    previous_snapshot = repository.get_latest_calibration_snapshot()
    raw_effective_parameters = (
        previous_snapshot.effective_parameters if previous_snapshot is not None else CalibrationParameterOverrides()
    )
    effective_parameters, _ = sanitize_calibration_overrides(base_request, raw_effective_parameters)

    snapshot_time_utc = datetime.now(tz=timezone.utc)
    stage_results: dict[str, CalibrationStageResult | None] = {
        "ufh_active": None,
        "dhw_standby": None,
        "dhw_active": None,
        "cop": None,
    }
    calibration_replay_dt_hours: float | None = None
    calibration_replay_dt_error: str | None = None

    try:
        calibration_replay_dt_hours = _infer_calibration_replay_dt_hours(repository)
    except Exception as exc:  # noqa: BLE001
        calibration_replay_dt_error = str(exc)

    if calibration_replay_dt_hours is None:
        stage_results["ufh_active"] = _stage_failure(
            "ufh_active",
            calibration_replay_dt_error or "Could not infer calibration replay timestep.",
        )
    else:
        try:
            effective_request = _apply_calibration_overrides(base_request, effective_parameters)
            ufh_reference_parameters = ThermalParameters(
                dt_hours=calibration_replay_dt_hours,
                C_r=effective_request.C_r,
                C_b=effective_request.C_b,
                R_br=effective_request.R_br,
                R_ro=effective_request.R_ro,
                alpha=effective_request.alpha,
                eta=effective_request.eta,
                A_glass=effective_request.A_glass,
            )
            ufh_active_settings = build_ufh_active_calibration_settings(
                reference_parameters=ufh_reference_parameters,
                reference_internal_gains_kw=effective_request.internal_gains_kw,
                reference_internal_gains_heat_fraction=effective_request.internal_gains_heat_fraction,
                fit_eta=settings.ufh_active_fit_eta,
                fit_internal_gains_heat_fraction=settings.ufh_active_fit_internal_gains_heat_fraction,
            )
            result = calibrate_ufh_active_from_repository(
                repository,
                ufh_active_settings,
            )
            ufh_active_diagnostics = _validate_automatic_ufh_active_fit(
                repository,
                result,
                active_settings=ufh_active_settings,
                automatic_settings=settings,
            )
            overrides = CalibrationParameterOverrides(
                C_r=result.fitted_parameters.C_r,
                C_b=result.fitted_parameters.C_b,
                R_br=result.fitted_parameters.R_br,
                R_ro=result.fitted_parameters.R_ro,
                eta=(
                    result.fitted_parameters.eta
                    if bool(getattr(result, "fit_eta", False))
                    else None
                ),
                internal_gains_heat_fraction=(
                    getattr(result, "fitted_internal_gains_heat_fraction")
                    if bool(getattr(result, "fit_internal_gains_heat_fraction", False))
                    else None
                ),
                room_temperature_bias_c=(
                    getattr(result, "fitted_room_temperature_bias_c")
                    if bool(getattr(result, "fit_room_temperature_bias", False))
                    else None
                ),
            )
            effective_parameters = _merge_runtime_safe_stage_overrides(
                base_request,
                effective_parameters,
                overrides,
            )
            stage_results["ufh_active"] = _stage_success(
                "ufh_active",
                f"UFH active RC calibrated (RMSE={result.rmse_room_temperature_c:.4f} °C).",
                overrides=overrides,
                sample_count=result.sample_count,
                segment_count=result.segment_count,
                dataset_start_utc=result.dataset_start_utc,
                dataset_end_utc=result.dataset_end_utc,
                optimizer_status=result.optimizer_status,
                diagnostics=ufh_active_diagnostics,
            )
        except _CalibrationValidationError as exc:
            stage_results["ufh_active"] = _stage_failure(
                "ufh_active",
                str(exc),
                diagnostics=exc.diagnostics,
            )
        except Exception as exc:  # noqa: BLE001
            stage_results["ufh_active"] = _stage_failure("ufh_active", str(exc))

    if calibration_replay_dt_hours is None:
        stage_results["dhw_standby"] = _stage_failure(
            "dhw_standby",
            calibration_replay_dt_error or "Could not infer calibration replay timestep.",
        )
    else:
        try:
            effective_request = _apply_calibration_overrides(base_request, effective_parameters)
            dhw_standby_settings = build_dhw_standby_calibration_settings(
                dt_hours=calibration_replay_dt_hours,
                reference_c_top_kwh_per_k=effective_request.dhw_C_top,
                reference_c_bot_kwh_per_k=effective_request.dhw_C_bot,
                fit_ambient_temperature_bias=True,
                initial_ambient_temperature_bias_c=effective_request.dhw_boiler_ambient_bias_c,
            )
            result = calibrate_dhw_standby_from_repository(
                repository,
                dhw_standby_settings,
            )
            dhw_standby_diagnostics = _validate_automatic_dhw_standby_fit(
                result,
                standby_settings=dhw_standby_settings,
                automatic_settings=settings,
            )
            overrides = CalibrationParameterOverrides(
                dhw_R_loss_top=result.suggested_r_loss_k_per_kw,
                dhw_R_loss_bot=result.suggested_r_loss_k_per_kw,
                dhw_boiler_ambient_bias_c=(
                    getattr(result, "fitted_ambient_temperature_bias_c")
                    if bool(getattr(result, "fit_ambient_temperature_bias", False))
                    else None
                ),
            )
            effective_parameters = _merge_runtime_safe_stage_overrides(
                base_request,
                effective_parameters,
                overrides,
            )
            stage_results["dhw_standby"] = _stage_success(
                "dhw_standby",
                f"DHW standby calibrated (R_loss={result.suggested_r_loss_k_per_kw:.4f} K/kW).",
                overrides=overrides,
                sample_count=result.sample_count,
                dataset_start_utc=result.dataset_start_utc,
                dataset_end_utc=result.dataset_end_utc,
                optimizer_status=result.optimizer_status,
                diagnostics=dhw_standby_diagnostics,
            )
        except _CalibrationValidationError as exc:
            stage_results["dhw_standby"] = _stage_failure(
                "dhw_standby",
                str(exc),
                diagnostics=exc.diagnostics,
            )
        except Exception as exc:  # noqa: BLE001
            stage_results["dhw_standby"] = _stage_failure("dhw_standby", str(exc))

    if calibration_replay_dt_hours is None:
        stage_results["dhw_active"] = _stage_failure(
            "dhw_active",
            calibration_replay_dt_error or "Could not infer calibration replay timestep.",
        )
    else:
        try:
            effective_request = _apply_calibration_overrides(base_request, effective_parameters)
            dhw_reference_parameters = DHWParameters(
                dt_hours=calibration_replay_dt_hours,
                C_top=effective_request.dhw_C_top,
                C_bot=effective_request.dhw_C_bot,
                R_strat=effective_request.dhw_R_strat,
                R_loss_top=effective_request.dhw_R_loss_top,
                R_loss_bot=effective_request.dhw_R_loss_bot,
            )
            dhw_active_settings = build_dhw_active_calibration_settings(
                reference_parameters=dhw_reference_parameters,
                fit_capacity_split=settings.dhw_active_fit_capacity_split,
                fit_temperature_biases=settings.dhw_active_fit_temperature_biases,
                ambient_temperature_bias_c=effective_request.dhw_boiler_ambient_bias_c,
                initial_t_top_bias_c=effective_request.dhw_top_temperature_bias_c,
                initial_t_bot_bias_c=effective_request.dhw_bottom_temperature_bias_c,
            )
            result = calibrate_dhw_active_from_repository(
                repository,
                dhw_active_settings,
            )
            dhw_active_diagnostics = _validate_automatic_dhw_active_fit(
                result,
                active_settings=dhw_active_settings,
                automatic_settings=settings,
            )
            overrides = CalibrationParameterOverrides(
                dhw_C_top=(
                    result.fitted_parameters.C_top
                    if bool(getattr(result, "fit_capacity_split", False))
                    else None
                ),
                dhw_C_bot=(
                    result.fitted_parameters.C_bot
                    if bool(getattr(result, "fit_capacity_split", False))
                    else None
                ),
                dhw_R_strat=result.fitted_parameters.R_strat,
                dhw_top_temperature_bias_c=(
                    getattr(result, "fitted_t_top_bias_c")
                    if bool(getattr(result, "fit_temperature_biases", False))
                    else None
                ),
                dhw_bottom_temperature_bias_c=(
                    getattr(result, "fitted_t_bot_bias_c")
                    if bool(getattr(result, "fit_temperature_biases", False))
                    else None
                ),
            )
            effective_parameters = _merge_runtime_safe_stage_overrides(
                base_request,
                effective_parameters,
                overrides,
            )
            stage_results["dhw_active"] = _stage_success(
                "dhw_active",
                f"DHW active stratification calibrated (RMSE_top={result.rmse_t_top_c:.4f} °C).",
                overrides=overrides,
                sample_count=result.sample_count,
                segment_count=result.segment_count,
                dataset_start_utc=result.dataset_start_utc,
                dataset_end_utc=result.dataset_end_utc,
                optimizer_status=result.optimizer_status,
                diagnostics=dhw_active_diagnostics,
            )
        except _CalibrationValidationError as exc:
            stage_results["dhw_active"] = _stage_failure(
                "dhw_active",
                str(exc),
                diagnostics=exc.diagnostics,
            )
        except Exception as exc:  # noqa: BLE001
            stage_results["dhw_active"] = _stage_failure("dhw_active", str(exc))

    try:
        effective_request = _apply_calibration_overrides(base_request, effective_parameters)
        result = calibrate_cop_from_repository(
            repository,
            build_cop_calibration_settings(
                t_ref_outdoor_c=effective_request.T_ref_outdoor_curve,
                delta_t_cond_k=effective_request.delta_T_cond,
                delta_t_evap_k=effective_request.delta_T_evap,
                cop_min=effective_request.cop_min,
                cop_max=effective_request.cop_max,
            ),
        )
        overrides = CalibrationParameterOverrides(
            eta_carnot_ufh=result.fitted_parameters.eta_carnot_ufh,
            eta_carnot_dhw=result.fitted_parameters.eta_carnot_dhw,
            T_supply_min=result.fitted_parameters.T_supply_min,
            T_ref_outdoor_curve=result.fitted_parameters.T_ref_outdoor,
            heating_curve_slope=result.fitted_parameters.heating_curve_slope,
        )
        effective_parameters = _merge_runtime_safe_stage_overrides(
            base_request,
            effective_parameters,
            overrides,
        )
        stage_results["cop"] = _stage_success(
            "cop",
            f"COP calibrated (RMSE_COP={result.rmse_actual_cop:.4f}).",
            overrides=overrides,
            sample_count=result.sample_count,
            dataset_start_utc=result.dataset_start_utc,
            dataset_end_utc=result.dataset_end_utc,
            optimizer_status=result.eta_optimizer_status,
        )
    except Exception as exc:  # noqa: BLE001
        stage_results["cop"] = _stage_failure("cop", str(exc))

    return CalibrationSnapshotPayload(
        generated_at_utc=snapshot_time_utc,
        effective_parameters=effective_parameters,
        ufh_active=stage_results["ufh_active"],
        dhw_standby=stage_results["dhw_standby"],
        dhw_active=stage_results["dhw_active"],
        cop=stage_results["cop"],
    )


def run_and_persist_automatic_calibration(
    repository: TelemetryRepository,
    *,
    base_request: RunRequest,
    settings: AutomaticCalibrationSettings,
) -> CalibrationSnapshotPayload | None:
    """Execute one automatic calibration cycle and persist the resulting snapshot.

    Returns ``None`` when not enough telemetry history exists yet; otherwise the
    newly stored snapshot payload is returned.
    """
    payload = build_automatic_calibration_snapshot(
        repository,
        base_request=base_request,
        settings=settings,
    )
    if payload is None:
        return None
    repository.add_calibration_snapshot(payload)
    return payload
