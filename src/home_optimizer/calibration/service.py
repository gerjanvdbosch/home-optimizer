"""High-level calibration services orchestrating repository access and fitting."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace
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
from ..optimizer import RunRequest
from ..telemetry.models import ForecastSnapshot, TelemetryAggregate
from ..telemetry.repository import TelemetryRepository
from ..types import CalibrationParameterOverrides, CalibrationSnapshotPayload, CalibrationStageResult, DHWParameters, ThermalParameters


def _parse_utc(value: object) -> datetime:
    """Parse SQLite/SQLAlchemy timestamp values into timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported timestamp type for calibration loader: {type(value)!r}")


def _load_calibration_aggregates(repository: TelemetryRepository) -> list[SimpleNamespace]:
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
        SimpleNamespace(
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


def _load_calibration_forecasts(repository: TelemetryRepository) -> list[SimpleNamespace]:
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
        SimpleNamespace(
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
    aggregates = _load_calibration_aggregates(repository)
    candidate_dt_seconds = [
        round((next_row.bucket_end_utc - previous_row.bucket_end_utc).total_seconds())
        for previous_row, next_row in zip(aggregates, aggregates[1:])
        if 0.0
        < (next_row.bucket_end_utc - previous_row.bucket_end_utc).total_seconds() / 3600.0
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
    """Return a request copy with the current effective calibration overrides applied."""
    update = overrides.as_run_request_updates()
    if not update:
        return base_request
    return base_request.model_copy(update=update)


def _stage_failure(stage_name: str, message: str) -> CalibrationStageResult:
    """Create a compact failed-stage summary for persistence/API observability."""
    return CalibrationStageResult(
        stage_name=stage_name,
        succeeded=False,
        message=message,
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
        overrides=overrides,
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
    effective_parameters = (
        previous_snapshot.effective_parameters if previous_snapshot is not None else CalibrationParameterOverrides()
    )

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
            result = calibrate_ufh_active_from_repository(
                repository,
                build_ufh_active_calibration_settings(reference_parameters=ufh_reference_parameters),
            )
            overrides = CalibrationParameterOverrides(
                C_r=result.fitted_parameters.C_r,
                C_b=result.fitted_parameters.C_b,
                R_br=result.fitted_parameters.R_br,
                R_ro=result.fitted_parameters.R_ro,
            )
            effective_parameters = effective_parameters.merged_with(overrides)
            stage_results["ufh_active"] = _stage_success(
                "ufh_active",
                f"UFH active RC calibrated (RMSE={result.rmse_room_temperature_c:.4f} °C).",
                overrides=overrides,
                sample_count=result.sample_count,
                segment_count=result.segment_count,
                dataset_start_utc=result.dataset_start_utc,
                dataset_end_utc=result.dataset_end_utc,
                optimizer_status=result.optimizer_status,
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
            result = calibrate_dhw_standby_from_repository(
                repository,
                build_dhw_standby_calibration_settings(
                    dt_hours=calibration_replay_dt_hours,
                    reference_c_top_kwh_per_k=effective_request.dhw_C_top,
                    reference_c_bot_kwh_per_k=effective_request.dhw_C_bot,
                ),
            )
            overrides = CalibrationParameterOverrides(dhw_R_loss=result.suggested_r_loss_k_per_kw)
            effective_parameters = effective_parameters.merged_with(overrides)
            stage_results["dhw_standby"] = _stage_success(
                "dhw_standby",
                f"DHW standby calibrated (R_loss={result.suggested_r_loss_k_per_kw:.4f} K/kW).",
                overrides=overrides,
                sample_count=result.sample_count,
                dataset_start_utc=result.dataset_start_utc,
                dataset_end_utc=result.dataset_end_utc,
                optimizer_status=result.optimizer_status,
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
                R_loss=effective_request.dhw_R_loss,
            )
            result = calibrate_dhw_active_from_repository(
                repository,
                build_dhw_active_calibration_settings(reference_parameters=dhw_reference_parameters),
            )
            overrides = CalibrationParameterOverrides(dhw_R_strat=result.fitted_parameters.R_strat)
            effective_parameters = effective_parameters.merged_with(overrides)
            stage_results["dhw_active"] = _stage_success(
                "dhw_active",
                f"DHW active stratification calibrated (RMSE_top={result.rmse_t_top_c:.4f} °C).",
                overrides=overrides,
                sample_count=result.sample_count,
                segment_count=result.segment_count,
                dataset_start_utc=result.dataset_start_utc,
                dataset_end_utc=result.dataset_end_utc,
                optimizer_status=result.optimizer_status,
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
            eta_carnot=result.fitted_parameters.eta_carnot,
            T_supply_min=result.fitted_parameters.T_supply_min,
            heating_curve_slope=result.fitted_parameters.heating_curve_slope,
        )
        effective_parameters = effective_parameters.merged_with(overrides)
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


