"""High-level calibration services orchestrating repository access and fitting."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import cast

from sqlalchemy import text

from .dataset import (
    build_cop_calibration_dataset,
    build_dhw_active_calibration_dataset,
    build_dhw_standby_calibration_dataset,
    build_ufh_active_calibration_dataset,
    build_ufh_off_calibration_dataset,
)
from .cop_offline import calibrate_cop_model
from .dhw_active import calibrate_dhw_active_stratification
from .models import (
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
)
from .dhw_standby import calibrate_dhw_standby_loss
from .ufh_active import calibrate_ufh_active_rc
from .ufh_offline import calibrate_ufh_off_envelope
from ..telemetry.models import ForecastSnapshot, TelemetryAggregate
from ..telemetry.repository import TelemetryRepository


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


def build_ufh_off_dataset_from_repository(
    repository: TelemetryRepository,
    settings: UFHOffCalibrationSettings,
) -> UFHCalibrationDataset:
    """Load telemetry history from the repository and build a UFH off-mode dataset."""
    return build_ufh_off_calibration_dataset(
        aggregates=cast(list[TelemetryAggregate], _load_calibration_aggregates(repository)),
        forecast_rows=cast(list[ForecastSnapshot], _load_calibration_forecasts(repository)),
        settings=settings,
    )


def build_cop_dataset_from_repository(
    repository: TelemetryRepository,
    settings: COPCalibrationSettings,
) -> COPCalibrationDataset:
    """Load telemetry history from the repository and build a COP calibration dataset."""
    return build_cop_calibration_dataset(
        aggregates=cast(list[TelemetryAggregate], _load_calibration_aggregates(repository)),
        settings=settings,
    )


def build_dhw_standby_dataset_from_repository(
    repository: TelemetryRepository,
    settings: DHWStandbyCalibrationSettings,
) -> DHWStandbyCalibrationDataset:
    """Load telemetry history from the repository and build a DHW standby dataset."""
    return build_dhw_standby_calibration_dataset(
        aggregates=cast(list[TelemetryAggregate], _load_calibration_aggregates(repository)),
        settings=settings,
    )


def build_dhw_active_dataset_from_repository(
    repository: TelemetryRepository,
    settings: DHWActiveCalibrationSettings,
) -> DHWActiveCalibrationDataset:
    """Load telemetry history from the repository and build an active DHW dataset."""
    return build_dhw_active_calibration_dataset(
        aggregates=cast(list[TelemetryAggregate], _load_calibration_aggregates(repository)),
        settings=settings,
    )


def build_ufh_active_dataset_from_repository(
    repository: TelemetryRepository,
    settings: UFHActiveCalibrationSettings,
) -> UFHActiveCalibrationDataset:
    """Load telemetry history from the repository and build an active UFH replay dataset."""
    return build_ufh_active_calibration_dataset(
        aggregates=cast(list[TelemetryAggregate], _load_calibration_aggregates(repository)),
        forecast_rows=cast(list[ForecastSnapshot], _load_calibration_forecasts(repository)),
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


