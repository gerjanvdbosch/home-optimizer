"""High-level calibration services orchestrating repository access and fitting."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from sqlalchemy import text

from .dataset import build_ufh_off_calibration_dataset
from .models import UFHCalibrationDataset, UFHOffCalibrationResult, UFHOffCalibrationSettings
from .ufh_offline import calibrate_ufh_off_envelope
from ..telemetry.repository import TelemetryRepository


def _parse_utc(value: object) -> datetime:
    """Parse SQLite/SQLAlchemy timestamp values into timezone-aware UTC datetimes."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"Unsupported timestamp type for calibration loader: {type(value)!r}")


def _load_calibration_aggregates(repository: TelemetryRepository) -> list[SimpleNamespace]:
    """Load only the telemetry columns required by the first-stage UFH calibrator."""
    statement = text(
        """
        SELECT
            bucket_end_utc,
            hp_mode_last,
            defrost_active_fraction,
            booster_heater_active_fraction,
            room_temperature_last_c,
            outdoor_temperature_mean_c,
            household_elec_power_mean_kw
        FROM telemetry_aggregates
        ORDER BY bucket_end_utc ASC
        """
    )
    with repository.engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()
    return [
        SimpleNamespace(
            bucket_end_utc=_parse_utc(row["bucket_end_utc"]),
            hp_mode_last=str(row["hp_mode_last"]),
            defrost_active_fraction=float(row["defrost_active_fraction"]),
            booster_heater_active_fraction=float(row["booster_heater_active_fraction"]),
            room_temperature_last_c=float(row["room_temperature_last_c"]),
            outdoor_temperature_mean_c=float(row["outdoor_temperature_mean_c"]),
            household_elec_power_mean_kw=float(row["household_elec_power_mean_kw"]),
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
        aggregates=_load_calibration_aggregates(repository),
        forecast_rows=_load_calibration_forecasts(repository),
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

