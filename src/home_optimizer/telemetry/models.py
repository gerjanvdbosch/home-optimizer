"""SQLAlchemy ORM models and validated settings for telemetry persistence.

The telemetry layer stores aggregated sensor windows so the project can later
estimate thermal parameters, train forecasting models, and backtest MPC runs
without writing every minute-level sample to disk.

Units
-----
Temperature : °C
Power       : kW
Flow        : L/min
Time        : UTC datetimes
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import Boolean, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DEFAULT_TELEMETRY_TIMEZONE_NAME: str = "UTC"
DEFAULT_SAMPLING_INTERVAL_SECONDS: int = 30
DEFAULT_FLUSH_INTERVAL_SECONDS: int = 300
DEFAULT_TELEMETRY_JOB_ID_PREFIX: str = "telemetry"
MAX_HP_MODE_LENGTH: int = 64


class TelemetryCollectorSettings(BaseModel):
    """Validated runtime settings for the telemetry collector.

    Parameters
    ----------
    database_url:
        SQLAlchemy database URL for telemetry storage, for example
        ``"sqlite:///telemetry.sqlite3"``.
    sampling_interval_seconds:
        APScheduler polling interval for live sensors [s].
    flush_interval_seconds:
        Aggregation window written to the database [s].  Must be an integer
        multiple of ``sampling_interval_seconds`` so each bucket represents a
        stable number of samples.
    timezone_name:
        APScheduler timezone name.  The database timestamps themselves are UTC.
    job_id_prefix:
        Prefix used for the APScheduler sampling and flush job identifiers.
    """

    model_config = ConfigDict(frozen=True)

    database_url: str
    sampling_interval_seconds: int = Field(default=DEFAULT_SAMPLING_INTERVAL_SECONDS, gt=0)
    flush_interval_seconds: int = Field(default=DEFAULT_FLUSH_INTERVAL_SECONDS, gt=0)
    timezone_name: str = DEFAULT_TELEMETRY_TIMEZONE_NAME
    job_id_prefix: str = DEFAULT_TELEMETRY_JOB_ID_PREFIX

    @field_validator("database_url", "timezone_name", "job_id_prefix")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        """Reject blank strings early so deployment errors are explicit."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank.")
        return stripped

    @model_validator(mode="after")
    def _validate_intervals(self) -> "TelemetryCollectorSettings":
        """Enforce that the flush interval cleanly contains whole sample periods."""
        if self.flush_interval_seconds < self.sampling_interval_seconds:
            raise ValueError(
                "flush_interval_seconds must be greater than or equal to "
                "sampling_interval_seconds."
            )
        if self.flush_interval_seconds % self.sampling_interval_seconds != 0:
            raise ValueError(
                "flush_interval_seconds must be an integer multiple of "
                "sampling_interval_seconds."
            )
        return self


class Base(DeclarativeBase):
    """Declarative SQLAlchemy base for telemetry tables."""


class TelemetryAggregate(Base):
    """Aggregated telemetry bucket persisted to SQL.

    Each row represents one aggregation window [bucket_start_utc, bucket_end_utc]
    built from ``sample_count`` live sensor samples.  For every numeric signal we
    persist both the arithmetic mean over the bucket and the last observed value.
    The mean supports robust energy- and trend-oriented model fitting; the last
    value preserves the end-of-window operating point needed for state-based
    training and replay.

    Buckets are expected to be mode-homogeneous: if the heat pump changes from
    UFH to DHW (or vice versa), the collector flushes the current buffer before
    starting the new mode bucket.
    """

    __tablename__ = "telemetry_aggregates"
    __table_args__ = (
        UniqueConstraint("bucket_start_utc", "bucket_end_utc", name="uq_telemetry_bucket"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bucket_start_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    bucket_end_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    sample_count: Mapped[int] = mapped_column(Integer)
    hp_mode_last: Mapped[str] = mapped_column(String(length=MAX_HP_MODE_LENGTH))

    # Room / weather telemetry ---------------------------------------------
    room_temperature_mean_c: Mapped[float] = mapped_column(Float)
    room_temperature_last_c: Mapped[float] = mapped_column(Float)
    outdoor_temperature_mean_c: Mapped[float] = mapped_column(Float)
    outdoor_temperature_last_c: Mapped[float] = mapped_column(Float)
    thermostat_setpoint_mean_c: Mapped[float] = mapped_column(Float)
    thermostat_setpoint_last_c: Mapped[float] = mapped_column(Float)

    # Heat-pump / electricity telemetry -----------------------------------
    hp_supply_temperature_mean_c: Mapped[float] = mapped_column(Float)
    hp_supply_temperature_last_c: Mapped[float] = mapped_column(Float)
    hp_supply_target_temperature_mean_c: Mapped[float] = mapped_column(Float)
    hp_supply_target_temperature_last_c: Mapped[float] = mapped_column(Float)
    hp_return_temperature_mean_c: Mapped[float] = mapped_column(Float)
    hp_return_temperature_last_c: Mapped[float] = mapped_column(Float)
    hp_flow_mean_lpm: Mapped[float] = mapped_column(Float)
    hp_flow_last_lpm: Mapped[float] = mapped_column(Float)
    hp_electric_power_mean_kw: Mapped[float] = mapped_column(Float)
    hp_electric_power_last_kw: Mapped[float] = mapped_column(Float)
    p1_net_power_mean_kw: Mapped[float] = mapped_column(Float)
    p1_net_power_last_kw: Mapped[float] = mapped_column(Float)
    pv_output_mean_kw: Mapped[float] = mapped_column(Float)
    pv_output_last_kw: Mapped[float] = mapped_column(Float)

    # DHW telemetry --------------------------------------------------------
    dhw_top_temperature_mean_c: Mapped[float] = mapped_column(Float)
    dhw_top_temperature_last_c: Mapped[float] = mapped_column(Float)
    dhw_bottom_temperature_mean_c: Mapped[float] = mapped_column(Float)
    dhw_bottom_temperature_last_c: Mapped[float] = mapped_column(Float)

    # Shutter telemetry ---------------------------------------------------
    # shutter_living_room_pct: fraction of the MPC horizon with sunlight entering
    # the room.  Mean captures the average obstruction over the bucket; last
    # reflects the end-of-window position used to initialise the next MPC step.
    shutter_living_room_mean_pct: Mapped[float] = mapped_column(Float)
    shutter_living_room_last_pct: Mapped[float] = mapped_column(Float)

    # Heat-pump status flags -----------------------------------------------
    # Fraction of samples within the bucket where the flag was True [0.0–1.0].
    # A non-zero fraction shows that a transient event (defrost, booster) occurred
    # during the window even if it ended before the flush.
    defrost_active_fraction: Mapped[float] = mapped_column(Float)
    # End-of-window boolean state: relevant for initialising the next Kalman step.
    defrost_active_last: Mapped[bool] = mapped_column(Boolean)
    booster_heater_active_fraction: Mapped[float] = mapped_column(Float)
    booster_heater_active_last: Mapped[bool] = mapped_column(Boolean)

    # Boiler ambient / refrigerant temperatures ----------------------------
    # These feed directly into COP pre-calculation (§14.1) and DHW standby-loss
    # estimation (§9.2).  Both mean and last are stored for model identification.
    boiler_ambient_temp_mean_c: Mapped[float] = mapped_column(Float)
    boiler_ambient_temp_last_c: Mapped[float] = mapped_column(Float)
    refrigerant_condensation_temp_mean_c: Mapped[float] = mapped_column(Float)
    refrigerant_condensation_temp_last_c: Mapped[float] = mapped_column(Float)
    refrigerant_temp_mean_c: Mapped[float] = mapped_column(Float)
    refrigerant_temp_last_c: Mapped[float] = mapped_column(Float)

    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=timezone.utc),
    )

