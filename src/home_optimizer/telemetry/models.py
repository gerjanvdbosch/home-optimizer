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
MAX_SOLVER_STATUS_LENGTH: int = 32


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

    # Weather / Open-Meteo (injected by WeatherAugmentedBackend) -----------
    # GTI for south-facing windows [W/m²] — used for Q_solar in §4.
    # Current forecast hour; constant within each 5-minute bucket.
    gti_mean_w_per_m2: Mapped[float] = mapped_column(Float)
    gti_last_w_per_m2: Mapped[float] = mapped_column(Float)
    # GTI for PV panels [W/m²] — used in effective_price() PV correction.
    # 0.0 when no PV surface is configured on the OpenMeteoClient.
    gti_pv_mean_w_per_m2: Mapped[float] = mapped_column(Float)
    gti_pv_last_w_per_m2: Mapped[float] = mapped_column(Float)
    # Cold mains water temperature estimate [°C] — DHW disturbance T_mains (§9.1).
    # Derived from SeasonalMainsModel (cosine seasonal model).
    t_mains_estimated_mean_c: Mapped[float] = mapped_column(Float)
    t_mains_estimated_last_c: Mapped[float] = mapped_column(Float)
    # Forecast outdoor temperature for the current UTC hour [°C] (step 0 of
    # the Open-Meteo forecast).  Compare with outdoor_temperature_last_c to
    # compute forecast error per bucket (§16, training requirement 7):
    #     error = outdoor_temperature_last_c − t_out_forecast_last_c
    t_out_forecast_mean_c: Mapped[float] = mapped_column(Float)
    t_out_forecast_last_c: Mapped[float] = mapped_column(Float)

    # Derived quantities (pre-computed at flush for training convenience) ---
    # HP thermal power Q_therm = V̇ × λ × ΔT [kW] (§15, hydraulic power formula).
    # Computed per sample as a property; aggregated here for easy model fitting.
    # Negative during defrost — use defrost_active_fraction to filter.
    hp_thermal_power_mean_kw: Mapped[float] = mapped_column(Float)
    hp_thermal_power_last_kw: Mapped[float] = mapped_column(Float)
    # Net non-HP household electricity [kW] = P1 + PV − HP_elec.
    # Q_int proxy: represents appliance heat dissipation (§2, §14.2).
    household_elec_power_mean_kw: Mapped[float] = mapped_column(Float)
    household_elec_power_last_kw: Mapped[float] = mapped_column(Float)

    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=timezone.utc),
    )


class ForecastSnapshot(Base):
    """One hourly Open-Meteo forecast step — the backbone of forecast-error training.

    Each row represents a single forecast step (step_k hours ahead) from a forecast
    that was fetched at ``fetched_at_utc`` (rounded down to the current UTC hour).

    Schema rationale
    ----------------
    Storing **one row per step** (normalised) rather than wide JSON arrays lets you:

    * Query "what did the model predict for this specific hour?" with a simple WHERE.
    * JOIN directly with ``telemetry_aggregates`` on ``valid_at_utc`` to compute
      per-hour forecast errors without parsing JSON blobs.
    * Efficiently train bias-correction models per lead time (e.g. 1-h, 6-h, 24-h).

    Forecast-error JOIN example
    ---------------------------
    ::

        SELECT
            fs.valid_at_utc,
            fs.step_k,
            ta.outdoor_temperature_last_c        AS t_actual,
            fs.t_out_c                           AS t_forecast,
            ta.outdoor_temperature_last_c - fs.t_out_c AS error_k
        FROM forecast_snapshots fs
        JOIN telemetry_aggregates ta
          ON ABS(JULIANDAY(ta.bucket_end_utc) - JULIANDAY(fs.valid_at_utc)) < (0.5/24.0)
        ORDER BY fs.valid_at_utc, fs.step_k;

    Attributes
    ----------
    fetched_at_utc:
        UTC timestamp of the Open-Meteo API call, rounded down to the start
        of the UTC hour (= ``WeatherForecast.valid_from``).  All steps of
        one fetch share the same ``fetched_at_utc``.
    valid_at_utc:
        Real-world UTC time this step is valid for:
            valid_at_utc = fetched_at_utc + step_k × dt_hours
    step_k:
        Forecast lead time in steps.  ``0`` = current hour, ``1`` = +1h, …
    dt_hours:
        Time step [h].  Always ``1.0`` for Open-Meteo hourly forecasts.
    t_out_c:
        Forecast outdoor temperature [°C].  Compare with
        ``telemetry_aggregates.outdoor_temperature_last_c`` to compute error.
    gti_w_per_m2:
        Forecast GTI for south-facing windows [W/m²].  Always ≥ 0.
    gti_pv_w_per_m2:
        Forecast GTI for PV panels [W/m²].  0.0 if no PV configured.
    """

    __tablename__ = "forecast_snapshots"
    __table_args__ = (
        # Prevent duplicate rows from re-runs within the same UTC hour.
        UniqueConstraint("fetched_at_utc", "step_k", name="uq_forecast_step"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fetched_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    valid_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    step_k: Mapped[int] = mapped_column(Integer)
    dt_hours: Mapped[float] = mapped_column(Float)
    t_out_c: Mapped[float] = mapped_column(Float)
    gti_w_per_m2: Mapped[float] = mapped_column(Float)
    gti_pv_w_per_m2: Mapped[float] = mapped_column(Float)

    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=timezone.utc),
    )
class MPCLog(Base):
    """Persisted record of each MPC solve cycle.

    Each row captures:
    * The **control outputs** (thermal setpoints P_UFH, P_dhw) for the next
      time step — the "what did the MPC decide?" record.
    * The **forecast inputs at step 0** that the MPC used — the "what did the
      MPC see?" record.  Comparing these with the nearest TelemetryAggregate row
      enables forecast-error analysis (§16, training requirement 7).
    * The **solver status** to distinguish optimal solutions from greedy fallbacks.

    Relationship to TelemetryAggregate
    ------------------------------------
    Join on ``solve_time_utc`` ≈ ``bucket_end_utc`` of the nearest aggregate row
    to reconstruct the closed-loop sequence:
        actual state → MPC decision → next actual state.
    """

    __tablename__ = "mpc_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    #: UTC timestamp when the MPC solve was initiated.
    solve_time_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    # --- Control outputs (MPC decision, step 0) ---------------------------
    #: UFH thermal power setpoint for the next Δt [kW].
    p_ufh_setpoint_kw: Mapped[float] = mapped_column(Float)
    #: DHW thermal power setpoint for the next Δt [kW].
    p_dhw_setpoint_kw: Mapped[float] = mapped_column(Float)
    #: CVXPY solver status: "optimal", "infeasible", or "greedy_fallback".
    solver_status: Mapped[str] = mapped_column(String(length=MAX_SOLVER_STATUS_LENGTH))

    # --- Forecast inputs "as seen by the MPC" at step 0 ------------------
    #: Outdoor temperature forecast T_out[0] [°C] — "what the MPC expected".
    #: Compare with outdoor_temperature_last_c from the nearest aggregate row
    #: to quantify forecast error (§16, training requirement 7).
    t_out_forecast_c: Mapped[float] = mapped_column(Float)
    #: GTI forecast for windows at step 0 [W/m²] — "what the MPC expected".
    gti_forecast_w_per_m2: Mapped[float] = mapped_column(Float)
    #: Electricity price used at step 0 [€/kWh] — primary cost signal (§14.2).
    electricity_price_eur_per_kwh: Mapped[float] = mapped_column(Float)

    # --- COP values used in the objective function (§14.1) ----------------
    #: Pre-computed UFH COP for this solve (Carnot-based, §14.1).
    cop_ufh: Mapped[float] = mapped_column(Float)
    #: Pre-computed DHW COP for this solve (Carnot-based, §14.1).
    cop_dhw: Mapped[float] = mapped_column(Float)

    # --- Horizon and initial states ---------------------------------------
    #: MPC prediction horizon N used for this solve.
    horizon_steps: Mapped[int] = mapped_column(Integer)
    #: Room air temperature T_r at solve time [°C] — initial state x_UFH[0].
    t_room_initial_c: Mapped[float] = mapped_column(Float)
    #: DHW top-layer temperature T_top at solve time [°C] — initial state x_DHW[0].
    t_dhw_top_initial_c: Mapped[float] = mapped_column(Float)

    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=timezone.utc),
    )

