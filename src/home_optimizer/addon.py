"""Home Assistant addon entry point for Home Optimizer.

Architecture
------------
1. Read ``/data/options.json`` injected by the HA supervisor.
2. Validate every option with Pydantic (fail-fast on missing or invalid values).
3. Build a :class:`~home_optimizer.sensors.HomeAssistantBackend` from the
   configured entity IDs.
4. Start the :class:`~home_optimizer.telemetry.BufferedTelemetryCollector` in
   the background via APScheduler (non-blocking).
5. Run the FastAPI application via Uvicorn (blocking main thread).
6. On shutdown (SIGTERM / SIGINT): flush the final telemetry bucket and stop
   the collector cleanly.

HA supervisor environment
--------------------------
- ``SUPERVISOR_TOKEN``  Long-lived token for the HA REST API.  The
  :class:`~home_optimizer.sensors.HomeAssistantBackend` reads it automatically
  from the environment so no explicit token is needed in options.
- ``/data/options.json``  Addon options written by the HA supervisor from the
  user's Configuration tab.  The path is a constant injected into the
  container's filesystem.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

import uvicorn
from pydantic import BaseModel, Field, field_validator, model_validator

from .api import api_service, app
from .application.runtime import OptimizerRuntime
from .application.optimizer import RunRequest
from .forecasting import ForecastService
from .application.optimizer import Optimizer
from .pricing import PriceConfig, PriceMode, build_price_model
from .sensors.ha_backend import HAEntityConfig, HomeAssistantBackend
from .sensors.open_meteo import OpenMeteoClient, SeasonalMainsModel
from .sensors.weather_backend import WeatherAugmentedBackend
from .telemetry import (
    BufferedTelemetryCollector,
    TelemetryRepository,
    ForecastPersister,
    TelemetryCollectorSettings,
)
from .types.constants import LAMBDA_WATER_KWH_PER_M3_K, M3_PER_LITER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Path where the HA supervisor writes the addon options.
_OPTIONS_PATH: Path = Path("/data/options.json")

#: Host to bind Uvicorn — bind to all interfaces inside the container.
_BIND_HOST: str = "0.0.0.0"

#: Automatic calibration interval [s] — slow background job over persisted telemetry.
DEFAULT_CALIBRATION_INTERVAL_SECONDS: int = 6 * 3600

#: Minimum telemetry history before the addon attempts offline calibration [h].
DEFAULT_CALIBRATION_MIN_HISTORY_HOURS: float = 24.0

DEFAULT_FORECAST_TRAINING_HOUR_UTC: int = 2
DEFAULT_FORECAST_TRAINING_MINUTE_UTC: int = 0
FORECAST_TRAINING_MISFIRE_GRACE_SECONDS: int = 3600

log = logging.getLogger("home_optimizer.addon")


# ---------------------------------------------------------------------------
# Validated options model
# ---------------------------------------------------------------------------


class AddonOptions(BaseModel):
    """Validated addon options loaded from ``/data/options.json``.

    All ``sensor_*`` fields map directly to Home Assistant entity IDs.
    Scale factors convert non-standard HA units to the model's canonical units:

    - Power [kW]: set ``*_scale=0.001`` when the entity reports in W.
    - Flow [L/min]: set ``sensor_hp_flow_scale=16.667`` when the entity
      reports in m³/h.

    Validation rules
    ----------------
    - ``sampling_interval_seconds > 0``
    - ``flush_interval_seconds`` is an integer multiple of
      ``sampling_interval_seconds`` (required by ``TelemetryCollectorSettings``).
    - All entity-ID strings must be non-blank (fail-fast).
    - All scale factors must be non-zero.
    """

    # --- Telemetry storage ---
    database_path: str = Field(
        ..., min_length=1, description="SQLite path, e.g. /config/database.sqlite3"
    )
    # Directory where persisted ML forecast-model artifacts are stored. When
    # running under the supervisor this is typically a writable folder under
    # /config such as /config/models. The path must be a non-empty string.
    models_path: str = Field(
        "/config/models",
        min_length=1,
        description="Directory path where ML forecast artifacts are persisted, e.g. /config/models",
    )
    sampling_interval_seconds: int = Field(10, gt=0, description="Sensor polling interval [s]")
    flush_interval_seconds: int = Field(
        300, gt=0, description="Aggregation window written to DB [s]"
    )
    api_port: int = Field(8099, gt=0, lt=65536, description="Uvicorn listen port")

    # --- MPC scheduling ---
    mpc_interval_seconds: int = Field(
        3600,
        gt=0,
        description=(
            "How often the MPC optimisation runs [s].  "
            "3600 = once per hour, 1800 = every 30 min.  "
            "Set to 0 in options.json to disable MPC scheduling entirely "
            "(simulator-only mode).  Must be > 0 when non-zero."
        ),
    )
    mpc_enabled: bool = Field(
        True,
        description=(
            "Enable periodic MPC scheduling.  When False the MPC is only "
            "available via the POST /api/simulate endpoint (simulator mode)."
        ),
    )
    calibration_enabled: bool = Field(
        True,
        description="Enable periodic automatic calibration from persisted telemetry.",
    )
    calibration_interval_seconds: int = Field(
        DEFAULT_CALIBRATION_INTERVAL_SECONDS,
        gt=0,
        description="How often the offline calibration pipeline runs [s].",
    )
    calibration_min_history_hours: float = Field(
        DEFAULT_CALIBRATION_MIN_HISTORY_HOURS,
        gt=0.0,
        description="Minimum persisted telemetry history required before auto-calibration [h].",
    )
    forecast_training_enabled: bool = Field(
        True,
        description="Enable nightly persisted ML forecast-model training (currently shutter_forecast).",
    )
    forecast_training_hour_utc: int = Field(
        DEFAULT_FORECAST_TRAINING_HOUR_UTC,
        ge=0,
        le=23,
        description="UTC hour for the nightly persisted ML forecast-model training job.",
    )
    forecast_training_minute_utc: int = Field(
        DEFAULT_FORECAST_TRAINING_MINUTE_UTC,
        ge=0,
        le=59,
        description="UTC minute for the nightly persisted ML forecast-model training job.",
    )

    # --- MPC physical parameters (§15 of the theory document) ---
    # These are the house / heat-pump parameters used by the periodic runner.
    # The simulator uses its own values from the browser form.
    mpc_C_r: float = Field(6.0, gt=0.0, description="Room thermal capacity C_r [kWh/K]")
    mpc_C_b: float = Field(10.0, gt=0.0, description="Slab thermal capacity C_b [kWh/K]")
    mpc_R_br: float = Field(1.0, gt=0.0, description="Floor-to-room resistance R_br [K/kW]")
    mpc_R_ro: float = Field(10.0, gt=0.0, description="Room-to-outside resistance R_ro [K/kW]")
    mpc_alpha: float = Field(0.25, ge=0.0, le=1.0, description="Solar fraction to room air α [-]")
    mpc_eta: float = Field(0.55, ge=0.0, le=1.0, description="Window transmittance η [-]")
    mpc_A_glass: float = Field(7.5, gt=0.0, description="South-facing glazing area [m²]")
    internal_gains_heat_fraction: float = Field(
        0.70,
        ge=0.0,
        le=1.0,
        description="Fraction of household baseload that becomes useful indoor heat gains [-]",
    )
    mpc_P_max: float = Field(4.5, gt=0.0, description="Max UFH thermal power P_UFH,max [kW]")
    mpc_P_min: float = Field(
        0.5,
        ge=0.0,
        description="Minimum UFH thermal power when the heat pump is switched on [kW]",
    )
    mpc_delta_P_max: float = Field(1.0, gt=0.0, description="Max UFH ramp-rate [kW/step]")
    mpc_T_min: float = Field(19.0, description="Min comfort temperature [°C]")
    mpc_T_max: float = Field(22.5, description="Max comfort temperature [°C]")
    mpc_T_ref: float = Field(20.5, description="Comfort setpoint T_ref [°C]")
    mpc_horizon_hours: int = Field(24, ge=4, le=48, description="MPC horizon N [steps]")
    mpc_dt_hours: float = Field(1.0, ge=0.25, le=2.0, description="MPC time step Δt [h]")
    mpc_Q_c: float = Field(8.0, ge=0.0, description="Comfort weight Q_c")
    mpc_R_c: float = Field(0.05, ge=0.0, description="Regularisation weight R_c")
    mpc_Q_N: float = Field(12.0, ge=0.0, description="Terminal comfort weight Q_N")
    mpc_ufh_on_off_control_enabled: bool = Field(
        True,
        description="Enable UFH mixed-integer on/off scheduling with minimum-power enforcement.",
    )
    mpc_ufh_switch_penalty_eur: float = Field(
        0.05,
        ge=0.0,
        description="Per-switch penalty for UFH on/off transitions [€ per switch].",
    )
    mpc_P_hp_max_elec: float = Field(2.5, gt=0.0, description="Max HP electrical power [kW]")
    mpc_heat_pump_topology: str = Field(
        "shared",
        description="Heat-pump topology policy for periodic MPC: shared or exclusive.",
    )
    mpc_exclusive_heat_pump_mode: str | None = Field(
        None,
        description="Active mode when mpc_heat_pump_topology=exclusive: ufh or dhw.",
    )
    mpc_eta_carnot_ufh: float = Field(0.45, ge=0.1, le=0.99, description="UFH Carnot efficiency η_ufh [-]")
    mpc_eta_carnot_dhw: float = Field(0.45, ge=0.1, le=0.99, description="DHW Carnot efficiency η_dhw [-]")
    mpc_delta_T_cond: float = Field(5.0, ge=0.0, le=15.0, description="Condensing approach ΔT [K]")
    mpc_delta_T_evap: float = Field(5.0, ge=0.0, le=15.0, description="Evaporating approach ΔT [K]")
    mpc_cop_min: float = Field(1.5, ge=1.01, le=5.0, description="COP lower bound [-]")
    mpc_cop_max: float = Field(7.0, ge=2.0, le=15.0, description="COP upper bound [-]")
    mpc_dhw_enabled: bool = Field(True, description="Include DHW in periodic MPC")
    mpc_dhw_P_max: float = Field(3.0, gt=0.0, description="Max DHW thermal power [kW]")
    mpc_dhw_P_min: float = Field(
        1.5,
        ge=0.0,
        description="Minimum DHW thermal power when the heat pump is switched on [kW]",
    )
    mpc_dhw_delta_P_max: float = Field(1.0, gt=0.0, description="Max DHW ramp-rate [kW/step]")
    mpc_dhw_on_off_control_enabled: bool = Field(
        True,
        description="Enable DHW mixed-integer on/off scheduling with minimum-power enforcement.",
    )
    mpc_dhw_switch_penalty_eur: float = Field(
        0.05,
        ge=0.0,
        description="Per-switch penalty for DHW on/off transitions [€ per switch].",
    )
    mpc_dhw_T_min: float = Field(30.0, description="Min DHW tap temperature [°C]")
    mpc_dhw_T_target: float = Field(55.0, description="Target DHW storage temperature used by the scheduled optimizer [°C]")
    mpc_dhw_T_legionella: float = Field(60.0, description="Legionella target temperature [°C]")
    mpc_dhw_target_rho_factor: float = Field(
        25.0,
        gt=0.0,
        description="Penalty weight for DHW shortfall relative to the scheduled target.",
    )
    mpc_dhw_schedule_enabled: bool = Field(
        True,
        description="Enable a daily DHW schedule window in which the target temperature becomes active.",
    )
    mpc_dhw_schedule_start_hour_local: int = Field(
        22,
        ge=0,
        le=23,
        description="Local clock hour at which the scheduled DHW target window starts.",
    )
    mpc_dhw_schedule_duration_hours: int = Field(
        2,
        ge=1,
        le=12,
        description="Duration of the scheduled DHW target window [h].",
    )
    mpc_dhw_schedule_target_c: float = Field(
        55.0,
        ge=20.0,
        le=85.0,
        description="Target DHW temperature during the scheduled window [°C].",
    )
    mpc_dhw_preheat_lead_steps: int = Field(
        0,
        ge=0,
        le=24,
        description="Legacy fallback lead time before a significant DHW draw when no explicit schedule is active.",
    )
    mpc_dhw_significant_tap_threshold_m3_per_h: float = Field(
        0.01,
        ge=0.0,
        le=0.5,
        description="Tap-flow threshold above which the controller treats a DHW draw as significant [m³/h].",
    )
    mpc_dhw_R_strat: float = Field(10.0, gt=0.0, description="DHW stratification resistance [K/kW]")
    mpc_dhw_R_loss_top: float = Field(50.0, gt=0.0, description="Top-node DHW standby-loss resistance [K/kW]")
    mpc_dhw_R_loss_bot: float = Field(50.0, gt=0.0, description="Bottom-node DHW standby-loss resistance [K/kW]")

    # ── Electricity price model ───────────────────────────────────────────
    price_mode: str = Field(
        "flat",
        description=(
            "Electricity price mode: 'flat' | 'dual' | 'nordpool'.  "
            "flat: constant rate.  "
            "dual: piek/dal tariff with terugleververgoeding.  "
            "nordpool: live Entso-E day-ahead prices."
        ),
    )
    price_flat_rate: float = Field(
        0.25, ge=0.0, le=5.0, description="Flat import tariff [€/kWh] (used when price_mode=flat)"
    )
    price_high_rate: float = Field(
        0.36,
        ge=0.0,
        le=5.0,
        description="Peak (high) import tariff [€/kWh] (used when price_mode=dual)",
    )
    price_low_rate: float = Field(
        0.21,
        ge=0.0,
        le=5.0,
        description="Off-peak (low) import tariff [€/kWh] (used when price_mode=dual)",
    )
    price_feed_in_rate: float = Field(
        0.09,
        ge=0.0,
        le=5.0,
        description=(
            "Feed-in / terugleververgoeding rate [€/kWh] for PV net export "
            "(used when price_mode=dual)"
        ),
    )
    price_low_tariff_hours: list[int] = Field(
        default_factory=lambda: [23, 0, 1, 2, 3, 4, 5, 6],
        description=(
            "Hour-of-day integers (0–23) that qualify as off-peak in dual-tariff mode.  "
            "Default: 23:00–06:00 inclusive."
        ),
    )
    nordpool_country_code: str = Field(
        "NL",
        min_length=2,
        description="Nordpool / Entso-E bidding zone code, e.g. 'NL' or 'DE-LU'",
    )
    nordpool_vat_factor: float = Field(
        1.21,
        ge=1.0,
        le=2.0,
        description="VAT + surcharge multiplier for Nordpool raw price (e.g. 1.21 = 21% BTW)",
    )

    # --- Temperature sensors (°C — scale 1.0) ---
    sensor_room_temperature: str = Field(..., min_length=1)
    sensor_outdoor_temperature: str = Field(..., min_length=1)
    sensor_hp_supply_temperature: str = Field(..., min_length=1)
    sensor_hp_supply_target_temperature: str = Field(..., min_length=1)
    sensor_hp_return_temperature: str = Field(..., min_length=1)
    sensor_thermostat_setpoint: str = Field(..., min_length=1)
    sensor_dhw_top_temperature: str = Field(..., min_length=1)
    sensor_dhw_bottom_temperature: str = Field(..., min_length=1)
    sensor_boiler_ambient_temperature: str = Field(..., min_length=1)
    sensor_refrigerant_condensation_temperature: str = Field(..., min_length=1)
    sensor_refrigerant_liquid_line_temperature: str = Field(
        ..., min_length=1, description="Buitenunit Vloeistofleiding [°C]"
    )
    sensor_discharge_temperature: str = Field(
        ..., min_length=1, description="Ontladingstemperatuur compressor [°C]"
    )

    # --- Flow sensor (L/min) ---
    sensor_hp_flow_lpm: str = Field(..., min_length=1)
    sensor_hp_flow_scale: float = Field(
        1.0, description="Unit scale: 1.0 if already L/min, 16.667 if m³/h"
    )

    # --- Power sensors (kW) ---
    sensor_hp_electric_power: str = Field(..., min_length=1)
    sensor_hp_electric_power_scale: float = Field(
        1.0, description="Unit scale: 1.0 if already kW, 0.001 if W"
    )
    sensor_p1_net_power: str = Field(
        ...,
        min_length=1,
        description="P1 smart-meter entity: positive = import [kW], negative = export [kW]",
    )
    sensor_p1_net_power_scale: float = Field(
        1.0, description="Unit scale: 1.0 if already kW, 0.001 if W"
    )
    sensor_pv_output: str = Field(..., min_length=1)
    sensor_pv_output_scale: float = Field(1.0)

    # --- Categorical sensor ---
    sensor_hp_mode: str = Field(
        ..., min_length=1, description="Entity reporting ufh/dhw/defrost/off"
    )

    # --- Shutter (0–100 %) ---
    sensor_shutter_living_room: str = Field(
        ..., min_length=1, description="Shutter/cover position entity (100 = fully open)"
    )

    # --- Binary sensors (on/off) ---
    sensor_defrost_active: str = Field(
        ..., min_length=1, description="binary_sensor for HP defrost cycle"
    )
    sensor_booster_heater_active: str = Field(
        ..., min_length=1, description="binary_sensor for DHW booster element"
    )

    # --- PV panel orientation (Open-Meteo GTI forecast) ---
    pv_tilt: float = Field(
        50.0,
        ge=0.0,
        le=90.0,
        description=(
            "PV panel tilt angle [°].  0 = horizontal surface, 90 = vertical wall.  "
            "Passed directly to Open-Meteo to compute irradiance on the PV surface."
        ),
    )
    pv_azimuth: float = Field(
        148.0,
        ge=0.0,
        lt=360.0,
        description=(
            "PV panel orientation as a compass bearing [°].  "
            "0 = North, 90 = East, 180 = South, 270 = West.  "
            "Converted internally to the Open-Meteo solar convention "
            "(0 = South, −90 = East, +90 = West) via azimuth_OM = pv_azimuth − 180."
        ),
    )

    # --- DHW boiler geometry ---
    boiler_tank_liters: int = Field(
        200,
        gt=0,
        description=(
            "Total DHW boiler tank volume [L].  "
            "Used to derive DHW thermal capacities C_top and C_bot via λ·V_tank."
        ),
    )

    # --- Monotonically-increasing energy counters [kWh] (required) ---
    # These counters enable accurate COP validation (§14.1), grid-cost
    # back-testing (§14.2), and PV self-consumption analysis.
    sensor_pv_total_kwh: str = Field(
        ...,
        min_length=1,
        description="HA entity ID for cumulative PV energy counter [kWh].",
    )
    sensor_hp_electric_total_kwh: str = Field(
        ...,
        min_length=1,
        description="HA entity ID for cumulative HP electrical energy counter [kWh].",
    )
    sensor_p1_import_total_kwh: str = Field(
        ...,
        min_length=1,
        description="HA entity ID for cumulative P1 import energy counter [kWh].",
    )
    sensor_p1_export_total_kwh: str = Field(
        ...,
        min_length=1,
        description="HA entity ID for cumulative P1 export energy counter [kWh].",
    )

    @field_validator(
        "sensor_hp_flow_scale",
        "sensor_hp_electric_power_scale",
        "sensor_p1_net_power_scale",
        "sensor_pv_output_scale",
    )
    @classmethod
    def _scale_nonzero(cls, v: float) -> float:
        """Scale factors of exactly 0 would silently zero-out every reading."""
        if v == 0.0:
            raise ValueError("Scale factor must be non-zero.")
        return v

    @model_validator(mode="after")
    def _flush_multiple_of_sample(self) -> "AddonOptions":
        """Enforce that the flush window contains a whole number of sample periods."""
        if self.flush_interval_seconds < self.sampling_interval_seconds:
            raise ValueError("flush_interval_seconds must be >= sampling_interval_seconds.")
        if self.flush_interval_seconds % self.sampling_interval_seconds != 0:
            raise ValueError(
                "flush_interval_seconds must be an integer multiple of "
                "sampling_interval_seconds."
            )
        if self.mpc_heat_pump_topology not in {"shared", "exclusive"}:
            raise ValueError("mpc_heat_pump_topology must be 'shared' or 'exclusive'.")
        if self.mpc_heat_pump_topology == "shared":
            if self.mpc_exclusive_heat_pump_mode is not None:
                raise ValueError(
                    "mpc_exclusive_heat_pump_mode must be omitted when mpc_heat_pump_topology='shared'."
                )
        elif self.mpc_exclusive_heat_pump_mode is not None and self.mpc_exclusive_heat_pump_mode not in {"ufh", "dhw"}:
            raise ValueError(
                "mpc_exclusive_heat_pump_mode must be 'ufh' or 'dhw' when provided."
            )
        if self.mpc_P_min > self.mpc_P_max:
            raise ValueError("mpc_P_min must be less than or equal to mpc_P_max.")
        if self.mpc_dhw_P_min > self.mpc_dhw_P_max:
            raise ValueError("mpc_dhw_P_min must be less than or equal to mpc_dhw_P_max.")
        if (
            self.mpc_heat_pump_topology == "exclusive"
            and self.mpc_exclusive_heat_pump_mode is None
            and not (self.mpc_ufh_on_off_control_enabled and self.mpc_dhw_on_off_control_enabled)
        ):
            raise ValueError(
                "exclusive topology without a fixed mode requires both UFH and DHW on/off control."
            )
        return self


def _build_runtime_base_request(
    opts: AddonOptions,
    *,
    defaults: RunRequest | None = None,
) -> RunRequest:
    """Build the canonical runtime base request from validated addon options.

    The returned request is the single source of truth for periodic MPC,
    calibration replay, and API defaults served by the addon runtime.
    """
    from .application.optimizer import RunRequest  # noqa: PLC0415

    runtime_defaults = defaults or RunRequest.model_validate({})
    price_cfg = {
        "mode": opts.price_mode,
        "flat_rate_eur_per_kwh": opts.price_flat_rate,
        "high_rate_eur_per_kwh": opts.price_high_rate,
        "low_rate_eur_per_kwh": opts.price_low_rate,
        "feed_in_rate_eur_per_kwh": opts.price_feed_in_rate,
        "low_tariff_hours": opts.price_low_tariff_hours,
        "nordpool_country_code": opts.nordpool_country_code,
        "nordpool_vat_factor": opts.nordpool_vat_factor,
    }
    return runtime_defaults.model_copy(
        update={
            "C_r": opts.mpc_C_r,
            "C_b": opts.mpc_C_b,
            "R_br": opts.mpc_R_br,
            "R_ro": opts.mpc_R_ro,
            "alpha": opts.mpc_alpha,
            "eta": opts.mpc_eta,
            "A_glass": opts.mpc_A_glass,
            "T_r_init": runtime_defaults.T_r_init,
            "T_b_init": runtime_defaults.T_b_init,
            "previous_power_kw": runtime_defaults.previous_power_kw,
            "horizon_hours": opts.mpc_horizon_hours,
            "dt_hours": opts.mpc_dt_hours,
            "Q_c": opts.mpc_Q_c,
            "R_c": opts.mpc_R_c,
            "Q_N": opts.mpc_Q_N,
            "P_max": opts.mpc_P_max,
            "P_min": opts.mpc_P_min,
            "delta_P_max": opts.mpc_delta_P_max,
            "T_min": opts.mpc_T_min,
            "T_max": opts.mpc_T_max,
            "T_ref": opts.mpc_T_ref,
            "ufh_on_off_control_enabled": opts.mpc_ufh_on_off_control_enabled,
            "ufh_switch_penalty_eur": opts.mpc_ufh_switch_penalty_eur,
            "outdoor_temperature_c": runtime_defaults.outdoor_temperature_c,
            "t_out_forecast": None,
            "gti_window_forecast": None,
            "gti_pv_forecast": None,
            "price_config": price_cfg,
            "internal_gains_kw": runtime_defaults.internal_gains_kw,
            "internal_gains_heat_fraction": opts.internal_gains_heat_fraction,
            "pv_enabled": runtime_defaults.pv_enabled,
            "pv_peak_kw": runtime_defaults.pv_peak_kw,
            "dhw_enabled": opts.mpc_dhw_enabled,
            "dhw_C_top": opts.boiler_tank_liters * M3_PER_LITER * LAMBDA_WATER_KWH_PER_M3_K / 2.0,
            "dhw_C_bot": opts.boiler_tank_liters * M3_PER_LITER * LAMBDA_WATER_KWH_PER_M3_K / 2.0,
            "dhw_R_strat": opts.mpc_dhw_R_strat,
            "dhw_R_loss_top": opts.mpc_dhw_R_loss_top,
            "dhw_R_loss_bot": opts.mpc_dhw_R_loss_bot,
            "dhw_lambda_water_kwh_per_m3k": LAMBDA_WATER_KWH_PER_M3_K,
            "dhw_T_top_init": runtime_defaults.dhw_T_top_init,
            "dhw_T_bot_init": runtime_defaults.dhw_T_bot_init,
            "dhw_P_max": opts.mpc_dhw_P_max,
            "dhw_P_min": opts.mpc_dhw_P_min,
            "dhw_delta_P_max": opts.mpc_dhw_delta_P_max,
            "dhw_on_off_control_enabled": opts.mpc_dhw_on_off_control_enabled,
            "dhw_switch_penalty_eur": opts.mpc_dhw_switch_penalty_eur,
            "dhw_T_min": opts.mpc_dhw_T_min,
            "dhw_T_target": opts.mpc_dhw_T_target,
            "dhw_T_legionella": opts.mpc_dhw_T_legionella,
            "dhw_target_rho_factor": opts.mpc_dhw_target_rho_factor,
            "dhw_schedule_enabled": opts.mpc_dhw_schedule_enabled,
            "dhw_schedule_start_hour_local": opts.mpc_dhw_schedule_start_hour_local,
            "dhw_schedule_duration_hours": opts.mpc_dhw_schedule_duration_hours,
            "dhw_schedule_target_c": opts.mpc_dhw_schedule_target_c,
            "dhw_preheat_lead_steps": opts.mpc_dhw_preheat_lead_steps,
            "dhw_significant_tap_threshold_m3_per_h": opts.mpc_dhw_significant_tap_threshold_m3_per_h,
            "dhw_legionella_period_steps": runtime_defaults.dhw_legionella_period_steps,
            "dhw_legionella_duration_steps": runtime_defaults.dhw_legionella_duration_steps,
            "dhw_t_mains_c": runtime_defaults.dhw_t_mains_c,
            "dhw_t_amb_c": runtime_defaults.dhw_t_amb_c,
            "P_hp_max_elec": opts.mpc_P_hp_max_elec,
            "heat_pump_topology": opts.mpc_heat_pump_topology,
            "exclusive_heat_pump_mode": opts.mpc_exclusive_heat_pump_mode,
            "eta_carnot_ufh": opts.mpc_eta_carnot_ufh,
            "eta_carnot_dhw": opts.mpc_eta_carnot_dhw,
            "delta_T_cond": opts.mpc_delta_T_cond,
            "delta_T_evap": opts.mpc_delta_T_evap,
            "T_supply_min": runtime_defaults.T_supply_min,
            "T_ref_outdoor_curve": runtime_defaults.T_ref_outdoor_curve,
            "heating_curve_slope": runtime_defaults.heating_curve_slope,
            "cop_min": opts.mpc_cop_min,
            "cop_max": opts.mpc_cop_max,
        }
    )


# ---------------------------------------------------------------------------
# Helper: load and validate options
# ---------------------------------------------------------------------------


def _load_options(path: Path = _OPTIONS_PATH) -> AddonOptions:
    """Read and validate ``/data/options.json``.

    Raises
    ------
    FileNotFoundError
        When the options file is absent (not running inside the HA supervisor).
    pydantic.ValidationError
        When any required option is missing or physically invalid.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Options file not found at {path}.  "
            "This entry point must be run inside the HA supervisor container.  "
            "For local development use 'python -m home_optimizer' instead."
        )
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return AddonOptions.model_validate(raw)


# ---------------------------------------------------------------------------
# Helper: build backend from options
# ---------------------------------------------------------------------------


def _build_backend(opts: AddonOptions) -> HomeAssistantBackend:
    """Construct a :class:`HomeAssistantBackend` from validated addon options.

    In addon mode the base URL and token are resolved automatically from the
    environment variables ``HA_BASE_URL`` (defaults to
    ``http://supervisor/core``) and ``SUPERVISOR_TOKEN``.  No explicit
    values are needed.

    Args:
        opts: Validated addon options.

    Returns:
        Configured :class:`HomeAssistantBackend`.
    """
    return HomeAssistantBackend(
        room_temperature=HAEntityConfig(opts.sensor_room_temperature),
        outdoor_temperature=HAEntityConfig(opts.sensor_outdoor_temperature),
        hp_supply_temperature=HAEntityConfig(opts.sensor_hp_supply_temperature),
        hp_supply_target_temperature=HAEntityConfig(opts.sensor_hp_supply_target_temperature),
        hp_return_temperature=HAEntityConfig(opts.sensor_hp_return_temperature),
        hp_flow_lpm=HAEntityConfig(opts.sensor_hp_flow_lpm, scale=opts.sensor_hp_flow_scale),
        hp_electric_power=HAEntityConfig(
            opts.sensor_hp_electric_power, scale=opts.sensor_hp_electric_power_scale
        ),
        hp_mode_entity_id=opts.sensor_hp_mode,
        p1_net_power=HAEntityConfig(opts.sensor_p1_net_power, scale=opts.sensor_p1_net_power_scale),
        pv_output=HAEntityConfig(opts.sensor_pv_output, scale=opts.sensor_pv_output_scale),
        thermostat_setpoint=HAEntityConfig(opts.sensor_thermostat_setpoint),
        dhw_top_temperature=HAEntityConfig(opts.sensor_dhw_top_temperature),
        dhw_bottom_temperature=HAEntityConfig(opts.sensor_dhw_bottom_temperature),
        shutter_living_room=HAEntityConfig(opts.sensor_shutter_living_room),
        defrost_active_entity_id=opts.sensor_defrost_active,
        booster_heater_active_entity_id=opts.sensor_booster_heater_active,
        boiler_ambient_temperature=HAEntityConfig(opts.sensor_boiler_ambient_temperature),
        refrigerant_condensation_temperature=HAEntityConfig(
            opts.sensor_refrigerant_condensation_temperature
        ),
        refrigerant_liquid_line_temperature=HAEntityConfig(
            opts.sensor_refrigerant_liquid_line_temperature
        ),
        discharge_temperature=HAEntityConfig(opts.sensor_discharge_temperature),
        # Required energy counters — entity IDs are validated non-blank by AddonOptions.
        pv_total_energy=HAEntityConfig(opts.sensor_pv_total_kwh),
        hp_electric_total_energy=HAEntityConfig(opts.sensor_hp_electric_total_kwh),
        p1_import_total_energy=HAEntityConfig(opts.sensor_p1_import_total_kwh),
        p1_export_total_energy=HAEntityConfig(opts.sensor_p1_export_total_kwh),
        # base_url and token are resolved automatically from the environment.
    )


def _build_automatic_calibration_settings(opts: AddonOptions):
    """Build validated automatic-calibration scheduler settings from addon options."""
    from .calibration.models import AutomaticCalibrationSettings  # noqa: PLC0415

    return AutomaticCalibrationSettings(
        min_history_hours=opts.calibration_min_history_hours,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Load options, start the telemetry scheduler, and run Uvicorn.

    Lifecycle
    ---------
    1. Read and validate ``/data/options.json`` (fail-fast on bad config).
    2. Create ``Database`` and obtain a ``TelemetryRepository`` via it.
    3. Start the collector (APScheduler runs in a daemon thread — non-blocking).
    4. Register SIGTERM handler so ``docker stop`` triggers a clean flush.
    5. Block on ``uvicorn.run()`` until the process is stopped.
    6. ``finally`` block flushes the last telemetry bucket and closes the
       HTTP client pool.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )

    # ── 1. Load and validate options ───────────────────────────────────────
    log.info("Loading addon options from %s", _OPTIONS_PATH)
    try:
        opts = _load_options()
    except (FileNotFoundError, Exception) as exc:
        log.critical("Failed to load addon options: %s", exc)
        sys.exit(1)

    log.info(
        "Options loaded: db=%s  sample=%ds  flush=%ds  port=%d",
        opts.database_path,
        opts.sampling_interval_seconds,
        opts.flush_interval_seconds,
        opts.api_port,
    )

    # ── 2. Set up telemetry storage ────────────────────────────────────────
    # TelemetryRepository.from_path() creates the parent directory, derives the
    # SQLite URL, validates the path, and initialises the schema in one call.
    repository = TelemetryRepository.from_path(opts.database_path, model_dir=opts.models_path)
    log.info("Telemetry database ready at %s", opts.database_path)

    # ── 3. Build HA sensor backend ─────────────────────────────────────────
    ha_backend = _build_backend(opts)
    log.info("HomeAssistantBackend configured with %d entity IDs", 20)

    # Wrap with WeatherAugmentedBackend so t_mains_estimated_c is populated
    # via the SeasonalMainsModel (date-only function, no network I/O).
    mains_model = SeasonalMainsModel.for_netherlands()
    backend = WeatherAugmentedBackend(ha_backend, mains_model)

    # ── 3b. Fetch home location from zone.home ─────────────────────────────
    # The latitude/longitude are stored as zone entity *attributes*, not as
    # config options, so they are fetched automatically at startup.
    log.info("Fetching home location from zone.home …")
    try:
        latitude, longitude = ha_backend.fetch_zone_location("zone.home")
    except (ConnectionError, ValueError) as exc:
        log.critical("Could not fetch lat/lon from zone.home: %s", exc)
        sys.exit(1)
    log.info("Home location: lat=%.5f  lon=%.5f", latitude, longitude)

    # ── 3c. Build OpenMeteoClient ──────────────────────────────────────────
    # Convert compass bearing to Open-Meteo solar azimuth convention:
    #   azimuth_OM = compass_bearing − 180
    #   (0° compass/North → −180° OM; 180° compass/South → 0° OM)
    open_meteo_pv_azimuth: float = opts.pv_azimuth - 180.0
    weather_client = OpenMeteoClient(
        latitude=latitude,
        longitude=longitude,
        pv_tilt=opts.pv_tilt,
        pv_azimuth=open_meteo_pv_azimuth,
    )
    log.info(
        "OpenMeteoClient: lat=%.5f  lon=%.5f  pv_tilt=%.1f°  pv_azimuth_compass=%.1f°  "
        "pv_azimuth_om=%.1f°",
        latitude,
        longitude,
        opts.pv_tilt,
        opts.pv_azimuth,
        open_meteo_pv_azimuth,
    )

    forecast_persister = ForecastPersister(
        weather_client=weather_client,
        repository=repository,
    )

    # ── 4. Start telemetry collector ───────────────────────────────────────
    # Build the price model early so the collector can stamp every telemetry
    # bucket with the applicable electricity tariff (§14.2).
    price_cfg = PriceConfig(
        mode=PriceMode(opts.price_mode),
        flat_rate_eur_per_kwh=opts.price_flat_rate,
        high_rate_eur_per_kwh=opts.price_high_rate,
        low_rate_eur_per_kwh=opts.price_low_rate,
        feed_in_rate_eur_per_kwh=opts.price_feed_in_rate,
        low_tariff_hours=opts.price_low_tariff_hours,
        nordpool_country_code=opts.nordpool_country_code,
        nordpool_vat_factor=opts.nordpool_vat_factor,
    )
    price_model = build_price_model(price_cfg)
    log.info("Price model ready (mode=%s)", opts.price_mode)

    collector_settings = TelemetryCollectorSettings(
        database_url=repository.url,
        sampling_interval_seconds=opts.sampling_interval_seconds,
        flush_interval_seconds=opts.flush_interval_seconds,
    )
    collector = BufferedTelemetryCollector(
        backend=backend,
        repository=repository,
        settings=collector_settings,
        price_model=price_model,
    )
    collector.start()
    log.info(
        "Telemetry collector started (sample every %ds, flush every %ds)",
        opts.sampling_interval_seconds,
        opts.flush_interval_seconds,
    )

    # ── 4b. Start forecast persister on the shared scheduler ──────────────
    # persist_once() is called immediately (run_immediately=True) so forecast
    # data is available from the first second rather than waiting one full hour.
    forecast_persister.start(collector.scheduler, run_immediately=True)
    log.info("ForecastPersister started (hourly Open-Meteo updates)")

    # ── 4c. Build the shared baseline MPC/calibration input ───────────────
    # Seed the request with the current Home Assistant sensor snapshot so the
    # simulator defaults mirror the real installation. Forecast arrays stay out
    # of the base request; they are injected from the persisted repository at
    # solve time.
    mpc_base_input = _build_runtime_base_request(opts)
    try:
        mpc_base_input = OptimizerRuntime.build_scheduled_input(
            base_input=mpc_base_input,
            backend=backend,
            repository=None,
        )
        log.info("MPC base request seeded from live Home Assistant sensors.")
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not seed MPC base request from live sensors; using config defaults: %s", exc)
    api_service.set_base_request(mpc_base_input)

    # ── 4d. Start nightly persisted ML forecast-model training (optional) ──
    if opts.forecast_training_enabled:
        forecast_service = ForecastService()

        def _run_forecast_training_job() -> None:
            results = forecast_service.train_and_persist_models(
                repository=repository,
                base_request_data=mpc_base_input.model_dump(mode="python"),
            )
            successful_models = 0
            for field_name, model_result in results.items():
                if model_result is None:
                    log.info("Forecast-model training skipped — insufficient history for %s.", field_name)
                    continue
                successful_models += 1
                trained_at_utc = getattr(model_result, "trained_at_utc", None)
                sample_count = getattr(model_result, "sample_count", None)
                log.info(
                    "Forecast model stored: %s trained at %s with %d samples.",
                    field_name,
                    trained_at_utc.isoformat() if trained_at_utc is not None else "unknown",
                    sample_count if sample_count is not None else -1,
                )
            if successful_models == 0:
                log.info("Forecast-model training produced no persisted artifacts this run.")
                return

        try:
            _run_forecast_training_job()
        except Exception as exc:  # noqa: BLE001
            log.exception("Initial forecast-model training failed: %s", exc)

        collector.scheduler.add_job(
            _run_forecast_training_job,
            trigger="cron",
            hour=opts.forecast_training_hour_utc,
            minute=opts.forecast_training_minute_utc,
            id="forecast_model_training_periodic",
            replace_existing=True,
            misfire_grace_time=FORECAST_TRAINING_MISFIRE_GRACE_SECONDS,
        )
        log.info(
            "Forecast-model training job scheduled daily at %02d:%02d UTC.",
            opts.forecast_training_hour_utc,
            opts.forecast_training_minute_utc,
        )
    else:
        log.info("Persisted ML forecast-model training disabled (forecast_training_enabled=false).")

    # ── 4e. Start periodic automatic calibration (optional) ───────────────
    if opts.calibration_enabled:
        from .calibration.service import run_and_persist_automatic_calibration  # noqa: PLC0415

        calibration_settings = _build_automatic_calibration_settings(opts)

        def _run_calibration_job() -> None:
            payload = run_and_persist_automatic_calibration(
                repository,
                base_request=mpc_base_input,
                settings=calibration_settings,
            )
            if payload is None:
                log.info(
                    "Automatic calibration skipped — waiting for %.1f h of telemetry history.",
                    calibration_settings.min_history_hours,
                )
                return
            log.info(
                "Automatic calibration snapshot stored at %s with %d effective overrides.",
                payload.generated_at_utc.isoformat(),
                len(payload.effective_parameters.as_run_request_updates()),
            )

        try:
            _run_calibration_job()
        except Exception as exc:  # noqa: BLE001
            log.exception("Initial automatic calibration run failed: %s", exc)

        collector.scheduler.add_job(
            _run_calibration_job,
            trigger="interval",
            seconds=opts.calibration_interval_seconds,
            id="calibration_periodic",
            replace_existing=True,
            misfire_grace_time=max(1, opts.calibration_interval_seconds // 2),
        )
        log.info(
            "Automatic calibration job scheduled: every %d s (%d min)",
            opts.calibration_interval_seconds,
            opts.calibration_interval_seconds // 60,
        )
    else:
        log.info("Automatic calibration scheduling disabled (calibration_enabled=false).")

    # ── 4f. Start periodic MPC (optional) ─────────────────────────────────
    if opts.mpc_enabled:
        optimizer = Optimizer()
        optimizer.schedule_periodic(
            base_input=mpc_base_input,
            backend=backend,
            repository=repository,
            scheduler=collector.scheduler,
            interval_seconds=opts.mpc_interval_seconds,
            run_immediately=True,
        )
        log.info(
            "MPC periodic runner started (every %d s / %d min)",
            opts.mpc_interval_seconds,
            opts.mpc_interval_seconds // 60,
        )
    else:
        log.info("MPC periodic scheduling disabled (mpc_enabled=false).")

    # ── 5. SIGTERM handler — graceful shutdown from `docker stop` ──────────
    # Only stop the APScheduler here so no new jobs are accepted.
    # Do NOT call sys.exit() or backend.close() from the signal handler:
    # that would race with an in-flight APScheduler thread that still holds
    # the httpx connection, causing [Errno 9] Bad file descriptor.
    # Uvicorn installs its own SIGTERM handler that cleanly stops the server;
    # the `finally` block below performs the definitive flush + close once
    # Uvicorn has exited.
    def _handle_sigterm(signum: int, frame: object) -> None:  # noqa: ANN001
        log.info(
            "Signal %d received — stopping scheduler (Uvicorn handles its own shutdown).",
            signum,
        )
        try:
            collector.shutdown(flush=False)
        except Exception:  # noqa: BLE001
            pass  # already stopped — safe to ignore

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # ── 6. Run FastAPI via Uvicorn (blocking) ──────────────────────────────
    # Export DATABASE_URL so the /api/forecast/latest endpoint finds the same
    # database that the addon uses for telemetry and forecast persistence.
    repository.export_to_env()

    # HA Ingress proxies requests via /api/hassio_ingress/<token>/.
    # Setting root_path tells FastAPI/Uvicorn the effective mount prefix so
    # that generated OpenAPI docs and redirects use the correct base path.
    # The supervisor injects the ingress path via the X-Ingress-Path header;
    # we read it at start-up so it is available before any request arrives.
    ingress_path: str = os.environ.get("INGRESS_PATH", "")
    log.info("Starting Uvicorn on %s:%d  root_path=%r", _BIND_HOST, opts.api_port, ingress_path)
    try:
        uvicorn.run(
            app,
            host=_BIND_HOST,
            port=opts.api_port,
            log_level="info",
            root_path=ingress_path,
            # Disable Uvicorn's default signal handlers so our SIGTERM handler
            # above controls the shutdown sequence.
            # (install_signal_handlers is False when not in the main thread;
            # explicitly set here for clarity.)
            loop="asyncio",
        )
    finally:
        log.info("Uvicorn stopped — performing final telemetry flush.")
        try:
            collector.shutdown(flush=True)
        except Exception:  # noqa: BLE001
            pass  # already stopped by SIGTERM handler — safe to ignore
        backend.close()
        log.info("Home Optimizer addon stopped cleanly.")


if __name__ == "__main__":
    main()
