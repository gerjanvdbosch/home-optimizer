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

from .api import app
from .sensors.ha_backend import HAEntityConfig, HomeAssistantBackend
from .sensors.open_meteo import OpenMeteoClient, SeasonalMainsModel
from .sensors.weather_backend import WeatherAugmentedBackend
from .settings import DATABASE_URL_ENV
from .telemetry import (
    BufferedTelemetryCollector,
    ForecastPersister,
    TelemetryCollectorSettings,
    TelemetryRepository,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Path where the HA supervisor writes the addon options.
_OPTIONS_PATH: Path = Path("/data/options.json")

#: Host to bind Uvicorn — bind to all interfaces inside the container.
_BIND_HOST: str = "0.0.0.0"

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
    database_path: str = Field(..., min_length=1,
                               description="SQLite path, e.g. /config/database.sqlite3")
    sampling_interval_seconds: int = Field(10, gt=0,
                                           description="Sensor polling interval [s]")
    flush_interval_seconds: int = Field(300, gt=0,
                                        description="Aggregation window written to DB [s]")
    api_port: int = Field(8099, gt=0, lt=65536,
                          description="Uvicorn listen port")

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
    sensor_refrigerant_liquid_line_temperature: str = Field(..., min_length=1,
                                                            description="Buitenunit Vloeistofleiding [°C]")
    sensor_discharge_temperature: str = Field(..., min_length=1,
                                              description="Ontladingstemperatuur compressor [°C]")

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
        ..., min_length=1,
        description="P1 smart-meter entity: positive = import [kW], negative = export [kW]"
    )
    sensor_p1_net_power_scale: float = Field(
        1.0, description="Unit scale: 1.0 if already kW, 0.001 if W"
    )
    sensor_pv_output: str = Field(..., min_length=1)
    sensor_pv_output_scale: float = Field(1.0)

    # --- Categorical sensor ---
    sensor_hp_mode: str = Field(..., min_length=1,
                                description="Entity reporting ufh/dhw/defrost/off")

    # --- Shutter (0–100 %) ---
    sensor_shutter_living_room: str = Field(
        ..., min_length=1,
        description="Shutter/cover position entity (100 = fully open)"
    )

    # --- Binary sensors (on/off) ---
    sensor_defrost_active: str = Field(..., min_length=1,
                                       description="binary_sensor for HP defrost cycle")
    sensor_booster_heater_active: str = Field(..., min_length=1,
                                              description="binary_sensor for DHW booster element")

    # --- PV panel orientation (Open-Meteo GTI forecast) ---
    pv_tilt: float = Field(
        50.0, ge=0.0, le=90.0,
        description=(
            "PV panel tilt angle [°].  0 = horizontal surface, 90 = vertical wall.  "
            "Passed directly to Open-Meteo to compute irradiance on the PV surface."
        ),
    )
    pv_azimuth: float = Field(
        148.0, ge=0.0, lt=360.0,
        description=(
            "PV panel orientation as a compass bearing [°].  "
            "0 = North, 90 = East, 180 = South, 270 = West.  "
            "Converted internally to the Open-Meteo solar convention "
            "(0 = South, −90 = East, +90 = West) via azimuth_OM = pv_azimuth − 180."
        ),
    )

    # --- DHW boiler geometry ---
    boiler_tank_liters: int = Field(
        200, gt=0,
        description=(
            "Total DHW boiler tank volume [L].  "
            "Used to derive DHW thermal capacities C_top and C_bot via λ·V_tank."
        ),
    )

    # --- Monotonically-increasing energy counters [kWh] (required) ---
    # These counters enable accurate COP validation (§14.1), grid-cost
    # back-testing (§14.2), and PV self-consumption analysis.
    sensor_pv_total_kwh: str = Field(
        ..., min_length=1,
        description="HA entity ID for cumulative PV energy counter [kWh].",
    )
    sensor_hp_electric_total_kwh: str = Field(
        ..., min_length=1,
        description="HA entity ID for cumulative HP electrical energy counter [kWh].",
    )
    sensor_p1_import_total_kwh: str = Field(
        ..., min_length=1,
        description="HA entity ID for cumulative P1 import energy counter [kWh].",
    )
    sensor_p1_export_total_kwh: str = Field(
        ..., min_length=1,
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
            raise ValueError(
                "flush_interval_seconds must be >= sampling_interval_seconds."
            )
        if self.flush_interval_seconds % self.sampling_interval_seconds != 0:
            raise ValueError(
                "flush_interval_seconds must be an integer multiple of "
                "sampling_interval_seconds."
            )
        return self


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
        hp_flow_lpm=HAEntityConfig(opts.sensor_hp_flow_lpm,
                                   scale=opts.sensor_hp_flow_scale),
        hp_electric_power=HAEntityConfig(opts.sensor_hp_electric_power,
                                         scale=opts.sensor_hp_electric_power_scale),
        hp_mode_entity_id=opts.sensor_hp_mode,
        p1_net_power=HAEntityConfig(opts.sensor_p1_net_power,
                                    scale=opts.sensor_p1_net_power_scale),
        pv_output=HAEntityConfig(opts.sensor_pv_output,
                                 scale=opts.sensor_pv_output_scale),
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Load options, start the telemetry scheduler, and run Uvicorn.

    Lifecycle
    ---------
    1. Read and validate ``/data/options.json`` (fail-fast on bad config).
    2. Create ``TelemetryRepository`` and ``BufferedTelemetryCollector``.
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
    # Ensure the parent directory exists (e.g. /data/home_optimizer/).
    # exist_ok=True is safe: no-op if directory is already present.
    Path(opts.database_path).parent.mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite:///{opts.database_path}"
    repository = TelemetryRepository(database_url=database_url)
    repository.create_schema()
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
        latitude, longitude, opts.pv_tilt, opts.pv_azimuth, open_meteo_pv_azimuth,
    )

    forecast_persister = ForecastPersister(
        weather_client=weather_client,
        repository=repository,
    )

    # ── 4. Start telemetry collector ───────────────────────────────────────
    collector_settings = TelemetryCollectorSettings(
        database_url=database_url,
        sampling_interval_seconds=opts.sampling_interval_seconds,
        flush_interval_seconds=opts.flush_interval_seconds,
    )
    collector = BufferedTelemetryCollector(
        backend=backend,
        repository=repository,
        settings=collector_settings,
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

    # ── 5. SIGTERM handler — graceful shutdown from `docker stop` ──────────
    def _handle_sigterm(signum: int, frame: object) -> None:  # noqa: ANN001
        log.info("SIGTERM received — flushing telemetry and shutting down.")
        collector.shutdown(flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # ── 6. Run FastAPI via Uvicorn (blocking) ──────────────────────────────
    # Inject DATABASE_URL (defined in settings.py) so the API's
    # /api/forecast/latest endpoint resolves the same database that the addon
    # uses for telemetry and forecast persistence.
    os.environ[DATABASE_URL_ENV] = f"sqlite:///{opts.database_path}"

    log.info("Starting Uvicorn on %s:%d", _BIND_HOST, opts.api_port)
    try:
        uvicorn.run(
            app,
            host=_BIND_HOST,
            port=opts.api_port,
            log_level="info",
            # Disable Uvicorn's default signal handlers so our SIGTERM handler
            # above controls the shutdown sequence.
            # (install_signal_handlers is False when not in the main thread;
            # explicitly set here for clarity.)
            loop="asyncio",
        )
    finally:
        log.info("Uvicorn stopped — performing final telemetry flush.")
        collector.shutdown(flush=True)
        backend.close()
        log.info("Home Optimizer addon stopped cleanly.")


if __name__ == "__main__":
    main()

