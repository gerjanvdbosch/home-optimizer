"""Local development runner for Home Optimizer.

Combines the APScheduler-based :class:`~home_optimizer.telemetry.ForecastPersister`
and the FastAPI/Uvicorn web server into a single process so the full stack can be
tested locally without a Home Assistant installation.

What this runner does
---------------------
1. Parses CLI arguments (latitude, longitude, database path, port, …).
2. Sets the ``DATABASE_URL`` environment variable so the FastAPI
   ``/api/forecast/latest`` endpoint finds the same database.
3. Creates the SQLite database and schema (idempotent).
4. Builds an :class:`~home_optimizer.sensors.OpenMeteoClient` for the site.
5. Runs :meth:`~home_optimizer.telemetry.ForecastPersister.persist_once`
   immediately so the first forecast is available right away.
6. Schedules hourly forecast refreshes via APScheduler.
7. Starts Uvicorn (blocking) on the configured host / port.
8. On Ctrl-C / SIGTERM: stops the scheduler cleanly before exiting.

Usage
-----
Direct::

    python -m home_optimizer.local_runner

With options::

    python -m home_optimizer.local_runner \\
        --lat 52.37 --lon 4.90 \\
        --database ./dev_data/forecast.db \\
        --port 8000 \\
        --horizon 48 \\
        --pv-tilt 35 --pv-azimuth 0

Run the MPC every hour using ``sensors.json`` for live initial conditions::

    python -m home_optimizer.local_runner \\
        --mpc-interval 3600 \\
        --sensors-json ./sensors.json

Environment variable override (alternative to CLI)::

    DATABASE_URL=sqlite:///my_local.db python -m home_optimizer.local_runner

The ``--database`` CLI argument always takes precedence over ``DATABASE_URL``.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

from .api import app
from .calibration import AutomaticCalibrationSettings, run_and_persist_automatic_calibration
from .forecasting import ForecastService
from .application.optimizer import Optimizer
from .pricing import PriceConfig, PriceMode, build_price_model
from .sensors.local_backend import LocalBackend
from .sensors.open_meteo import OpenMeteoClient
from .telemetry import (
    DATABASE_URL_DEFAULT,
    BufferedTelemetryCollector,
    ForecastPersister,
    TelemetryCollectorSettings,
    TelemetryRepository,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default host to bind (localhost-only for local dev — not 0.0.0.0).
_DEFAULT_HOST: str = "127.0.0.1"
#: Default Uvicorn port.
_DEFAULT_PORT: int = 8000
#: Default forecast horizon [h].
_DEFAULT_HORIZON_HOURS: int = 48
#: Default Open-Meteo window surface tilt [°] — south-facing vertical glass.
_DEFAULT_WINDOW_TILT: float = 90.0
#: Default window azimuth [°] — 0 = South (Open-Meteo solar convention).
_DEFAULT_WINDOW_AZIMUTH: float = 0.0
# Default pv tilt
_DEFAULT_PV_TILT: float = 50.0
#: Default site latitude [°N] — Amsterdam.
_DEFAULT_LATITUDE: float = 52.37
#: Default site longitude [°E] — Amsterdam.
_DEFAULT_LONGITUDE: float = 4.90
_DEFAULT_CALIBRATION_INTERVAL_SECONDS: int = 6 * 3600
_DEFAULT_CALIBRATION_MIN_HISTORY_HOURS: float = 24.0
_DEFAULT_FORECAST_TRAINING_HOUR_UTC: int = 2
_DEFAULT_FORECAST_TRAINING_MINUTE_UTC: int = 0
_FORECAST_TRAINING_MISFIRE_GRACE_SECONDS: int = 3600

log = logging.getLogger("home_optimizer.local_runner")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the local runner.

    All arguments are optional; sensible defaults are provided for every
    parameter so the runner starts with zero configuration.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m home_optimizer.local_runner",
        description=(
            "Local development runner — starts ForecastPersister + FastAPI "
            "in a single process without Home Assistant."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Location ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--lat",
        type=float,
        default=_DEFAULT_LATITUDE,
        metavar="DEGREES",
        help="Site latitude [°N].",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=_DEFAULT_LONGITUDE,
        metavar="DEGREES",
        help="Site longitude [°E].",
    )

    # ── Database ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the SQLite database file, e.g. './database.sqlite3'. "
            "Overrides the DATABASE_URL environment variable. "
            f"Default: uses DATABASE_URL env var or '{DATABASE_URL_DEFAULT}'."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default='./models',
        metavar="PATH",
        help=(
            "Directory where persisted ML forecast-model artifacts are stored. "
            "When omitted artifacts are stored next to the SQLite database file."
        ),
    )

    # ── Server ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--host",
        type=str,
        default=_DEFAULT_HOST,
        help="Uvicorn bind host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_DEFAULT_PORT,
        help="Uvicorn bind port.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable Uvicorn auto-reload (useful during template development).",
    )

    # ── Forecast ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--horizon",
        type=int,
        default=_DEFAULT_HORIZON_HOURS,
        metavar="HOURS",
        help="Open-Meteo forecast horizon [h] to fetch and persist.",
    )
    parser.add_argument(
        "--window-tilt",
        type=float,
        default=_DEFAULT_WINDOW_TILT,
        metavar="DEGREES",
        help="South-facing window tilt [°] for solar-gain GTI (90 = vertical).",
    )
    parser.add_argument(
        "--window-azimuth",
        type=float,
        default=_DEFAULT_WINDOW_AZIMUTH,
        metavar="DEGREES",
        help="Window surface azimuth [°] (0 = South, Open-Meteo convention).",
    )
    parser.add_argument(
        "--pv-tilt",
        type=float,
        default=_DEFAULT_PV_TILT,
        metavar="DEGREES",
        help="PV panel tilt [°]. Omit to disable PV GTI forecast.",
    )
    parser.add_argument(
        "--pv-azimuth",
        type=float,
        default=_DEFAULT_WINDOW_AZIMUTH,
        metavar="DEGREES",
        help="PV panel azimuth [] (0 = South).",
    )

    # ── MPC scheduling ────────────────────────────────────────────────
    parser.add_argument(
        "--mpc-interval",
        type=int,
        default=30,
        metavar="SECONDS",
        help=(
            "How often the MPC runs [s].  0 (default) disables scheduling — "
            "the MPC is then only available via POST /api/simulate.  "
            "Example: --mpc-interval 3600 runs the MPC every hour."
        ),
    )
    parser.add_argument(
        "--mpc-t-ref",
        type=float,
        default=20.5,
        metavar="CELSIUS",
        help="MPC comfort setpoint T_ref [°C].",
    )
    parser.add_argument(
        "--mpc-t-min",
        type=float,
        default=19.0,
        metavar="CELSIUS",
        help="MPC minimum comfort temperature [°C].",
    )
    parser.add_argument(
        "--mpc-t-max",
        type=float,
        default=22.5,
        metavar="CELSIUS",
        help="MPC maximum comfort temperature [°C].",
    )
    parser.add_argument(
        "--mpc-t-out",
        type=float,
        default=8.0,
        metavar="CELSIUS",
        help="Fixed outdoor temperature used as MPC disturbance when no forecast [°C].",
    )

    # ── Sensors JSON (MPC live-readings source) ───────────────────────────
    parser.add_argument(
        "--sensors-json",
        type=str,
        default="sensors.json",
        metavar="PATH",
        help=(
            "Path to a sensors JSON file (e.g. './sensors.json').  "
            "When provided the MPC runner reads live initial conditions "
            "(T_r, T_out, DHW temperatures, …) from this file at every "
            "solve interval instead of using the fixed CLI defaults.  "
            "The file is re-read on every MPC call so values can be "
            "updated while the runner is active.  "
            "Must contain all fields required by LocalBackend.from_json_file()."
        ),
    )
    parser.add_argument(
        "--calibration-interval",
        type=int,
        default=_DEFAULT_CALIBRATION_INTERVAL_SECONDS,
        metavar="SECONDS",
        help=(
            "How often automatic calibration runs [s].  0 disables automatic calibration. "
            "Calibration uses persisted telemetry from the local collector / existing database."
        ),
    )
    parser.add_argument(
        "--calibration-min-history-hours",
        type=float,
        default=_DEFAULT_CALIBRATION_MIN_HISTORY_HOURS,
        metavar="HOURS",
        help="Minimum persisted telemetry history required before automatic calibration [h].",
    )
    parser.add_argument(
        "--forecast-training-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable persisted ML forecast-model training (currently shutter_forecast).",
    )
    parser.add_argument(
        "--forecast-training-hour-utc",
        type=int,
        default=_DEFAULT_FORECAST_TRAINING_HOUR_UTC,
        metavar="HOUR",
        help="UTC hour (0–23) for the nightly ML forecast-model training job.",
    )
    parser.add_argument(
        "--forecast-training-minute-utc",
        type=int,
        default=_DEFAULT_FORECAST_TRAINING_MINUTE_UTC,
        metavar="MINUTE",
        help="UTC minute (0–59) for the nightly ML forecast-model training job.",
    )

    # ── Electricity price model ───────────────────────────────────────────
    parser.add_argument(
        "--price-mode",
        type=str,
        default="flat",
        choices=["flat", "dual", "nordpool"],
        help="Electricity price mode: flat | dual | nordpool.",
    )
    parser.add_argument(
        "--price-flat-rate",
        type=float,
        default=0.25,
        metavar="EUR_PER_KWH",
        help="Flat import tariff [€/kWh] (used when --price-mode=flat).",
    )
    parser.add_argument(
        "--price-high-rate",
        type=float,
        default=0.36,
        metavar="EUR_PER_KWH",
        help="Peak (high) import tariff [€/kWh] (used when --price-mode=dual).",
    )
    parser.add_argument(
        "--price-low-rate",
        type=float,
        default=0.21,
        metavar="EUR_PER_KWH",
        help="Off-peak (low) import tariff [€/kWh] (used when --price-mode=dual).",
    )
    parser.add_argument(
        "--price-feed-in-rate",
        type=float,
        default=0.09,
        metavar="EUR_PER_KWH",
        help="Feed-in / terugleververgoeding rate [€/kWh] (used when --price-mode=dual).",
    )
    parser.add_argument(
        "--nordpool-country",
        type=str,
        default="NL",
        metavar="CODE",
        help="Nordpool bidding-zone code, e.g. 'NL' or 'DE-LU' (used when --price-mode=nordpool).",
    )
    parser.add_argument(
        "--nordpool-vat",
        type=float,
        default=1.21,
        metavar="FACTOR",
        help="VAT + surcharge multiplier for Nordpool raw price (e.g. 1.21 = 21%% BTW).",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Start the local development server.

    Lifecycle
    ---------
    1. Parse CLI args.
    2. Resolve and export ``DATABASE_URL`` (CLI ``--database`` > env var > default).
    3. Create database schema.
    4. Fetch first forecast immediately via :meth:`ForecastPersister.persist_once`.
    5. Schedule hourly forecast refresh.
    6. Block on ``uvicorn.run()``.
    7. On exit: stop the APScheduler cleanly.

    Args:
        argv: Optional argument list for programmatic invocation / testing.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )

    args = _parse_args(argv)
    if not 0 <= args.forecast_training_hour_utc <= 23:
        raise ValueError("--forecast-training-hour-utc must be in [0, 23].")
    if not 0 <= args.forecast_training_minute_utc <= 59:
        raise ValueError("--forecast-training-minute-utc must be in [0, 59].")

    # ── 1. Resolve repository — class handles path, mkdir, URL, and schema ──
    # Priority: --database CLI arg > DATABASE_URL env var > default.
    if args.database is not None:
        repository = TelemetryRepository.from_path(args.database, model_dir=args.models_dir)
        log.info("Using database from --database: %s  models_dir=%s", args.database, args.models_dir)
    else:
        repository = TelemetryRepository.from_env(model_dir=args.models_dir)
        log.info("Using database from env/default: %s  models_dir=%s", repository.url, args.models_dir)

    # Export so the FastAPI /api/forecast/latest endpoint finds the same DB.
    repository.export_to_env()

    # ── 2. Repository is already initialised by the constructor helpers ────
    log.info("Database schema ready.")

    # ── 3. Build Open-Meteo client ─────────────────────────────────────────
    weather_client = OpenMeteoClient(
        latitude=args.lat,
        longitude=args.lon,
        tilt=args.window_tilt,
        azimuth=args.window_azimuth,
        pv_tilt=args.pv_tilt,
        pv_azimuth=args.pv_azimuth,
    )
    log.info(
        "OpenMeteoClient: lat=%.4f lon=%.4f  window=%.0f°/%.0f°  pv=%s",
        args.lat,
        args.lon,
        args.window_tilt,
        args.window_azimuth,
        f"{args.pv_tilt}°/{args.pv_azimuth}°" if args.pv_tilt is not None else "disabled",
    )

    # ── 4. Shared scheduler + optional local telemetry collection ──────────
    scheduler = BackgroundScheduler(timezone="UTC")
    telemetry_collector: BufferedTelemetryCollector | None = None
    mpc_sensor_backend: LocalBackend | None = None

    # Build price model from CLI arguments so both telemetry persistence and MPC
    # cost evaluation use the same tariff assumptions (§14.2).
    price_cfg = PriceConfig(
        mode=PriceMode(args.price_mode),
        flat_rate_eur_per_kwh=args.price_flat_rate,
        high_rate_eur_per_kwh=args.price_high_rate,
        low_rate_eur_per_kwh=args.price_low_rate,
        feed_in_rate_eur_per_kwh=args.price_feed_in_rate,
        nordpool_country_code=args.nordpool_country,
        nordpool_vat_factor=args.nordpool_vat,
    )
    price_model = build_price_model(price_cfg)
    log.info("Price model ready (mode=%s)", args.price_mode)

    sensors_path: Path | None = Path(args.sensors_json).resolve() if args.sensors_json is not None else None
    sensors_file_exists = sensors_path is not None and sensors_path.exists()
    if sensors_file_exists and sensors_path is not None:
        mpc_sensor_backend = LocalBackend.from_json_file(sensors_path)
        telemetry_settings = TelemetryCollectorSettings(database_url=repository.url)
        telemetry_collector = BufferedTelemetryCollector(
            backend=LocalBackend.from_json_file(sensors_path),
            repository=repository,
            settings=telemetry_settings,
            scheduler=scheduler,
            price_model=price_model,
        )
        telemetry_collector.start()
        log.info(
            "Local telemetry collector started from %s (sample=%ds, flush=%ds).",
            sensors_path,
            telemetry_settings.sampling_interval_seconds,
            telemetry_settings.flush_interval_seconds,
        )
    else:
        scheduler.start()
        if sensors_path is not None:
            log.info(
                "No sensors-json file at %s — local telemetry collection disabled; existing database history remains available.",
                sensors_path,
            )

    # ── 5. ForecastPersister — fetch immediately + schedule hourly ─────────
    persister = ForecastPersister(
        weather_client=weather_client,
        repository=repository,
        horizon_hours=args.horizon,
    )

    log.info("Fetching initial forecast from Open-Meteo (horizon=%dh)…", args.horizon)
    try:
        inserted = persister.persist_once()
        log.info("Initial forecast stored: %d new steps inserted.", inserted)
    except Exception as exc:  # noqa: BLE001
        # Log but do not abort: the API will return 404 until the next retry.
        log.warning("Initial forecast fetch failed: %s", exc)

    persister.start(scheduler, run_immediately=False)  # already ran above
    log.info("Hourly forecast refresh scheduled.")

    # ── 5b. Build shared baseline RunRequest for calibration + MPC ─────────
    from .application.optimizer import RunRequest  # noqa: PLC0415

    t_out_init = args.mpc_t_out
    if mpc_sensor_backend is not None:
        try:
            readings = mpc_sensor_backend.read_all()
            t_out_init = readings.outdoor_temperature_c
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Could not read sensors.json at startup (will retry later): %s",
                exc,
            )

    _defaults = RunRequest.model_validate({})
    mpc_base_input = _defaults.model_copy(
        update={
            "outdoor_temperature_c": t_out_init,
            "T_ref": args.mpc_t_ref,
            "T_min": args.mpc_t_min,
            "T_max": args.mpc_t_max,
            "price_config": price_cfg,
        }
    )

    # ── 5c. Nightly ML forecast-model training (currently shutter model) ───
    if args.forecast_training_enabled:
        forecast_service = ForecastService()

        def _run_forecast_training_job() -> None:
            results = forecast_service.train_and_persist_models(repository=repository)
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

        scheduler.add_job(
            _run_forecast_training_job,
            trigger="cron",
            hour=args.forecast_training_hour_utc,
            minute=args.forecast_training_minute_utc,
            id="forecast_model_training_periodic",
            replace_existing=True,
            misfire_grace_time=_FORECAST_TRAINING_MISFIRE_GRACE_SECONDS,
        )
        log.info(
            "Forecast-model training job scheduled daily at %02d:%02d UTC.",
            args.forecast_training_hour_utc,
            args.forecast_training_minute_utc,
        )
    else:
        log.info("Persisted ML forecast-model training disabled (--no-forecast-training-enabled).")

    # ── 5d. Automatic calibration on persisted telemetry ───────────────────
    if args.calibration_interval > 0:
        calibration_settings = AutomaticCalibrationSettings(
            min_history_hours=args.calibration_min_history_hours,
        )

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

        scheduler.add_job(
            _run_calibration_job,
            trigger="interval",
            seconds=args.calibration_interval,
            id="calibration_periodic",
            replace_existing=True,
            misfire_grace_time=max(1, args.calibration_interval // 2),
        )
        log.info(
            "Automatic calibration job scheduled: every %d s (%d min)",
            args.calibration_interval,
            args.calibration_interval // 60,
        )
    else:
        log.info("Automatic calibration disabled (--calibration-interval=0).")

    # ── 5e. Optional: schedule periodic MPC ───────────────────────────────
    # When --sensors-json is provided, the MPC reads live initial conditions
    # from the JSON file at every solve interval (LocalBackend re-reads the
    # file on each call, so values updated on disk are picked up immediately).
    # Without --sensors-json the MPC uses the fixed CLI defaults as initial
    # conditions — useful for smoke-testing without real sensor data.
    if args.mpc_interval > 0:
        # Build the sensor backend when a sensors.json path is supplied.
        if sensors_path is not None:
            if not sensors_path.exists():
                log.critical("sensors-json file not found: %s — aborting MPC setup.", sensors_path)
                sys.exit(1)
            local_backend = mpc_sensor_backend or LocalBackend.from_json_file(sensors_path)
            log.info("MPC sensor backend: LocalBackend reading %s", sensors_path)
        else:
            local_backend = None
            log.info(
                "MPC sensor backend: none — using CLI defaults " "(T_out=%.1f C, T_ref=%.1f C).",
                mpc_base_input.outdoor_temperature_c,
                args.mpc_t_ref,
            )
        optimizer = Optimizer()
        optimizer.schedule_periodic(
            base_input=mpc_base_input,
            backend=local_backend,
            repository=repository,
            scheduler=scheduler,
            interval_seconds=args.mpc_interval,
            run_immediately=True,
        )
        log.info(
            "MPC periodic runner started (every %d s / %d min, sensors=%s)",
            args.mpc_interval,
            args.mpc_interval // 60,
            args.sensors_json or "none (CLI defaults)",
        )
    else:
        log.info(
            "MPC scheduling disabled (--mpc-interval not set).  "
            "Use POST /api/simulate for on-demand optimisation."
        )

    # ── 5. SIGTERM handler ─────────────────────────────────────────────────
    # Only stop the APScheduler — Uvicorn handles its own shutdown on SIGTERM.
    # Do NOT call sys.exit() here: that raises SystemExit inside Uvicorn's
    # capture_signals context manager, which then re-raises the signal and
    # triggers a SchedulerNotRunningError on the second invocation.
    def _shutdown(signum: int, frame: object) -> None:  # noqa: ANN001
        log.info("Signal %d received — stopping scheduler.", signum)
        try:
            scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass  # already stopped — safe to ignore

    signal.signal(signal.SIGTERM, _shutdown)

    # ── 6. Start Uvicorn (blocking) ────────────────────────────────────────
    log.info(
        "Starting Uvicorn on http://%s:%d  (reload=%s)",
        args.host,
        args.port,
        args.reload,
    )
    try:
        uvicorn.run(
            # Pass the import string when reload=True (required by Uvicorn's
            # reloader which needs to re-import the module in a subprocess).
            "home_optimizer.api:app" if args.reload else app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    finally:
        log.info("Uvicorn stopped — shutting down scheduler.")
        try:
            if telemetry_collector is not None:
                telemetry_collector.shutdown(flush=True, wait=False)
            elif scheduler.running:
                scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass  # already stopped by SIGTERM handler — safe to ignore
        if mpc_sensor_backend is not None:
            mpc_sensor_backend.close()


if __name__ == "__main__":
    main()
