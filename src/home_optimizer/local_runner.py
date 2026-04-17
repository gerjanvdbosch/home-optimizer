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
from .database import DATABASE_URL_DEFAULT, Database
from .optimizer import Optimizer
from .sensors.open_meteo import OpenMeteoClient
from .telemetry import ForecastPersister

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
#: Default site latitude [°N] — Amsterdam.
_DEFAULT_LATITUDE: float = 52.37
#: Default site longitude [°E] — Amsterdam.
_DEFAULT_LONGITUDE: float = 4.90

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
            "Path to the SQLite database file, e.g. './dev_data/local.db'. "
            "Overrides the DATABASE_URL environment variable. "
            f"Default: uses DATABASE_URL env var or '{DATABASE_URL_DEFAULT}'."
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
        default=None,
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

    # ── 1. Resolve database — Database class handles path, mkdir, and URL ──
    # Priority: --database CLI arg > DATABASE_URL env var > default.
    if args.database is not None:
        db = Database.from_path(args.database)
        log.info("Using database from --database: %s", args.database)
    else:
        db = Database.from_env()
        log.info("Using database from env/default: %s", db.url)

    # Export so the FastAPI /api/forecast/latest endpoint finds the same DB.
    db.export_to_env()

    # ── 2. Initialise database ─────────────────────────────────────────────
    repository = db.repository()
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

    # ── 4. ForecastPersister — fetch immediately + schedule hourly ─────────
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

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.start()
    persister.start(scheduler, run_immediately=False)  # already ran above
    log.info("Hourly forecast refresh scheduled.")

    # ── 4b. Optional: schedule periodic MPC ───────────────────────────────
    # When --sensors-json is provided, the MPC reads live initial conditions
    # from the JSON file at every solve interval (LocalBackend re-reads the
    # file on each call, so values updated on disk are picked up immediately).
    # Without --sensors-json the MPC uses the fixed CLI defaults as initial
    # conditions — useful for smoke-testing without real sensor data.
    if args.mpc_interval > 0:
        from .optimizer import RunRequest  # noqa: PLC0415
        from .sensors.local_backend import LocalBackend  # noqa: PLC0415

        # Build the sensor backend when a sensors.json path is supplied.
        if args.sensors_json is not None:
            sensors_path = Path(args.sensors_json).resolve()
            if not sensors_path.exists():
                log.critical("sensors-json file not found: %s — aborting MPC setup.", sensors_path)
                sys.exit(1)
            local_backend: LocalBackend | None = LocalBackend.from_json_file(sensors_path)
            log.info("MPC sensor backend: LocalBackend reading %s", sensors_path)
            # Read current outdoor temp from file to use as base_input default.
            try:
                readings = local_backend.read_all()
                t_out_init = readings.outdoor_temperature_c
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Could not read sensors.json at startup (will retry at first MPC step): %s",
                    exc,
                )
                t_out_init = args.mpc_t_out
        else:
            local_backend = None
            t_out_init = args.mpc_t_out
            log.info(
                "MPC sensor backend: none — using CLI defaults " "(T_out=%.1f C, T_ref=%.1f C).",
                t_out_init,
                args.mpc_t_ref,
            )

        # Start from Pydantic defaults, then override runtime fields immutably.
        _defaults = RunRequest.model_validate({})
        mpc_base_input = _defaults.model_copy(
            update={
                "outdoor_temperature_c": t_out_init,
                "T_ref": args.mpc_t_ref,
                "T_min": args.mpc_t_min,
                "T_max": args.mpc_t_max,
            }
        )
        optimizer = Optimizer()
        optimizer.schedule_periodic(
            base_input=mpc_base_input,
            backend=local_backend,
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
            scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass  # already stopped by SIGTERM handler — safe to ignore


if __name__ == "__main__":
    main()
